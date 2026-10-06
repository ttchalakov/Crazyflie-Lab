"""Mocap test kit: NatNet wire builders, a fake Motive UDP stack, a fake clock and real flight data."""
from __future__ import annotations

import csv
import json
import queue
import socket
import struct
import sys
import time
import types
from pathlib import Path

import pytest

from cf_lab.mocap import natnet
from cf_lab.mocap.types import PoseSample, Quaternion

DATA = Path(__file__).parent / "data"
SERVER_IP = "10.0.0.5"


# -- NatNet wire format, written from the NatNet SDK packet layout -------------


def _sized(version) -> bool:
    return tuple(version) >= (4, 1)


def _cstr(text: str) -> bytes:
    return text.encode() + b"\0"


def frame_body(version, rb_id, position, quat, *, tracked=True, n_markers=2) -> bytes:
    major, minor = version
    out = struct.pack("<i3f4f", rb_id, *position, *quat)
    if 0 < major < 3:
        out += struct.pack("<i", n_markers)
        out += b"".join(struct.pack("<3f", 0.01 * i, 0.02, 0.03) for i in range(n_markers))
        if major >= 2:
            out += b"".join(struct.pack("<i", 100 + i) for i in range(n_markers))
            out += b"".join(struct.pack("<f", 0.014) for _ in range(n_markers))
    out += struct.pack("<f", 0.0003)  # mean marker error
    if major > 2 or (major == 2 and minor >= 6):
        # bit 0 = tracking valid; other bits set so only bit 0 may decide.
        out += struct.pack("<h", 0x06 | (0x01 if tracked else 0))
    return out


def frame_packet(version, frame_number, bodies, *, stamps=(1_000_000, 1_010_000, 1_040_000)) -> bytes:
    """A FrameOfData; ``bodies`` is a list of (id, position, quat, tracked)."""
    major, minor = version
    sized = _sized(version)

    def section(count: int, payload: bytes) -> bytes:
        return struct.pack("<i", count) + (struct.pack("<i", len(payload)) if sized else b"") + payload

    marker_sets = [("cf21bl_01", 4), ("all", 6)]
    sets = b"".join(
        _cstr(name) + struct.pack("<i", n) + b"".join(struct.pack("<3f", 0.1 * i, 0.2, 0.3) for i in range(n))
        for name, n in marker_sets
    )
    out = struct.pack("<i", frame_number)
    out += section(len(marker_sets), sets)
    out += section(3, b"".join(struct.pack("<3f", 1.0, 2.0, 3.0) for _ in range(3)))
    out += section(
        len(bodies),
        b"".join(frame_body(version, i, p, q, tracked=t) for i, p, q, t in bodies),
    )
    if sized:
        # Sections the parser must jump by byte size: junk payloads of odd length.
        out += section(1, b"\xa5" * 37)  # skeletons
        out += section(1, b"\x5a" * 23)  # assets
        out += section(2, b"\x11" * 52)  # labeled markers
        out += section(0, b"")  # force plates
        out += section(1, b"\x22" * 9)  # devices
    else:
        out += section(0, b"")  # skeletons
        out += section(0, b"")  # labeled markers
        out += section(0, b"")  # force plates
        out += section(0, b"")  # devices
    out += struct.pack("<IId", 0x01020304, 0, 12.5)  # timecode, sub-frame, timestamp
    if major >= 3:
        out += struct.pack("<QQQ", *stamps)
    if sized:
        out += struct.pack("<II", 1_700_000_000, 123)
    out += struct.pack("<hi", 0, 0)  # params, end-of-data tag
    return struct.pack("<HH", natnet.NAT_FRAMEOFDATA, len(out)) + out


def _rb_desc(version, name, ident, n_markers=3) -> bytes:
    major = version[0]
    out = _cstr(name) + struct.pack("<ii3f", ident, -1, 0.0, 0.0, 0.0)
    if major >= 3:
        out += struct.pack("<i", n_markers)
        out += b"".join(struct.pack("<3f", 0.01 * i, 0.0, 0.0) for i in range(n_markers))
        out += b"".join(struct.pack("<i", i + 1) for i in range(n_markers))
        if major >= 4:
            out += b"".join(_cstr(f"Marker{i + 1}") for i in range(n_markers))
    return out


def modeldef_packet(version, assets) -> bytes:
    """Model descriptions; ``assets`` items are tuples tagged by kind:
    ("markers", name), ("rb", name, id), ("skeleton", name, id, [(bone, id), ...]),
    ("force_plate",), ("device",), ("camera", name), ("raw", kind, payload).
    """
    records = []
    for asset in assets:
        tag = asset[0]
        if tag == "markers":
            kind, body = 0, _cstr(asset[1]) + struct.pack("<i", 2) + _cstr("m1") + _cstr("m2")
        elif tag == "rb":
            kind, body = 1, _rb_desc(version, asset[1], asset[2])
        elif tag == "skeleton":
            bones = asset[3]
            body = _cstr(asset[1]) + struct.pack("<ii", asset[2], len(bones))
            body += b"".join(_rb_desc(version, n, i) for n, i in bones)
            kind = 2
        elif tag == "force_plate":
            body = struct.pack("<i", 1) + _cstr("FP-0001") + struct.pack("<2f3f", 0.6, 0.4, 0, 0, 0)
            body += struct.pack("<144f", *range(144)) + struct.pack("<12f", *range(12))
            body += struct.pack("<ii", 1, 1) + struct.pack("<i", 2) + _cstr("Fx") + _cstr("Fy")
            kind = 3
        elif tag == "device":
            body = struct.pack("<i", 2) + _cstr("NIDAQ") + _cstr("SN-9") + struct.pack("<ii", 1, 1)
            body += struct.pack("<i", 1) + _cstr("ch0")
            kind = 4
        elif tag == "camera":
            kind, body = 5, _cstr(asset[1]) + struct.pack("<3f4f", 1, 2, 3, 0, 0, 0, 1)
        elif tag == "raw":
            kind, body = asset[1], asset[2]
        else:
            raise ValueError(tag)
        header = struct.pack("<i", kind) + (struct.pack("<i", len(body)) if _sized(version) else b"")
        records.append(header + body)
    payload = struct.pack("<i", len(records)) + b"".join(records)
    return struct.pack("<HH", natnet.NAT_MODELDEF, len(payload)) + payload


def server_info_packet(version, *, name="Motive", clock_hz=10_000_000) -> bytes:
    major, minor = version
    payload = name.encode().ljust(256, b"\0") + bytes([3, 1, 0, 0]) + bytes([major, minor, 0, 0])
    if major >= 3:
        payload += struct.pack("<QH?4s", clock_hz, 1511, False, socket.inet_aton("239.255.42.99"))
    return struct.pack("<HH", natnet.NAT_SERVERINFO, len(payload)) + payload


@pytest.fixture
def wire():
    return types.SimpleNamespace(
        frame=frame_packet, modeldef=modeldef_packet, server_info=server_info_packet
    )


# -- fake UDP stack standing in for Motive ------------------------------------


class FakeSocket:
    def __init__(self, net: "FakeMotive") -> None:
        self.net = net
        self.inbox: queue.Queue[bytes] = queue.Queue()
        self.bound = None
        self.peer = None
        self.options: dict = {}
        self.closed = False

    def setsockopt(self, level, option, value) -> None:
        self.options[(level, option)] = value

    def settimeout(self, _timeout) -> None:
        pass

    def bind(self, address) -> None:
        self.bound = address
        if address[1] != 0:
            self.net.data_socket = self
            for packet in self.net.frames:
                self.inbox.put(packet)

    def connect(self, address) -> None:
        self.peer = address

    def send(self, data) -> None:
        self.sendto(data, self.peer)

    def sendto(self, data, address) -> None:
        self.net.sent.append((self.bound, address, bytes(data)))
        self.net.reply(self, bytes(data), address)

    def recv(self, _size):
        return self.recvfrom(_size)[0]

    def recvfrom(self, _size):
        deadline = time.monotonic() + 0.05
        while True:
            if self.closed or (self.net.data_link_down and self is self.net.data_socket):
                raise OSError("socket closed")
            try:
                return self.inbox.get(timeout=0.005), (SERVER_IP, 1511)
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise socket.timeout("timed out") from None

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class FakeMotive:
    """Answers NatNet commands sent to (server_ip, command_port) like Motive does."""

    def __init__(self, *, server=(SERVER_IP, 1510), server_info=None, modeldefs=(), frames=()):
        self.server = server
        self.server_info = server_info
        self.modeldefs = list(modeldefs)
        self.frames = list(frames)
        self.sent: list[tuple] = []
        self.sockets: list[FakeSocket] = []
        self.data_socket: FakeSocket | None = None
        self.data_link_down = False

    def socket(self, *_args, **_kwargs) -> FakeSocket:
        sock = FakeSocket(self)
        self.sockets.append(sock)
        return sock

    def reply(self, sock: FakeSocket, data: bytes, address) -> None:
        if tuple(address) != self.server:
            return
        (message_id,) = struct.unpack_from("<H", data)
        if message_id == natnet.NAT_CONNECT and sock is not self.data_socket and self.server_info:
            sock.inbox.put(self.server_info)
        elif message_id == natnet.NAT_REQUEST_MODELDEF:
            for packet in self.modeldefs:
                sock.inbox.put(packet)

    def commands_from(self, sock: FakeSocket | None) -> list[int]:
        return [
            struct.unpack_from("<H", data)[0]
            for bound, _addr, data in self.sent
            if sock is not None and bound == sock.bound
        ]


@pytest.fixture
def fake_motive(monkeypatch):
    """Install a FakeMotive behind cf_lab.mocap.natnet's socket module; returns a builder."""

    def install(**kwargs) -> FakeMotive:
        net = FakeMotive(**kwargs)
        fake_socket = types.SimpleNamespace(
            **{name: getattr(socket, name) for name in dir(socket) if not name.startswith("__")}
        )
        fake_socket.socket = net.socket
        monkeypatch.setattr(natnet, "socket", fake_socket)
        return net

    return install


@pytest.fixture
def natnet_parser(fake_motive, wire):
    """A NatNetPoseSource connected (then closed) against a fake Motive of ``version``,
    with the target asset 'cf21bl_01' resolved to streaming id 7."""

    def build(version, *, clock_hz=10_000_000) -> natnet.NatNetPoseSource:
        fake_motive(
            server_info=wire.server_info(version, clock_hz=clock_hz),
            modeldefs=[wire.modeldef(version, [("rb", "other", 2), ("rb", "cf21bl_01", 7)])],
        )
        source = natnet.NatNetPoseSource(server_ip=SERVER_IP, rigid_body="cf21bl_01")
        source.connect()
        source.close()
        return source

    return build


# -- clocks, poses, scripted sources -------------------------------------------


@pytest.fixture
def fake_clock(monkeypatch):
    """Replace time.monotonic inside cf_lab.mocap.stream with a settable clock."""
    from cf_lab.mocap import stream

    clock = types.SimpleNamespace(t=0.0)
    monkeypatch.setattr(stream, "time", types.SimpleNamespace(monotonic=lambda: clock.t))
    return clock


def make_pose(position=(0.0, 0.0, 0.0), orientation=None, *, seq=None, t=0.0) -> PoseSample:
    return PoseSample(
        timestamp_s=t,
        position_m=tuple(position),
        orientation=orientation or Quaternion.identity(),
        rigid_body="cf21bl_01",
        source_seq=seq,
    )


class ScriptedSource:
    """samples() yields ``poses``; each item may be (arrival_s, pose) to drive a fake clock."""

    def __init__(self, items, clock=None) -> None:
        self.items = list(items)
        self.clock = clock
        self.calls: list[dict] = []

    def samples(self, *, start_s=0.0, dt_s=0.01):
        self.calls.append({"start_s": start_s, "dt_s": dt_s})
        for item in self.items:
            if self.clock is not None:
                self.clock.t, item = item
            yield item


class RecordingBackend:
    def __init__(self, *, stop_after=None, streamer=None, fail_on=None) -> None:
        self.poses: list[PoseSample] = []
        self.stop_after = stop_after
        self.streamer = streamer
        self.fail_on = fail_on

    def send_extpose(self, pose: PoseSample) -> None:
        if self.fail_on is not None and len(self.poses) == self.fail_on:
            raise ConnectionError("radio link lost")
        self.poses.append(pose)
        if self.stop_after is not None and len(self.poses) >= self.stop_after:
            self.streamer.stop()


@pytest.fixture
def kit():
    return types.SimpleNamespace(
        pose=make_pose, Source=ScriptedSource, Backend=RecordingBackend, server_ip=SERVER_IP
    )


# -- real NatNet flight (see data/make_flight_sample.py) -----------------------


@pytest.fixture(scope="session")
def flight_sample():
    folder = DATA / "flight_sample"

    def rows(name):
        with open(folder / name, newline="") as handle:
            out = []
            for row in csv.DictReader(handle):
                out.append(
                    {
                        "segment": row["segment"],
                        "host_ns": int(row["host_ns"]),
                        "seq": int(row["source_seq"]),
                        "position": tuple(float(row[k]) for k in ("x", "y", "z")),
                        "quat": Quaternion(*(float(row[k]) for k in ("qx", "qy", "qz", "qw"))),
                    }
                )
            return out

    return types.SimpleNamespace(
        mocap=rows("mocap.csv"),
        extpose=rows("extpose.csv"),
        origin=json.loads((folder / "origin.json").read_text()),
    )


# -- roslibpy stand-in (the websocket boundary) ---------------------------------


@pytest.fixture
def fake_roslibpy(monkeypatch):
    """Replace the roslibpy module; records every Ros/Topic the code opens."""
    opened = types.SimpleNamespace(ros=[], topics=[])

    class Ros:
        def __init__(self, host, port):
            self.host, self.port = host, port
            self.running = False
            opened.ros.append(self)

        def run(self):
            self.running = True

        def terminate(self):
            self.running = False

    class Topic:
        def __init__(self, ros, name, message_type, throttle_rate=0, queue_length=0):
            self.ros, self.name, self.message_type = ros, name, message_type
            self.throttle_rate, self.queue_length = throttle_rate, queue_length
            self.callback = None
            opened.topics.append(self)

        def subscribe(self, callback):
            self.callback = callback

        def unsubscribe(self):
            self.callback = None

    monkeypatch.setitem(sys.modules, "roslibpy", types.SimpleNamespace(Ros=Ros, Topic=Topic))
    return opened
