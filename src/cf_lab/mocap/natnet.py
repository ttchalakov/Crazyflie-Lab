"""Pure-Python NatNet pose source for OptiTrack/Motive — no ROS, no deps.

Motive streams rigid-body pose over its native NatNet protocol (UDP). This
module joins the NatNet stream directly with the standard library and yields
``PoseSample`` objects, so the host can read OptiTrack pose without
``vrpn_client_ros`` + ``rosbridge``. It is a drop-in replacement for
:class:`RosbridgePoseSource`.

Conventions: the stream is passed through unchanged, so set Motive's streaming
**Up Axis to Z** (Data Streaming pane) — exactly the gotcha the VRPN path also
relies on — and the frame is already +z up in metres to match the estimator.

Only the NatNet 4.2 bitstream is parsed. Motive's output bitstream is one
Motive-wide setting that any client can change (a ROS driver may leave it at
4.0, for unicast and multicast alike), and its server info reports 4.2 whatever
it is set to, so connect() asks Motive for 4.2 every time.

Both modes query Motive's command port to resolve the asset name before reading
poses. ``use_multicast`` must match Motive's Transmission Type, which connect()
checks against the server info. Multicast data needs no registration; unicast
registers for streaming.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time
from collections.abc import Iterator
from queue import Queue

from cf_lab.config import NatNetConfig
from cf_lab.mocap.types import PoseSample, Quaternion

# NatNet message ids (header is "<HH" = message_id, payload_byte_count).
NAT_CONNECT = 0
NAT_SERVERINFO = 1
NAT_REQUEST = 2
NAT_RESPONSE = 3
NAT_REQUEST_MODELDEF = 4
NAT_MODELDEF = 5
NAT_FRAMEOFDATA = 7
NAT_KEEPALIVE = 10


def _rigid_body_descriptions(data: bytes) -> list[tuple[str, int]]:
    """Read standalone asset names/IDs from NatNet 4.2 model descriptions, where
    each record is prefixed with its kind and byte size."""
    offset = 0

    def skip(size: int) -> None:
        nonlocal offset
        if size < 0 or offset + size > len(data):
            raise ValueError("Truncated NatNet model description")
        offset += size

    def integer() -> int:
        start = offset
        skip(4)
        return struct.unpack_from("<i", data, start)[0]

    def count() -> int:
        value = integer()
        if value < 0 or value > len(data):
            raise ValueError("Invalid NatNet description count")
        return value

    def string() -> str:
        start = offset
        end = data.find(b"\0", start)
        if end < 0:
            raise ValueError("Unterminated NatNet description name")
        skip(end - start + 1)
        return data[start:end].decode("utf-8")

    bodies = []
    for _ in range(count()):
        kind = integer()
        size = count()
        end = offset + size
        if kind == 1:  # a rigid body; skeleton bones are not standalone assets
            bodies.append((string(), integer()))
        skip(end - offset)  # the rest of the record, whatever its kind
    return bodies


class NatNetPoseSource:
    """Read rigid-body pose straight from a Motive NatNet stream."""

    def __init__(
        self,
        *,
        server_ip: str,
        rigid_body: str = "cf21bl_01",
        multicast_ip: str = NatNetConfig.multicast_ip,
        command_port: int = NatNetConfig.command_port,
        data_port: int = NatNetConfig.data_port,
        local_ip: str = "",
        use_multicast: bool = NatNetConfig.use_multicast,
    ) -> None:
        self._server_ip = server_ip
        self._rigid_body = rigid_body
        if not rigid_body.strip():
            raise ValueError("mocap.rigid_body must be a Motive asset name")
        self._rigid_id: int | None = None
        self._multicast_ip = multicast_ip
        self._command_port = command_port
        self._data_port = data_port
        self._local_ip = local_ip or "0.0.0.0"
        self._use_multicast = use_multicast
        # Only the newest pose; None is close()'s wake-up for a waiting consumer.
        self._queue: Queue[PoseSample | None] = Queue(maxsize=1)
        self._data_sock: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.server_name = ""
        # From Motive's server info: its high-resolution clock, for frame latency,
        # and whether it streams multicast (its Transmission Type).
        self._clock_hz = 0
        self._server_multicast = False
        self._arrival_ns: int | None = None

    # -- lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        self._rigid_id = None
        self._probe_server_info()
        if not self.server_name:
            raise RuntimeError("Cannot query Motive: check the NatNet server IP and command port")
        if self._server_multicast != self._use_multicast:
            mode, flag = ("multicast", "true") if self._server_multicast else ("unicast", "false")
            raise ValueError(
                f"Motive is streaming {mode}: set mocap.natnet.use_multicast to {flag} "
                "to match Motive's Transmission Type"
            )
        self._request_bitstream()
        self._rigid_id = self._resolve_rigid_body()
        self._data_sock = self._open_data_socket()
        if not self._use_multicast:
            # Unicast: register with Motive (NAT_CONNECT) so it starts streaming
            # frames to us; the recv loop then keeps the registration alive with
            # periodic keepalives. (Multicast needs no handshake at all.)
            self._send_command(NAT_CONNECT)
            self._send_command(NAT_KEEPALIVE)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._recv_loop, name="natnet-recv", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._data_sock is not None:
            try:
                self._data_sock.close()
            finally:
                self._data_sock = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._publish(None)  # wake a consumer waiting in samples()

    def __enter__(self) -> "NatNetPoseSource":
        self.connect()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def samples(self, *, start_s: float = 0.0, dt_s: float = 0.01) -> Iterator[PoseSample]:
        del start_s, dt_s
        if self._thread is None:
            self.connect()
        while not self._stop.is_set():
            pose = self._queue.get()
            if pose is not None:
                yield pose

    def connected(self, within_s: float) -> bool:
        """Whether Motive sent anything in the last ``within_s``.

        Motive streams a frame per camera frame even while the rigid body is
        untracked, so this tells an untracked body from a lost connection.
        """
        thread = self._thread
        arrival_ns = self._arrival_ns
        return (
            thread is not None
            and thread.is_alive()
            and arrival_ns is not None
            and time.monotonic_ns() - arrival_ns <= within_s * 1e9
        )

    # -- sockets -------------------------------------------------------------

    def _open_data_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        if self._use_multicast:
            sock.bind(("", self._data_port))
            mreq = socket.inet_aton(self._multicast_ip) + socket.inet_aton(self._local_ip)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        else:
            sock.bind((self._local_ip, self._data_port))
        sock.settimeout(1.0)
        return sock

    def _probe_server_info(self) -> None:
        """Ping the command port to confirm reachability, and read Motive's clock and mode.

        connect() requires a successful reply before decoding model descriptions.
        """
        self.server_name = ""
        try:
            cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            cmd.settimeout(0.75)
            with cmd:
                cmd.bind((self._local_ip, 0))
                cmd.sendto(struct.pack("<HH", NAT_CONNECT, 0), (self._server_ip, self._command_port))
                data, _ = cmd.recvfrom(2048)
            message_id = struct.unpack("<H", data[0:2])[0]
            if message_id == NAT_SERVERINFO:
                payload = data[4:]
                self.server_name = payload[0:256].split(b"\0", 1)[0].decode("utf-8", "replace")
                (self._clock_hz,) = struct.unpack_from("<Q", payload, 264)
                self._server_multicast = bool(payload[274])
        except OSError:
            pass  # connect() reports failure before starting the receive task.

    def _request_bitstream(self) -> None:
        """Ask Motive to stream the NatNet 4.2 bitstream, the one parsed here.

        Motive applies the request to its whole output, in either Transmission
        Type, but in unicast it takes it only from a connected client, so the
        socket connects (NAT_CONNECT) first. A description request sent the
        instant Motive accepts can still be answered in the old bitstream (3 of 20
        tries; never one sent a millisecond later), so wait briefly before the
        asset lookup.
        """
        command = b"Bitstream,4.2\0"
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as cmd:
            cmd.bind((self._local_ip, 0))
            cmd.connect((self._server_ip, self._command_port))
            cmd.settimeout(0.75)
            cmd.send(struct.pack("<HH", NAT_CONNECT, 0))
            cmd.send(struct.pack("<HH", NAT_REQUEST, len(command)) + command)
            while struct.unpack_from("<H", reply := cmd.recv(65535))[0] != NAT_RESPONSE:
                pass  # the server info answering NAT_CONNECT
            (result,) = struct.unpack_from("<i", reply, 4)
        if result != 0:
            raise RuntimeError("Motive did not accept the request for the NatNet 4.2 bitstream")
        time.sleep(0.1)

    def _resolve_rigid_body(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as cmd:
            cmd.bind((self._local_ip, 0))
            cmd.connect((self._server_ip, self._command_port))
            for _ in range(3):
                cmd.send(struct.pack("<HH", NAT_REQUEST_MODELDEF, 0))
                deadline = time.monotonic() + 0.75
                while (remaining := deadline - time.monotonic()) > 0:
                    cmd.settimeout(remaining)
                    try:
                        packet = cmd.recv(65535)
                    except socket.timeout:
                        break
                    if len(packet) < 4:
                        continue
                    kind, size = struct.unpack_from("<HH", packet)
                    if kind != NAT_MODELDEF:
                        continue
                    if len(packet) != size + 4:
                        raise ValueError("Truncated NatNet model descriptions")
                    bodies = _rigid_body_descriptions(packet[4:])
                    matches = [ident for name, ident in bodies if name == self._rigid_body]
                    if len(matches) == 1:
                        return matches[0]
                    names = ", ".join(name for name, _ in bodies) or "(none)"
                    raise ValueError(
                        f"Motive rigid body {self._rigid_body!r} must match exactly one asset; "
                        f"found {len(matches)}. Available rigid bodies: {names}"
                    )
        raise RuntimeError("Motive did not return model descriptions for rigid-body name lookup")

    def _send_command(self, message_id: int) -> None:
        """Send a header-only NatNet command (e.g. NAT_CONNECT/NAT_KEEPALIVE)."""
        sock = self._data_sock
        if sock is None:
            return
        try:
            sock.sendto(
                struct.pack("<HH", message_id, 0), (self._server_ip, self._command_port)
            )
        except OSError:
            pass

    # -- receive + parse -----------------------------------------------------

    def _recv_loop(self) -> None:
        last_keepalive = 0.0
        while not self._stop.is_set():
            if not self._use_multicast and time.monotonic() - last_keepalive > 1.0:
                self._send_command(NAT_KEEPALIVE)
                last_keepalive = time.monotonic()
            sock = self._data_sock
            if sock is None:
                return
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            self._arrival_ns = time.monotonic_ns()
            pose = self._pose_from_packet(data)
            if pose is not None:
                self._publish(pose)

    def _publish(self, pose: PoseSample | None) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Exception:
                pass
        self._queue.put_nowait(pose)

    def _pose_from_packet(self, data: bytes) -> PoseSample | None:
        if len(data) < 4:
            return None
        message_id, _size = struct.unpack("<HH", data[0:4])
        if message_id != NAT_FRAMEOFDATA:
            return None
        try:
            (frame_number,) = struct.unpack_from("<i", data, 4)
            rb = self._first_matching_rigid_body(data, 4)
        except (struct.error, IndexError, ValueError):
            return None
        if rb is None:
            return None
        _rb_id, position, quat = rb
        # A corrupt frame must neither reach the drone nor end this thread, which
        # a zero quaternion's normalization would.
        if not all(map(math.isfinite, (*position, *quat))) or not any(quat):
            return None
        return PoseSample(
            timestamp_s=time.time(),
            position_m=position,
            orientation=Quaternion(qx=quat[0], qy=quat[1], qz=quat[2], qw=quat[3]).normalized(),
            rigid_body=self._rigid_body,
            source_seq=frame_number,
            host_ns=self._arrival_ns,
            latency_s=self._latency_s(data),
        )

    def _latency_s(self, data: bytes) -> float | None:
        """Motive's mid-exposure-to-transmit time for this frame, or None.

        The frame suffix holds both stamps on Motive's high-resolution clock. It
        ends every frame at a fixed size, so the three stamps (mid-exposure,
        received, transmit) start 38 bytes before the end of the packet, ahead of
        the precision timestamp (two uint32), the frame params (int16) and the
        end-of-data tag (int32).
        """
        mid_exposure, _received, transmit = struct.unpack_from("<QQQ", data, len(data) - 38)
        latency = (transmit - mid_exposure) / self._clock_hz
        return latency if 0.0 <= latency < 1.0 else None

    def _first_matching_rigid_body(self, data: bytes, offset: int):
        """Parse a FrameOfData far enough to return one rigid body.

        Returns ``(id, (x,y,z), (qx,qy,qz,qw))`` for the configured streaming id
        resolved from the configured asset name, else ``None``.
        Untracked bodies (Motive streams them as a zero pose) are skipped so we
        never forward a stale origin pose.
        """
        offset += 4  # frame number (int32)

        # Jump the marker-set and legacy "other" marker sections: NatNet 4.2
        # prefixes each section with (count, byte_size).
        offset = self._skip_sized_section(data, offset)  # marker sets
        offset = self._skip_sized_section(data, offset)  # other markers
        (n_rigid,) = struct.unpack_from("<i", data, offset)
        offset += 8  # rigid-body count + block byte_size (we parse items)

        for _ in range(n_rigid):
            (rb_id,) = struct.unpack_from("<i", data, offset)
            offset += 4
            x, y, z, qx, qy, qz, qw = struct.unpack_from("<7f", data, offset)
            offset += 28
            offset += 4  # mean marker error (float)
            (params,) = struct.unpack_from("<h", data, offset)
            offset += 2
            tracked = bool(params & 0x01)
            if not tracked:
                continue
            rb = (rb_id, (x, y, z), (qx, qy, qz, qw))
            if rb_id == self._rigid_id:
                return rb
        return None

    @staticmethod
    def _skip_sized_section(data: bytes, offset: int) -> int:
        """Skip a NatNet 4.2 section: int32 count, int32 byte_size, then data."""
        (byte_size,) = struct.unpack_from("<i", data, offset + 4)
        return offset + 8 + byte_size
