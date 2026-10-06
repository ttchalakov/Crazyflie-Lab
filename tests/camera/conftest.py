"""Builders for the deck's wire format and real captured frames, shared by the camera tests."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from cf_lab.imgstream import IMG_HEADER

DATA = Path(__file__).parent / "data"
REPO = Path(__file__).resolve().parents[2]
QVGA_PGM = REPO / "examples" / "jpeg_cluster_bench" / "frames"

# What the GAP8 sends (lib/gap8/cpx): function 5 = app, 2 = console; 1020 data bytes max.
CPX_APP = 5
CPX_CONSOLE = 2
CPX_DATA_MTU = 1020


class EndOfStream(Exception):
    """A canned packet source ran dry."""


def cpx_packet(data: bytes, *, function: int = CPX_APP, version: int = 0) -> bytes:
    """One CPX packet as it appears on the deck's TCP socket / in one UDP datagram."""
    routing = (4 << 4) | 3  # GAP8 -> WiFi host; the host ignores it
    return struct.pack("<HBB", len(data) + 2, routing, (version << 6) | function) + data


def frame_payloads(
    payload: bytes, *, width: int, height: int, type_: int = 1, ext: bytes = b"",
    size: int | None = None, depth: int = 1,
) -> list[bytes]:
    """The CPX payloads imgstream_send emits: one header(+ext) packet, then MTU chunks."""
    header = IMG_HEADER.pack(0xBC, type_, depth, len(ext), width, height,
                             len(payload) if size is None else size)
    chunks = [payload[i:i + CPX_DATA_MTU] for i in range(0, len(payload), CPX_DATA_MTU)]
    return [header + ext, *chunks]


def packet_source(packets):
    """``recv_packet`` for imgstream.frames over a list; raises EndOfStream when empty."""
    it = iter(list(packets))

    def recv_packet() -> bytes:
        try:
            return next(it)
        except StopIteration:
            raise EndOfStream from None

    return recv_packet


def byte_recv(data: bytes, *, max_part: int | None = None):
    """``recv(n)`` over bytes, like feed.socket_recv; ConnectionError at the end."""
    view = memoryview(data)
    pos = 0

    def recv(size: int) -> bytes:
        nonlocal pos
        if pos + size > len(view):
            raise ConnectionError("image stream closed")
        out = bytes(view[pos:pos + size])
        pos += size
        return out

    return recv


def read_pgm(path: Path) -> np.ndarray:
    """A binary (P5) 8-bit PGM as an (h, w) uint8 array."""
    raw = path.read_bytes()
    fields, pos = [], 0
    while len(fields) < 4:
        while raw[pos:pos + 1].isspace():
            pos += 1
        if raw[pos:pos + 1] == b"#":
            pos = raw.index(b"\n", pos) + 1
            continue
        end = pos
        while not raw[end:end + 1].isspace():
            end += 1
        fields.append(raw[pos:end])
        pos = end
    assert fields[0] == b"P5" and fields[3] == b"255"
    width, height = int(fields[1]), int(fields[2])
    return np.frombuffer(raw, np.uint8, width * height, pos + 1).reshape(height, width)


@pytest.fixture(scope="session")
def deck_frames() -> list[dict]:
    """30 consecutive real QVGA JPEG frames from a flight, with their 36-byte extension."""
    data = np.load(DATA / "deck_jpeg_frames.npz")
    blob = data["payload"].tobytes()
    frames, offset = [], 0
    for i, size in enumerate(data["size"].tolist()):
        frames.append({
            "payload": blob[offset:offset + size],
            "ext": data["ext"][i].tobytes(),
            "host_ns": int(data["host_ns"][i]),
            "frame_id": int(data["frame_id"][i]),
            "deck_ms": int(data["deck_ms"][i]),
            "width": int(data["width"]),
            "height": int(data["height"]),
        })
        offset += size
    return frames


@pytest.fixture(scope="session")
def qvga_frames() -> list[np.ndarray]:
    """Real QVGA grayscale captures (tracked in examples/jpeg_cluster_bench/frames)."""
    return [read_pgm(QVGA_PGM / f"qvga_0{i}.pgm") for i in range(4)]


class FakeClock:
    """Stands in for a module's ``time``: sleep() advances monotonic() instantly."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def monotonic_ns(self) -> int:
        return int(self.now * 1e9)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(scope="session")
def wire():
    """The wire builders above, for test modules (conftest is not importable by name)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        cpx_packet=cpx_packet, frame_payloads=frame_payloads, packet_source=packet_source,
        byte_recv=byte_recv, EndOfStream=EndOfStream, APP=CPX_APP, CONSOLE=CPX_CONSOLE,
        MTU=CPX_DATA_MTU,
    )
