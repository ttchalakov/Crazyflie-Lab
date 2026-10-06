"""Example-local wire protocol: one checksummed benchmark record per CPX packet."""

from __future__ import annotations

import socket
import struct
import time
import zlib
from dataclasses import dataclass

HEADER = struct.Struct("<4sBBHIIII")
PING, PONG, START, STOP, DATA, STATS = range(1, 7)
TO_GAP8 = bytes((4 | (3 << 3) | 0x40, 5))
FROM_GAP8 = bytes((3 | (4 << 3) | 0x40, 5))


def record(kind, session, sequence=0, body=b""):
    header = HEADER.pack(b"WFB1", kind, 1, HEADER.size + len(body), session, sequence, 0, 0)
    crc = zlib.crc32(body, zlib.crc32(header[:20]))
    return header[:20] + struct.pack("<I", crc) + body


def wire(payload):
    return struct.pack("<H", len(payload) + 2) + TO_GAP8 + payload


@dataclass
class Packet:
    kind: int
    session: int
    sequence: int
    tick_us: int
    body: bytes


def decode(payload):
    if len(payload) < HEADER.size:
        raise ValueError("short benchmark header")
    magic, kind, version, size, session, seq, tick, crc = HEADER.unpack_from(payload)
    if magic != b"WFB1" or version != 1 or size != len(payload):
        raise ValueError("invalid benchmark header")
    body = payload[HEADER.size :]
    if zlib.crc32(body, zlib.crc32(payload[:20])) != crc:
        raise ValueError("benchmark checksum mismatch")
    if kind == DATA:
        pattern = bytes((seq + i) & 255 for i in range(HEADER.size, HEADER.size + 256))
        if body != (pattern * 4)[: len(body)]:
            raise ValueError("synthetic payload mismatch")
    return Packet(kind, session, seq, tick, body)


class Link:
    def __init__(self, ip, port, transport, timeout=5):
        self.transport = transport
        self.target = (socket.gethostbyname(ip), port)
        self.buffer = bytearray()
        self.sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM if transport == "udp" else socket.SOCK_STREAM
        )
        try:
            self.sock.settimeout(timeout)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            self.sock.connect(self.target)
            if transport == "udp":
                self.sock.send(b"FER")
            else:
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except BaseException:
            self.sock.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self, reset=False):
        if reset and self.transport == "tcp":
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        self.sock.close()

    def send(self, payload):
        self.raw(wire(payload))

    def raw(self, data):
        self.sock.settimeout(2)
        self.sock.sendall(data)

    def receive(self, timeout=2):
        deadline = time.monotonic() + timeout
        while True:
            if self.transport == "tcp" and len(self.buffer) >= 2:
                length = struct.unpack_from("<H", self.buffer)[0]
                if not 2 <= length <= 1022:
                    raise ValueError(f"TCP framing lost: length {length}")
                if len(self.buffer) >= length + 2:
                    data = bytes(self.buffer[: length + 2])
                    del self.buffer[: length + 2]
                    if data[2:4] == FROM_GAP8:
                        return decode(data[4:])
                    continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("waiting for benchmark response")
            self.sock.settimeout(remaining)
            data = self.sock.recv(65535)
            if self.transport == "tcp":
                if not data:
                    raise ConnectionError("TCP peer closed")
                self.buffer.extend(data)
            else:
                if len(data) < 4 or struct.unpack_from("<H", data)[0] != len(data) - 2:
                    raise ValueError("invalid UDP CPX framing")
                if data[2:4] == FROM_GAP8:
                    return decode(data[4:])

    def request(self, kind, session, sequence=0, body=b"", timeout=3, on_packet=None):
        self.send(record(kind, session, sequence, body))
        deadline = time.monotonic() + timeout
        expected = PONG if kind == PING else STATS
        while time.monotonic() < deadline:
            packet = self.receive(deadline - time.monotonic())
            if (packet.kind, packet.session, packet.sequence) == (expected, session, sequence):
                return packet
            if on_packet is not None:
                on_packet(packet)
        raise TimeoutError("request timed out")
