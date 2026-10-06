"""CPX framing, console demux and the TCP/UDP byte readers under cf_lab.camera.feed."""

import random
import socket

import pytest

from cf_lab.camera import feed
from cf_lab.camera.ext import CameraStreamExt
from cf_lab.imgstream import StreamDesync


def _frame_wire(wire, frames, *, console_after=None):
    """Bytes of a TCP stream carrying ``frames``; ``console_after`` maps packet index -> text."""
    out, index = [], 0
    for f in frames:
        for payload in wire.frame_payloads(f["payload"], width=320, height=240, ext=f["ext"]):
            out.append(wire.cpx_packet(payload))
            if console_after and index in console_after:
                out.append(wire.cpx_packet(console_after[index].encode() + b"\0",
                                           function=wire.CONSOLE))
            index += 1
    return b"".join(out)


class ChunkySocket:
    """A stream socket that returns at most a few random bytes per recv()."""

    def __init__(self, data: bytes, seed: int = 0) -> None:
        self.data = memoryview(data)
        self.pos = 0
        self.rng = random.Random(seed)

    def recv(self, n: int) -> bytes:
        take = min(n, self.rng.randint(1, 7), len(self.data) - self.pos)
        out = bytes(self.data[self.pos:self.pos + take])
        self.pos += take
        return out


class FakeUdpSocket:
    """recvfrom() replays a script of datagrams and timeouts; sendto() is recorded."""

    def __init__(self, script) -> None:
        self.script = list(script)
        self.sent = []

    def recvfrom(self, _size):
        item = self.script.pop(0) if self.script else socket.timeout()
        if isinstance(item, BaseException):
            raise item
        return item, ("192.0.2.1", 5000)

    def sendto(self, data, address):
        self.sent.append((data, address))


def test_cpx_payloads_diverts_console_text_and_returns_image_packets(wire):
    lines = []
    stream = b"".join([
        wire.cpx_packet(b"\xbc image"),
        wire.cpx_packet(b"status frames=3\r\n\0garbage", function=wire.CONSOLE),
        wire.cpx_packet(b"x"),  # a one-byte tail chunk: CPX length 3
        wire.cpx_packet(b"hello\0", function=wire.CONSOLE, version=1),
    ])
    next_payload = feed.cpx_payloads(wire.byte_recv(stream), lines.append)

    assert [next_payload(), next_payload()] == [b"\xbc image", b"x"]
    with pytest.raises(ConnectionError):
        next_payload()
    assert lines == ["status frames=3", "hello"]


@pytest.mark.parametrize("length", [0, 1, 1023, 0xFFFF])
def test_cpx_payloads_rejects_an_implausible_length_without_reading_it(length):
    calls = []

    def recv(n):
        calls.append(n)
        return length.to_bytes(2, "little") + b"\x43\x05"

    with pytest.raises(StreamDesync, match="CPX length"):
        feed.cpx_payloads(recv)()
    assert calls == [4]


def test_frames_decodes_real_stream_with_console_lines_between_chunks(wire, deck_frames):
    lines = []
    data = _frame_wire(wire, deck_frames[:5], console_after={0: "camera config: fps=60",
                                                             6: "status frames=2"})
    got = feed.frames(wire.byte_recv(data), on_console=lines.append)

    decoded = [next(got) for _ in range(5)]

    assert [d.payload for d in decoded] == [f["payload"] for f in deck_frames[:5]]
    assert [d.raw_ext for d in decoded] == [f["ext"] for f in deck_frames[:5]]
    assert all(isinstance(d.ext, CameraStreamExt) for d in decoded)
    assert [d.ext.frame_id for d in decoded] == [f["frame_id"] for f in deck_frames[:5]]
    assert lines == ["camera config: fps=60", "status frames=2"]


def test_socket_recv_reassembles_partial_reads_exactly(wire, deck_frames):
    data = _frame_wire(wire, deck_frames[:3])
    recv = feed.socket_recv(ChunkySocket(data, seed=3))

    got = feed.frames(recv)

    assert [next(got).payload for _ in range(3)] == [f["payload"] for f in deck_frames[:3]]
    with pytest.raises(ConnectionError):
        next(got)


def test_udp_recv_lost_datagram_drops_one_frame_and_keeps_the_stream(wire, deck_frames):
    datagrams = [wire.cpx_packet(p) for f in deck_frames[:3]
                 for p in wire.frame_payloads(f["payload"], width=320, height=240, ext=f["ext"])]
    del datagrams[7]  # inside frame 1 (each frame here is 5 datagrams)
    sock = FakeUdpSocket(datagrams[1:])
    recv = feed.udp_recv(sock, ("192.0.2.1", 5000), datagrams[0])

    got = feed.frames(recv, recover_incomplete=True)

    assert [next(got).payload for _ in range(2)] == [deck_frames[0]["payload"],
                                                     deck_frames[2]["payload"]]


def test_udp_recv_reannounces_to_the_freshly_looked_up_address(wire):
    addresses = iter(["10.0.0.7", "10.0.0.9"])
    sock = FakeUdpSocket([socket.timeout(), socket.timeout(), wire.cpx_packet(b"data")])
    recv = feed.udp_recv(sock, ("10.0.0.5", 5000), b"", look_up=lambda: next(addresses))

    assert recv(4 + 4)[4:] == b"data"
    assert sock.sent == [(b"FER", ("10.0.0.7", 5000)), (b"FER", ("10.0.0.9", 5000))]


def test_udp_recv_keeps_the_old_address_when_the_lookup_fails():
    def look_up():
        raise OSError("no deck answers")

    sock = FakeUdpSocket([socket.timeout(), b"abcd"])
    recv = feed.udp_recv(sock, ("10.0.0.5", 5000), b"", look_up=look_up)

    assert recv(4) == b"abcd"
    assert sock.sent == [(b"FER", ("10.0.0.5", 5000))]


def test_udp_recv_gives_up_after_three_quiet_timeouts():
    sock = FakeUdpSocket([socket.timeout()] * 3)
    recv = feed.udp_recv(sock, ("10.0.0.5", 5000), b"")

    with pytest.raises(ConnectionError, match="quiet"):
        recv(4)
    assert len(sock.sent) == 2


def test_captured_raw_frame_with_wrong_size_decodes_to_none(qvga_frames):
    from cf_lab.imgstream import ImgHeader

    image = qvga_frames[0]
    good = feed.CapturedFrame(ImgHeader(0xBC, 0, 1, 0, 320, 240, image.size),
                              image.tobytes(), None, b"", 0)
    short = feed.CapturedFrame(ImgHeader(0xBC, 0, 1, 0, 320, 240, image.size - 1),
                               image.tobytes()[:-1], None, b"", 0)

    assert (good.decode() == image).all()
    assert short.decode() is None
