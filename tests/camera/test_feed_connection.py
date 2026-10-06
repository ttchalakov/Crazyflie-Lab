"""CameraFeed and open_stream against fake decks on loopback: retries, re-lookup, fan-out."""

import socket
import threading
import time

import pytest

from cf_lab.camera import feed, find_deck
from cf_lab.config import CameraConfig, LabConfig, RadioConfig

URI = "radio://0/80/2M/E7E7E7E7E7"


def _wait_until(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.01)


@pytest.fixture
def listener():
    """A TCP listener on 127.0.0.1, port chosen by the kernel."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    sock.settimeout(5.0)
    yield sock
    sock.close()


def _serve(listener, data: bytes, *, hold: threading.Event | None = None):
    """Accept one client, send ``data``, then close (or wait for ``hold`` first)."""
    def run():
        conn, _ = listener.accept()
        with conn:
            conn.sendall(data)
            if hold is not None:
                hold.wait(5.0)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _stream(wire, frames, *, config="camera config: fps=60 format=jpeg"):
    packets = [wire.cpx_packet(config.encode() + b"\0", function=wire.CONSOLE)]
    for f in frames:
        packets += [wire.cpx_packet(p) for p in
                    wire.frame_payloads(f["payload"], width=320, height=240, ext=f["ext"])]
    return b"".join(packets)


class ListSink:
    def __init__(self):
        self.frames, self.lines = [], []

    def offer(self, frame):
        self.frames.append(frame)

    def offer_console(self, line):
        self.lines.append(line)


class BrokenSink:
    def offer(self, frame):
        raise RuntimeError("viewer died")

    def offer_console(self, line):
        raise RuntimeError("viewer died")


def test_open_stream_retries_at_the_address_look_up_gives(listener, fake_clock, monkeypatch):
    monkeypatch.setattr(feed, "time", fake_clock)
    port = listener.getsockname()[1]
    server = _serve(listener, b"")
    looked_up = []

    def look_up():
        looked_up.append(1)
        return "127.0.0.1"

    # Nothing listens on 127.0.0.2: the deck "moved" after a reboot.
    sock, _recv = feed.open_stream("127.0.0.2", port, connect_timeout_s=5.0, look_up=look_up)
    try:
        assert sock.getpeername() == ("127.0.0.1", port)
        assert looked_up == [1] and fake_clock.sleeps == [0.5]
    finally:
        feed.close_stream(sock)
        server.join(5.0)


def test_open_stream_raises_the_connect_error_once_the_deadline_passes(listener, fake_clock,
                                                                       monkeypatch):
    monkeypatch.setattr(feed, "time", fake_clock)
    port = listener.getsockname()[1]

    with pytest.raises(ConnectionRefusedError):
        feed.open_stream("127.0.0.2", port, connect_timeout_s=2.0)
    assert sum(fake_clock.sleeps) == pytest.approx(2.0)


def test_feed_fans_out_frames_and_console_lines_despite_a_broken_sink(listener, wire,
                                                                      deck_frames):
    frames = deck_frames[:4]
    server = _serve(listener, _stream(wire, frames))
    good = ListSink()
    observed = ListSink()
    stream = feed.CameraFeed(ip="127.0.0.1", port=listener.getsockname()[1],
                             connect_timeout_s=2.0)
    stream.subscribe(BrokenSink())
    stream.subscribe(good)
    stream.add_observer(observed)
    broken_observer = BrokenSink()
    stream.add_observer(broken_observer)
    stream.start()
    try:
        assert stream.wait_for_config(5.0) == {"fps": 60, "format": "jpeg"}
        _wait_until(lambda: stream.error is not None)  # the fake deck closes after sending
    finally:
        stream.stop()
        server.join(5.0)

    assert [f.payload for f in good.frames] == [f["payload"] for f in frames]
    assert [f.payload for f in observed.frames] == [f["payload"] for f in frames]
    assert [f.ext.frame_id for f in good.frames] == [f["frame_id"] for f in frames]
    assert [line.kind for line in good.lines] == ["config"]
    assert stream.frames_received == 4
    assert stream.sink_errors == 4 + 1 and stream.observer_errors == 4
    assert isinstance(stream.error, ConnectionError)


def test_removed_observer_gets_no_more_frames(listener, wire, deck_frames):
    hold = threading.Event()
    server = _serve(listener, _stream(wire, deck_frames[:1]), hold=hold)
    observed = ListSink()
    stream = feed.CameraFeed(ip="127.0.0.1", port=listener.getsockname()[1])
    unsubscribe = stream.add_observer(observed)
    unsubscribe()
    unsubscribe()  # idempotent
    stream.start()
    try:
        assert stream.wait_for_frame(5.0)
    finally:
        hold.set()
        stream.stop()
        server.join(5.0)

    assert observed.frames == []


def test_from_config_looks_the_deck_up_again_before_retrying(listener, wire, deck_frames,
                                                             fake_clock, monkeypatch):
    real_getaddrinfo = socket.getaddrinfo
    answers = iter(["127.0.0.2", "127.0.0.1"])  # avahi's cached old address, then the new one
    asked = []

    def getaddrinfo(host, *args, **kwargs):
        if not str(host).endswith(".local"):
            return real_getaddrinfo(host, *args, **kwargs)
        asked.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), 0))]

    monkeypatch.setattr(find_deck.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(feed, "time", fake_clock)
    port = listener.getsockname()[1]
    hold = threading.Event()
    server = _serve(listener, _stream(wire, deck_frames[:1]), hold=hold)
    cfg = LabConfig(radio=RadioConfig(uri=URI),
                    camera=CameraConfig(ssid="dummy-ssid", deck_port=port))

    stream = feed.CameraFeed.from_config(cfg, connect_timeout_s=5.0)
    stream.start()
    try:
        assert stream.wait_for_frame(5.0) and stream.error is None
    finally:
        hold.set()
        stream.stop()
        server.join(5.0)

    assert asked == ["cf-80-2m-e7e7e7e7e7.local"] * 2


def test_stop_does_not_report_its_own_shutdown_as_a_stream_error(listener):
    hold = threading.Event()
    server = _serve(listener, b"", hold=hold)
    stream = feed.CameraFeed(ip="127.0.0.1", port=listener.getsockname()[1])
    stream.start()
    try:
        stream.stop()
    finally:
        hold.set()
        server.join(5.0)

    assert stream.error is None


def test_udp_feed_reannounces_to_a_deck_that_came_back_on_a_new_address(wire, deck_frames):
    old_deck = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    old_deck.bind(("127.0.0.1", 0))
    port = old_deck.getsockname()[1]
    new_deck = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    new_deck.bind(("127.0.0.2", port))
    for deck in (old_deck, new_deck):
        deck.settimeout(5.0)
    heard = {}

    def deck_sends(deck, frame, name):
        magic, host = deck.recvfrom(64)
        heard[name] = magic
        for payload in wire.frame_payloads(frame["payload"], width=320, height=240,
                                           ext=frame["ext"]):
            deck.sendto(wire.cpx_packet(payload), host)

    decks = [threading.Thread(target=deck_sends, args=(old_deck, deck_frames[0], "old")),
             threading.Thread(target=deck_sends, args=(new_deck, deck_frames[1], "new"))]
    for thread in decks:
        thread.start()
    sock, recv = feed.open_stream("127.0.0.1", port, transport="udp", connect_timeout_s=2.0,
                                  read_timeout_s=0.2, look_up=lambda: "127.0.0.2")
    try:
        got = feed.frames(recv, recover_incomplete=True)
        assert next(got).payload == deck_frames[0]["payload"]
        assert next(got).payload == deck_frames[1]["payload"]  # after the old deck went quiet
    finally:
        sock.close()
        for thread in decks:
            thread.join(5.0)
        old_deck.close()
        new_deck.close()

    assert heard == {"old": b"FER", "new": b"FER"}


def test_udp_feed_drops_only_the_frame_that_lost_a_datagram(wire, deck_frames):
    deck = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    deck.bind(("127.0.0.1", 0))
    deck.settimeout(5.0)
    datagrams = [wire.cpx_packet(b"camera config: fps=60 transport=udp\0",
                                 function=wire.CONSOLE)]
    for i, f in enumerate(deck_frames[:3]):
        packets = wire.frame_payloads(f["payload"], width=320, height=240, ext=f["ext"])
        if i == 1:
            del packets[2]  # one payload datagram of frame 1 never arrives
        datagrams += [wire.cpx_packet(p) for p in packets]

    def deck_sends():
        _magic, host = deck.recvfrom(64)
        for datagram in datagrams:
            deck.sendto(datagram, host)

    sender = threading.Thread(target=deck_sends, daemon=True)
    sender.start()
    sink = ListSink()
    stream = feed.CameraFeed(ip="127.0.0.1", port=deck.getsockname()[1], transport="udp",
                             connect_timeout_s=2.0, read_timeout_s=5.0)
    stream.subscribe(sink)
    stream.start()
    try:
        _wait_until(lambda: len(sink.frames) >= 2 or stream.error is not None)
        error = stream.error
    finally:
        stream.stop()
        sender.join(5.0)
        deck.close()

    assert error is None
    assert [f.payload for f in sink.frames] == [deck_frames[0]["payload"],
                                                deck_frames[2]["payload"]]
    assert stream.deck_config == {"fps": 60, "transport": "udp"}
