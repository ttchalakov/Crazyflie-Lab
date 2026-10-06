"""One socket, one decode, many sinks.

The AI-deck's ESP32 does ``listen(serverSock, 1)`` and keeps a single
``clientConnection`` (external/aideck-esp-firmware/main/wifi.c). There is
exactly one client, ever. A viewer and a telemetry recorder therefore cannot
each open their own connection -- the second is refused, or steals the first.
So this module owns the only connection and fans decoded frames out locally.

That constraint turns out to be the good design anyway. The frame payload is
immutable ``bytes`` handed to every sink without copying, and nothing decodes an
image here: the recorder stores frames as they arrived, and the viewer decodes
only the frames it actually paints. Decode cost tracks what a human looks at,
not what the deck sends.

Sinks are deliberately asymmetric, which is why one shared queue would not do:

* a recorder wants *every* frame, so it queues and counts what it drops;
* a viewer wants only the *newest* frame, so it keeps one slot and silently
  overwrites. A viewer showing a stale frame is worse than one skipping ahead.

The receive thread must never block on a sink. ``offer()`` is the whole contract.

The deck also writes console lines onto the same socket (CPX console packets,
between frames): camera_stream describes its build in one "config" line when a
client connects, and with LOG=1 sends a "status" line every second. They are
parsed here (``cf_lab.camera.ext.parse_deck_line``), kept as
``CameraFeed.deck_config``, and handed to any sink with an ``offer_console``.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterator, Protocol

from cf_lab.camera.ext import DeckLine, ExtDecoder, FrameExt, decode_deck_ext, parse_deck_line
from cf_lab.camera.find_deck import resolve_deck
from cf_lab.config import LabConfig
from cf_lab.imgstream import ImgHeader, StreamDesync
from cf_lab.imgstream import frames as img_frames

CPX_HEADER = struct.Struct("<HBB")  # length, routing, function
# The function byte's low 6 bits (the top 2 are the CPX version). Console packets
# carry one NUL-terminated text line each; everything else is image stream.
CPX_FUNCTION_MASK = 0x3F
CPX_FUNCTION_CONSOLE = 2
# The GAP8 CPX length field counts the 2 packed routing bytes plus payload, but
# not the 2-byte length field itself. A one-byte JPEG tail chunk is a valid
# CPX length of 3.
CPX_ROUTE_BYTES = 2
# Max CPX packet length on the wire (MTU in lib/gap8/cpx/inc/com.h). Anything
# larger means we are reading mis-aligned bytes, not a real packet.
CPX_MTU = 1022

# The deck learns the host's address from the source of this datagram.
UDP_MAGIC = b"FER"
DEFAULT_READ_TIMEOUT_S = 15.0


def decode_frame(payload: bytes, width: int, height: int, is_jpeg: bool):
    """Decode a frame's payload to an ``(h, w)`` uint8 ndarray, or None if undecodable.

    Costs real work (a JPEG decode, or a reshape for raw), so only callers that
    actually look at pixels should call it -- never the recorder.
    """
    import numpy as np

    if not is_jpeg:
        flat = np.frombuffer(payload, dtype=np.uint8)
        if flat.size != width * height:
            return None
        return flat.reshape(height, width)

    import cv2

    return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)


@dataclass(frozen=True)
class CapturedFrame:
    """One frame as it arrived, plus the host clock it arrived on.

    ``payload`` is exactly the bytes the deck sent -- JPEG or raw grayscale, at
    whatever resolution the GAP8 was compiled for. It is never re-encoded or
    resized on the way to disk. ``host_ns`` is ``time.monotonic_ns()`` at
    decode, the same clock every other telemetry stream is stamped with, which
    is what lets camera frames line up with state/mocap rows.
    """

    header: ImgHeader
    payload: bytes
    # Decoded extension, or None when this app sends one this feed cannot read.
    ext: FrameExt | None
    # The same extension undecoded. Recorded verbatim so an app's timings
    # survive to analysis even when nothing here knows how to read them.
    raw_ext: bytes
    host_ns: int

    @property
    def width(self) -> int:
        return self.header.width

    @property
    def height(self) -> int:
        return self.header.height

    @property
    def is_jpeg(self) -> bool:
        return self.header.is_jpeg

    def decode(self):
        """Decode to an ``(h, w)`` uint8 ndarray, or None; see :func:`decode_frame`."""
        return decode_frame(self.payload, self.width, self.height, self.is_jpeg)


class FrameSink(Protocol):
    """A consumer of frames. ``offer`` runs on the receive thread.

    It MUST NOT block: no disk I/O, no GUI call, no lock held by a slow thread.
    Hand the frame to a queue or a slot and return. A sink may also define
    ``offer_console(line: DeckLine)``, called the same way for each console line.
    """

    def offer(self, frame: CapturedFrame) -> None: ...


def socket_recv(sock) -> Callable[[int], bytes]:
    """``recv(n)`` returning exactly n bytes from a stream socket."""

    def recv(size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            part = sock.recv(size - len(data))
            if not part:
                raise ConnectionError("image stream closed")
            data.extend(part)
        return bytes(data)

    return recv


def udp_recv(sock, deck_addr, primed: bytes, look_up=None) -> Callable[[int], bytes]:
    """``recv(n)`` over UDP.

    Each datagram is one whole CPX wire packet, so concatenating datagrams
    reproduces the TCP byte stream: a lost datagram removes whole packets and
    leaves the parser packet-aligned. On a receive timeout the magic datagram is
    re-sent (the deck forgets us if it reboots), to the address ``look_up`` gives
    now if there is one (it may come back on a new one); a quiet link fails only
    after several timeouts.
    """
    ip, port = deck_addr
    buf = bytearray(primed)

    def recv(size: int) -> bytes:
        nonlocal ip
        timeouts = 0
        while len(buf) < size:
            try:
                data, _addr = sock.recvfrom(4096)
            except socket.timeout:
                timeouts += 1
                if timeouts >= 3:
                    raise ConnectionError("UDP stream went quiet") from None
                ip = _look_up_again(look_up, ip)
                sock.sendto(UDP_MAGIC, (ip, port))
                continue
            buf.extend(data)
        out = bytes(buf[:size])
        del buf[:size]
        return out

    return recv


def close_stream(sock) -> None:
    """Close cleanly (FIN) so the deck frees us as its client; never RST."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def open_stream(
    ip: str,
    port: int,
    *,
    transport: str = "tcp",
    connect_timeout_s: float = 20.0,
    read_timeout_s: float = DEFAULT_READ_TIMEOUT_S,
    look_up: Callable[[], str] | None = None,
    log=lambda _msg: None,
):
    """Connect to the deck's stream. Returns ``(sock, recv)``.

    Right after the deck boots (or after joining its AP) the server may not be
    up yet, so the connection is retried for ``connect_timeout_s``. Before each
    retry ``look_up``, if given, is asked for the deck's address again: a
    rebooted deck can come back on a new one.
    """
    if transport == "udp":
        sock, primed, ip = _open_udp(
            ip, port, connect_timeout_s=connect_timeout_s,
            read_timeout_s=read_timeout_s, look_up=look_up, log=log,
        )
        return sock, udp_recv(sock, (ip, port), primed, look_up=look_up)

    deadline = time.monotonic() + connect_timeout_s
    announced = False
    while True:
        try:
            sock = socket.create_connection((ip, port), timeout=5.0)
            sock.settimeout(read_timeout_s)
            return sock, socket_recv(sock)
        except OSError:
            if time.monotonic() >= deadline:
                raise
            if not announced:
                announced = True
                log(f"Deck not answering yet; retrying for up to {connect_timeout_s:.0f}s ...")
            ip = _look_up_again(look_up, ip)
            time.sleep(0.5)


def _look_up_again(look_up, ip: str) -> str:
    """The address ``look_up`` gives now; ``ip`` without one, or if the deck does not answer."""
    if look_up is None:
        return ip
    try:
        return look_up()
    except OSError:
        return ip


def _open_udp(ip, port, *, connect_timeout_s, read_timeout_s, look_up, log):
    """Teach the deck our address, wait for first data, return it unconsumed.

    Returns ``(sock, data, ip)``: ``ip`` is the address that answered.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # A raw QVGA frame is ~76 datagrams and the deck bursts them back to back.
    # Grow the receive buffer so a scheduling hiccup here does not make the
    # kernel drop whole frames' worth of datagrams; UDP has no retransmit.
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    except OSError:
        pass
    sock.settimeout(1.0)
    deadline = time.monotonic() + connect_timeout_s
    announced = False
    while True:
        try:
            sock.sendto(UDP_MAGIC, (ip, port))
            data, _addr = sock.recvfrom(4096)
            sock.settimeout(read_timeout_s)
            return sock, data, ip
        except OSError:
            if time.monotonic() >= deadline:
                sock.close()
                raise
            if not announced:
                announced = True
                log(f"Deck not streaming yet; retrying for up to {connect_timeout_s:.0f}s ...")
            ip = _look_up_again(look_up, ip)
            time.sleep(0.5)


def cpx_payloads(
    recv: Callable[[int], bytes],
    on_console: Callable[[str], None] | None = None,
) -> Callable[[], bytes]:
    """Adapt a byte-stream ``recv(n)`` into "one whole image-stream CPX payload per call".

    Console packets never reach the image decoder: each one's text goes to
    ``on_console``, or is dropped.
    """

    def next_payload() -> bytes:
        while True:
            length, _routing, function = CPX_HEADER.unpack(recv(CPX_HEADER.size))
            if not CPX_ROUTE_BYTES <= length <= CPX_MTU:
                # CPX framing itself is lost -- never blindly recv() a huge length.
                raise StreamDesync(f"implausible CPX length {length}")
            data = recv(length - CPX_ROUTE_BYTES)
            if function & CPX_FUNCTION_MASK != CPX_FUNCTION_CONSOLE:
                return data
            if on_console is not None:
                on_console(data.split(b"\0", 1)[0].decode("utf-8", "replace").rstrip())

    return next_payload


def frames(
    recv: Callable[[int], bytes],
    *,
    ext_decoder: ExtDecoder | None = decode_deck_ext,
    recover_incomplete: bool = False,
    on_console: Callable[[str], None] | None = None,
) -> Iterator[CapturedFrame]:
    """Yield ``CapturedFrame`` from a CPX image stream.

    Framing is delegated to ``cf_lab.imgstream`` -- the one decoder shared by
    every cf_lab image stream. This adds the CPX transport underneath and the
    app extension on top. ``recover_incomplete`` belongs to lossy UDP only;
    reliable TCP keeps strict body assembly. Console lines go to ``on_console``.
    """
    payloads = cpx_payloads(recv, on_console)
    for frame in img_frames(payloads, recover_incomplete=recover_incomplete):
        ext = ext_decoder(frame.ext) if ext_decoder is not None else None
        yield CapturedFrame(
            header=frame.header,
            payload=frame.payload,
            ext=ext,
            raw_ext=frame.ext,
            host_ns=time.monotonic_ns(),
        )


class CameraFeed:
    """The single subscriber to the deck's stream; fans out to sinks.

    The receive thread owns the socket and calls each sink's ``offer`` in turn;
    a sink that raises is counted and skipped rather than being allowed to take
    the feed (and the flight) down. A sink subscribed while the feed runs gets
    frames from the next one on, so a recorder can join after startup checks.
    """

    def __init__(
        self,
        *,
        ip: str,
        port: int,
        transport: str = "tcp",
        ext_decoder: ExtDecoder | None = decode_deck_ext,
        connect_timeout_s: float = 20.0,
        read_timeout_s: float = DEFAULT_READ_TIMEOUT_S,
        look_up: Callable[[], str] | None = None,
        log=lambda _msg: None,
    ) -> None:
        self._ip = ip
        self._port = port
        self._transport = transport
        self._ext_decoder = ext_decoder
        self._connect_timeout_s = connect_timeout_s
        self._read_timeout_s = read_timeout_s
        # Finds the deck's address again before a retry; None keeps ``ip``.
        self._look_up = look_up
        self._log = log
        self._sinks: tuple[FrameSink, ...] = ()
        self._observer_lock = threading.Lock()
        self._observers: list[FrameSink] = []
        self._sock = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._first_frame = threading.Event()
        self._config_received = threading.Event()
        self.frames_received = 0
        self.sink_errors = 0
        self.observer_errors = 0
        self.error: BaseException | None = None
        # The deck's latest "config" line: its build's settings, e.g. fps.
        self.deck_config_line: DeckLine | None = None

    @classmethod
    def from_config(
        cls, cfg: LabConfig, *, connect_timeout_s: float = 20.0, look_up_again: bool = True,
        **kwargs,
    ) -> "CameraFeed":
        """The feed of the deck on the drone ``cfg.radio.uri`` names.

        Looks the deck up by name first, for up to ``connect_timeout_s`` (a deck
        still booting); raises OSError if it does not answer. With
        ``look_up_again`` it looks the deck up again before each connect retry,
        and over UDP before each re-announce to a quiet stream: for ~15 s after a
        deck reboots, avahi can answer from its cache with the old address, and
        the deck may have a new one.
        """
        return cls(
            ip=resolve_deck(cfg.radio.uri, connect_timeout_s),
            port=cfg.camera.deck_port,
            transport=cfg.camera.transport,
            connect_timeout_s=connect_timeout_s,
            look_up=partial(resolve_deck, cfg.radio.uri) if look_up_again else None,
            **kwargs,
        )

    def subscribe(self, sink: FrameSink) -> None:
        # Replaced, never mutated, so the receive thread iterates a stable tuple.
        with self._observer_lock:
            self._sinks = (*self._sinks, sink)

    def add_observer(self, observer: FrameSink) -> Callable[[], None]:
        """Temporarily observe frames without opening a second deck connection.

        Like a sink, but removable: the returned unsubscribe function is
        idempotent. Its ``offer`` runs on the receive thread and must not block.
        This is for short diagnostics such as preflight checks; recorders and
        viewers use :meth:`subscribe`.
        """

        with self._observer_lock:
            self._observers.append(observer)

        def unsubscribe() -> None:
            with self._observer_lock:
                try:
                    self._observers.remove(observer)
                except ValueError:
                    pass

        return unsubscribe

    def start(self) -> None:
        """Connect, then receive on a background thread. Raises if it cannot connect."""
        self._sock, recv = open_stream(
            self._ip,
            self._port,
            transport=self._transport,
            connect_timeout_s=self._connect_timeout_s,
            read_timeout_s=self._read_timeout_s,
            look_up=self._look_up,
            log=self._log,
        )
        self._thread = threading.Thread(
            target=self._run, args=(recv,), name="camera-feed", daemon=True
        )
        self._thread.start()

    def wait_for_frame(self, timeout_s: float) -> bool:
        return self._first_frame.wait(timeout_s)

    @property
    def deck_config(self) -> dict[str, int | str] | None:
        """The fields of the deck's config line, or None if it sent none."""
        line = self.deck_config_line
        return None if line is None else line.fields

    def wait_for_config(self, timeout_s: float) -> dict[str, int | str] | None:
        """The deck's config fields, which it sends before the first frame; None if none came."""
        self._config_received.wait(timeout_s)
        return self.deck_config

    def _on_console(self, text: str) -> None:
        line = parse_deck_line(text)
        if line.kind == "config":
            self.deck_config_line = line
            self._config_received.set()
        for sink in self._sinks:
            offer = getattr(sink, "offer_console", None)
            if offer is None:
                continue
            try:
                offer(line)
            except Exception:
                self.sink_errors += 1

    def _run(self, recv) -> None:
        try:
            for frame in frames(
                recv,
                ext_decoder=self._ext_decoder,
                recover_incomplete=self._transport == "udp",
                on_console=self._on_console,
            ):
                self.frames_received += 1
                for sink in self._sinks:
                    try:
                        sink.offer(frame)
                    except Exception:
                        # A broken sink must not end the feed: telemetry keeps
                        # recording when a viewer dies, and vice versa.
                        self.sink_errors += 1
                with self._observer_lock:
                    observers = tuple(self._observers)
                for observer in observers:
                    try:
                        observer.offer(frame)
                    except Exception:
                        # An optional diagnostic must not stop the shared feed.
                        self.observer_errors += 1
                self._first_frame.set()
                if self._stop.is_set():
                    break
        except BaseException as exc:
            if not self._stop.is_set():  # stop() closing the socket is not a stream error
                self.error = exc
            self._first_frame.set()
            self._config_received.set()

    def stop(self) -> None:
        """Stop receiving and close the socket. Idempotent."""
        self._stop.set()
        if self._sock is not None:
            close_stream(self._sock)
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
