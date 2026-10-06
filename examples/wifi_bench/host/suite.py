"""The wifi_bench tests. Each takes a Context and returns {"summary", "checks"}.

A check is {"name", "passed", "measured", "want"}, with "advisory": True when a
miss is reported but does not fail the case (camera_bench's convention).
"""

from __future__ import annotations

import json
import math
import re
import secrets
import socket
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from link import DATA, HEADER, PING, PONG, START, STOP, Link, record, wire

HOST_DIR = Path(__file__).resolve().parent
# Request kind that makes the GAP8 send a malformed SPI frame (wifi_bench.c SPI_BAD).
SPI_BAD = 7


@dataclass
class Context:
    """What a test needs: the deck's address, the case folder, the radio console."""

    ip: str
    port: int
    transport: str
    directory: Path
    console: Path | None  # the radio console log, None when the bench runs without it
    access_point: object  # cf_lab.bench.AccessPoint
    log: object
    options: dict = field(default_factory=dict)

    def link(self) -> Link:
        return Link(self.ip, self.port, self.transport)


def check(name, passed, measured, want, advisory=False) -> dict:
    return {"name": name, "passed": bool(passed), "measured": measured, "want": want,
            **({"advisory": True} if advisory else {})}


def _echo(ctx: Context, count: int = 10, size: int = 64) -> dict:
    session = secrets.randbits(32)
    body = (bytes(range(256)) * 4)[: size - HEADER.size]
    times = []
    with ctx.link() as link:
        for seq in range(count):
            start = time.monotonic()
            packet = link.request(PING, session, seq, body)
            if packet.body != body:
                raise ValueError("echo payload changed")
            times.append(1000 * (time.monotonic() - start))
    return {"echoes": count, "rtt_median_ms": round(statistics.median(times), 2),
            "rtt_max_ms": round(max(times), 2)}


def recovers(ctx: Context) -> dict:
    """The server still answers a new client."""
    try:
        return check("recovers", True, _echo(ctx, 1)["rtt_median_ms"], "an echo")
    except Exception as exc:
        return check("recovers", False, f"{type(exc).__name__}: {exc}", "an echo")


def stop_stream(ctx: Context) -> None:
    """Stop the GAP8's stream and drain what is in flight. A test that leaves one
    running would load the cases after it: a stream is bounded by time, not by
    the client, and on UDP it follows each new client."""
    with ctx.link() as link:
        link.request(STOP, secrets.randbits(32))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                link.receive(min(0.2, deadline - time.monotonic()))
            except TimeoutError:
                break


def echo(ctx: Context) -> dict:
    summary = _echo(ctx)
    return {"summary": summary, "checks": [check("echo", True, summary["echoes"], "10 echoes")]}


def stream(ctx: Context, size: int, rate: int, duration_s: float) -> dict:
    """The GAP8 produces `rate` packets/s of `size` bytes; every one is checked."""
    session = secrets.randbits(32)
    seen = bytearray((math.ceil(duration_s * rate) + 1031) // 8)  # one bit per sequence
    unique = duplicates = reordered = corrupt = foreign = 0
    last_seq = -1
    received_bytes = 0
    began = last_sample = time.monotonic()
    with ctx.link() as link, (ctx.directory / "samples.jsonl").open("a") as samples:

        def collect(packet):
            nonlocal unique, duplicates, reordered, foreign, last_seq, received_bytes, last_sample
            if packet.kind != DATA:
                return
            if packet.session != session:
                foreign += 1
                return
            index, bit = divmod(packet.sequence, 8)
            if index >= len(seen):
                raise ValueError("sequence exceeds the run's capacity")
            if seen[index] & (1 << bit):
                duplicates += 1
            else:
                seen[index] |= 1 << bit
                unique += 1
                received_bytes += len(packet.body) + HEADER.size
            if packet.sequence < last_seq:
                reordered += 1
            last_seq = max(last_seq, packet.sequence)
            now = time.monotonic()
            if now - last_sample >= 1:
                samples.write(json.dumps({"t_s": round(now - began, 3), "packets": unique,
                                          "bytes": received_bytes}) + "\n")
                last_sample = now

        link.request(START, session, body=struct.pack("<III", rate, size, round(duration_s * 1000)),
                     on_packet=collect)
        deadline = time.monotonic() + duration_s + 0.7
        while time.monotonic() < deadline:
            try:
                packet = link.receive(min(0.5, deadline - time.monotonic()))
            except TimeoutError:
                continue
            except ValueError:
                corrupt += 1
                if ctx.transport == "tcp":
                    raise
                continue
            collect(packet)
        stats = link.request(STOP, session, sequence=1, on_packet=collect)
        (attempted, accepted, dropped, rx, bad, active_session, running,
         _connected, _transport, _channel) = struct.unpack("<10I", stats.body)
        deadline = time.monotonic() + (3 if ctx.transport == "tcp" else 0.3)
        while unique < accepted and time.monotonic() < deadline:
            try:
                collect(link.receive(deadline - time.monotonic()))
            except TimeoutError:
                break
    missing = accepted - unique
    summary = {"size": size, "offered_pps": rate, "duration_s": duration_s,
               "attempted": attempted, "gap8_accepted": accepted, "gap8_queue_drops": dropped,
               "received": unique, "missing": missing, "duplicates": duplicates,
               "reordered": reordered, "corrupt": corrupt, "foreign": foreign,
               "mbps": round(received_bytes * 8 / duration_s / 1e6, 3),
               "gap8_rx": rx, "gap8_invalid": bad}
    checks = [check("data", unique > 0, unique, "> 0"),
              check("corrupt", corrupt == 0, corrupt, "0"),
              check("producer_stopped", active_session == session and not running,
                    "stopped" if not running else "running", "stopped")]
    if ctx.transport == "tcp":
        checks.append(check("tcp_lossless", missing == 0 and not duplicates and not reordered,
                            {"missing": missing, "duplicates": duplicates, "reordered": reordered},
                            "all accepted packets once, in order"))
    else:
        loss = missing / accepted if accepted else 0.0
        limit = ctx.options.get("udp_loss_max", 0.02)
        checks.append(check("udp_loss", loss <= limit, round(loss, 4), f"<= {limit}", advisory=True))
    return {"summary": summary, "checks": checks}


def reconnect(ctx: Context, count: int = 15) -> dict:
    """Connect and echo repeatedly, closing alternately gracefully and with a reset.

    Runs idle: data flowing to the host would reveal a reset through a send error
    and hide a receive-side fault.
    """
    stop_stream(ctx)
    durations = []
    for cycle in range(count):
        start = time.monotonic()
        try:
            link = ctx.link()
            try:
                link.request(PING, secrets.randbits(32), cycle)
            finally:
                link.close(reset=bool(cycle % 2))
        except Exception as exc:
            after = "reset" if cycle and (cycle - 1) % 2 else "graceful close"
            return {"summary": {"cycles": cycle},
                    "checks": [check("reconnects", False,
                                     f"cycle {cycle + 1} after a {after}: {type(exc).__name__}: {exc}",
                                     f"{count} cycles")]}
        durations.append(time.monotonic() - start)
    return {"summary": {"cycles": count, "max_s": round(max(durations), 3)},
            "checks": [check("reconnects", True, count, f"{count} cycles")]}


def stalled_reader(ctx: Context) -> dict:
    """A client that stops reading must not keep the server from the next one."""
    with ctx.link() as link:
        link.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        link.request(START, secrets.randbits(32), body=struct.pack("<III", 2000, 1020, 10000))
        time.sleep(8)
        # Keep the stalled socket open: a new client being served proves the
        # server freed the stalled session by itself (its send timeout).
        start = time.monotonic()
        recovery = recovers(ctx)
        recovery_s = time.monotonic() - start
    stop_stream(ctx)
    return {"summary": {"recovery_s": round(recovery_s, 3)}, "checks": [recovery]}


def host_exit(ctx: Context) -> dict:
    """A host process that dies mid-stream, without closing its socket."""
    code = ("import os,struct,sys,time\nfrom link import Link,START\n"
            "with Link(sys.argv[1],int(sys.argv[2]),sys.argv[3]) as link:\n"
            "    link.request(START,12345,body=struct.pack('<III',1000,1020,5000))\n"
            "    time.sleep(.3)\n    os._exit(0)\n")
    subprocess.run([sys.executable, "-c", code, ctx.ip, str(ctx.port), ctx.transport],
                   cwd=HOST_DIR, check=True, timeout=15)
    recovery = recovers(ctx)
    stop_stream(ctx)
    return {"summary": {}, "checks": [recovery]}


def fragmented(ctx: Context) -> dict:
    """A request split across four TCP segments arrives whole."""
    session = secrets.randbits(32)
    message = wire(record(PING, session, body=b"fragmented"))
    with ctx.link() as link:
        for chunk in (message[:1], message[1:2], message[2:7], message[7:]):
            link.raw(chunk)
            time.sleep(0.03)
        reply = link.receive()
    whole = reply.kind == PONG and reply.session == session and reply.body == b"fragmented"
    return {"summary": {"fragments": 4}, "checks": [check("reassembled", whole, whole, True)]}


def incomplete(ctx: Context, payload: bytes) -> dict:
    """A request cut short by the host's FIN is dropped and the connection closed."""
    with ctx.link() as link:
        link.raw(payload)
        time.sleep(0.05)
        link.sock.shutdown(socket.SHUT_WR)
        try:
            link.receive(2)
            closed = "answered"
        except TimeoutError:
            closed = "left open"
        except (ConnectionError, OSError):
            closed = "closed"
    return {"summary": {}, "checks": [check("closed", closed == "closed", closed, "closed"),
                                      recovers(ctx)]}


def malformed(ctx: Context, payload) -> dict:
    """A request with a bad length is rejected and the server keeps serving."""
    session = secrets.randbits(32)
    with ctx.link() as link:
        if payload == "oversized":  # a valid maximum-size request plus five bytes
            payload = wire(record(PING, session, body=b"x" * (1020 - HEADER.size))) + b"extra"
        link.raw(payload)
        outcome = "rejected"
        deadline = time.monotonic() + 0.7
        try:
            while time.monotonic() < deadline:
                packet = link.receive(deadline - time.monotonic())
                if packet.kind == PONG and packet.session == session:
                    outcome = "answered"
                    break
        except (TimeoutError, ConnectionError, OSError):
            pass
    return {"summary": {}, "checks": [check("rejected", outcome == "rejected", outcome, "rejected"),
                                      recovers(ctx)]}


def _spi_rejects(ctx: Context) -> int | None:
    values = re.findall(r"spiRej (\d+)", ctx.console.read_text()) if ctx.console else []
    return int(values[-1]) if values else None


def spi_invalid_length(ctx: Context) -> dict:
    """A malformed SPI frame from the GAP8 is counted and dropped by the ESP32."""
    if ctx.console is None:
        return {"summary": {}, "checks": [], "skipped": "needs the radio console (spiRej)"}
    deadline = time.monotonic() + 6
    before = _spi_rejects(ctx)
    while before is None and time.monotonic() < deadline:
        time.sleep(0.1)
        before = _spi_rejects(ctx)
    if before is None:
        return {"summary": {}, "checks": [check("spi_rejects", False, "no ESP32 status line",
                                                "a rejection counter")]}
    with ctx.link() as link:
        link.request(SPI_BAD, secrets.randbits(32))
    deadline = time.monotonic() + 6  # the ESP32 reports every 5 s
    while time.monotonic() < deadline and _spi_rejects(ctx) == before:
        time.sleep(0.1)
    delta = _spi_rejects(ctx) - before
    return {"summary": {"spi_reject_delta": delta},
            "checks": [check("spi_rejects", delta == 1, delta, 1), recovers(ctx)]}


def wifi_rejoin(ctx: Context) -> dict:
    """The host leaves and rejoins the access point in the middle of a stream."""
    with ctx.link() as link:
        link.request(START, secrets.randbits(32), body=struct.pack("<III", 1000, 1020, 15000))
        ctx.access_point.down()
        time.sleep(3)
        start = time.monotonic()
        ctx.access_point.up(time.monotonic() + 30)
    recovery = recovers(ctx)
    rejoin_s = time.monotonic() - start
    stop_stream(ctx)
    return {"summary": {"rejoin_and_echo_s": round(rejoin_s, 3)}, "checks": [recovery]}


# name -> (function, transports, keyword arguments)
TESTS = {
    "stream-64x50": (stream, ("tcp", "udp"), {"size": 64, "rate": 50}),
    "stream-512x500": (stream, ("tcp", "udp"), {"size": 512, "rate": 500}),
    "stream-1020x1000": (stream, ("tcp", "udp"), {"size": 1020, "rate": 1000}),
    "echo": (echo, ("tcp", "udp"), {}),
    "host-exit": (host_exit, ("tcp", "udp"), {}),
    "stalled-reader": (stalled_reader, ("tcp", "udp"), {}),
    "fragmented": (fragmented, ("tcp",), {}),
    "incomplete-header": (incomplete, ("tcp",), {"payload": b"\x20"}),
    "incomplete-payload": (incomplete, ("tcp",), {"payload": b"\x20\x00\x5c\x05abc"}),
    "short-length": (malformed, ("tcp", "udp"), {"payload": b"\x01\x00x"}),
    "long-length": (malformed, ("tcp", "udp"), {"payload": b"\xff\xff"}),
    "truncated-datagram": (malformed, ("udp",), {"payload": b"\x20\x00abc"}),
    "oversized-datagram": (malformed, ("udp",), {"payload": "oversized"}),
    "spi-invalid-length": (spi_invalid_length, ("tcp", "udp"), {}),
    "wifi-rejoin": (wifi_rejoin, ("tcp", "udp"), {}),
    # Last: a wedged idle server would fail every case after it.
    "reconnect": (reconnect, ("tcp", "udp"), {}),
}
