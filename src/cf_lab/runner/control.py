"""Shared control-flow helpers for example host runners.

Every demo runner watches the same "stop now" conditions while it drives the
drone (operator SPACE, mocap stream death, radio loss, lost mocap) and prints
the same timestamped log lines. These helpers capture that common plumbing so
each runner keeps only its demo-specific logic.
"""

from __future__ import annotations

import datetime
import threading
import time
from collections.abc import Callable
from pathlib import Path


class KillSwitchEngaged(Exception):
    """Raised when the operator presses SPACE."""


class ExtposeStreamFailed(Exception):
    """Raised when the background mocap/extpose stream dies."""


class RadioLinkLost(Exception):
    """Raised when the Crazyflie radio link drops."""


def poll_kill(keyboard, streamer, session) -> None:
    """Raise the matching kill exception if any stop condition is active.

    Checks, in order: SPACE on the keyboard, a captured streamer error, a lost
    radio link, and, once the runner has started the streamer's pose watch, a gap
    in the poses (``cf_lab.mocap.MocapLost``: ``MocapBodyLost`` or
    ``MocapConnectionLost``). ``ExtposeStreamFailed`` chains the underlying error
    as its cause so callers can log ``exc.__cause__``. ``streamer`` may be
    ``None`` for a flight with no mocap stream; the stream checks are skipped then.
    """
    if " " in keyboard.pressed():
        raise KillSwitchEngaged
    if streamer is not None and streamer.error is not None:
        raise ExtposeStreamFailed from streamer.error
    if session.link_lost:
        raise RadioLinkLost
    if streamer is not None:
        streamer.check_poses()


def kill_aware_wait(
    keyboard,
    streamer,
    session,
    duration_s: float,
    *,
    until: Callable[[], bool] | None = None,
    poll_s: float = 0.05,
) -> bool:
    """Sleep ``duration_s`` while polling the kill conditions every ``poll_s``.

    A standalone building block: it waits and watches for SPACE / stream death /
    radio loss (raising via :func:`poll_kill`), nothing more. With ``until`` it
    returns ``True`` as soon as the predicate holds, else ``False`` at timeout;
    without ``until`` it always returns ``True`` once the time elapses.
    """
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        poll_kill(keyboard, streamer, session)
        if until is not None and until():
            return True
        time.sleep(poll_s)
    return until is None


def stream_segment(
    duration_s: float,
    command: Callable[[float], None],
    *,
    keyboard,
    streamer,
    session,
    rate_hz: float,
) -> None:
    """Call ``command(fraction)`` from 0.0 to 1.0 over ``duration_s`` at ``rate_hz``.

    A standalone building block for streaming a ramped setpoint (position,
    altitude, ...): the caller's ``command`` computes and sends one setpoint for
    the given progress fraction. The kill conditions are polled each tick. A
    ``duration_s <= 0`` sends a single ``command(1.0)``. This advances no flight
    state of its own — sequencing segments is entirely the caller's job.
    """
    period = 1.0 / rate_hz
    start = tick = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        poll_kill(keyboard, streamer, session)
        fraction = 1.0 if duration_s <= 0 else min(elapsed / duration_s, 1.0)
        command(fraction)
        if fraction >= 1.0:
            return
        # On a fixed schedule, so command()'s own time is not added to every
        # period; once behind, it carries on from now rather than catch up.
        tick = max(tick + period, time.monotonic())
        time.sleep(max(0.0, tick - time.monotonic()))


def mocap_settle_and_gate(
    streamer,
    *,
    keyboard,
    session,
    cfg,
    settle_s: float,
    log,
    gate_label: str = "ARMING BLOCKED",
) -> bool:
    """Settle the estimator for ``settle_s`` and check the mocap gate.

    Convenience wrapper over :func:`kill_aware_wait` + :func:`mocap_gate_required`:
    waits the settle window (polling kill), then confirms enough mocap samples
    were forwarded so a dead mocap system can't go live on a stale estimate.
    Logs the standard pass/fail lines and returns ``True`` only on pass. Callers
    that want a different shape can use the two lower-level helpers directly.
    """
    rate_hz = cfg.mocap.stream_rate_hz
    log(f"Settling estimator on the start frame for {settle_s}s.")
    kill_aware_wait(keyboard, streamer, session, settle_s)
    required = mocap_gate_required(settle_s, rate_hz)
    if streamer.sent < required:
        log(
            f"{gate_label}: only {streamer.sent} mocap sample(s) forwarded during "
            f"settle (need >= {required}). Is the mocap system running?"
        )
        return False
    log(f"Mocap gate passed: {streamer.sent} sample(s) during settle.")
    return True


# The STM32 forwards GAP8/ESP32 console text through its own console module, so on
# the host's single CRTP console stream those lines arrive marked (DEBUG_MODULE
# "CPX" + cpx.c re-emit): "CPX: GAP8: ..." / "CPX: ESP32: ...". Anything else on the
# stream is the STM32 itself, which is where the console physically originates.
_CONSOLE_SOURCE_MARKERS = (("CPX: GAP8: ", "GAP8"), ("CPX: ESP32: ", "ESP32"))


def tag_console_line(line: str) -> tuple[str, str]:
    """Split one firmware console line into ``(source_tag, message)``.

    Strips the STM32-added ``CPX: GAP8: ``/``CPX: ESP32: `` marker to ``"GAP8"``/
    ``"ESP32"``; any other line is the STM32's own console (``"STM32"``).
    """
    for marker, tag in _CONSOLE_SOURCE_MARKERS:
        if line.startswith(marker):
            return tag, line[len(marker):]
    return "STM32", line


def make_run_logger(
    log_dir: Path | None = None,
    *,
    timestamped: bool = False,
    echo_console: bool = False,
):
    """Open a run's debug log; return ``(log, console_sink, close)``.

    One shared ``[t=  1.23s]`` clock and source tag scheme across every tier:

    - ``log(message, echo=True)`` — host tier: writes ``[t=..] [HOST] message`` to
      the terminal **and** the file. Pass ``echo=False`` to write only to the file
      (e.g. high-rate telemetry that would otherwise spam the terminal).
    - ``console_sink(line)`` — firmware tiers: tags the line via
      :func:`tag_console_line` and writes ``[t=..] [GAP8|STM32|ESP32] message`` to
      the file. Set ``echo_console=True`` to also print firmware lines.
    - ``close()`` — close the file.

    With ``log_dir=None`` no file is opened: ``log`` prints to the terminal,
    ``console_sink`` prints only when ``echo_console=True``, and ``close`` is safe.
    ``timestamped=True`` keeps one
    ``debug_log_<ISO>.txt`` per run instead of overwriting ``debug_log.txt``. The
    file is shared by two threads (``console_sink`` fires on cflib's link thread),
    so writes are serialised under a lock.
    """
    start = time.monotonic()
    handle = None
    path = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        if timestamped:
            stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            path = log_dir / f"debug_log_{stamp}.txt"
        else:
            path = log_dir / "debug_log.txt"
        handle = open(path, "w", buffering=1)
    lock = threading.Lock()

    def emit(tag: str, message: str, *, echo: bool) -> None:
        line = f"[t={time.monotonic() - start:6.2f}s] [{tag}] {message}"
        with lock:
            if echo:
                print(line, flush=True)
            if handle is not None:
                handle.write(line + "\n")

    def log(message: str, *, echo: bool = True) -> None:
        emit("HOST", message, echo=echo)

    def console_sink(line: str) -> None:
        tag, message = tag_console_line(line)
        emit(tag, message, echo=echo_console)

    def close() -> None:
        if handle is not None:
            handle.close()

    if path is not None:
        log(f"Capturing firmware console -> {path}")
    return log, console_sink, close


def mocap_gate_required(settle_s: float, rate_hz: float) -> int:
    """Minimum mocap samples that must arrive during settle to go live.

    A quarter of the nominal settle-window count, floored at 10, so a dead mocap
    system can never satisfy the gate.
    """
    return max(10, int(0.25 * settle_s * rate_hz))
