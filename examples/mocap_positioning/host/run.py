#!/usr/bin/env python3
"""Mocap positioning demo: motion capture -> extpose -> LED safety-box indicator.

The propellers never spin and the drone is never armed: the out-of-tree
controller runs on every stabilizer tick regardless of arm state, and an
un-armed drone has its motors force-stopped by the firmware, so they physically
cannot turn. The drone is purely a position indicator.

Place it on the floor in the centre of the room and start the runner. The first
mocap sample fixes the world origin at the drone's starting pose (0, 0, 0), +z
up. A configurable axis-aligned safety box sits in that frame (default centre
(0, 0, 1.5) m, side widths (2.0, 1.0, 1.0) m). As you carry the drone around by
hand:

  inside the box  -> the front LEDs are GREEN
  outside the box -> the front LEDs are RED

The box check runs onboard in the controller (against the estimator state the
host feeds from mocap, exposed as box.inside); the runner lights the front LEDs
through the firmware's led.bitmask override from the same in/out decision it
computes from the mocap stream, and prints that view so you can follow along
without looking at the drone.

There is no auto-stop of any kind: tilting the drone or carrying it out of the
box only changes the LED. The demo ends only when you say so:

  SPACE  -> emergency stop (firmware cuts/locks motors) and exit, code 2.
  Ctrl-C -> stop streaming and exit, code 0.

(With --duration > 0 the runner also stops when the timer elapses; the default,
--duration 0, runs until Ctrl-C / SPACE.)

Mocap precondition:
  - The runner refuses to go live unless mocap samples were actually forwarded
    during the settle phase (so a dead mocap system can't leave the LED stuck on
    a stale estimate). Dry-run that check with --source fake-dead.
  - If the mocap stream dies mid-run with the radio alive, the runner emergency
    stops and exits.

The mocap system in this lab is OptiTrack. --source optitrack reads pose from
Motive over the config's mocap.transport: NatNet, or VRPN -> vrpn_client_ros ->
rosbridge (see the example README). Run --source fake --backend fake for a
hardware-free dry run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cf_lab.backends.crazyflie import CrazyflieRadioBackend
from cf_lab.backends.fake import FakeCrazyflieBackend
from cf_lab.config import load_config
from cf_lab.mocap import BoxSweepSource, build_pose_source, origin_zeroed_streamer
from cf_lab.runner import (
    ExtposeStreamFailed,
    FakeSessionBase,
    KeyboardMonitor,
    KillSwitchEngaged,
    RadioLinkLost,
    RadioSessionBase,
    make_run_logger,
    mocap_settle_and_gate,
    poll_kill,
)

# Safety box in the host-zeroed mocap frame (origin = drone's start pose on the
# floor, +z up): centre (x, y, z) and the full side widths (x, y, z), in metres.
BOX_CENTER_M = (0.0, 0.0, 1.5)
BOX_SIZE_M = (2.0, 1.0, 1.0)

# Estimator settle before the mocap gate lets the indicator go live.
MOCAP_SETTLE_S = 2.0

# Front-LED indicator via the firmware's `led.bitmask` override (bit 7 = enable;
# the firmware drives the GPIOs directly, bypassing the LED sequencer):
#   | 7:ENABLE | 5:BLUE_R | 4:RED_R | 3:GREEN_R | 2:RED_L | 1:GREEN_L | 0:BLUE_L |
# Both front greens on = inside the box; both front reds on = outside.
LED_ENABLE = 1 << 7
LED_GREEN_PAIR = LED_ENABLE | (1 << 1) | (1 << 3)  # GREEN_L + GREEN_R
LED_RED_PAIR = LED_ENABLE | (1 << 2) | (1 << 4)  # RED_L + RED_R
LED_OFF = 0  # hand the LEDs back to the firmware's normal sequencer

# --- Firmware console capture (DEBUG_PRINT) ------------------------------------
# The firmware streams console packets over the radio whenever it prints, whether
# or not the host listens, so capturing them costs no extra link bandwidth — we
# capture by default on the radio backend (disable with --no-debug-log) and write
# them under this example's own artifacts/ dir.
#
# DEBUG_LOG_TIMESTAMPED is a deliberate non-CLI knob: False (default) overwrites a
# single debug_log.txt each run; set True to keep one debug_log_<ISO>.txt per run.
DEBUG_LOG_TIMESTAMPED = False
DEBUG_LOG_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "debug_log"


def inside_box(position_m, center_m, size_m) -> bool:
    """True when position is within the axis-aligned box (inclusive faces)."""
    return all(
        abs(p - c) <= s / 2.0 for p, c, s in zip(position_m, center_m, size_m)
    )


class FakeSession(FakeSessionBase):
    """Print-only drone session backed by the in-memory fake backend."""

    def set_box(self, center_m, size_m) -> None:
        pass

    def set_led(self, inside: bool) -> None:
        pass

    def restore_leds(self) -> None:
        pass


class RadioSession(RadioSessionBase):
    """Drive a real Crazyflie over the open cflib radio link (no arming)."""

    def set_box(self, center_m, size_m) -> None:
        cx, cy, cz = center_m
        sx, sy, sz = size_m
        self._cf.param.set_value("box.centerX", str(cx))
        self._cf.param.set_value("box.centerY", str(cy))
        self._cf.param.set_value("box.centerZ", str(cz))
        self._cf.param.set_value("box.sizeX", str(sx))
        self._cf.param.set_value("box.sizeY", str(sy))
        self._cf.param.set_value("box.sizeZ", str(sz))

    def set_led(self, inside: bool) -> None:
        # Drive the front LEDs through the firmware's led.bitmask override:
        # both greens inside the box, both reds outside.
        bitmask = LED_GREEN_PAIR if inside else LED_RED_PAIR
        try:
            self._cf.param.set_value("led.bitmask", str(bitmask))
        except Exception:
            pass

    def emergency_stop(self) -> None:
        super().emergency_stop()
        self.restore_leds()

    def restore_leds(self) -> None:
        try:
            self._cf.param.set_value("led.bitmask", str(LED_OFF))
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/drone.yaml")
    parser.add_argument(
        "--source",
        choices=("fake", "optitrack", "fake-dead"),
        default="fake",
        help=(
            "optitrack = the live mocap (transport + connection from the config); "
            "fake-dead simulates a mocap system that is off (mocap gate)"
        ),
    )
    parser.add_argument("--backend", choices=("fake", "radio"), default="fake")
    parser.add_argument(
        "--no-debug-log",
        action="store_true",
        help="Do not capture the firmware console (DEBUG_PRINT) to artifacts/debug_log/ (radio only)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run time in seconds; 0 (default) runs until Ctrl-C / SPACE",
    )
    args = parser.parse_args(argv)

    center_m = BOX_CENTER_M
    size_m = BOX_SIZE_M

    if any(s <= 0.0 for s in size_m):
        print(f"Box side widths must be positive, got {size_m}.")
        return 1

    cfg = load_config(args.config)
    # The drone never arms here, but it still streams mocap to drive the LED, so
    # the live link (transport, host/IP, ports, rate) comes from the config; the
    # runner only supplies its own synthetic source for fake runs. z sweeps
    # 0 -> peak -> 0 so the in/out-of-box indicator toggles.
    fake = BoxSweepSource(rigid_body=cfg.mocap.rigid_body, z_peak_m=BOX_CENTER_M[2])
    source = build_pose_source(args.source, cfg=cfg, fake=fake)

    print(__doc__.split("\n\n")[0])
    print("Frame: +z up, origin and zero yaw at the drone's starting pose on the floor.")
    print(f"Safety box: centre {center_m} m, side widths {size_m} m.")
    print("Propellers OFF, never armed. SPACE = emergency stop, Ctrl-C = stop.")
    print()

    # One logger for the run: [HOST] lines to the terminal (+ file), the firmware
    # console tagged [GAP8]/[STM32]/[ESP32] to the file only. Capture is radio-only
    # and can be turned off with --no-debug-log.
    log_dir = DEBUG_LOG_DIR if (args.backend == "radio" and not args.no_debug_log) else None
    log, debug_sink, debug_close = make_run_logger(log_dir, timestamped=DEBUG_LOG_TIMESTAMPED)
    console_sink = debug_sink if log_dir is not None else None

    try:
        if args.backend == "fake":
            backend = FakeCrazyflieBackend()
            session = FakeSession(backend)
            return _run(
                args, cfg, source=source, backend=backend, session=session,
                center_m=center_m, size_m=size_m, log=log,
            )
        uri = cfg.radio.uri
        try:
            with CrazyflieRadioBackend(uri, console_sink=console_sink) as backend:
                session = RadioSession(backend)
                return _run(
                    args, cfg, source=source, backend=backend, session=session,
                    center_m=center_m, size_m=size_m, log=log,
                )
        finally:
            debug_close()
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()


def _run(args, cfg, *, source, backend, session, center_m, size_m, log) -> int:
    streamer = origin_zeroed_streamer(
        source=source, source_kind=args.source, backend=backend, cfg=cfg
    )

    with KeyboardMonitor() as keyboard:

        def watch(duration_s: float) -> None:
            """Drive the live indicator, polling kill/stream/link health.

            duration_s <= 0 waits forever (until a kill condition is raised).
            """
            indefinite = duration_s <= 0.0
            deadline = time.monotonic() + duration_s
            last_inside: bool | None = None
            while indefinite or time.monotonic() < deadline:
                poll_kill(keyboard, streamer, session)
                pose = streamer.latest
                if pose is not None:
                    inside = inside_box(pose.position_m, center_m, size_m)
                    if inside != last_inside:
                        session.set_led(inside)
                        log(f"{'INSIDE' if inside else 'OUTSIDE'} box "
                            f"(pos {_fmt(pose.position_m)} m) -> "
                            f"LED {'GREEN' if inside else 'RED'}.")
                        last_inside = inside
                time.sleep(0.02)

        try:
            log(f"Streaming origin-zeroed extpose at {cfg.mocap.stream_rate_hz:.0f} Hz.")
            streamer.start()
            if not mocap_settle_and_gate(streamer, keyboard=keyboard, session=session,
                                         cfg=cfg, settle_s=MOCAP_SETTLE_S, log=log,
                                         gate_label="NO MOCAP"):
                return 1
            session.prepare()
            session.set_box(center_m, size_m)
            log("Indicator live (motors off, never armed). Move the drone; the LED tracks the box.")
            if args.duration > 0.0:
                log(f"Running for {args.duration:.1f}s, then stopping.")
            else:
                log("Running until SPACE / Ctrl-C.")
            watch(args.duration)
            log("Duration elapsed; stopping.")
        except KillSwitchEngaged:
            session.emergency_stop()
            log("EMERGENCY STOP: spacebar pressed, motors cut.")
            return 2
        except ExtposeStreamFailed as exc:
            session.emergency_stop()
            log(f"EMERGENCY STOP: extpose stream failed ({exc.__cause__}).")
            return 2
        except RadioLinkLost:
            log("RADIO LINK LOST: stopping (the drone was never armed; motors stay off).")
            return 3
        except KeyboardInterrupt:
            log("Ctrl-C: stopping.")
            return 0
        finally:
            streamer.stop()
            session.restore_leds()

    dropped = f", dropped {streamer.dropped} stale" if streamer.dropped else ""
    log(f"Done. Forwarded {streamer.sent} extpose sample(s){dropped}.")
    return 0


def _fmt(position_m) -> str:
    return "(" + ", ".join(f"{c:+.2f}" for c in position_m) + ")"


if __name__ == "__main__":
    raise SystemExit(main())
