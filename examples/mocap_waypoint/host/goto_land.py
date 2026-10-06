#!/usr/bin/env python3
"""Take off in place, fly to a global (x, y), and land there -- slow and accurate.

Unlike the mocap_waypoint demo (which flies in a start-origin, yaw-zeroed frame),
this drives the drone in the **global mocap frame**: extpose is streamed raw (no
origin/yaw zeroing), so the target you pass is an absolute position in Motive's
coordinates. The flight is: climb straight up from wherever the drone sits to a
hold altitude, translate at a capped speed to the target x/y, settle until it is
within tolerance, then descend and cut motors on the spot. Heading is held at
whatever the drone is pointing now, unless you pass --yaw (a global heading, deg):
then it lifts off straight up first and only turns to that heading in place at
altitude, so a heading change never fights the ground during takeoff.

Safety:
  * Refuses to arm unless the drone is within 10 deg of level -- an inverted or
    tilted mocap orientation would arm the controller believing down is up and
    flip it on the pad.
  * Speed is capped (--speed), setpoints are ramped (never stepped) so the drone
    tracks closely with little lag/windup, and it settles to a position tolerance
    before descending -- so it lands with little steady-state error.
  * No radio-loss deadman is armed, so a clean disarm leaves the drone
    re-armable without a reboot (the deadman latches until reboot). On a link
    drop the firmware's commander-setpoint timeout stops the motors instead.
  * SPACE = emergency stop (motors cut). Ctrl-C = stop + disarm in place.
  * Once armed, no new mocap pose for mocap.pose_timeout_s (1 s by default) ->
    descend in place from the last pose at --speed, then stop + disarm; shorter
    gaps are left to the estimator.

  python examples/mocap_waypoint/host/goto_land.py --target 0.0 0.0
  python examples/mocap_waypoint/host/goto_land.py --target -0.5 0.8 --yaw 0 --speed 0.2 --height 0.5 --tol 0.03
"""
from __future__ import annotations

import argparse
import functools
import math
import time

from cf_lab.backends.crazyflie import CrazyflieRadioBackend
from cf_lab.config import load_config
from cf_lab.mocap import MocapLost, build_live_pose_source
from cf_lab.mocap.stream import ExtposeStreamer
from cf_lab.preflight import (
    check_mocap_quality,
    run_preflight,
    standard_flight_checks,
)
from cf_lab.runner import (
    ExtposeStreamFailed,
    FlightRadioSession,
    KeyboardMonitor,
    KillSwitchEngaged,
    RadioLinkLost,
    WatchdogHeartbeat,
    kill_aware_wait,
    make_run_logger,
    mocap_settle_and_gate,
    poll_kill,
    stream_segment,
)

SETPOINT_RATE_HZ = 50.0
MAX_TILT_DEG = 10.0          # refuse to arm above this roll/pitch
ARM_SETTLE_S = 1.0
MOCAP_SETTLE_S = 2.0         # estimator settle before the mocap arming gate
MIN_SEGMENT_S = 0.5          # floor on a ramp's duration, so tiny moves stay gentle
YAW_RATE_DPS = 45.0          # cap on in-place heading changes
# Outer-loop x/y gains: the lab manual tune. xVelMax is overridden by --speed to
# enforce the speed cap; a non-zero Ki trims the steady-state position bias.
POS_XY_KP, POS_XY_KI = 2.0, 0.25
VEL_XY_KP, VEL_XY_KI = 20.0, 2.0


class GotoSession(FlightRadioSession):
    """FlightRadioSession plus an absolute-position setpoint and the gain setup."""

    @property
    def cf(self):
        return self._cf

    def goto(self, x: float, y: float, z: float, yaw_deg: float) -> None:
        self._cf.commander.send_position_setpoint(x, y, z, yaw_deg)

    def apply_gains(self, vel_max: float) -> None:
        p = self._cf.param
        for axis in ("x", "y"):
            p.set_value(f"posCtlPid.{axis}Kp", POS_XY_KP)
            p.set_value(f"posCtlPid.{axis}Ki", POS_XY_KI)
            p.set_value(f"velCtlPid.v{axis}Kp", VEL_XY_KP)
            p.set_value(f"velCtlPid.v{axis}Ki", VEL_XY_KI)
            p.set_value(f"posCtlPid.{axis}VelMax", vel_max)  # the speed cap

    def arm(self) -> None:
        # Arm WITHOUT the emergency-stop watchdog that FlightRadioSession.arm()
        # starts -- that deadman latches until reboot, forcing a power-cycle
        # between runs. But a heartbeat is still REQUIRED: the supervisor
        # auto-disarms a not-yet-flying drone after a short timeout, so without one
        # the motors cut mid-takeoff and it drops. Feed the ARMING request (not the
        # watchdog) to keep it armed through the slow takeoff ramp -- that keeps
        # reruns reboot-free (nothing latches) while still lifting off. Trade-off:
        # no explicit radio-loss motor cut; a link drop falls back on the firmware
        # commander-setpoint timeout, and SPACE still hard-stops. An old arming
        # heartbeat left running would re-arm the drone after disarm.
        self.stop_heartbeat()
        self._cf.supervisor.send_arming_request(True)
        self._heartbeat = WatchdogHeartbeat(
            lambda: self._cf.supervisor.send_arming_request(True)
        )
        self._heartbeat.start()
        time.sleep(0.5)

    def disarm(self) -> None:
        # Stop the arming-refresh heartbeat FIRST or it immediately re-arms.
        self.stop_heartbeat()
        self._cf.supervisor.send_arming_request(False)


def _lerp_yaw(y0: float, y1: float, f: float) -> float:
    """Interpolate absolute yaw (deg) the short way round."""
    delta = ((y1 - y0 + 180.0) % 360.0) - 180.0
    return y0 + delta * f


def _ramp(session, keyboard, streamer, p0, p1, yaw0, yaw1, speed, log, label) -> None:
    """Stream a straight-line setpoint p0->p1 (and yaw yaw0->yaw1), speed- and yaw-rate-capped."""
    dyaw = abs(_lerp_yaw(yaw0, yaw1, 1.0) - yaw0)
    dur = max(math.dist(p0, p1) / max(speed, 1e-3), dyaw / YAW_RATE_DPS, MIN_SEGMENT_S)
    yaw_note = f" yaw {yaw0:+.0f}->{yaw1:+.0f}" if dyaw > 1 else ""
    log(f"{label}: ({p0[0]:+.2f},{p0[1]:+.2f},{p0[2]:.2f}) -> "
        f"({p1[0]:+.2f},{p1[1]:+.2f},{p1[2]:.2f}){yaw_note} over {dur:.1f}s (<= {speed:.2f} m/s)")

    def command(f: float) -> None:
        session.goto(
            p0[0] + (p1[0] - p0[0]) * f,
            p0[1] + (p1[1] - p0[1]) * f,
            p0[2] + (p1[2] - p0[2]) * f,
            _lerp_yaw(yaw0, yaw1, f),
        )

    stream_segment(dur, command, keyboard=keyboard, streamer=streamer,
                   session=session, rate_hz=SETPOINT_RATE_HZ)


def _settle(session, keyboard, streamer, target, yaw_deg, *, tol, timeout_s, log, label) -> None:
    """Hold `target` (still sending setpoints) until within `tol`, or timeout."""
    period = 1.0 / SETPOINT_RATE_HZ
    start = time.monotonic()
    while True:
        poll_kill(keyboard, streamer, session)
        session.goto(target[0], target[1], target[2], yaw_deg)
        pose = streamer.latest
        err = math.dist(pose.position_m, target) if pose is not None else math.inf
        elapsed = time.monotonic() - start
        if err < tol:
            log(f"{label}: converged to {err * 1000:.0f} mm in {elapsed:.1f}s.")
            return
        if elapsed > timeout_s:
            log(f"{label}: settle timeout ({timeout_s:.0f}s), still {err * 1000:.0f} mm out; "
                "proceeding.")
            return
        time.sleep(period)


def _fly(cfg, args, session, streamer, log) -> int:
    tx, ty = args.target
    with KeyboardMonitor() as keyboard:
        try:
            log(f"Streaming GLOBAL-frame extpose at {cfg.mocap.stream_rate_hz:.0f} Hz.")
            streamer.start()
            if not mocap_settle_and_gate(streamer, keyboard=keyboard, session=session,
                                         cfg=cfg, settle_s=MOCAP_SETTLE_S, log=log):
                return 1
            mocap_report = run_preflight(
                (
                    functools.partial(
                        check_mocap_quality,
                        streamer,
                        expected_rate_hz=cfg.mocap.stream_rate_hz,
                        expected_rigid_body=cfg.mocap.rigid_body,
                    ),
                )
            )
            _log_preflight(mocap_report, log)
            if not mocap_report:
                log("Mocap quality preflight failed; keeping the drone disarmed.")
                return 1

            pose = streamer.latest
            if pose is None:
                log("No mocap pose after settle; aborting.")
                return 1
            x0, y0, z0 = pose.position_m
            e = pose.euler_rad
            roll, pitch, yaw0 = (math.degrees(e.roll), math.degrees(e.pitch),
                                 math.degrees(e.yaw))
            log(f"Current global pose: x={x0:+.2f} y={y0:+.2f} z={z0:+.2f} | "
                f"roll={roll:+.1f} pitch={pitch:+.1f} yaw={yaw0:+.1f} deg")

            # --- level gate, before anything arms ---
            if abs(roll) > MAX_TILT_DEG or abs(pitch) > MAX_TILT_DEG:
                log(f"REFUSING TO FLY: not level. |roll|/|pitch| must be < {MAX_TILT_DEG:.0f} "
                    f"deg, got roll {roll:+.1f}, pitch {pitch:+.1f}. The mocap orientation may "
                    "be flipped, or the drone is not sitting flat -- check before arming.")
                return 1
            log(f"Level check passed (roll/pitch within {MAX_TILT_DEG:.0f} deg).")

            session.prepare()  # Kalman + reset so it converges onto the global frame

            yaw_target = yaw0 if args.yaw is None else args.yaw
            session.apply_gains(args.speed)
            if args.yaw is None:
                log(f"Speed capped at {args.speed:.2f} m/s; holding current heading "
                    f"yaw={yaw0:+.1f} deg.")
            else:
                log(f"Speed capped at {args.speed:.2f} m/s; will take off straight up then "
                    f"turn to yaw={yaw_target:+.1f} deg (from {yaw0:+.1f}) at "
                    f"<= {YAW_RATE_DPS:.0f} deg/s.")
            session.arm()
            streamer.start_pose_watch(cfg.mocap.pose_timeout_s)
            log("Armed (arming-request heartbeat keeps it armed through takeoff; no "
                "emergency-stop deadman, so reruns need no reboot; SPACE and the "
                "commander timeout are the radio-loss fallbacks).")
            kill_aware_wait(keyboard, streamer, session, ARM_SETTLE_S)
            session.unlock()

            hz = args.height
            log(f"Flying to global target (x={tx:+.2f}, y={ty:+.2f}) via {hz:.2f} m.")
            # Straight up first, holding the CURRENT heading -- turning on the pad
            # fights the ground and can pin the drone down instead of lifting off.
            _ramp(session, keyboard, streamer, (x0, y0, z0), (x0, y0, hz), yaw0, yaw0,
                  args.speed, log, "takeoff")
            _settle(session, keyboard, streamer, (x0, y0, hz), yaw0,
                    tol=args.tol, timeout_s=args.settle_timeout, log=log, label="takeoff-hold")
            # Then turn in place to the requested heading (skipped if unchanged).
            if abs(_lerp_yaw(yaw0, yaw_target, 1.0) - yaw0) > 1.0:
                _ramp(session, keyboard, streamer, (x0, y0, hz), (x0, y0, hz), yaw0, yaw_target,
                      args.speed, log, "turn")
                _settle(session, keyboard, streamer, (x0, y0, hz), yaw_target,
                        tol=args.tol, timeout_s=args.settle_timeout, log=log, label="turn-hold")
            _ramp(session, keyboard, streamer, (x0, y0, hz), (tx, ty, hz), yaw_target, yaw_target,
                  args.speed, log, "transit")
            _settle(session, keyboard, streamer, (tx, ty, hz), yaw_target,
                    tol=args.tol, timeout_s=args.settle_timeout, log=log, label="arrive")
            streamer.stop_pose_watch()  # a mocap loss must not interrupt the landing
            _ramp(session, keyboard, streamer, (tx, ty, hz), (tx, ty, args.land_z), yaw_target, yaw_target,
                  args.speed, log, "descend")

            session.stop()
            session.disarm()
            log(f"Landed at global (x={tx:+.2f}, y={ty:+.2f}) and disarmed.")
            return 0
        except KillSwitchEngaged:
            session.emergency_stop()
            log("EMERGENCY STOP: spacebar; motors cut.")
            return 2
        except ExtposeStreamFailed as exc:
            session.emergency_stop()
            log(f"EMERGENCY STOP: mocap/extpose stream died ({exc.__cause__}); motors cut.")
            return 2
        except RadioLinkLost:
            log("RADIO LINK LOST: the onboard watchdog stops the motors.")
            return 3
        except MocapLost as exc:
            streamer.stop_pose_watch()  # a mocap loss must not interrupt the landing
            log(f"MOCAP LOST: {exc}. Descending in place.")
            pose = streamer.latest
            yaw = math.degrees(pose.euler_rad.yaw)
            x, y, z = pose.position_m
            try:
                _ramp(session, keyboard, streamer, (x, y, z), (x, y, args.land_z), yaw, yaw,
                      args.speed, log, "descend")
                session.stop()
                session.disarm()
            except KillSwitchEngaged:
                session.emergency_stop()
                log("EMERGENCY STOP: spacebar during the descent; motors cut.")
                return 2
            return 1
        except KeyboardInterrupt:
            log("Interrupted: stop + disarm in place.")
            try:
                session.stop()
                session.disarm()
            except Exception:
                pass
            return 1
        finally:
            streamer.stop()
            session.stop_heartbeat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target", nargs=2, type=float, required=True, metavar=("X", "Y"),
                        help="global-frame x y (m) to land at")
    parser.add_argument("--config", default="configs/drone.yaml")
    parser.add_argument("--yaw", type=float, default=None, metavar="DEG",
                        help="global heading to hold, deg (lifts off straight up, then turns "
                        "to it in place); default: keep the current heading")
    parser.add_argument("--height", type=float, default=0.5,
                        help="hold/transit altitude, global z (m); default 0.5")
    parser.add_argument("--speed", type=float, default=0.25,
                        help="max translational speed (m/s); default 0.25 -- slow is accurate")
    parser.add_argument("--land-z", type=float, default=0.05,
                        help="global z (m) at which motors are cut; default 0.05")
    parser.add_argument("--tol", type=float, default=0.05,
                        help="arrival tolerance (m) before descending; default 0.05")
    parser.add_argument("--settle-timeout", type=float, default=12.0,
                        help="max seconds to wait for convergence at a hold; default 12")
    args = parser.parse_args(argv)

    if args.speed <= 0 or args.height <= 0:
        parser.error("--speed and --height must be positive")

    cfg = load_config(args.config)

    print(__doc__.split("\n\n")[0])
    source = build_live_pose_source(cfg=cfg)
    log, _console, close_log = make_run_logger(None)  # terminal-only, no run dir
    try:
        with CrazyflieRadioBackend(cfg.radio.uri) as backend:
            hardware_report = _run_crazyflie_preflight(backend.crazyflie, log=log)
            if not hardware_report:
                log("Crazyflie preflight failed; keeping the drone disarmed.")
                return 1
            session = GotoSession(backend)
            # GLOBAL frame: identity transform (no origin/yaw zeroing).
            streamer = ExtposeStreamer(
                source=source, backend=backend,
                max_rate_hz=cfg.mocap.stream_rate_hz, pace=False, transform=None,
            )
            return _fly(cfg, args, session, streamer, log)
    finally:
        close_log()
        close = getattr(source, "close", None)
        if close is not None:
            close()


def _log_preflight(report, log) -> None:
    for line in report.format().splitlines():
        log(line)


def _run_crazyflie_preflight(cf, *, log):
    report = run_preflight(standard_flight_checks(cf))
    _log_preflight(report, log)
    return report


if __name__ == "__main__":
    raise SystemExit(main())
