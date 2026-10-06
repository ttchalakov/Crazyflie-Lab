#!/usr/bin/env python3
"""Mocap waypoint demo: fly a sequence of absolute position setpoints.

Uses full motion-capture pose so the estimator fuses external position and
orientation while the built-in PID position controller tracks absolute setpoints
streamed from the host. Frames are +z up with the origin at the drone's starting
pose (the first mocap sample fixes (0, 0, 0)).

Flight profile: fly the TIMED_SETPOINTS list (below) in order. Each absolute position
setpoint is streamed for its configured duration, then the schedule advances
regardless of whether the drone reached it. After the last point it lands
straight down from its latest measured position. The first waypoint doubles as
the takeoff, so give it enough time and keep it straight up over the origin.

Edit TIMED_SETPOINTS at the top to configure the flight.

Kill controls, live during every phase:
  SPACE  -> emergency stop (supervisor cuts motors immediately), exit code 2.
  Ctrl-C -> graceful shutdown (straight-down land in place, then disarm).

Mocap-loss policy (always mocap-fused): refuses to arm unless mocap samples
arrived during settle. Once armed, no new pose for mocap.pose_timeout_s (1 s by
default) -> the Ctrl-C landing, with a mocap_lost event saying whether the rigid
body was untracked or the mocap connection lost; shorter gaps are left to the
estimator. An error sending poses with the radio alive -> host emergency stop;
radio death -> the firmware deadman (fed at 4 Hz from arming) cuts the motors
onboard. The deadman stays armed until reboot; run cf-lab reboot
before the next flight. This runner never reboots the drone or AI deck.

The lab mocap is OptiTrack, read using the transport configured in drone.yaml.
The Crazyflie connection is always the configured radio link.
"""

from __future__ import annotations

import argparse
import functools
import math
from pathlib import Path

from cf_lab.backends.crazyflie import CrazyflieRadioBackend
from cf_lab.config import load_config
from cf_lab.crazyflie import read_rate_pid_parameters
from cf_lab.mocap import MocapLost, build_live_pose_source, origin_zeroed_streamer
from cf_lab.preflight import (
    check_camera_mocap_coexistence,
    check_configuration,
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
    kill_aware_wait,
    make_run_logger,
    mocap_settle_and_gate,
    stream_segment,
)
from cf_lab.telemetry import TelemetryRecorder
from cf_lab.trajectory import figure_eight_setpoints

# Absolute position setpoints are streamed at this rate.
SETPOINT_RATE_HZ = 50.0
# "Landed" setpoint height: a touch above the floor before the motors stop.
LAND_SETPOINT_Z_M = 0.01
LAND_S = 3.0  # straight-down descent duration
# After arming, hold at idle before the first setpoint.
ARM_SETTLE_S = 1.0
# Estimator settle before the mocap arming gate.
MOCAP_SETTLE_S = 2.0
# Horizontal x/y PID values applied in RAM before arming.
POS_XY_KP = 3.0  # posCtlPid.xKp/yKp        (stock 2.0)  position P -> desired vel
POS_XY_KI = 0.25  # posCtlPid.xKi/yKi        (stock 0.0)  position I -> kills EKF vel bias
VEL_XY_KP = 20.0  # velCtlPid.vxKp/vyKp      (stock 25.0) velocity P -> attitude
VEL_XY_KI = 2.0  # velCtlPid.vxKi/vyKi      (stock 1.0)  velocity I
XY_VEL_MAX = 2.0  # posCtlPid.xVelMax/yVelMax (stock 1.0) max commanded speed (m/s)
# Vertical position PID values applied in RAM before arming.
POS_Z_KP = 2.0  # posCtlPid.zKp     (stock 2.0) position P -> desired vertical velocity
POS_Z_KI = 0.0  # posCtlPid.zKi     (stock 0.5) off: no windup on altitude steps
Z_VEL_MAX = 1.0  # posCtlPid.zVelMax (stock 1.0) max commanded vertical speed (m/s)

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = EXAMPLE_DIR / "artifacts"

# Each profile uses (duration_s, x, y, z, yaw_deg) in the start-zeroed mocap
# frame; leave exactly one TIMED_SETPOINTS assignment active.

# Hover at 0.4 m and make two small +x adjustments before recentering to land.
# TIMED_SETPOINTS = [
#     (3.0, 0.0, 0.0, 0.4, 0.0),
#     (12.0, 0.0, 0.0, 0.4, 0.0),
#     (2.5, 0.10, 0.0, 0.4, 0.0),
#     (2.5, 0.0, 0.0, 0.4, 0.0),
#     (2.5, 0.15, 0.0, 0.4, 0.0),
#     (2.5, 0.0, 0.0, 0.4, 0.0),
# ]

# Sweep each translation axis and the full yaw range from a centered 0.8 m hover.
# TIMED_SETPOINTS = [
#     (4.0, 0.0, 0.0, 0.8, 0.0),
#     (4.0, -0.6, 0.0, 0.8, 0.0),
#     (4.0, 0.6, 0.0, 0.8, 0.0),
#     (4.0, 0.0, 0.0, 0.8, 0.0),
#     (4.0, 0.0, -0.6, 0.8, 0.0),
#     (4.0, 0.0, 0.6, 0.8, 0.0),
#     (4.0, 0.0, 0.0, 0.8, 0.0),
#     (4.0, 0.0, 0.0, 0.5, 0.0),
#     (4.0, 0.0, 0.0, 1.1, 0.0),
#     (4.0, 0.0, 0.0, 0.8, 0.0),
#     (4.0, 0.0, 0.0, 0.8, -180.0),
#     (4.0, 0.0, 0.0, 0.8, -90.0),
#     (4.0, 0.0, 0.0, 0.8, 0.0),
#     (4.0, 0.0, 0.0, 0.8, 90.0),
#     (4.0, 0.0, 0.0, 0.8, 180.0),
#     (4.0, 0.0, 0.0, 0.8, 0.0),
# ]

# Couple aggressive translations, altitude changes, and large yaw reversals.
# TIMED_SETPOINTS = [
#     (4.0, 0.0, 0.0, 0.8, 0.0),
#     (4.0, 0.0, 0.0, 0.8, 180.0),
#     (4.0, 0.0, 0.0, 0.8, -90.0),
#     (5.0, 0.8, 0.0, 0.9, -90.0),
#     (5.0, 0.8, 0.8, 0.9, 0.0),
#     (6.0, -0.2, 0.8, 1.3, 45.0),
#     (6.0, -0.2, -0.2, 0.5, 135.0),
#     (6.0, 0.8, -0.2, 0.9, -45.0),
#     (5.0, 0.0, 0.0, 0.8, 0.0),
#     (8.0, 0.0, 0.0, 0.8, 0.0),
# ]

# Fill the configured x/y/z/yaw ranges with the generated three-dimensional figure eight.
# Two loops at 19.2 s each hold the peak demand to ~0.43 m/s, slow enough that
# the position loop tracks the curve instead of saturating on it.
TIMED_SETPOINTS = figure_eight_setpoints(
    x_range=(-0.5, 0.5),
    y_range=(-0.5, 0.5),
    z_range=(0.3, 2.0),
    yaw_range_deg=(-180, 180),
    loops=2,
    duration_s=38.4,
)


class RadioSession(FlightRadioSession):
    """Drive a real Crazyflie over the open cflib radio link, mocap-fused.

    Inherits arm/unlock/disarm/stop/emergency_stop and the Kalman estimator from
    FlightRadioSession; adds the absolute-position setpoint and the flight
    telemetry streams.
    """

    def __init__(
        self,
        backend,
        recorder: TelemetryRecorder,
    ) -> None:
        super().__init__(backend)
        self._recorder = recorder

    def goto(self, x: float, y: float, z: float, yaw_deg: float) -> None:
        self._recorder.record_setpoint(x, y, z, yaw_deg)
        self._cf.commander.send_position_setpoint(x, y, z, yaw_deg)

    def apply_pid_values(self) -> str:
        """Apply the exposed RAM-only x/y and z PID values while disarmed."""
        p = self._cf.param
        for axis in ("x", "y"):
            p.set_value(f"posCtlPid.{axis}Kp", POS_XY_KP)
            p.set_value(f"posCtlPid.{axis}Ki", POS_XY_KI)
            p.set_value(f"velCtlPid.v{axis}Kp", VEL_XY_KP)
            p.set_value(f"velCtlPid.v{axis}Ki", VEL_XY_KI)
            p.set_value(f"posCtlPid.{axis}VelMax", XY_VEL_MAX)
        p.set_value("posCtlPid.zKp", POS_Z_KP)
        p.set_value("posCtlPid.zKi", POS_Z_KI)
        p.set_value("posCtlPid.zVelMax", Z_VEL_MAX)
        if self._recorder.enabled:
            # pid_rate parameters are persistent and may differ from the build
            # defaults. Preserve the actual values alongside this run.
            self._recorder.event(
                "controller_parameters",
                "active values after startup overrides",
                rate_pid=read_rate_pid_parameters(p),
                outer_xy={
                    "pos_kp": POS_XY_KP,
                    "pos_ki": POS_XY_KI,
                    "vel_kp": VEL_XY_KP,
                    "vel_ki": VEL_XY_KI,
                    "vel_max": XY_VEL_MAX,
                },
                outer_z={
                    "pos_kp": POS_Z_KP,
                    "pos_ki": POS_Z_KI,
                    "vel_max": Z_VEL_MAX,
                },
            )
        return (
            f"Applied x/y PID values: posKp={POS_XY_KP}, posKi={POS_XY_KI}, "
            f"velKp={VEL_XY_KP}, velKi={VEL_XY_KI}, xyVelMax={XY_VEL_MAX}; "
            f"z position values: posKp={POS_Z_KP}, posKi={POS_Z_KI}, "
            f"zVelMax={Z_VEL_MAX}."
        )

    def start_flight_log(self) -> None:
        """Start the firmware-pushed telemetry log blocks (state + controls).

        The recorder registers one LogConfig per stream; the firmware's log
        task then samples and pushes at the configured period on its own —
        rates and toggles come from the config's `telemetry:` section.
        """
        self._recorder.attach_stm32(self._cf)

    def stop_flight_log(self) -> None:
        self._recorder.detach_stm32()


def validate_waypoints(waypoints) -> None:
    """Reject malformed timed setpoints before opening mocap or radio."""
    if not waypoints:
        raise ValueError("TIMED_SETPOINTS must contain at least one timed setpoint")
    for index, waypoint in enumerate(waypoints, 1):
        if len(waypoint) != 5:
            raise ValueError(
                f"TIMED_SETPOINTS entry {index} must be (duration_s, x, y, z, yaw_deg)"
            )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in waypoint):
            raise ValueError(f"TIMED_SETPOINTS entry {index} must contain only finite numbers")
        if waypoint[0] <= 0.0:
            raise ValueError(f"TIMED_SETPOINTS entry {index} duration_s must be greater than zero")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/drone.yaml")
    args = parser.parse_args(argv)

    try:
        validate_waypoints(TIMED_SETPOINTS)
    except ValueError as exc:
        print(f"Waypoint validation failed; refusing to connect: {exc}")
        return 1

    cfg = load_config(args.config)

    # Transport, host/IP, ports, rigid body, and rate all come from the config.
    source = build_live_pose_source(cfg=cfg)

    recorder = TelemetryRecorder(
        cfg.telemetry, run_root=ARTIFACTS_DIR, lab_config=cfg, example="mocap_waypoint"
    )
    recorder.start()
    source = recorder.wrap_pose_source(source)
    # With telemetry on, the debug log joins the parquet streams in the run dir
    # (already timestamped); otherwise it gets a timestamped file in artifacts/.
    log, _console_sink, close_log = make_run_logger(
        recorder.run_dir or ARTIFACTS_DIR, timestamped=not recorder.enabled
    )
    print(__doc__.split("\n\n")[0])
    print("Frame: +z up, origin and zero yaw at the drone's starting pose on the floor.")
    print(
        f"{len(TIMED_SETPOINTS)} timed setpoint(s), then straight-down land. "
        "SPACE = emergency stop, Ctrl-C = land."
    )
    print("Gain source: five x/y PID constants in this file.")
    print()

    try:
        static_report = run_preflight(
            (
                functools.partial(check_configuration, cfg),
                functools.partial(check_camera_mocap_coexistence, cfg),
            ),
            telemetry=recorder,
        )
        _log_preflight(static_report, log)
        if not static_report:
            log("Preflight failed before hardware startup; refusing to fly.")
            return 1

        # The route check runs before opening the camera connection. The
        # recorder is the single feed owner, and camera loss is a soft failure
        # for this experiment.
        recorder.attach_camera_from_config(log=log)

        with CrazyflieRadioBackend(cfg.radio.uri) as backend:
            hardware_report = _run_crazyflie_preflight(
                backend.crazyflie,
                recorder=recorder,
                log=log,
            )
            if not hardware_report:
                log("Crazyflie preflight failed; keeping the drone disarmed.")
                return 1

            session = RadioSession(backend, recorder)
            result = _fly(
                cfg,
                source=source,
                backend=backend,
                session=session,
                log=log,
                waypoints=TIMED_SETPOINTS,
                recorder=recorder,
            )
        return result
    finally:
        # Recorder first: it finalizes parquet + manifest on every exit path.
        recorder.close()
        close_log()
        close = getattr(source, "close", None)
        if close is not None:
            close()


def _log_preflight(report, log) -> None:
    for line in report.format().splitlines():
        log(line)


def _run_crazyflie_preflight(cf, *, recorder, log):
    # Save cflib's downloaded log/parameter TOCs even if a later check refuses
    # the flight, so the attempted run remains diagnosable.
    recorder.capture_crazyflie_toc(cf)
    report = run_preflight(standard_flight_checks(cf), telemetry=recorder)
    _log_preflight(report, log)
    return report


def _record_frame_metadata(recorder) -> None:
    """Record how live OptiTrack poses are transformed and fused."""
    recorder.event(
        "extpose_config",
        position_only=False,
        orientation_baseline=True,
        rezeroed=True,
        coordinate_frame="start_origin_yaw_zeroed",
        origin_rule="first mocap sample position and yaw",
    )


def _fly(cfg, *, source, backend, session, log, waypoints, recorder) -> int:
    # The backend tap records the zeroed poses actually sent up the radio, and
    # the origin event records the transform — together they let analysis
    # recreate the exact extpose signal the estimator received.
    _record_frame_metadata(recorder)
    streamer = origin_zeroed_streamer(
        source=source,
        source_kind="optitrack",
        backend=recorder.wrap_extpose_backend(backend),
        cfg=cfg,
        on_origin=lambda position_m, yaw_rad: recorder.event(
            "origin_zeroed",
            x=position_m[0],
            y=position_m[1],
            z=position_m[2],
            yaw_rad=yaw_rad,
        ),
    )
    current_target = [0.0, 0.0, 0.0, 0.0]

    with KeyboardMonitor() as keyboard:

        def straight_down_land() -> None:
            # A landing also answers a mocap loss, so that loss must not interrupt it.
            streamer.stop_pose_watch()
            pose = streamer.latest
            if pose is None:
                x, y, z, yaw = current_target
                anchor_source = "last_commanded_setpoint"
            else:
                x, y, z = pose.position_m
                yaw = current_target[3]
                anchor_source = "latest_mocap_position"
            recorder.event("land_start", x=x, y=y, z=z, yaw_deg=yaw, anchor_source=anchor_source)
            log(f"Straight-down landing at ({x:+.2f}, {y:+.2f}) over {LAND_S}s.")
            stream_segment(
                LAND_S,
                lambda f: session.goto(x, y, z + (LAND_SETPOINT_Z_M - z) * f, yaw),
                keyboard=keyboard,
                streamer=streamer,
                session=session,
                rate_hz=SETPOINT_RATE_HZ,
            )
            session.stop()
            session.disarm()
            log("Landed and disarmed.")

        def land_unless_space() -> int:
            """The straight-down landing; SPACE during it still cuts the motors."""
            try:
                straight_down_land()
            except KillSwitchEngaged:
                session.emergency_stop()
                recorder.event("emergency_stop", "spacebar during land")
                log("EMERGENCY STOP: spacebar pressed during land.")
                return 2
            return 1

        def run_waypoints() -> None:
            n = len(waypoints)
            for i, (duration_s, x, y, z, yaw) in enumerate(waypoints, 1):
                current_target[:] = [x, y, z, yaw]
                recorder.event(
                    "waypoint",
                    f"{i}/{n}",
                    duration_s=duration_s,
                    x=x,
                    y=y,
                    z=z,
                    yaw_deg=yaw,
                )
                log(
                    f"Waypoint {i}/{n}: stream ({x:+.2f}, {y:+.2f}, {z:+.2f}) "
                    f"yaw {yaw:+.0f} for {duration_s:.1f}s."
                )
                stream_segment(
                    duration_s,
                    lambda _fraction, target=(x, y, z, yaw): session.goto(*target),
                    keyboard=keyboard,
                    streamer=streamer,
                    session=session,
                    rate_hz=SETPOINT_RATE_HZ,
                )
            straight_down_land()

        try:
            log(f"Streaming origin-zeroed extpose at {cfg.mocap.stream_rate_hz:.0f} Hz.")
            streamer.start()
            if not mocap_settle_and_gate(
                streamer, keyboard=keyboard, session=session, cfg=cfg,
                settle_s=MOCAP_SETTLE_S, log=log,
            ):
                return 1
            mocap_report = run_preflight(
                (
                    functools.partial(
                        check_mocap_quality,
                        streamer,
                        expected_rate_hz=cfg.mocap.stream_rate_hz,
                        expected_rigid_body=cfg.mocap.rigid_body,
                    ),
                ),
                telemetry=recorder,
            )
            _log_preflight(mocap_report, log)
            if not mocap_report:
                log("Mocap quality preflight failed; keeping the drone disarmed.")
                return 1
            session.prepare()
            log(session.apply_pid_values())
            session.arm()
            streamer.start_pose_watch(cfg.mocap.pose_timeout_s)
            recorder.event("armed")
            log("Armed; onboard radio-loss deadman active (4 Hz heartbeat).")

            session.start_flight_log()
            if recorder.enabled:
                log(f"Recording telemetry streams to {recorder.run_dir}.")
            else:
                log("Telemetry disabled (configs `telemetry.enabled: false`).")

            log(f"Settling at idle for {ARM_SETTLE_S:.1f}s before setpoints.")
            kill_aware_wait(keyboard, streamer, session, ARM_SETTLE_S)
            session.unlock()

            run_waypoints()
            result = 0
        except KillSwitchEngaged:
            session.emergency_stop()
            recorder.event("emergency_stop", "spacebar")
            log("EMERGENCY STOP: spacebar pressed, motors cut.")
            result = 2
        except ExtposeStreamFailed as exc:
            session.emergency_stop()
            recorder.event("emergency_stop", f"extpose stream failed: {exc.__cause__}")
            log(f"EMERGENCY STOP: extpose stream failed ({exc.__cause__}).")
            result = 2
        except RadioLinkLost:
            recorder.event("radio_lost")
            log("RADIO LINK LOST: aborting. The onboard watchdog stops the motors.")
            result = 3
        except MocapLost as exc:
            recorder.event("mocap_lost", str(exc))
            log(f"MOCAP LOST: {exc}. Landing.")
            result = land_unless_space()
        except KeyboardInterrupt:
            recorder.event("interrupt", "Ctrl-C, graceful land")
            log("Interrupted, attempting graceful land.")
            result = land_unless_space()
        finally:
            session.stop_flight_log()
            streamer.stop()
            session.stop_heartbeat()

    dropped = f", dropped {streamer.dropped} stale" if streamer.dropped else ""
    log(f"Done. Forwarded {streamer.sent} extpose sample(s){dropped}.")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
