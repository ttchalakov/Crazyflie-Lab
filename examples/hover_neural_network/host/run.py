#!/usr/bin/env python3
"""Mocap hover demo: PID takeoff, neural hover, then PID landing.

The first mocap pose fixes a +z-up frame at the drone's starting position and
zero yaw. The built-in PID controller rises to ``HOVER_HEIGHT_M`` and settles,
then the host immediately enables the out-of-tree neural controller. On the
STM32, the policy state is the 11-float yaw-invariant observation -- body-frame
position error and velocity, world-up in the body frame, and heading as
(sin, cos) -- built against the position setpoint the host is already commanding.
The GAP8 returns collective thrust and a body-rate setpoint, which the STM32
tracks with the stock attitude-rate PID.

After ``EXPERIMENT_S`` the host immediately returns to the warm built-in PID,
recenters over the recorded start, descends slowly, stops the motors, and
disarms. Camera, mocap, telemetry, and radio settings come from ``drone.yaml``.

Kill controls, live during every phase:
  SPACE  -> emergency stop (supervisor cuts motors immediately), exit code 2.
  Ctrl-C -> disable neural control and land under the built-in PID.

Once armed, no new mocap pose for mocap.pose_timeout_s (1 s by default) lands
the same way as Ctrl-C, with a mocap_lost event saying whether the rigid body
was untracked or the mocap connection lost; shorter gaps are left to the
estimator.

TODO -- both unverified, and neither is checkable on a bench. Confirm them in
the mocap room before trusting a flight:

  1. Body-frame pitch sign. hover_nn_controller.c sends the policy's body-rate
     setpoint as ``.pitch = -wy``: the firmware's pitch convention is inverted
     relative to the FLU frame the policy trains in (it feeds -gyro.y to the rate
     PID). If it is wrong, pitch diverges while roll behaves normally -- so watch
     a low, short hover and be ready on the spacebar.

  2. Standing yaw bias. At the exact hover equilibrium the trained policy still
     commands a small yaw rate, so expect a slow steady rotation in hover. It is
     a training artifact, not a firmware bug: the lever is LOSS_YAW_WEIGHT in
     model/hover_policy.py followed by a retrain, not anything on the drone.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import threading
import time
from pathlib import Path

from cf_lab.backends.crazyflie import CrazyflieRadioBackend
from cf_lab.config import CustomStreamConfig, load_config
from cf_lab.mocap import MocapLost, build_live_pose_source, origin_zeroed_streamer
from cf_lab.preflight import (
    check_camera_mocap_coexistence,
    check_camera_stream,
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


HOVER_HEIGHT_M = 0.8
TAKEOFF_S = 4.0
PID_SETTLE_S = 3.0
EXPERIMENT_S = 10.0
PID_RECOVERY_S = 1.0
LAND_S = 4.0
LAND_SETPOINT_Z_M = 0.01
ARM_SETTLE_S = 1.0
MOCAP_SETTLE_S = 2.0  # estimator settle before the mocap arming gate
SETPOINT_RATE_HZ = 50.0
NEURAL_READY_TIMEOUT_S = 25.0
NEURAL_READY_WINDOW_S = 1.0
NEURAL_READY_MIN_RATE_HZ = 95.0
CAMERA_PREFLIGHT_TIMEOUT_S = 8.0
# Enough frames (about half a second) that the logged FPS is a measurement.
CAMERA_PREFLIGHT_FRAMES = 30
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
DRONE_CONFIG = "configs/drone.yaml"

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = EXAMPLE_DIR / "artifacts"

# Demo-owned logs are present whenever telemetry itself is enabled. They are
# deliberately not toggles in drone.yaml: the physical NN command and CPX
# timing are essential context for interpreting this experiment.
DEMO_TELEMETRY_STREAMS = (
    CustomStreamConfig(
        name="hnn_control",
        period_ms=10,
        variables=(
            "hnn.active:uint8_t",
            "hnn.thrustN:float",
            "hnn.wx:float",
            "hnn.wy:float",
            "hnn.wz:float",
        ),
    ),
    CustomStreamConfig(
        name="hnn_link",
        period_ms=20,
        variables=("hnn.rxCount:uint32_t",),
    ),
)


class RadioSession(FlightRadioSession):
    """Real-radio session with neural-controller and telemetry verbs."""

    def __init__(self, backend, recorder: TelemetryRecorder) -> None:
        super().__init__(backend)
        self._recorder = recorder

    def goto(self, x: float, y: float, z: float, yaw_deg: float = 0.0) -> None:
        self._recorder.record_setpoint(x, y, z, yaw_deg)
        self._cf.commander.send_position_setpoint(x, y, z, yaw_deg)

    def set_neural_controller(self, enabled: bool) -> None:
        self._cf.param.set_value("hnn.enable", 1 if enabled else 0)

    def wait_until_neural_ready(self, timeout_s: float) -> tuple[bool, dict[str, float]]:
        """Require a sustained stream of GAP8 policy replies.

        The reply rate is the readiness signal. The STM32 only accepts a reply
        that is well-formed, in-range and finite, so a sustained
        NEURAL_READY_MIN_RATE_HZ means the whole chain -- state out, inference,
        command back, validation -- is running at rate.
        """
        from cflib.crazyflie.log import LogConfig

        ready = threading.Event()
        latest: dict[str, float] = {}
        config = LogConfig(name="hnn_ready", period_in_ms=100)
        config.add_variable("hnn.rxCount", "uint32_t")
        window_start: tuple[float, int] | None = None

        def on_sample(_timestamp, data, _config) -> None:
            nonlocal window_start
            latest.update(data)
            now = time.monotonic()
            count = int(data["hnn.rxCount"])
            if window_start is None:
                window_start = (now, count)
                return
            elapsed = now - window_start[0]
            if elapsed <= 0.0:
                return
            response_rate_hz = (count - window_start[1]) / elapsed
            latest["responseRateHz"] = response_rate_hz
            if response_rate_hz < NEURAL_READY_MIN_RATE_HZ:
                window_start = (now, count)  # restart the window on any stall
                return
            if elapsed >= NEURAL_READY_WINDOW_S:
                ready.set()

        config.data_received_cb.add_callback(on_sample)
        self._cf.log.add_config(config)
        config.start()
        try:
            return ready.wait(timeout_s), dict(latest)
        finally:
            try:
                config.stop()
                config.delete()
            except Exception:
                pass

    def start_flight_log(self) -> None:
        self._recorder.attach_stm32(self._cf)

    def stop_flight_log(self) -> None:
        self._recorder.detach_stm32()

    def emergency_stop(self) -> None:
        try:
            self.set_neural_controller(False)
        except Exception:
            pass
        super().emergency_stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DRONE_CONFIG)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    source = build_live_pose_source(cfg=cfg)
    telemetry = dataclasses.replace(
        cfg.telemetry,
        custom=cfg.telemetry.custom + DEMO_TELEMETRY_STREAMS,
    )
    cfg = dataclasses.replace(cfg, telemetry=telemetry)
    recorder = TelemetryRecorder(
        cfg.telemetry,
        run_root=ARTIFACTS_DIR,
        lab_config=cfg,
        example="hover_neural_network",
    )
    recorder.start()
    source = recorder.wrap_pose_source(source)
    log, _console_sink, close_log = make_run_logger(
        recorder.run_dir or ARTIFACTS_DIR,
        timestamped=not recorder.enabled,
    )

    print(__doc__.split("\n\n")[0])
    print("Frame: +z up; origin and zero yaw are the drone's starting floor pose.")
    print(
        f"PID to {HOVER_HEIGHT_M:.2f} m, neural hover for {EXPERIMENT_S:.1f} s, "
        "then PID landing at the start."
    )
    print("SPACE = emergency stop, Ctrl-C = PID landing.")
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

        # Blanking the camera SSID flies this control-only: the deck still captures,
        # but with no client it never encodes, so the cluster serves only the
        # policy. No reflash of the deck is needed.
        if not cfg.camera.enabled:
            log("Camera disabled in config (SSID blank): flying control-only.")
            log("No video will be recorded and the camera preflight is skipped.")
            recorder.event("camera_disabled", reason="camera.ssid blank")
        else:
            if not recorder.attach_camera_from_config(
                connect_timeout_s=CAMERA_PREFLIGHT_TIMEOUT_S,
                log=log,
            ):
                log("Camera connection failed; this combined demo will not arm.")
                return 1
            camera_report = run_preflight(
                (
                    functools.partial(
                        check_camera_stream,
                        feed=recorder.camera_feed,
                        timeout_s=CAMERA_PREFLIGHT_TIMEOUT_S,
                        sample_frames=CAMERA_PREFLIGHT_FRAMES,
                        expected_width=CAMERA_WIDTH,
                        expected_height=CAMERA_HEIGHT,
                        expected_format="jpeg",
                    ),
                ),
                telemetry=recorder,
            )
            _log_preflight(camera_report, log)
            if not camera_report:
                log("Camera stream preflight failed; this combined demo will not arm.")
                return 1
        with CrazyflieRadioBackend(cfg.radio.uri, console_sink=_console_sink) as backend:
            hardware_report = _run_crazyflie_preflight(
                backend.crazyflie,
                recorder=recorder,
                log=log,
            )
            if not hardware_report:
                log("Crazyflie preflight failed; keeping the drone disarmed.")
                return 1
            session = RadioSession(backend, recorder)
            return _fly(
                cfg,
                source=source,
                backend=backend,
                session=session,
                recorder=recorder,
                log=log,
            )
    finally:
        recorder.close()
        close_log()
        close = getattr(source, "close", None)
        if close is not None:
            close()


def _log_preflight(report, log) -> None:
    for line in report.format().splitlines():
        log(line)


def _run_crazyflie_preflight(cf, *, recorder, log):
    recorder.capture_crazyflie_toc(cf)
    report = run_preflight(
        standard_flight_checks(cf, required_decks=("bcAI",)),
        telemetry=recorder,
    )
    _log_preflight(report, log)
    return report


def _record_frame_metadata(recorder: TelemetryRecorder) -> None:
    recorder.event(
        "extpose_config",
        position_only=False,
        orientation_baseline=True,
        rezeroed=True,
        coordinate_frame="start_origin_yaw_zeroed",
        origin_rule="first mocap sample position and yaw",
    )
    recorder.event(
        "neural_equilibrium",
        x_m=0.0,
        y_m=0.0,
        z_m=HOVER_HEIGHT_M,
        yaw_rad=0.0,
        state_order="ex_b,ey_b,ez_b,vx_b,vy_b,vz_b,upx,upy,upz,sin_yaw_err,cos_yaw_err",
        action_order="thrust_N,wx,wy,wz",
    )


def _fly(cfg, *, source, backend, session, recorder, log) -> int:
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

    with KeyboardMonitor() as keyboard:

        def pid_hold(duration_s: float, z_m: float = HOVER_HEIGHT_M) -> None:
            stream_segment(
                duration_s,
                lambda _fraction: session.goto(0.0, 0.0, z_m, 0.0),
                keyboard=keyboard,
                streamer=streamer,
                session=session,
                rate_hz=SETPOINT_RATE_HZ,
            )

        def land_at_start() -> None:
            # A landing also answers a mocap loss, so that loss must not interrupt it.
            streamer.stop_pose_watch()
            session.set_neural_controller(False)
            pose = streamer.latest
            start_z = pose.position_m[2] if pose is not None else HOVER_HEIGHT_M
            recorder.event("land_start", x=0.0, y=0.0, z=start_z, yaw_deg=0.0)
            log(f"PID landing over the recorded start over {LAND_S:.1f}s.")
            stream_segment(
                LAND_S,
                lambda fraction: session.goto(
                    0.0,
                    0.0,
                    start_z + (LAND_SETPOINT_Z_M - start_z) * fraction,
                    0.0,
                ),
                keyboard=keyboard,
                streamer=streamer,
                session=session,
                rate_hz=SETPOINT_RATE_HZ,
            )
            session.stop()
            session.disarm()
            recorder.event("disarmed")
            log("Landed at the start and disarmed.")

        def land_unless_space() -> int:
            """The PID landing; SPACE during it still cuts the motors."""
            try:
                land_at_start()
            except KillSwitchEngaged:
                session.emergency_stop()
                recorder.event("emergency_stop", "spacebar during landing")
                log("EMERGENCY STOP: spacebar pressed during landing.")
                return 2
            return 1

        try:
            log(f"Streaming origin-zeroed full extpose at {cfg.mocap.stream_rate_hz:.0f} Hz.")
            streamer.start()
            if not mocap_settle_and_gate(
                streamer,
                keyboard=keyboard,
                session=session,
                cfg=cfg,
                settle_s=MOCAP_SETTLE_S,
                log=log,
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
            session.set_neural_controller(False)
            session.start_flight_log()
            log(f"Waiting up to {NEURAL_READY_TIMEOUT_S:.0f}s for GAP8 policy outputs.")
            neural_ready, status = session.wait_until_neural_ready(NEURAL_READY_TIMEOUT_S)
            recorder.event("neural_readiness", ready=neural_ready, **status)
            if not neural_ready:
                log(f"Neural pipeline not ready ({status}); keeping the drone disarmed.")
                return 1
            log(
                "Neural pipeline ready "
                f"({status.get('responseRateHz', 0.0):.1f} on-deadline responses/s)."
            )

            session.arm()
            streamer.start_pose_watch(cfg.mocap.pose_timeout_s)
            recorder.event("armed")
            log("Armed; onboard radio-loss deadman active (4 Hz heartbeat).")
            log(f"Settling at idle for {ARM_SETTLE_S:.1f}s.")
            kill_aware_wait(keyboard, streamer, session, ARM_SETTLE_S)
            session.unlock()

            recorder.event("pid_takeoff_start")
            log(f"PID takeoff to {HOVER_HEIGHT_M:.2f}m over {TAKEOFF_S:.1f}s.")
            stream_segment(
                TAKEOFF_S,
                lambda fraction: session.goto(0.0, 0.0, HOVER_HEIGHT_M * fraction, 0.0),
                keyboard=keyboard,
                streamer=streamer,
                session=session,
                rate_hz=SETPOINT_RATE_HZ,
            )
            log(f"Holding under PID for {PID_SETTLE_S:.1f}s before the handover.")
            pid_hold(PID_SETTLE_S)

            recorder.event("neural_control_start", duration_s=EXPERIMENT_S)
            log(f"Immediate handover to neural control for {EXPERIMENT_S:.1f}s.")
            session.set_neural_controller(True)
            pid_hold(EXPERIMENT_S)

            recorder.event("neural_control_stop")
            session.set_neural_controller(False)
            log(f"Immediate return to PID; recentering for {PID_RECOVERY_S:.1f}s.")
            pid_hold(PID_RECOVERY_S)
            land_at_start()
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
            log(f"MOCAP LOST: {exc}. Disabling neural control and landing under PID.")
            result = land_unless_space()
        except KeyboardInterrupt:
            recorder.event("interrupt", "Ctrl-C, PID landing")
            log("Interrupted; disabling neural control and attempting PID landing.")
            result = land_unless_space()
        finally:
            try:
                session.set_neural_controller(False)
            except Exception:
                pass
            session.stop_flight_log()
            streamer.stop()
            session.stop_heartbeat()

    dropped = f", dropped {streamer.dropped} stale" if streamer.dropped else ""
    log(f"Done. Forwarded {streamer.sent} extpose sample(s){dropped}.")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
