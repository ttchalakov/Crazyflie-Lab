#!/usr/bin/env python3
"""Telemetry stress bench: every stream type at once, benchtop, props never arm.

Loads the telemetry pipeline the way a worst-case run would and then verifies
the recording against ground truth:

  STM32 --radio--> host: 7 base log blocks at --log-hz (12D state, PID commands,
                         and the tba/tbb STM32 and tga/tgb GAP8 waveforms), plus
                         telemetry.custom blocks from the YAML config
  host fake mocap:       self-paced orbit source at --mocap-hz, tapped raw,
                         origin-zeroed, decimated and sent as extpose uplink
  host synth setpoints:  recorded at 50 Hz (never sent to the commander)

The drone must be flashed with examples/telemetry_bench/controller (stock
firmware + an app publishing log groups tba/tbb whose values have closed-form
ground truth) and its AI deck with examples/telemetry_bench/gap8 (the same
waveforms, relayed as tga/tgb). The runner never arms and never sends motor
setpoints.

After the run it calls the shared ``cf_lab.telemetry.validate_run`` integrity
checker, then adds the bench's own checks: the mocap clock fit recovers the
orbit's frame period, and the tba and tga ticks advance and their waveforms
match ground truth computed from the tick. Exit code 0 means every applicable
check passed; hardware-only checks are explicitly skipped under --backend fake.

Dry run without hardware: --backend fake (mocap/extpose/setpoints paths only).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import time
from pathlib import Path

from cf_lab.backends.crazyflie import CrazyflieRadioBackend
from cf_lab.backends.fake import FakeCrazyflieBackend
from cf_lab.config import CustomStreamConfig, TelemetryConfig, load_config
from cf_lab.mocap.relative import OriginZeroer
from cf_lab.mocap.stream import ExtposeStreamer
from cf_lab.mocap.types import EulerAngles, PoseSample, Quaternion
from cf_lab.runner import make_run_logger
from cf_lab.runner.session import RadioSessionBase
from cf_lab.telemetry import TelemetryRecorder, validate_run
from cf_lab.telemetry.schema import log_period_ms

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
SETPOINT_RATE_HZ = 50.0
STATUS_PERIOD_S = 5.0

# Waveforms are computed in float32 on the STM32 (tba/tbb) and the GAP8 (tga) and
# compared against float64 host math: sinf's absolute error plus the float32
# round-trip stays under 1e-3.
WAVEFORM_TOL = 1e-3


class OrbitSource:
    """Self-paced synthetic mocap: a circular orbit with rotating yaw.

    Unlike the ScriptedTakeoffSource (paced by the streamer), this source paces
    itself like a real mocap system, so the streamer runs pace=False: the tap
    records the full --mocap-hz stream while the uplink is decimated to the
    config cap — the exact live-mocap data path. ``source_seq`` counts frames
    like NatNet does, exercising the frame-number clock fit.
    """

    def __init__(self, *, rigid_body: str, rate_hz: float, radius_m: float = 0.5) -> None:
        self._rigid_body = rigid_body
        self._dt_s = 1.0 / rate_hz
        self._radius_m = radius_m
        self._stop = False

    def samples(self, *, start_s: float = 0.0, dt_s: float = 0.01):
        seq = 0
        next_s = time.monotonic()
        while not self._stop:
            next_s += self._dt_s
            delay = next_s - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            seq += 1
            t = seq * self._dt_s
            theta = 2.0 * math.pi * 0.25 * t  # quarter-turn per second
            yield PoseSample(
                timestamp_s=time.time(),
                position_m=(
                    self._radius_m * math.cos(theta),
                    self._radius_m * math.sin(theta),
                    0.5 + 0.1 * math.sin(2.0 * math.pi * 0.5 * t),
                ),
                orientation=Quaternion.from_euler_rad(
                    EulerAngles(roll=0.0, pitch=0.0, yaw=theta)
                ),
                rigid_body=self._rigid_body,
                source_seq=seq,
            )

    def close(self) -> None:
        self._stop = True


def bench_telemetry_config(
    log_hz: float,
    configured_custom: tuple[CustomStreamConfig, ...] = (),
) -> TelemetryConfig:
    """Build the stress profile and append custom streams from the lab YAML."""
    period_ms = log_period_ms(log_hz)
    return TelemetryConfig(
        enabled=True,
        state_hz=log_hz,
        controls_hz=log_hz,
        setpoints=True,
        mocap=True,
        extpose=True,
        custom=(
            CustomStreamConfig(
                name="tba",
                period_ms=period_ms,
                variables=("tba.tick:uint32_t", "tba.sin1:float",
                           "tba.cos1:float", "tba.saw:float"),
            ),
            CustomStreamConfig(
                name="tbb",
                period_ms=period_ms,
                variables=("tbb.blk:uint32_t", "tbb.ramp:float",
                           "tbb.square:float", "tbb.tri:float"),
            ),
            CustomStreamConfig(
                name="tga",
                period_ms=period_ms,
                variables=("tga.tick:uint32_t", "tga.sin1:float",
                           "tga.cos1:float", "tga.saw:float"),
            ),
            CustomStreamConfig(
                name="tgb",
                period_ms=period_ms,
                variables=("tgb.blk:uint32_t", "tgb.ramp:float",
                           "tgb.square:float", "tgb.tri:float"),
            ),
            *configured_custom,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/drone.yaml")
    parser.add_argument("--backend", choices=("fake", "radio"), default="radio")
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--mocap-hz", type=float, default=500.0,
                        help="fake mocap frame rate through the telemetry tap")
    parser.add_argument("--log-hz", type=float, default=100.0,
                        help="rate of each STM32 log stream (max 100)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    telemetry = bench_telemetry_config(args.log_hz, cfg.telemetry.custom)
    recorder = TelemetryRecorder(
        telemetry,
        run_root=ARTIFACTS_DIR,
        lab_config=dataclasses.replace(cfg, telemetry=telemetry),
        example="telemetry_bench",
    )
    recorder.start()
    log, console_sink, close_log = make_run_logger(recorder.run_dir)

    source = OrbitSource(rigid_body=cfg.mocap.rigid_body, rate_hz=args.mocap_hz)
    tapped = recorder.wrap_pose_source(source)

    custom_count = len(cfg.telemetry.custom)
    log(f"Telemetry stress bench: {args.duration_s:.0f}s, mocap {args.mocap_hz:.0f} Hz "
        f"(uplink capped {cfg.mocap.stream_rate_hz:.0f} Hz), 7 base STM32 streams at "
        f"{args.log_hz:.0f} Hz plus {custom_count} config custom stream(s). "
        "Props stay off; the drone is never armed.")
    log(f"Recording to {recorder.run_dir}")

    try:
        if args.backend == "fake":
            backend = FakeCrazyflieBackend()
            _stress(args, cfg, recorder=recorder, source=tapped, backend=backend,
                    session=None, log=log)
        else:
            with CrazyflieRadioBackend(cfg.radio.uri, console_sink=console_sink) as backend:
                session = RadioSessionBase(backend)
                session.prepare()  # kalman select + reset, exercises param writes
                recorder.attach_stm32(backend.crazyflie)
                try:
                    _stress(args, cfg, recorder=recorder, source=tapped, backend=backend,
                            session=session, log=log)
                finally:
                    recorder.detach_stm32()
    finally:
        recorder.close()
        close_log()
        source.close()

    return analyze(
        recorder.run_dir,
        args=args,
        uplink_hz=cfg.mocap.stream_rate_hz,
        telemetry=telemetry,
    )


def _stress(args, cfg, *, recorder, source, backend, session, log) -> None:
    streamer = ExtposeStreamer(
        source=source,
        backend=recorder.wrap_extpose_backend(backend),
        max_rate_hz=cfg.mocap.stream_rate_hz,
        pace=False,  # OrbitSource paces itself, like a live mocap system
        transform=OriginZeroer(
            on_origin=lambda position_m, yaw_rad: recorder.event(
                "origin_zeroed",
                x=position_m[0], y=position_m[1], z=position_m[2], yaw_rad=yaw_rad,
            )
        ).zero,
    )
    recorder.event("extpose_config", position_only=False)
    streamer.start()
    recorder.event("stress_start", duration_s=args.duration_s, mocap_hz=args.mocap_hz,
                   log_hz=args.log_hz)

    setpoint_period = 1.0 / SETPOINT_RATE_HZ
    next_status = STATUS_PERIOD_S
    start = time.monotonic()
    i = 0
    try:
        while (elapsed := time.monotonic() - start) < args.duration_s:
            if session is not None and session.link_lost:
                recorder.event("radio_lost")
                log("RADIO LINK LOST: ending stress early.")
                break
            if streamer.error is not None:
                recorder.event("stream_error", str(streamer.error))
                log(f"Mocap stream died: {streamer.error}")
                break
            i += 1
            phase = elapsed
            recorder.record_setpoint(
                0.4 * math.cos(phase), 0.4 * math.sin(phase), 0.8, math.degrees(phase) % 360.0
            )
            if elapsed >= next_status:
                next_status += STATUS_PERIOD_S
                stats = recorder.stats()
                summary = " ".join(
                    f"{name}={s['rows']}r/{s['dropped']}d" for name, s in sorted(stats.items())
                )
                log(f"[{elapsed:5.1f}s] {summary}")
            time.sleep(setpoint_period)
    except KeyboardInterrupt:
        recorder.event("interrupt")
        log("Interrupted; finalizing recording.")
    finally:
        recorder.event("stress_end")
        streamer.stop()
    log(f"Uplink: {streamer.sent} extpose sent, {streamer.dropped} decimated (cap "
        f"{cfg.mocap.stream_rate_hz:.0f} Hz).")


def analyze(run_dir, *, args, uplink_hz: float, telemetry: TelemetryConfig) -> int:
    """Run the shared checks plus the mocap clock fit and tba/tga ground truth."""
    duration = args.duration_s
    stream_rates = {
        "state_pos": args.log_hz,
        "state_att": args.log_hz,
        "ctrl": args.log_hz,
        **{custom.name: 1000.0 / custom.period_ms for custom in telemetry.custom},
    }
    stm32_streams = tuple(stream_rates)
    host_streams = ("events", "mocap", "extpose", "setpoints")
    expected_rows = {"mocap": duration * args.mocap_hz}
    skipped_streams = {}
    if args.backend == "radio":
        expected_rows.update(
            {name: duration * rate_hz for name, rate_hz in stream_rates.items()}
        )
    else:
        skipped_streams = {
            name: "fake backend: no STM32 log source is attached" for name in stm32_streams
        }

    run, report = validate_run(
        run_dir,
        required_streams=host_streams + (stm32_streams if args.backend == "radio" else ()),
        skipped_streams=skipped_streams,
        expected_rows=expected_rows,
        minimum_delivery=0.9,
        maximum_rows={"extpose": duration * uplink_hz * 1.2},
        required_events=("origin_zeroed",),
        source_sequence_streams=("mocap",),
        recreate_extpose=True,
        clock_rms_limits_ns={"stm32_ms": 5e6} if args.backend == "radio" else {},
    )

    fit = run.fits.get("natnet_frame")
    report.add(fit is not None, "mocap clock fit available")
    if fit is not None:
        frame_period_ms = fit.slope / 1e6
        report.add(
            abs(frame_period_ms - 1000.0 / args.mocap_hz) < 0.5,
            "mocap clock fit recovers frame period",
            f"{frame_period_ms:.3f} ms/frame, rms {fit.rms_ns / 1e6:.2f} ms",
        )

    if args.backend == "radio":
        tba = run.streams["tba"]
        if tba.num_rows:
            ticks = tba.column("tba.tick").to_pylist()
            deltas = [b - a for a, b in zip(ticks, ticks[1:])]
            period = log_period_ms(args.log_hz)
            on_time = sum(1 for d in deltas if d == period) / max(1, len(deltas))
            report.add(
                all(delta > 0 for delta in deltas),
                "tba.tick strictly increasing",
                f"{on_time:.1%} exactly {period} ms; min {min(deltas)}, max {max(deltas)}",
            )
            err = waveform_error(tba)
            report.add(
                err < WAVEFORM_TOL,
                "waveforms match ground truth from tick",
                f"max |err| = {err:.2e}",
            )

    if args.backend == "radio":
        tga = run.streams["tga"]
        if tga.num_rows:
            ticks = tga.column("tga.tick").to_pylist()
            report.add(
                len(ticks) > 1 and ticks[-1] > ticks[0]
                and all(b >= a for a, b in zip(ticks, ticks[1:])),
                "GAP8 tick advances without reversing",
            )
            err = waveform_error(tga, prefix="tga")
            report.add(err < WAVEFORM_TOL, "GAP8 waveforms match tick",
                       f"max |err| = {err:.2e}")

    print(f"\n{report.format()}")
    return 0 if report.passed else 1


def waveform_error(tba, *, prefix: str = "tba") -> float:
    """Max deviation of logged waveforms from ground truth computed off tick.

    The log task samples the block asynchronously to the app's 1 kHz update,
    so a sample can catch the variables one tick staler than ``tick`` itself
    (update order: tick first, waveforms after). Each value therefore gets to
    match either tick or tick-1; what remains is genuine corruption.
    """
    worst = 0.0
    for tick, sin1, cos1, saw in zip(
        tba.column(f"{prefix}.tick").to_pylist(),
        tba.column(f"{prefix}.sin1").to_pylist(),
        tba.column(f"{prefix}.cos1").to_pylist(),
        tba.column(f"{prefix}.saw").to_pylist(),
    ):
        phases = [(t % 1000) / 1000.0 for t in (tick, tick - 1)]
        worst = max(
            worst,
            min(abs(sin1 - math.sin(2.0 * math.pi * p)) for p in phases),
            min(abs(cos1 - math.cos(2.0 * math.pi * p)) for p in phases),
            min(abs(saw - p) for p in phases),
        )
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
