#!/usr/bin/env python3
"""Align a Motive rigid body with the drone's IMU: guided, about two minutes.

Reads the asset's marker offsets from Motive, reboots the drone for a fresh gyro
calibration, records mocap and the IMU while you rotate the drone by hand, then prints
the pivot edit to enter in Motive and checks it once you have.

  run.py calibrate             record and analyze (the default workflow)
  run.py analyze RUN_DIR       analyze an existing recording again
  run.py verify RUN_DIR        after the Motive edit: did it land where it should?

The drone is never armed and no parameter is written: the radio only logs. Nothing
assumes a level floor or a level Motive ground plane; Motive's tilt is measured.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cf_lab.backends.crazyflie import CrazyflieRadioBackend, power_cycle_until_healthy
from cf_lab.config import CustomStreamConfig, load_config
from cf_lab.mocap import build_live_pose_source
from cf_lab.mocap.alignment import (
    GRAVITY_MPS2,
    MAX_DECK_RMS_M,
    AlignmentResult,
    MotionCoverage,
    analyze_motion,
    fit_deck,
    load_geometry,
    load_motion,
    markers_after_edit,
    measure_pivot_edit,
    pivot_edit,
)
from cf_lab.telemetry import TelemetryRecorder

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
DEFAULT_GEOMETRY = "configs/marker_geometry/cf21bl_mocap_deck.yaml"
ASSET_FILE = "asset.json"  # in the run: the asset as Motive described it for this recording
RECORD_FILE = "alignment.json"  # in the run: the result, read back by `verify`

STILL_BEFORE_S = 5.0  # at rest before the motion: gyro bias and the closure test
MAX_MOTION_S = 120.0  # stop waiting for full coverage after this long
STILL_AFTER_S = 5.0  # at rest after the motion
MAX_REST_WAIT_S = 30.0

# Motive sends marker offsets in its native Y-up axes, so its edit fields may use them too:
# (x, y, z) in the streamed Z-up axes is (x, z, -y) in Motive's.
_Z_UP_TO_MOTIVE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


class _CoverageSource:
    """Passes poses through to the recorder, updating the live coverage on the way."""

    def __init__(self, source, coverage: MotionCoverage) -> None:
        self._source = source
        self._coverage = coverage

    def samples(self, *, start_s: float = 0.0, dt_s: float = 0.01):
        for pose in self._source.samples(start_s=start_s, dt_s=dt_s):
            self._coverage.update(pose.orientation, time.monotonic())
            yield pose

    def connected(self, within_s: float) -> bool:
        return self._source.connected(within_s)

    def close(self) -> None:
        self._source.close()


def _imu_stream() -> CustomStreamConfig:
    # 100 Hz is the ceiling of the firmware's log framework; 24 bytes fits one log block.
    names = ("acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z")
    return CustomStreamConfig(
        name="imu", period_ms=10, variables=tuple(f"{name}:float" for name in names)
    )


def _wait_for_motion(coverage: MotionCoverage) -> None:
    print("\nPick the drone up now. Hold it by the battery or underside, with the beads")
    print("in view, and turn it through every orientation: roll, pitch, yaw, past 60 deg.")
    start = time.monotonic()
    while not coverage.complete and time.monotonic() - start < MAX_MOTION_S:
        axes = " ".join(f"{name} {value:4.0f}" for name, value in zip("xyz", coverage.axes_rad))
        print(
            f"\r  {coverage.fraction * 100:3.0f}%  rotation (rad) {axes}  "
            f"tilt {coverage.max_tilt_deg:3.0f} deg  turning {coverage.turning_s:3.0f} s ",
            end="",
            flush=True,
        )
        time.sleep(0.2)
    print()
    print(
        "Enough motion. Put the drone down anywhere and let go."
        if coverage.complete
        else f"Stopping after {MAX_MOTION_S:.0f} s without full coverage. "
        "Put the drone down and let go."
    )


def _wait_for_rest(coverage: MotionCoverage) -> None:
    start = time.monotonic()
    while time.monotonic() - start < MAX_REST_WAIT_S:
        last = coverage.last_turning_s
        if last is not None and time.monotonic() - last >= STILL_AFTER_S:
            return
        time.sleep(0.1)


def _record(cfg, source, markers_m: np.ndarray) -> Path:
    telemetry = dataclasses.replace(
        cfg.telemetry,
        enabled=True,
        state_hz=0.0,
        controls_hz=0.0,
        setpoints=False,
        mocap=True,
        extpose=False,
        camera=False,
        custom=(_imu_stream(),),
    )
    recorder = TelemetryRecorder(
        telemetry, run_root=ARTIFACTS_DIR, lab_config=cfg, example="mocap_alignment"
    )
    recorder.start()
    coverage = MotionCoverage()
    try:
        (recorder.run_dir / ASSET_FILE).write_text(
            json.dumps(
                {"rigid_body": cfg.mocap.rigid_body, "markers_m": markers_m.tolist()}, indent=1
            )
        )
        with CrazyflieRadioBackend(cfg.radio.uri) as backend:
            recorder.attach_stm32(backend.crazyflie)
            recorder.attach_mocap(_CoverageSource(source, coverage))
            print(
                f"Recording to {recorder.run_dir}. "
                f"Leave the drone still for {STILL_BEFORE_S:.0f} s."
            )
            time.sleep(STILL_BEFORE_S)
            _wait_for_motion(coverage)
            _wait_for_rest(coverage)
            recorder.detach_stm32()
    finally:
        recorder.close()
    return recorder.run_dir


def _vector(values, *, scale: float = 1.0, digits: int = 2) -> str:
    return ", ".join(f"{value * scale:+.{digits}f}" for value in values)


def _print_report(result: AlignmentResult, rigid_body: str) -> None:
    euler = Rotation.from_matrix(result.rotation_asset_to_imu).as_euler("xyz", degrees=True)
    print(f"\n=== {rigid_body} ===")
    print(
        f"Time sync      gyro lag {result.gyro_lag_s * 1e3:+.1f} ms, "
        f"accelerometer lag {result.accel_lag_s * 1e3:+.1f} ms"
    )
    print(
        f"Asset -> IMU   roll {euler[0]:+.2f}, pitch {euler[1]:+.2f}, yaw {euler[2]:+.2f} deg "
        f"(chunks agree within {result.rotation_spread_deg.max():.2f} deg)"
    )
    print(
        f"Gyro           bias {_vector(np.degrees(result.gyro_bias_rad_s), digits=3)} deg/s, "
        f"rate ratio {result.gyro_rate_ratio:.4f}"
    )
    print(
        "Accelerometer  bias "
        f"{_vector(result.accel_bias_mps2 / GRAVITY_MPS2, scale=1e3, digits=1)} mg, "
        f"fit residual {result.force_rms_mps2:.3f} m/s^2"
    )
    print(
        f"Motive's up    {result.motive_tilt_deg:.2f} deg from true vertical "
        f"(true up in Motive's axes: {_vector(result.true_up_in_motive, digits=4)})"
    )
    print(
        f"Deck           markers {result.deck.marker_indices} in the holes, centre "
        f"{_vector(result.deck.center_m, scale=1e3)} mm in the asset, "
        f"{result.deck_yaw_from_imu_deg:+.2f} deg of yaw from the IMU"
    )
    print("IMU position, in the asset (mm):")
    print(
        f"  motion fit   {_vector(result.imu_position_m, scale=1e3)}  "
        f"(chunks agree within {result.imu_position_spread_m.max() * 1e3:.1f}; z from the z axis "
        f"{result.imu_z_by_channel_m[0] * 1e3:+.1f}, "
        f"from x and y {result.imu_z_by_channel_m[1] * 1e3:+.1f})"
    )
    print(f"  geometry     {_vector(result.prior_position_m, scale=1e3)}")
    print(f"  using        {result.origin_source}")
    print("\nChecks:")
    for check in result.checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")


def _print_edit(result: AlignmentResult, run_dir: Path) -> None:
    translate, euler = pivot_edit(result.rotation_asset_to_imu, result.origin_m)
    frame = result.rotation_asset_to_imu.T
    native_euler = Rotation.from_matrix(_Z_UP_TO_MOTIVE @ frame @ _Z_UP_TO_MOTIVE.T).as_euler(
        "XYZ", degrees=True
    )
    print("\nPivot edit for Motive's Builder pane (Edit tab, this rigid body):")
    print(f"  translate, streamed Z-up axes (mm):   {_vector(translate, scale=1e3)}")
    print(f"  rotate about X, Y, Z (deg):           {_vector(euler)}")
    print(
        f"  the same in Motive's Y-up axes (mm):  {_vector(_Z_UP_TO_MOTIVE @ translate, scale=1e3)}"
    )
    print(f"  rotate about X, Y, Z (deg):           {_vector(native_euler)}")
    print(
        "Which axes Motive's fields use is checked by `verify`, which measures what the edit did:"
    )
    print(f"  run.py verify {run_dir}")


def _analyze(run_dir: Path, geometry_path: str) -> int:
    asset = json.loads((run_dir / ASSET_FILE).read_text())
    markers = np.array(asset["markers_m"])
    print("Analyzing the recording...")
    result = analyze_motion(load_motion(run_dir), markers, load_geometry(geometry_path))
    _print_report(result, asset["rigid_body"])
    _print_edit(result, run_dir)
    (run_dir / RECORD_FILE).write_text(
        json.dumps(
            {
                "rigid_body": asset["rigid_body"],
                "markers_before_m": markers.tolist(),
                "markers_expected_m": markers_after_edit(
                    markers, result.rotation_asset_to_imu, result.origin_m
                ).tolist(),
                "rotation_asset_to_imu": result.rotation_asset_to_imu.tolist(),
                "origin_m": result.origin_m.tolist(),
                "origin_source": result.origin_source,
                "imu_position_fit_m": result.imu_position_m.tolist(),
                "imu_position_prior_m": result.prior_position_m.tolist(),
                "motive_tilt_deg": result.motive_tilt_deg,
                "true_up_in_motive": result.true_up_in_motive.tolist(),
                "accel_bias_mps2": result.accel_bias_mps2.tolist(),
                "gyro_bias_rad_s": result.gyro_bias_rad_s.tolist(),
                "checks": [dataclasses.asdict(check) for check in result.checks],
            },
            indent=1,
        )
    )
    print(f"\nSaved {run_dir / RECORD_FILE}")
    return 0 if result.passed else 1


def _calibrate(args) -> int:
    cfg = load_config(args.config)
    if cfg.mocap.transport != "natnet":
        print("mocap.transport must be natnet: only NatNet describes the asset's markers.")
        return 1
    geometry = load_geometry(args.geometry)
    source = build_live_pose_source(cfg=cfg)
    source.connect()
    try:
        markers = np.array(source.marker_offsets_m)
        deck = fit_deck(markers, geometry)
        print(
            f"{cfg.mocap.rigid_body}: {len(markers)} markers; "
            f"deck beads are markers {deck.marker_indices} "
            f"({deck.rms_m * 1e3:.2f} mm rms)"
        )
        if deck.rms_m > MAX_DECK_RMS_M:
            print(
                f"The deck beads do not fit the holes in {args.geometry}. "
                "Check the file against the drone."
            )
            return 1
        if not args.no_reboot and not power_cycle_until_healthy(cfg.radio.uri, log=print):
            print("The drone's boot self-test failed.")
            return 1
        run_dir = _record(cfg, source, markers)
    finally:
        source.close()
    return _analyze(run_dir, args.geometry)


def _verify(args) -> int:
    record = json.loads((Path(args.run_dir) / RECORD_FILE).read_text())
    cfg = load_config(args.config)
    source = build_live_pose_source(cfg=cfg)
    source.connect()
    source.close()
    before = np.array(record["markers_before_m"])
    expected = np.array(record["markers_expected_m"])
    after = np.array(source.marker_offsets_m)
    if after.shape != before.shape:
        print(
            f"The asset now has {len(after)} markers, not {len(before)}; it was remade. "
            "Calibrate again."
        )
        return 1
    translate, euler, fit_rms = measure_pivot_edit(before, after)
    wanted_translate, wanted_euler = pivot_edit(
        np.array(record["rotation_asset_to_imu"]), np.array(record["origin_m"])
    )
    miss = math.sqrt(np.mean(np.sum((after - expected) ** 2, axis=1)))
    print("                 asked for                 Motive applied")
    print(
        f"translate (mm)   {_vector(wanted_translate, scale=1e3):26s}"
        f"{_vector(translate, scale=1e3)}"
    )
    print(f"rotate (deg)     {_vector(wanted_euler):26s}{_vector(euler)}")
    print(
        f"Markers are {miss * 1e3:.2f} mm rms from where they should be "
        f"(the old and new layouts match to {fit_rms * 1e3:.2f} mm)."
    )
    if miss <= MAX_DECK_RMS_M:
        print(
            "PASS: the pivot is on the IMU. "
            "Running `calibrate` again should now report almost no rotation."
        )
        return 0
    print(
        "FAIL: if the applied values are the requested ones with axes swapped or signs "
        "flipped, undo the"
    )
    print("edit and enter the other set of printed values.")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate", help="reboot, record the hand motion, analyze")
    calibrate.add_argument("--config", default="configs/drone.yaml")
    calibrate.add_argument("--geometry", default=DEFAULT_GEOMETRY)
    calibrate.add_argument(
        "--no-reboot", action="store_true", help="keep the current gyro calibration"
    )
    analyze = commands.add_parser("analyze", help="analyze an existing recording")
    analyze.add_argument("run_dir")
    analyze.add_argument("--geometry", default=DEFAULT_GEOMETRY)
    verify = commands.add_parser(
        "verify", help="check the Motive edit against a recording's result"
    )
    verify.add_argument("run_dir")
    verify.add_argument("--config", default="configs/drone.yaml")
    args = parser.parse_args(argv)

    try:
        if args.command == "calibrate":
            return _calibrate(args)
        if args.command == "analyze":
            return _analyze(Path(args.run_dir), args.geometry)
        return _verify(args)
    except (RuntimeError, ValueError) as exc:  # Motive unreachable or mismatched, no motion, ...
        print(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
