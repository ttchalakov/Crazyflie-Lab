"""Aligning an asset with the IMU: a simulated hand motion with known truth, and the real deck."""

import importlib.util
import json
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.signal import butter, lfilter
from scipy.spatial.transform import Rotation

from cf_lab.config import CustomStreamConfig
from cf_lab.mocap.alignment import (
    GRAVITY_MPS2,
    MAX_DECK_RMS_M,
    MarkerGeometry,
    Motion,
    MotionCoverage,
    analyze_motion,
    fit_deck,
    load_geometry,
    load_motion,
    markers_after_edit,
    measure_pivot_edit,
    pivot_edit,
)
from cf_lab.mocap.types import PoseSample, Quaternion
from cf_lab.telemetry import schema
from cf_lab.telemetry.writer import RunWriter

REPO = Path(__file__).resolve().parents[2]

# cf_teo as Motive described it on 2026-11-03 (Z-up, mm), with the holes its deck beads sit in.
CF_TEO_MARKERS = (
    np.array(
        [
            [-17.92, 59.85, -2.40],
            [-0.65, 17.37, 3.21],
            [27.91, -13.47, 2.63],
            [61.31, 51.74, -2.91],
            [-69.16, -72.11, -3.55],
            [-1.50, -43.39, 2.96],
        ]
    )
    / 1e3
)
GEOMETRY = MarkerGeometry(
    deck_holes_m=np.array([[30.0, 0.0], [0.0, 30.0], [0.0, -30.0]]) / 1e3,
    bead_radius_m=0.00635,
    imu_xy_m=np.zeros(2),
    imu_z_from_deck_top_m=-0.010,
)


def _truth():
    """Asset frame relative to the IMU: what the fit must recover."""
    return types.SimpleNamespace(
        rotation=Rotation.from_euler(
            "xyz", [-1.2, -2.4, -3.4], degrees=True
        ).as_matrix(),  # asset -> IMU
        imu_position=np.array([0.005, -0.008, -0.026]),  # in the asset
        dg_world=np.array([-0.25, 0.04, 0.0]),  # true gravity minus Motive's -z
        accel_bias_imu=np.array([0.012, -0.008, 0.004]) * GRAVITY_MPS2,
        gyro_bias_imu=np.radians([0.03, -0.02, 0.05]),
    )


def _simulate(truth, *, seconds=46.0, rest_s=3.0, seed=4) -> Motion:
    """A drone rotated by hand between two rests, seen by Motive and by its own filtered IMU."""
    rng = np.random.default_rng(seed)
    fs = 1000.0
    t = np.arange(0.0, seconds, 1.0 / fs)
    envelope = np.clip((t - rest_s) / 1.0, 0.0, 1.0) * np.clip(
        (seconds - rest_s - t) / 1.0, 0.0, 1.0
    )
    freqs = np.array([[0.31, 0.83, 1.71], [0.43, 1.09, 2.03], [0.23, 0.67, 1.37]])
    amps = np.array([[2.6, 1.6, 0.8], [2.2, 1.8, 0.7], [2.4, 1.3, 0.9]])
    w = envelope[:, None] * np.stack(
        [
            sum(a * np.sin(2 * np.pi * f * t + i) for f, a in zip(freqs[i], amps[i]))
            for i in range(3)
        ],
        axis=1,
    )
    alpha = np.gradient(w, 1.0 / fs, axis=0)
    # Integrate the attitude in the asset frame.
    q = np.empty((len(t), 4))
    q[0] = Rotation.from_euler("xyz", [1.0, -2.0, 30.0], degrees=True).as_quat()
    half = 0.5 * w / fs
    for k in range(len(t) - 1):
        x, y, z, s = q[k]
        a, b, c = half[k]
        step = np.array(
            [
                s * a + y * c - z * b,
                s * b + z * a - x * c,
                s * c + x * b - y * a,
                -x * a - y * b - z * c,
            ]
        )
        q[k + 1] = q[k] + step
        q[k + 1] /= np.linalg.norm(q[k + 1])
    rotations = Rotation.from_quat(q)
    wobble = envelope[:, None] * np.stack(
        [0.15 * np.sin(2 * np.pi * f * t) for f in (0.37, 0.53, 0.71)], axis=1
    )
    position = np.array([0.8, -0.6, 0.9]) + wobble
    acc_w = np.gradient(np.gradient(position, 1.0 / fs, axis=0), 1.0 / fs, axis=0)

    # The IMU, at the IMU's point on the drone, through the firmware's filters.
    r = truth.imu_position
    gravity = np.array([0.0, 0.0, -GRAVITY_MPS2]) + truth.dg_world
    force_asset = (
        rotations.inv().apply(acc_w - gravity) + np.cross(alpha, r) + np.cross(w, np.cross(w, r))
    )
    force = force_asset @ truth.rotation.T + truth.accel_bias_imu
    gyro = w @ truth.rotation.T + truth.gyro_bias_imu
    b30, a30 = butter(2, 30.0 / (fs / 2))
    b80, a80 = butter(2, 80.0 / (fs / 2))
    force = lfilter(b30, a30, force - force[0], axis=0) + force[0]
    gyro = lfilter(b80, a80, gyro - gyro[0], axis=0) + gyro[0]
    imu_index = np.arange(0, len(t), 10)
    imu_clock_offset = 0.0065  # the IMU's clock against mocap, after the sync
    # Motive, at 360 Hz with its noise, mid-exposure stamped.
    mocap_index = np.round(np.arange(0.0, seconds - 0.01, 1 / 360.0) * fs).astype(int)
    noise = Rotation.from_rotvec(rng.normal(0.0, np.radians(0.02), (len(mocap_index), 3)))
    return Motion(
        mocap_t_s=t[mocap_index],
        quaternion_xyzw=(rotations[mocap_index] * noise).as_quat(),
        position_m=position[mocap_index] + rng.normal(0.0, 3e-5, (len(mocap_index), 3)),
        imu_t_s=t[imu_index] + imu_clock_offset,
        gyro_rad_s=gyro[imu_index] + rng.normal(0.0, np.radians(0.2), (len(imu_index), 3)),
        acc_mps2=force[imu_index] + rng.normal(0.0, 0.012, (len(imu_index), 3)),
    )


def _markers_for(truth, geometry=GEOMETRY):
    """Deck beads in their holes and guard beads lower down, in the asset frame."""
    height = geometry.bead_radius_m - geometry.imu_z_from_deck_top_m  # bead centres above the IMU
    deck = np.c_[
        geometry.deck_holes_m - geometry.imu_xy_m, np.full(len(geometry.deck_holes_m), height)
    ]
    guards = np.array(
        [
            [0.065, 0.06, height - 0.008],
            [-0.015, 0.065, height - 0.009],
            [-0.055, -0.07, height - 0.008],
        ]
    )
    in_imu = np.vstack([guards, deck])
    return in_imu @ truth.rotation + truth.imu_position


@pytest.fixture(scope="module")
def solved():
    truth = _truth()
    return truth, analyze_motion(_simulate(truth), _markers_for(truth), GEOMETRY)


def test_the_motion_fit_recovers_the_rotation_between_asset_and_imu(solved):
    truth, result = solved
    error = Rotation.from_matrix(result.rotation_asset_to_imu @ truth.rotation.T).magnitude()
    assert np.degrees(error) < 0.1
    assert result.gyro_bias_rad_s == pytest.approx(truth.gyro_bias_imu, abs=np.radians(0.05))
    assert result.gyro_rate_ratio == pytest.approx(1.0, abs=0.01)
    assert result.rate_correlation > 0.99


def test_the_motion_fit_separates_accelerometer_bias_from_motives_tilt(solved):
    truth, result = solved
    assert result.accel_bias_mps2 == pytest.approx(truth.accel_bias_imu, abs=0.02)  # 2 mg
    expected_up = -(np.array([0.0, 0.0, -GRAVITY_MPS2]) + truth.dg_world)
    expected_up /= np.linalg.norm(expected_up)
    assert result.true_up_in_motive == pytest.approx(expected_up, abs=0.002)
    assert result.motive_tilt_deg == pytest.approx(
        np.degrees(np.arctan(0.253 / GRAVITY_MPS2)), abs=0.1
    )
    assert result.closure_mg < 5.0


def test_the_motion_fit_finds_the_imu_through_the_firmware_filter(solved):
    truth, result = solved
    assert result.imu_position_m == pytest.approx(truth.imu_position, abs=0.002)
    assert result.imu_z_by_channel_m == pytest.approx((truth.imu_position[2],) * 2, abs=0.003)


def test_a_trusted_motion_fit_sets_the_origin_and_every_check_passes(solved):
    truth, result = solved
    assert result.origin_source == "motion fit"
    assert result.origin_m == pytest.approx(result.imu_position_m)
    assert result.prior_position_m == pytest.approx(truth.imu_position, abs=1e-4)
    assert result.passed, [check for check in result.checks if not check.passed]


def test_a_motion_fit_far_from_the_geometry_leaves_the_origin_on_the_geometry():
    truth = _truth()
    # The bench measurement says 15 mm further down than the drone really is.
    measured = MarkerGeometry(
        GEOMETRY.deck_holes_m, GEOMETRY.bead_radius_m, GEOMETRY.imu_xy_m, -0.038
    )
    result = analyze_motion(_simulate(truth), _markers_for(truth), measured)
    assert result.origin_source == "geometry"
    assert result.origin_m == pytest.approx(result.prior_position_m)
    assert result.imu_position_m == pytest.approx(truth.imu_position, abs=0.002)  # still reported


def test_a_motion_without_rest_fails_the_closure_check():
    truth = _truth()
    motion = _simulate(truth, rest_s=0.0)
    result = analyze_motion(motion, _markers_for(truth), GEOMETRY)
    closure = next(check for check in result.checks if check.name == "closure at rest")
    assert not closure.passed and result.closure_mg is None


def test_a_drone_that_never_moved_is_refused():
    truth = _truth()
    motion = _simulate(truth, seconds=6.0, rest_s=3.0)
    with pytest.raises(ValueError, match="move it by hand"):
        analyze_motion(motion, _markers_for(truth), GEOMETRY)


def test_an_occlusion_is_excluded_rather_than_differentiated():
    truth = _truth()
    motion = _simulate(truth)
    keep = (motion.mocap_t_s < 20.0) | (motion.mocap_t_s > 20.5)
    occluded = Motion(
        motion.mocap_t_s[keep],
        motion.quaternion_xyzw[keep],
        motion.position_m[keep],
        motion.imu_t_s,
        motion.gyro_rad_s,
        motion.acc_mps2,
    )
    result = analyze_motion(occluded, _markers_for(truth), GEOMETRY)
    assert result.imu_position_m == pytest.approx(truth.imu_position, abs=0.002)


def test_the_time_grid_stays_inside_the_recording_at_any_clock_offset():
    # Frames exactly 1/360 s apart, at an offset where np.arange put the grid's last point
    # past the last mocap sample.
    truth = _truth()
    motion = _simulate(truth)
    offset = -178785.47045894465
    frames = 16000
    shifted = Motion(
        offset + np.arange(frames) / 360.0,
        motion.quaternion_xyzw[:frames],
        motion.position_m[:frames],
        motion.imu_t_s + offset - motion.mocap_t_s[0],
        motion.gyro_rad_s,
        motion.acc_mps2,
    )
    result = analyze_motion(shifted, _markers_for(truth), GEOMETRY)
    assert result.rotation_asset_to_imu == pytest.approx(truth.rotation, abs=0.002)


# -- the deck ---------------------------------------------------------------------------


def test_the_real_cf_teo_deck_beads_are_found_in_their_holes():
    deck = fit_deck(CF_TEO_MARKERS, GEOMETRY)
    assert deck.marker_indices == (2, 1, 5)
    assert deck.rms_m == pytest.approx(0.00057, abs=0.0001)
    assert deck.center_m * 1e3 == pytest.approx([-1.41, -13.02, 3.09], abs=0.02)
    assert np.degrees(np.arctan2(deck.rotation[1, 0], deck.rotation[0, 0])) == pytest.approx(
        -0.82, abs=0.05
    )
    assert deck.runner_up_rms_m > 5 * deck.rms_m


def test_a_wrong_hole_shows_up_as_a_large_deck_residual():
    wrong = MarkerGeometry(
        np.array([[25.0, 0.0], [0.0, 30.0], [0.0, -30.0]]) / 1e3,
        GEOMETRY.bead_radius_m,
        GEOMETRY.imu_xy_m,
        GEOMETRY.imu_z_from_deck_top_m,
    )
    assert fit_deck(CF_TEO_MARKERS, wrong).rms_m > MAX_DECK_RMS_M


def test_geometry_loads_in_metres_and_rejects_too_few_holes(tmp_path):
    path = tmp_path / "geometry.yaml"
    path.write_text(
        "deck_holes_mm: [[10, 5], [0, -30], [-20, 0]]\nbead_diameter_mm: 12.7\n"
        "imu_xy_mm: [1, 2]\nimu_z_from_deck_top_mm: -23\n"
    )
    geometry = load_geometry(path)
    assert geometry.deck_holes_m[1] == pytest.approx([0.0, -0.03])
    assert (geometry.bead_radius_m, geometry.imu_z_from_deck_top_m) == pytest.approx(
        (0.00635, -0.023)
    )
    assert geometry.imu_xy_m == pytest.approx([0.001, 0.002])
    path.write_text(
        "deck_holes_mm: [[10, 5], [0, -30]]\nbead_diameter_mm: 12.7\n"
        "imu_xy_mm: [0, 0]\nimu_z_from_deck_top_mm: -23\n"
    )
    with pytest.raises(ValueError, match="at least three"):
        load_geometry(path)


def test_the_shipped_geometry_file_loads_and_fits_the_real_asset():
    geometry = load_geometry(REPO / "configs" / "marker_geometry" / "cf21bl_mocap_deck.yaml")
    assert fit_deck(CF_TEO_MARKERS, geometry).marker_indices == (2, 1, 5)


# -- the pivot edit ---------------------------------------------------------------------


def test_a_pivot_edit_can_be_measured_back_from_the_marker_offsets():
    truth = _truth()
    translate, euler = pivot_edit(truth.rotation, truth.imu_position)
    after = markers_after_edit(CF_TEO_MARKERS, truth.rotation, truth.imu_position)
    applied_translate, applied_euler, rms = measure_pivot_edit(CF_TEO_MARKERS, after)
    assert applied_translate == pytest.approx(translate, abs=1e-9)
    assert applied_euler == pytest.approx(euler, abs=1e-6)
    assert rms < 1e-9


def test_after_the_edit_the_markers_sit_where_the_imu_axes_put_them():
    truth = _truth()
    in_imu = np.array([[0.01, 0.02, 0.03]])
    in_asset = in_imu @ truth.rotation + truth.imu_position
    assert markers_after_edit(in_asset, truth.rotation, truth.imu_position) == pytest.approx(in_imu)


# -- live coverage ----------------------------------------------------------------------


def _quaternion(rotation):
    return Quaternion(*rotation.as_quat())


def test_coverage_ignores_pose_noise_on_a_still_drone():
    rng = np.random.default_rng(1)
    coverage = MotionCoverage()
    for k in range(3600):
        noise = Rotation.from_rotvec(rng.normal(0.0, np.radians(0.03), 3))
        coverage.update(_quaternion(noise), k / 360.0)
    assert coverage.axes_rad == [0.0, 0.0, 0.0]
    assert coverage.turning_s == 0.0 and coverage.fraction == 0.0


def test_coverage_completes_only_with_every_axis_and_enough_tilt():
    coverage = MotionCoverage()
    now = 0.0
    attitude = Rotation.identity()
    for axis in ([1, 0, 0], [0, 1, 0]):  # roll and pitch only: yaw is still missing
        for _ in range(12 * 360):
            attitude = attitude * Rotation.from_rotvec(np.array(axis) * 2.0 / 360.0)
            now += 1 / 360.0
            coverage.update(_quaternion(attitude), now)
    assert coverage.max_tilt_deg > 60.0 and not coverage.complete
    for _ in range(12 * 360):
        attitude = attitude * Rotation.from_rotvec(np.array([0, 0, 2.0]) / 360.0)
        now += 1 / 360.0
        coverage.update(_quaternion(attitude), now)
    assert coverage.complete
    assert coverage.last_turning_s == pytest.approx(now, abs=0.1)


# -- from a telemetry run ---------------------------------------------------------------


def test_a_recorded_run_loads_on_one_clock_in_si_units(tmp_path):
    imu = schema.custom_spec(
        CustomStreamConfig(
            name="imu",
            period_ms=10,
            variables=tuple(
                f"{v}:float" for v in ("acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z")
            ),
        )
    )
    writer = RunWriter(tmp_path / "run", [schema.mocap_spec("natnet"), imu, schema.events_spec()])
    writer.start()
    for i in range(200):  # both streams fit the writer's bounded queues in one burst
        writer.append_at(
            "imu",
            (1000 + 10 * i, 0.0, 0.0, 1.0, 90.0, 0.0, 0.0),
            host_ns=5_000_000_000 + i * 10_000_000,
        )
    for i in range(720):
        q = (0.0, 0.0, 0.0, 1.0)
        writer.append_at(
            "mocap",
            (0.0, 5000 + i, 0.1, 0.2, 0.3, *q, 4000.0),
            host_ns=5_000_000_000 + int(i * 1e9 / 360),
        )
    writer.close()

    motion = load_motion(tmp_path / "run")
    assert len(motion.mocap_t_s) == 720 and len(motion.imu_t_s) == 200
    assert motion.acc_mps2[0] == pytest.approx([0.0, 0.0, GRAVITY_MPS2])
    assert motion.gyro_rad_s[0] == pytest.approx([np.pi / 2, 0.0, 0.0])
    assert motion.position_m[0] == pytest.approx([0.1, 0.2, 0.3])
    assert np.all(np.diff(motion.mocap_t_s) > 0.0)
    assert (
        json.loads((tmp_path / "run" / "manifest.json").read_text())["streams"]["imu"]["rows"]
        == 200
    )


# -- the guided script, end to end ------------------------------------------------------


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "mocap_alignment_run", REPO / "examples/mocap_alignment/host/run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_simulated_run(run_dir, motion, markers):
    """A simulated motion, through the real writer: load_run must align the clocks itself."""
    imu = schema.custom_spec(
        CustomStreamConfig(
            name="imu",
            period_ms=10,
            variables=tuple(
                f"{v}:float" for v in ("acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z")
            ),
        )
    )
    writer = RunWriter(
        run_dir, [schema.mocap_spec("natnet"), imu, schema.events_spec()], queue_size=100_000
    )
    writer.start()
    host_start = 1_000_000_000_000
    latency_s = 0.004
    for t, q, p, seq in zip(
        motion.mocap_t_s, motion.quaternion_xyzw, motion.position_m, range(len(motion.mocap_t_s))
    ):
        host_ns = host_start + int((t + latency_s + 0.002) * 1e9)
        writer.append_at("mocap", (0.0, 7000 + seq, *p, *q, latency_s * 1e6), host_ns=host_ns)
    for t, a, g in zip(motion.imu_t_s, motion.acc_mps2, motion.gyro_rad_s):
        host_ns = host_start + int((t + 0.003) * 1e9)
        row = (int(5000 + t * 1e3), *(a / GRAVITY_MPS2), *np.degrees(g))
        writer.append_at("imu", row, host_ns=host_ns)
    writer.close()
    (run_dir / "asset.json").write_text(
        json.dumps({"rigid_body": "cf_sim", "markers_m": markers.tolist()})
    )


GEOMETRY_YAML = (
    "deck_holes_mm: [[30, 0], [0, 30], [0, -30]]\nbead_diameter_mm: 12.7\n"
    "imu_xy_mm: [0, 0]\nimu_z_from_deck_top_mm: -10\n"
)


@pytest.fixture(scope="module")
def analyzed_run(tmp_path_factory, runner):
    truth = _truth()
    root = tmp_path_factory.mktemp("alignment")
    (root / "geometry.yaml").write_text(GEOMETRY_YAML)
    run_dir = root / "run"
    _write_simulated_run(run_dir, _simulate(truth), _markers_for(truth))
    code = runner.main(["analyze", str(run_dir), "--geometry", str(root / "geometry.yaml")])
    return truth, run_dir, code


def test_the_script_analyzes_a_recorded_run_and_saves_the_result(analyzed_run, capsys):
    truth, run_dir, code = analyzed_run
    assert code == 0
    record = json.loads((run_dir / "alignment.json").read_text())
    error = Rotation.from_matrix(
        np.array(record["rotation_asset_to_imu"]) @ truth.rotation.T
    ).magnitude()
    assert np.degrees(error) < 0.2
    assert record["origin_source"] == "motion fit"
    assert record["origin_m"] == pytest.approx(truth.imu_position, abs=0.003)
    assert all(check["passed"] for check in record["checks"])


def _stub_motive(monkeypatch, runner, markers):
    source = types.SimpleNamespace(
        connect=lambda: None, close=lambda: None, marker_offsets_m=tuple(map(tuple, markers))
    )
    monkeypatch.setattr(runner, "build_live_pose_source", lambda *, cfg: source)


def test_verify_passes_once_the_edit_landed(analyzed_run, runner, monkeypatch, capsys):
    _, run_dir, _ = analyzed_run
    record = json.loads((run_dir / "alignment.json").read_text())
    _stub_motive(monkeypatch, runner, record["markers_expected_m"])
    assert runner.main(["verify", str(run_dir), "--config", str(REPO / "configs/drone.yaml")]) == 0
    assert "PASS" in capsys.readouterr().out


def test_verify_fails_and_shows_what_was_applied_when_the_axes_were_wrong(
    analyzed_run, runner, monkeypatch, capsys
):
    _, run_dir, _ = analyzed_run
    record = json.loads((run_dir / "alignment.json").read_text())
    before = np.array(record["markers_before_m"])
    origin = np.array(record["origin_m"])
    swapped = origin[[0, 2, 1]] * np.array([1.0, 1.0, -1.0])  # entered in Motive's axes by mistake
    _stub_motive(monkeypatch, runner, before - swapped)
    assert runner.main(["verify", str(run_dir), "--config", str(REPO / "configs/drone.yaml")]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "Motive applied" in out


def test_verify_notices_a_remade_asset(analyzed_run, runner, monkeypatch, capsys):
    _, run_dir, _ = analyzed_run
    record = json.loads((run_dir / "alignment.json").read_text())
    _stub_motive(monkeypatch, runner, record["markers_before_m"][:-1])
    assert runner.main(["verify", str(run_dir), "--config", str(REPO / "configs/drone.yaml")]) == 1
    assert "remade" in capsys.readouterr().out


def test_a_drone_that_never_moved_gets_a_message_not_a_traceback(tmp_path, runner, capsys):
    truth = _truth()
    (tmp_path / "geometry.yaml").write_text(GEOMETRY_YAML)
    run_dir = tmp_path / "run"
    _write_simulated_run(run_dir, _simulate(truth, seconds=6.0, rest_s=3.0), _markers_for(truth))
    assert (
        runner.main(["analyze", str(run_dir), "--geometry", str(tmp_path / "geometry.yaml")]) == 1
    )
    assert "move it by hand" in capsys.readouterr().out


def test_calibrate_reboots_then_records_until_covered_and_at_rest(
    tmp_path, runner, monkeypatch, capsys
):
    """The live flow with the radio and Motive stubbed: the first real run should not find typos."""
    events = []

    class QuickCoverage(MotionCoverage):
        MIN_AXIS_ROTATION_RAD = 0.1
        MIN_TILT_DEG = 5.0
        MIN_TURNING_S = 0.1

    class Source:
        marker_offsets_m = tuple(map(tuple, CF_TEO_MARKERS))

        def __init__(self):
            self.closed = False

        def connect(self):
            events.append("connect")

        def connected(self, within_s):
            return True

        def close(self):
            self.closed = True

        def samples(self, *, start_s=0.0, dt_s=0.01):
            turn = Rotation.from_rotvec([1.0, 1.0, 1.0])
            for seq in range(100_000):
                if self.closed:
                    return
                time.sleep(0.002)
                moving = seq < 300
                attitude = (
                    Rotation.from_rotvec(np.array([1.0, 1.0, 1.0]) * 0.004 * min(seq, 300))
                    if moving
                    else turn
                )
                yield PoseSample(
                    timestamp_s=0.0,
                    position_m=(0.0, 0.0, 0.0),
                    orientation=_quaternion(attitude),
                    rigid_body="cf_teo",
                    source_seq=seq,
                )

    class Radio:
        crazyflie = object()

        def __init__(self, uri):
            pass

        def __enter__(self):
            events.append("radio")
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(runner, "build_live_pose_source", lambda *, cfg: Source())
    monkeypatch.setattr(
        runner, "power_cycle_until_healthy", lambda uri, log: events.append("reboot") or True
    )
    monkeypatch.setattr(runner, "CrazyflieRadioBackend", Radio)
    monkeypatch.setattr(runner, "MotionCoverage", QuickCoverage)
    monkeypatch.setattr(runner, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(runner, "STILL_BEFORE_S", 0.05)
    monkeypatch.setattr(runner, "STILL_AFTER_S", 0.2)
    monkeypatch.setattr(
        runner.TelemetryRecorder, "attach_stm32", lambda self, cf: events.append("imu")
    )
    monkeypatch.setattr(runner, "_analyze", lambda run_dir, geometry: events.append(run_dir) or 0)

    args = [
        "calibrate",
        "--config",
        str(REPO / "configs/drone.yaml"),
        "--geometry",
        str(REPO / "configs/marker_geometry/cf21bl_mocap_deck.yaml"),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the stubbed IMU stream records no rows
        assert runner.main(args) == 0

    run_dir = events[-1]
    assert events[:4] == ["connect", "reboot", "radio", "imu"]
    assert json.loads((run_dir / "asset.json").read_text())["markers_m"][0] == pytest.approx(
        CF_TEO_MARKERS[0].tolist()
    )
    assert pq.read_table(run_dir / "mocap.parquet").num_rows > 100
    out = capsys.readouterr().out
    assert "Pick the drone up" in out and "Enough motion" in out


def test_calibrate_stops_before_the_reboot_when_the_deck_does_not_fit(
    tmp_path, runner, monkeypatch, capsys
):
    source = types.SimpleNamespace(
        connect=lambda: None, close=lambda: None, marker_offsets_m=tuple(map(tuple, CF_TEO_MARKERS))
    )
    monkeypatch.setattr(runner, "build_live_pose_source", lambda *, cfg: source)
    monkeypatch.setattr(
        runner, "power_cycle_until_healthy", lambda uri, log: pytest.fail("rebooted")
    )
    (tmp_path / "wrong.yaml").write_text(GEOMETRY_YAML.replace("[30, 0]", "[25, 0]"))
    args = [
        "calibrate",
        "--config",
        str(REPO / "configs/drone.yaml"),
        "--geometry",
        str(tmp_path / "wrong.yaml"),
    ]
    assert runner.main(args) == 1
    assert "do not fit the holes" in capsys.readouterr().out
