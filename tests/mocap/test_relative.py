"""Quaternion maths and the origin-zeroing transform that defines the frame streamed to the drone."""
import math

import numpy as np
import pytest

from cf_lab.mocap.relative import OriginZeroer
from cf_lab.mocap.types import EulerAngles, PoseSample, Quaternion


def _rot(q: Quaternion) -> np.ndarray:
    x, y, z, w = q.qx, q.qy, q.qz, q.qw
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _zyx(roll, pitch, yaw) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def _random_euler(rng):
    return EulerAngles(
        roll=rng.uniform(-math.pi, math.pi), pitch=rng.uniform(-1.5, 1.5), yaw=rng.uniform(-math.pi, math.pi)
    )


def _sample(position, orientation, seq=None):
    return PoseSample(timestamp_s=1.5, position_m=tuple(position), orientation=orientation,
                      rigid_body="cf21bl_01", source_seq=seq)


# -- quaternion maths -----------------------------------------------------------


def test_euler_to_quaternion_is_the_zyx_rotation_and_round_trips():
    rng = np.random.default_rng(1)
    for _ in range(200):
        euler = _random_euler(rng)
        q = Quaternion.from_euler_rad(euler)
        np.testing.assert_allclose(_rot(q), _zyx(euler.roll, euler.pitch, euler.yaw), atol=1e-12)
        back = q.to_euler_rad()
        assert (back.roll, back.pitch, back.yaw) == pytest.approx((euler.roll, euler.pitch, euler.yaw), abs=1e-9)


def test_hamilton_product_composes_rotations_left_after_right():
    rng = np.random.default_rng(2)
    for _ in range(100):
        a = Quaternion.from_euler_rad(_random_euler(rng))
        b = Quaternion.from_euler_rad(_random_euler(rng))
        np.testing.assert_allclose(_rot(a.multiply(b)), _rot(a) @ _rot(b), atol=1e-12)


@pytest.mark.parametrize("pitch", [math.pi / 2, -math.pi / 2])
def test_to_euler_survives_gimbal_lock(pitch):
    # sqrt(0.5)**2 rounds above 0.5, so sin(pitch) lands just past 1.
    q = Quaternion(0.0, math.copysign(math.sqrt(0.5), pitch), 0.0, math.sqrt(0.5))
    assert q.to_euler_rad().pitch == pytest.approx(pitch, abs=1e-6)


def test_to_euler_normalizes_a_scaled_quaternion():
    q = Quaternion.from_euler_rad(EulerAngles(0.1, -0.2, 2.5))
    euler = Quaternion(q.qx * 3, q.qy * 3, q.qz * 3, q.qw * 3).to_euler_rad()
    assert (euler.roll, euler.pitch, euler.yaw) == pytest.approx((0.1, -0.2, 2.5), abs=1e-12)


def test_zero_quaternion_is_rejected():
    with pytest.raises(ValueError):
        Quaternion(0.0, 0.0, 0.0, 0.0).normalized()


# -- origin zeroing ----------------------------------------------------------------


def test_first_pose_becomes_origin_with_zero_yaw_and_unchanged_roll_pitch():
    origins = []
    zeroer = OriginZeroer(on_origin=lambda p, yaw: origins.append((p, yaw)))
    raw = _sample((0.9, -0.6, 0.05), Quaternion.from_euler_rad(EulerAngles(0.03, -0.02, 2.0)), seq=11)
    out = zeroer.zero(raw)
    assert out.position_m == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    euler = out.orientation.to_euler_rad()
    assert (euler.roll, euler.pitch, euler.yaw) == pytest.approx((0.03, -0.02, 0.0), abs=1e-12)
    assert origins == [((0.9, -0.6, 0.05), pytest.approx(2.0))]
    assert (out.source_seq, out.timestamp_s, out.rigid_body) == (11, 1.5, "cf21bl_01")


def test_zeroing_keeps_the_source_arrival_time_and_latency():
    pose = PoseSample(timestamp_s=1.5, position_m=(1.0, 2.0, 0.0), orientation=Quaternion.identity(),
                      rigid_body="cf21bl_01", source_seq=3, host_ns=6_932_239_680_158, latency_s=0.004)
    out = OriginZeroer().zero(pose)
    assert (out.host_ns, out.latency_s) == (6_932_239_680_158, 0.004)


def test_origin_locks_once_and_on_origin_fires_once():
    origins = []
    zeroer = OriginZeroer(on_origin=lambda p, yaw: origins.append(p))
    zeroer.zero(_sample((1.0, 2.0, 0.0), Quaternion.identity()))
    second = zeroer.zero(_sample((1.5, 2.0, 0.3), Quaternion.identity()))
    assert second.position_m == pytest.approx((0.5, 0.0, 0.3))
    assert origins == [(1.0, 2.0, 0.0)]


@pytest.mark.parametrize(
    "yaw, displacement, expected",
    [
        (math.pi / 2, (0.0, 1.0, 0.2), (1.0, 0.0, 0.2)),  # move along the start heading -> +x
        (math.pi / 2, (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # start's left -> +y
        (-math.pi / 2, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        (math.pi, (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    ],
)
def test_positions_are_expressed_in_the_start_heading_frame(yaw, displacement, expected):
    origin = (0.3, -0.7, 0.1)
    start_q = Quaternion.from_euler_rad(EulerAngles(0.0, 0.0, yaw))
    zeroer = OriginZeroer()
    zeroer.zero(_sample(origin, start_q))
    moved = zeroer.zero(_sample(np.add(origin, displacement), start_q))
    assert moved.position_m == pytest.approx(expected, abs=1e-12)
    assert moved.orientation.to_euler_rad().yaw == pytest.approx(0.0, abs=1e-12)


def test_zeroing_is_one_rigid_transform_for_every_later_pose():
    rng = np.random.default_rng(3)
    start = _sample(rng.normal(size=3), Quaternion.from_euler_rad(_random_euler(rng)))
    yaw0 = start.orientation.to_euler_rad().yaw
    unyaw = _zyx(0.0, 0.0, -yaw0)
    zeroer = OriginZeroer()
    zeroer.zero(start)
    for _ in range(100):
        euler = _random_euler(rng)
        raw = _sample(rng.normal(size=3), Quaternion.from_euler_rad(euler))
        out = zeroer.zero(raw)
        np.testing.assert_allclose(out.position_m, unyaw @ (np.array(raw.position_m) - start.position_m), atol=1e-12)
        np.testing.assert_allclose(_rot(out.orientation), unyaw @ _rot(raw.orientation), atol=1e-12)
        got = out.orientation.to_euler_rad()
        assert (got.roll, got.pitch) == pytest.approx((euler.roll, euler.pitch), abs=1e-9)


def test_set_origin_replays_live_zeroing_exactly():
    rng = np.random.default_rng(4)
    poses = [_sample(rng.normal(size=3), Quaternion.from_euler_rad(_random_euler(rng))) for _ in range(50)]
    recorded = []
    live = OriginZeroer(on_origin=lambda p, yaw: recorded.append((p, yaw)))
    replay = OriginZeroer()
    live_out = [live.zero(p) for p in poses]
    replay.set_origin(*recorded[0])
    assert [replay.zero(p) for p in poses] == live_out


def test_real_flight_uplink_is_reproduced_from_raw_mocap(flight_sample):
    """A real mocap_waypoint flight: raw NatNet poses + origin event recreate every sent extpose."""
    raw = {row["seq"]: row for row in flight_sample.mocap}
    origins = []
    zeroer = OriginZeroer(on_origin=lambda p, yaw: origins.append((p, yaw)))
    first = flight_sample.mocap[0]
    zeroer.zero(_sample(first["position"], first["quat"], first["seq"]))
    origin = flight_sample.origin
    assert origins == [((origin["x"], origin["y"], origin["z"]), pytest.approx(origin["yaw_rad"], abs=1e-12))]
    assert len(flight_sample.extpose) > 150
    for sent in flight_sample.extpose:
        source = raw[sent["seq"]]
        out = zeroer.zero(_sample(source["position"], source["quat"], source["seq"]))
        assert out.position_m == pytest.approx(sent["position"], abs=1e-9)
        q, ref = out.orientation, sent["quat"]
        assert (q.qx, q.qy, q.qz, q.qw) == pytest.approx((ref.qx, ref.qy, ref.qz, ref.qw), abs=1e-9)
        assert out.source_seq == sent["seq"]
