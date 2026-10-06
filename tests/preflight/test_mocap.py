"""check_mocap_quality pass/fail boundaries over a replayed, clocked extpose stream."""
import math
import time

import pytest

from cf_lab.backends.fake import FakeCrazyflieBackend
from cf_lab.mocap.fake import ScriptedTakeoffSource
from cf_lab.mocap.stream import ExtposeStreamer
from cf_lab.preflight.mocap import check_mocap_quality

LAB = {"expected_rate_hz": 100.0, "expected_rigid_body": "cf21bl_01"}  # what the runners pass


def _check(fakes, clock, events, **kwargs):
    streamer = fakes.ReplayStreamer(clock, events)
    return check_mocap_quality(streamer, **kwargs), streamer


def test_real_pad_recording_passes_with_the_runner_arguments(fakes, mocap_clock, through_uplink):
    # 360 Hz NatNet with the drone on its pad, through the real 100 Hz latest-wins uplink cap.
    events = through_uplink(fakes.recorded_poses("mocap_pad_natnet.csv"))

    result, streamer = _check(fakes, mocap_clock, events, **LAB)

    assert result.passed, result.summary
    assert result.details["sample_count"] > 340
    assert 80.0 <= result.details["observed_rate_hz"] <= 100.0
    assert streamer.unsubscribed


def test_default_window_is_one_second(fakes, mocap_clock):
    started = mocap_clock.now

    _check(fakes, mocap_clock, fakes.steady_stream())

    assert mocap_clock.now - started == pytest.approx(1.0)


def test_real_flight_recording_fails_the_stationary_spread_but_passes_as_a_moving_body(
    fakes, mocap_clock, through_uplink
):
    events = through_uplink(fakes.recorded_poses("mocap_flight_natnet.csv"))

    stationary, _ = _check(fakes, mocap_clock, events, **LAB)
    moving, _ = _check(fakes, mocap_clock, events, max_position_spread_m=None, **LAB)

    assert not stationary.passed
    assert stationary.details["failures"][0].startswith("stationary position spread")
    assert moving.passed, moving.summary


@pytest.mark.parametrize(("age_ms", "passed"), [(49.0, True), (50.0, True), (51.0, False)])
def test_receipt_freshness_limit_is_fifty_ms(fakes, mocap_clock, age_ms, passed):
    last = 1.0 - age_ms / 1000.0
    events = [(t, t, fakes.pose(seq=i)) for i, t in enumerate((0.0, last / 2, last))]

    result, _ = _check(fakes, mocap_clock, events)

    assert result.passed is passed
    if not passed:
        assert result.details["failures"][0] == "latest received pose is 51.0 ms old (limit 50.0 ms)"


def test_stream_that_stalls_mid_window_fails_freshness(fakes, mocap_clock):
    events = fakes.steady_stream(duration_s=0.7)

    result, _ = _check(fakes, mocap_clock, events)

    assert not result.passed
    assert result.summary.startswith("Mocap quality failed: latest received pose is 300.")


def test_uplink_stall_fails_even_while_mocap_keeps_arriving(fakes, mocap_clock):
    # Poses keep arriving but none was forwarded to the drone in the last 200 ms.
    events = [
        (t, t if t < 0.8 else None, pose)
        for t, _, pose in fakes.steady_stream()
    ]

    result, _ = _check(fakes, mocap_clock, events)

    assert not result.passed
    assert len(result.details["failures"]) == 1
    assert result.details["failures"][0].startswith("latest forwarded pose is 200.")


def test_no_poses_at_all_fails(fakes, mocap_clock):
    result, _ = _check(fakes, mocap_clock, [])

    assert not result.passed
    assert result.details["failures"][0] == "only 0 forwarded sample(s), need at least 2"


def test_freshness_without_poses_in_the_window_uses_the_last_pose_seen_before_it(fakes, mocap_clock):
    streamer = fakes.ReplayStreamer(
        mocap_clock, [], latest_received_s=-0.01, latest_forwarded_s=-0.01
    )

    result = check_mocap_quality(streamer, minimum_samples=1)

    assert not result.passed
    assert result.details["receipt_freshness_ms"] == pytest.approx(1010.0)


@pytest.mark.parametrize(("rate_hz", "passed"), [(79, False), (81, True)])
def test_forwarded_rate_must_reach_eighty_percent_of_expected(fakes, mocap_clock, rate_hz, passed):
    result, _ = _check(
        fakes, mocap_clock, fakes.steady_stream(rate_hz=rate_hz), expected_rate_hz=100.0
    )

    assert result.passed is passed


def test_decimated_poses_do_not_count_toward_the_forwarded_rate(fakes, mocap_clock):
    # 150 Hz received, every other pose dropped by the uplink cap: 75 Hz reach the drone.
    events = [
        (t, t if i % 2 else None, pose)
        for i, (t, _, pose) in enumerate(fakes.steady_stream(rate_hz=150))
    ]

    result, _ = _check(fakes, mocap_clock, events, expected_rate_hz=100.0)

    assert not result.passed
    assert result.details["failures"] == ("forwarded rate 75.0 Hz is below 80.0 Hz",)


@pytest.mark.parametrize(("dx", "passed"), [(0.049, True), (0.051, False)])
def test_stationary_spread_limit_is_five_cm(fakes, mocap_clock, dx, passed):
    events = fakes.steady_stream(
        poses=lambda i: fakes.pose((0.43 + dx * i / 99, -0.44, 0.06), seq=i)
    )

    result, _ = _check(fakes, mocap_clock, events)

    assert result.passed is passed


def test_a_120hz_motive_through_the_real_uplink_cap_passes_the_runner_rate_gate(
    fakes, mocap_clock, through_uplink
):
    # Motive's 120 Hz preset, still on the pad, with the runners' expected_rate_hz=100 (80 Hz floor).
    events = through_uplink([(i / 120.0, fakes.pose(seq=i)) for i in range(120)])

    result, _ = _check(fakes, mocap_clock, events, **LAB)

    assert result.passed, result.summary


@pytest.mark.parametrize(("step_m", "passed"), [(0.04, True), (0.06, False)])
def test_a_single_position_jump_over_five_cm_fails_even_for_a_moving_body(
    fakes, mocap_clock, step_m, passed
):
    events = fakes.steady_stream(
        poses=lambda i: fakes.pose((0.43 + (step_m if i >= 50 else 0.0), -0.44, 0.06), seq=i)
    )

    result, _ = _check(fakes, mocap_clock, events, max_position_spread_m=None)

    assert result.passed is passed


def _yaw(deg):
    half = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


@pytest.mark.parametrize(("step_deg", "passed"), [(15.0, True), (25.0, False)])
def test_orientation_jump_limit_is_twenty_degrees(fakes, mocap_clock, step_deg, passed):
    events = fakes.steady_stream(
        poses=lambda i: fakes.pose(q=_yaw(step_deg if i >= 50 else 0.0), seq=i)
    )

    result, _ = _check(fakes, mocap_clock, events)

    assert result.passed is passed


def test_quaternion_sign_flip_is_not_an_orientation_jump(fakes, mocap_clock):
    # q and -q are the same attitude; NatNet may emit either.
    q = _yaw(90.0)
    events = fakes.steady_stream(
        poses=lambda i: fakes.pose(q=q if i % 2 else tuple(-v for v in q), seq=i)
    )

    result, _ = _check(fakes, mocap_clock, events)

    assert result.passed, result.summary
    assert result.details["max_orientation_jump_deg"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("bad_pose", "failure"),
    [
        ({"position": (math.nan, -0.44, 0.06)}, "1 pose(s) have non-finite position"),
        ({"q": (0.0, 0.0, 0.0, 0.0)}, "1 pose(s) have invalid quaternion"),
        ({"q": (0.0, 0.0, 0.0, 1.06)}, "1 pose(s) have invalid quaternion"),
        ({"q": (0.0, 0.0, 0.0, math.inf)}, "1 pose(s) have invalid quaternion"),
    ],
    ids=["nan_position", "zero_quaternion", "unnormalized_quaternion", "inf_quaternion"],
)
def test_one_invalid_pose_fails_the_check(fakes, mocap_clock, bad_pose, failure):
    events = fakes.steady_stream(
        poses=lambda i: fakes.pose(seq=i, **(bad_pose if i == 30 else {}))
    )

    result, _ = _check(fakes, mocap_clock, events, max_position_jump_m=None,
                       max_orientation_jump_deg=None)

    assert not result.passed
    assert failure in result.details["failures"]


def test_quaternion_within_default_norm_tolerance_passes(fakes, mocap_clock):
    events = fakes.steady_stream(poses=lambda i: fakes.pose(q=(0.0, 0.0, 0.0, 1.04), seq=i))

    assert _check(fakes, mocap_clock, events)[0].passed


@pytest.mark.parametrize(
    ("seqs", "passed"),
    [
        (lambda i: 10 + 3 * i, True),
        (lambda i: i if i != 40 else 39, False),
        (lambda i: i if i != 40 else 5, False),
    ],
    ids=["gaps_are_fine", "duplicate", "regression"],
)
def test_source_sequence_must_strictly_increase(fakes, mocap_clock, seqs, passed):
    events = fakes.steady_stream(poses=lambda i: fakes.pose(seq=seqs(i)))

    assert _check(fakes, mocap_clock, events)[0].passed is passed


@pytest.mark.parametrize(
    "bodies", [("cf21bl_02",), ("cf21bl_01", "cf21bl_02")], ids=["wrong", "two_bodies"]
)
def test_rigid_body_must_be_exactly_the_expected_one(fakes, mocap_clock, bodies):
    events = fakes.steady_stream(poses=lambda i: fakes.pose(seq=i, body=bodies[i % len(bodies)]))

    result, _ = _check(fakes, mocap_clock, events, **LAB)

    assert not result.passed
    assert result.details["failures"][-1].startswith("observed rigid body/bodies")


def test_streamer_error_fails_the_check(fakes, mocap_clock):
    streamer = fakes.ReplayStreamer(
        mocap_clock, fakes.steady_stream(), error=ConnectionResetError("natnet socket closed")
    )

    result = check_mocap_quality(streamer)

    assert not result.passed
    assert result.error == "ConnectionResetError: natnet socket closed"


def test_object_that_cannot_be_observed_fails(mocap_clock):
    result = check_mocap_quality(None)

    assert not result.passed
    assert result.error.startswith("AttributeError")


@pytest.mark.parametrize(
    "threshold",
    [
        "duration_s",
        "expected_rate_hz",
        "minimum_rate_hz",
        "max_freshness_ms",
        "max_position_spread_m",
        "max_position_jump_m",
        "max_orientation_jump_deg",
    ],
)
@pytest.mark.parametrize("value", [math.nan, -1.0])
def test_invalid_threshold_fails_instead_of_disabling_its_comparison(
    fakes, mocap_clock, threshold, value
):
    # Every comparison against NaN is False, so a NaN limit would otherwise always pass.
    result, streamer = _check(fakes, mocap_clock, fakes.steady_stream(), **{threshold: value})

    assert not result.passed
    assert result.summary == "Invalid mocap-quality check configuration."
    assert streamer.observer is None


def test_live_streamer_observer_path_passes_on_a_still_fake_source():
    """End to end with the real ExtposeStreamer thread, a paced fake source and the fake radio."""
    streamer = ExtposeStreamer(
        source=ScriptedTakeoffSource(ground_s=60.0),
        backend=FakeCrazyflieBackend(),
        max_rate_hz=100.0,
        pace=True,
    )
    streamer.start()
    try:
        deadline = time.monotonic() + 1.0
        while streamer.sent == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        # Generous rate and freshness: this asserts the wiring, not the scheduling of a busy machine.
        result = check_mocap_quality(
            streamer, duration_s=0.3, minimum_rate_hz=40.0, max_freshness_ms=250.0,
            expected_rigid_body="cf21bl_01",
        )
    finally:
        streamer.stop()
        streamer.join(timeout=1.0)

    assert result.passed, result.summary
