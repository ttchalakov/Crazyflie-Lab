"""figure_eight_setpoints: the timed setpoint list mocap_waypoint flies."""
import math

import numpy as np
import pytest

from cf_lab.trajectory import figure_eight_setpoints

# The mocap_waypoint flight and an off-centre, lopsided box.
CASES = [
    dict(x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), z_range=(0.3, 2.0), yaw_range_deg=(-180, 180), loops=2, duration_s=38.4),
    dict(x_range=(1.0, 1.4), y_range=(-3.0, -1.0), z_range=(0.8, 1.1), yaw_range_deg=(10, 50), loops=3, duration_s=12.3),
]


def eight(traj):
    return np.array(traj[1:-1])


@pytest.mark.parametrize("kwargs", CASES)
def test_every_setpoint_stays_inside_the_box(kwargs):
    traj = np.array(figure_eight_setpoints(**kwargs))
    for column, key in zip(range(1, 5), ("x_range", "y_range", "z_range", "yaw_range_deg")):
        lo, hi = kwargs[key]
        assert lo <= traj[:, column].min() and traj[:, column].max() <= hi, key
    # The eight fills its box rather than shrinking into a corner of it.
    path = eight(traj)
    for column, key in zip(range(1, 5), ("x_range", "y_range", "z_range", "yaw_range_deg")):
        lo, hi = kwargs[key]
        assert path[:, column].max() - path[:, column].min() > 0.98 * (hi - lo), key


@pytest.mark.parametrize("kwargs", CASES)
def test_takeoff_and_landing_segments_frame_the_eight(kwargs):
    traj = figure_eight_setpoints(**kwargs)
    xc, yc = (sum(kwargs["x_range"]) / 2, sum(kwargs["y_range"]) / 2)
    z_lo, z_hi = kwargs["z_range"]
    takeoff, last, land = traj[0], traj[-2], traj[-1]
    assert takeoff[0] == 5.0 and takeoff[1:3] == pytest.approx((xc, yc))
    assert land[0] == 8.0 and land[1:3] == pytest.approx((xc, yc))
    assert land[3] == pytest.approx(min(max(0.6, z_lo), z_hi))
    # The eight starts and ends over the centre, so both are pure vertical moves.
    assert last[1:5] == pytest.approx(takeoff[1:5])
    assert land[4] == pytest.approx(last[4])


@pytest.mark.parametrize("z_range, land_z", [((0.3, 2.0), 0.6), ((1.0, 2.0), 1.0), ((0.1, 0.5), 0.5)])
def test_land_height_is_clamped_into_the_z_range(z_range, land_z):
    assert figure_eight_setpoints(z_range=z_range)[-1][3] == pytest.approx(land_z)


@pytest.mark.parametrize("duration_s, loops", [(20.0, 1), (38.4, 2), (12.34, 3), (0.4, 1)])
def test_eight_samples_every_tenth_of_a_second_and_lasts_duration_s(duration_s, loops):
    traj = figure_eight_setpoints(loops=loops, duration_s=duration_s)
    dwell = eight(traj)[:, 0]
    assert len(dwell) == round(duration_s / 0.1)
    assert dwell.sum() == pytest.approx(duration_s)
    assert np.allclose(dwell, dwell[0]) and abs(dwell[0] - 0.1) < 0.1 / 3


def test_loops_repeat_the_same_eight():
    one = eight(figure_eight_setpoints(loops=1, duration_s=10.0))
    three = eight(figure_eight_setpoints(loops=3, duration_s=30.0))
    np.testing.assert_allclose(three, np.tile(one, (3, 1)), atol=1e-9)


def test_eight_is_a_lissajous_figure_of_eight():
    path = eight(figure_eight_setpoints(duration_s=10.0))
    t = 2 * math.pi * np.arange(1, 101) / 100
    np.testing.assert_allclose(path[:, 1], np.sin(t), atol=1e-12)
    np.testing.assert_allclose(path[:, 2], np.sin(2 * t), atol=1e-12)
    np.testing.assert_allclose(path[:, 3], 1.25 + 0.75 * np.cos(t), atol=1e-12)
    np.testing.assert_allclose(path[:, 4], 180 * np.sin(t), atol=1e-9)


@pytest.mark.parametrize("kwargs", CASES)
def test_consecutive_setpoints_never_jump(kwargs):
    traj = np.array(figure_eight_setpoints(**kwargs))
    steps = np.abs(np.diff(traj[:-1, 1:5], axis=0))  # takeoff point through the last eight point
    dphase = 2 * math.pi * kwargs["loops"] / round(kwargs["duration_s"] / 0.1)
    half = [(hi - lo) / 2 for lo, hi in (kwargs[k] for k in ("x_range", "y_range", "z_range", "yaw_range_deg"))]
    # |d/dphase| of sin(t), sin(2t), cos(t), sin(t) bounds each step.
    bound = np.array([half[0], 2 * half[1], half[2], half[3]]) * dphase
    assert (steps <= bound * (1 + 1e-9)).all()


def test_fractional_loops_lands_straight_down_from_where_the_eight_ends():
    # 1.25 loops ends the eight 1 m off-centre in x; the landing must not step there sideways.
    traj = figure_eight_setpoints(loops=1.25, duration_s=10.0)
    assert traj[-2][1] == pytest.approx(1.0)
    assert traj[-1][1:3] == pytest.approx(traj[-2][1:3], abs=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(x_range=(1.0, 1.0)),
        dict(y_range=(1.0, -1.0)),
        dict(z_range=(0.5, math.inf)),
        dict(yaw_range_deg=(math.nan, 10.0)),
        dict(loops=0),
        dict(duration_s=0.3),
        dict(loops=2, duration_s=0.7),
        dict(duration_s=-5.0),
        dict(duration_s=math.nan),
        dict(duration_s=math.inf),
    ],
)
def test_rejects_unflyable_requests(kwargs):
    with pytest.raises(ValueError):
        figure_eight_setpoints(**kwargs)
