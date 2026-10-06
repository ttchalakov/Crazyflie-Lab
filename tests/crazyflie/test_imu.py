"""summarize_static_imu: the stationary / one-g / disarmed verdict behind the static IMU preflight."""
import csv
import math
import random
from pathlib import Path

import pytest

from cf_lab.crazyflie.imu import ImuSample, summarize_static_imu

PAD_IMU = Path(__file__).parent / "data" / "pad_imu.csv"


def _pad_window() -> list[ImuSample]:
    """5 s of a real drone resting disarmed on the pad (telemetry_bench, 50 Hz)."""
    with PAD_IMU.open() as handle:
        return [
            ImuSample(
                acceleration_g=(float(r["acc.x"]), float(r["acc.y"]), float(r["acc.z"])),
                angular_velocity_dps=(float(r["gyro.x"]), float(r["gyro.y"]), float(r["gyro.z"])),
                supervisor_info=0x0001,  # canArm, not ARMED
            )
            for r in csv.DictReader(handle)
        ]


def _map(samples, *, acc=lambda a: a, gyro=lambda g: g, info=None):
    return [
        ImuSample(
            acceleration_g=tuple(acc(s.acceleration_g)),
            angular_velocity_dps=tuple(gyro(s.angular_velocity_dps)),
            supervisor_info=s.supervisor_info if info is None else info,
        )
        for s in samples
    ]


def _rotate_x(vector, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    x, y, z = vector
    return (x, c * y - s * z, s * y + c * z)


def _noisy(n, *, acc=(0.0, 0.0, 1.0), acc_noise=0.002, gyro=(0.0, 0.0, 0.0), gyro_noise=0.2, seed=1):
    rng = random.Random(seed)
    return [
        ImuSample(
            acceleration_g=tuple(a + rng.gauss(0.0, acc_noise) for a in acc),
            angular_velocity_dps=tuple(g + rng.gauss(0.0, gyro_noise) for g in gyro),
            supervisor_info=0,
        )
        for _ in range(n)
    ]


def test_real_pad_window_is_usable():
    summary = summarize_static_imu(_pad_window())

    assert summary.samples == 250
    assert summary.stationary and summary.reads_one_g and summary.disarmed
    assert summary.usable
    assert summary.acceleration_magnitude_g == pytest.approx(1.0, abs=0.01)


def test_tilted_but_still_drone_is_usable():
    tilted = _map(_pad_window(), acc=lambda a: _rotate_x(a, 35.0))

    summary = summarize_static_imu(tilted)

    assert summary.usable
    assert summary.mean_acceleration_g[1] == pytest.approx(-math.sin(math.radians(35.0)), abs=0.03)


def test_dead_imu_reading_exactly_zero_is_still_but_not_usable():
    dead = _map(_pad_window(), acc=lambda a: (0.0, 0.0, 0.0), gyro=lambda g: (0.0, 0.0, 0.0))

    summary = summarize_static_imu(dead)

    assert summary.stationary
    assert not summary.reads_one_g
    assert not summary.usable


def test_gyro_bias_above_rate_limit_is_not_stationary():
    biased = _map(_pad_window(), gyro=lambda g: (g[0] + 1.5, g[1], g[2]))

    summary = summarize_static_imu(biased)

    assert summary.angular_velocity_magnitude_dps > 1.0
    assert summary.angular_velocity_motion_dps < 2.0
    assert not summary.stationary and not summary.usable


def test_one_armed_sample_fails_the_window():
    samples = _pad_window()
    samples[100] = ImuSample(samples[100].acceleration_g, samples[100].angular_velocity_dps, 0x0003)

    summary = summarize_static_imu(samples)

    assert summary.armed_samples == 1
    assert not summary.disarmed and not summary.usable


def test_empty_window_is_not_usable():
    summary = summarize_static_imu([])

    assert summary.samples == 0
    assert not summary.disarmed and not summary.usable


@pytest.mark.parametrize(
    ("samples", "stationary"),
    [
        (_noisy(200, gyro_noise=0.3), True),
        (_noisy(200, gyro_noise=3.0), False),  # handled, rotating
        (_noisy(200, acc_noise=0.08), False),  # shaken, gyro still quiet
        (_noisy(200, gyro=(0.0, 0.0, 0.9)), True),  # bias residual inside the 1 dps limit
    ],
)
def test_motion_and_bias_decide_stationary(samples, stationary):
    assert summarize_static_imu(samples).stationary is stationary


def test_statistics_are_mean_norm_and_population_spread():
    samples = [
        ImuSample((0.0, 0.0, 0.9), (1.0, 0.0, 0.0), 0),
        ImuSample((0.0, 0.0, 1.1), (-1.0, 0.0, 2.0), 0),
    ]

    summary = summarize_static_imu(samples)

    assert summary.mean_acceleration_g == pytest.approx((0.0, 0.0, 1.0))
    assert summary.acceleration_magnitude_g == pytest.approx(1.0)
    assert summary.acceleration_motion_g == pytest.approx(0.1)
    assert summary.mean_angular_velocity_dps == pytest.approx((0.0, 0.0, 1.0))
    assert summary.angular_velocity_magnitude_dps == pytest.approx(1.0)
    assert summary.angular_velocity_motion_dps == pytest.approx(math.sqrt(2.0))


@pytest.mark.parametrize(("z_g", "reads_one_g"), [(0.86, True), (1.14, True), (0.84, False), (1.16, False)])
def test_reads_one_g_within_0_15_g_of_gravity(z_g, reads_one_g):
    samples = [ImuSample((0.0, 0.0, z_g), (0.0, 0.0, 0.0), 0)] * 10

    assert summarize_static_imu(samples).reads_one_g is reads_one_g


def test_nan_sample_fails_closed():
    samples = _noisy(50)
    samples[10] = ImuSample((float("nan"), 0.0, 1.0), (0.0, 0.0, 0.0), 0)

    assert not summarize_static_imu(samples).usable


def test_negative_threshold_is_rejected():
    with pytest.raises(ValueError, match="max_gyro_rate_dps"):
        summarize_static_imu(_noisy(5), max_gyro_rate_dps=-1.0)
