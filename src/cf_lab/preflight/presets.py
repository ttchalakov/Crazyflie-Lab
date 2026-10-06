"""Small, explicit check collections for common experiment entry points."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence

from cf_lab.preflight.core import Check
from cf_lab.preflight.crazyflie import (
    check_battery,
    check_expected_parameters,
    check_firmware_self_test,
    check_required_decks,
    check_static_imu,
    check_supervisor_ready,
)


def standard_flight_checks(
    cf,
    *,
    minimum_battery_voltage_v: float = 3.7,
    static_imu_seconds: float = 1.5,
    expected_parameters: Mapping[str, str | int | float | bool] | None = None,
    required_decks: Sequence[str] = (),
) -> tuple[Check, ...]:
    """Build the read-only baseline used immediately before a real flight.

    The returned callables do nothing until passed to :func:`run_preflight`,
    so an application may remove, append, or reorder checks explicitly.  An
    IMU that failed to start fails both the firmware self-test and the
    stationary IMU check, which reads 0 g.

    Experiment-specific parameter and deck expectations are included only when
    supplied.  No check in this collection writes a parameter, resets the
    estimator, arms, commands a motor, reboots, or power-cycles the vehicle.
    """

    checks: list[Check] = [
        functools.partial(check_firmware_self_test, cf),
        functools.partial(check_supervisor_ready, cf),
        functools.partial(
            check_battery,
            cf,
            minimum_voltage_v=minimum_battery_voltage_v,
        ),
        functools.partial(check_static_imu, cf, seconds=static_imu_seconds),
    ]
    if expected_parameters is not None:
        checks.append(functools.partial(check_expected_parameters, cf, expected_parameters))
    if required_decks:
        checks.append(functools.partial(check_required_decks, cf, required_decks))
    return tuple(checks)


__all__ = ["standard_flight_checks"]
