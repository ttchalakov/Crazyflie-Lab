"""Firmware power-on self-test gate.

The firmware runs a sensor/deck self-test once at boot and latches the verdict
in ``system.selftestPassed`` for the life of that boot. When it fails, the
supervisor refuses every arming request for as long as the drone stays powered:
``supervisor.info`` never sets its ``canArm`` bit, so a runner can stream
setpoints for minutes at a drone that is quietly sitting on the pad.

This is worth a dedicated gate because the failure is *silent and intermittent*.
A drone whose IMU failed to enumerate reports ``acc``/``gyro``/``baro`` as
exactly 0.0. The gyro then looks perfectly still, but the accelerometer reads
0 g instead of 1 g, so :func:`cf_lab.preflight.crazyflie.check_static_imu`
fails as well.

A radio power-cycle (:func:`cf_lab.backends.crazyflie.power_cycle`) can leave the
IMU unenumerated. Another power cycle clears it; see ``cf-lab reboot``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cf_lab.crazyflie.parameters import read_parameter

SELF_TEST_PARAMETER = "system.selftestPassed"


class TelemetryEventSink(Protocol):
    """Minimal telemetry interface accepted by :func:`self_test_passed`."""

    def event(self, kind: str, message: str = "", **data: object) -> None: ...


@dataclass(frozen=True)
class SelfTestResult:
    """The firmware's latched boot self-test verdict.

    ``passed`` is true only on a confirmed pass. A read that fails (link down,
    parameter missing on an older firmware) reports ``passed=False`` with
    ``error`` set, so an unreadable gate never reads as permission to arm.
    """

    passed: bool
    raw: str | None = None
    error: str | None = None


def read_self_test(cf, *, timeout_s: float = 3.0) -> SelfTestResult:
    """Read ``system.selftestPassed``, reporting failures rather than raising."""
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")
    try:
        raw = read_parameter(cf, SELF_TEST_PARAMETER, timeout_s)
    except Exception as exc:  # noqa: BLE001 - any read failure is a failed gate
        return SelfTestResult(passed=False, error=f"{type(exc).__name__}: {exc}")
    return SelfTestResult(passed=str(raw).strip() == "1", raw=str(raw))


def self_test_passed(
    cf,
    *,
    timeout_s: float = 3.0,
    telemetry: TelemetryEventSink | None = None,
) -> bool:
    """Return whether the firmware's boot self-test passed.

    Passing a telemetry recorder opts in to a ``self_test_check`` event, so a
    run that was refused leaves the same audit trail as one that flew.
    """
    result = read_self_test(cf, timeout_s=timeout_s)
    if telemetry is not None:
        telemetry.event(
            "self_test_check",
            passed=result.passed,
            parameter=SELF_TEST_PARAMETER,
            value=result.raw,
            error=result.error,
        )
    return result.passed
