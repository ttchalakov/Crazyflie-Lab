"""Composable, read-only preflight checks for a connected Crazyflie.

Every function in this module observes an already-connected cflib Crazyflie
and returns a :class:`~cf_lab.preflight.core.PreflightResult`.  Hardware and
transport failures fail closed in the result; invalid caller arguments raise
``ValueError``.  None of these checks writes parameters, resets an estimator,
arms, commands a motor, reboots, or power-cycles the vehicle.
"""

from __future__ import annotations

import math
import struct
import time
from collections.abc import Mapping, Sequence
from numbers import Integral, Real

from cf_lab.backends.crazyflie import SUPERVISOR_INFO_BITS
from cf_lab.crazyflie.capture import capture_log_window
from cf_lab.crazyflie.health import SELF_TEST_PARAMETER, read_self_test
from cf_lab.crazyflie.parameters import read_parameter
from cf_lab.crazyflie.imu import (
    DEFAULT_MAX_ACCELERATION_MAGNITUDE_ERROR_G,
    DEFAULT_MAX_ACCELERATION_MOTION_G,
    DEFAULT_MAX_GYRO_MOTION_DPS,
    DEFAULT_MAX_GYRO_RATE_DPS,
    IMU_LOG_VARIABLES,
    ImuSample,
    summarize_static_imu,
)
from cf_lab.preflight.core import PreflightResult

_SUPERVISOR_CAN_ARM = SUPERVISOR_INFO_BITS["canArm"]
_SUPERVISOR_FORBIDDEN = {
    name: SUPERVISOR_INFO_BITS[name] for name in ("ARMED", "TUMBLED", "LOCKED", "crashed")
}
_BATTERY_LOG_VARIABLES = (
    ("pm.vbat", "float"),
    ("pm.batteryLevel", "uint8_t"),
    ("pm.state", "int8_t"),
)
# A dead IMU reads exactly 0 g; a working but badly scaled one reads well above this.
_DEAD_ACCELEROMETER_MAX_G = 0.1


def check_firmware_self_test(cf, *, timeout_s: float = 3.0) -> PreflightResult:
    """Adapt the firmware's latched boot self-test to the common result API."""
    _require_finite_number("timeout_s", timeout_s, positive=True)

    started = time.perf_counter()
    result = read_self_test(cf, timeout_s=timeout_s)
    if result.passed:
        summary = "Firmware boot self-test passed"
        remediation = ""
    else:
        summary = "Firmware boot self-test did not pass"
        remediation = (
            "Keep the drone still, cold-cycle it, and rerun the check. Repeated failures "
            "indicate a sensor, deck, or firmware startup problem."
        )
    return PreflightResult(
        check_id="firmware_self_test",
        passed=result.passed,
        summary=summary,
        details={"parameter": SELF_TEST_PARAMETER, "raw": result.raw},
        error=result.error,
        remediation=remediation,
        elapsed_s=time.perf_counter() - started,
    )


def check_static_imu(
    cf,
    *,
    seconds: float = 1.5,
    period_ms: int = 20,
    minimum_samples: int = 10,
    max_gyro_rate_dps: float = DEFAULT_MAX_GYRO_RATE_DPS,
    max_gyro_motion_dps: float = DEFAULT_MAX_GYRO_MOTION_DPS,
    max_acceleration_motion_g: float = DEFAULT_MAX_ACCELERATION_MOTION_G,
    max_acceleration_magnitude_error_g: float = DEFAULT_MAX_ACCELERATION_MAGNITUDE_ERROR_G,
) -> PreflightResult:
    """Check a stationary, disarmed IMU window.

    This function does not run :func:`check_firmware_self_test`: experiments
    select and order their own checks.  An IMU that failed to start reads 0 g
    and fails here too.
    """
    _validate_capture_args(seconds=seconds, period_ms=period_ms, minimum_samples=minimum_samples)
    thresholds = {
        "max_gyro_rate_dps": max_gyro_rate_dps,
        "max_gyro_motion_dps": max_gyro_motion_dps,
        "max_acceleration_motion_g": max_acceleration_motion_g,
        "max_acceleration_magnitude_error_g": max_acceleration_magnitude_error_g,
    }
    for name, value in thresholds.items():
        _require_finite_number(name, value, non_negative=True)

    started = time.perf_counter()
    try:
        rows = capture_log_window(
            cf, name="cf_lab_imu", variables=IMU_LOG_VARIABLES, seconds=seconds, period_ms=period_ms
        )
        summary = summarize_static_imu([ImuSample.from_log(row) for row in rows], **thresholds)
    except Exception as exc:  # noqa: BLE001 - an incomplete observation fails closed
        return _failed_operation(
            "static_imu",
            "Static IMU observation could not complete",
            exc,
            started,
            remediation="Check the radio link and firmware log TOC, then rerun while still.",
        )

    enough_samples = summary.samples >= minimum_samples
    # A dead IMU fails however loose the one-g tolerance is set.
    responding = summary.acceleration_magnitude_g > _DEAD_ACCELEROMETER_MAX_G
    passed = enough_samples and summary.usable and responding
    details = {
        "seconds": seconds,
        "period_ms": period_ms,
        "minimum_samples": minimum_samples,
        "samples": summary.samples,
        "armed_samples": summary.armed_samples,
        "disarmed": summary.disarmed,
        "stationary": summary.stationary,
        "reads_one_g": summary.reads_one_g,
        "mean_acceleration_g": list(summary.mean_acceleration_g),
        "acceleration_magnitude_g": summary.acceleration_magnitude_g,
        "acceleration_motion_g": summary.acceleration_motion_g,
        "mean_angular_velocity_dps": list(summary.mean_angular_velocity_dps),
        "angular_velocity_magnitude_dps": summary.angular_velocity_magnitude_dps,
        "angular_velocity_motion_dps": summary.angular_velocity_motion_dps,
        **thresholds,
    }
    if passed:
        text = f"Static, disarmed IMU window passed with {summary.samples} samples"
        remediation = ""
    elif not enough_samples:
        text = f"Static IMU window had only {summary.samples}/{minimum_samples} required samples"
        remediation = "Check radio/log bandwidth and collect a longer stationary window."
    elif not summary.disarmed:
        text = f"Drone was armed in {summary.armed_samples} IMU sample(s)"
        remediation = "Disarm the drone before running a preflight observation."
    elif not responding:
        text = "Accelerometer reads about 0 g: the IMU is not responding"
        remediation = "Cold-cycle the drone (cf-lab reboot) and rerun the check."
    else:
        text = "IMU window was not sufficiently still or close to one-g gravity"
        remediation = (
            "Keep the drone still. If the result persists, "
            "cold-cycle it without moving it during gyro calibration."
        )
    return PreflightResult(
        check_id="static_imu",
        passed=passed,
        summary=text,
        details=details,
        remediation=remediation,
        elapsed_s=time.perf_counter() - started,
    )


def check_supervisor_ready(
    cf,
    *,
    seconds: float = 0.25,
    period_ms: int = 50,
    minimum_samples: int = 1,
) -> PreflightResult:
    """Require ``canArm`` and reject unsafe supervisor state bits."""
    _validate_capture_args(seconds=seconds, period_ms=period_ms, minimum_samples=minimum_samples)
    started = time.perf_counter()
    try:
        rows = capture_log_window(
            cf,
            name="cf_lab_supervisor",
            variables=(("supervisor.info", "uint16_t"),),
            seconds=seconds,
            period_ms=period_ms,
        )
        values = tuple(_exact_int(row["supervisor.info"]) for row in rows)
    except Exception as exc:  # noqa: BLE001 - an incomplete observation fails closed
        return _failed_operation(
            "supervisor_ready",
            "Supervisor state could not be read",
            exc,
            started,
            remediation="Check the radio link and that supervisor.info exists in the log TOC.",
        )

    latest = values[-1] if values else None
    observed_forbidden = sorted(
        name for name, mask in _SUPERVISOR_FORBIDDEN.items() if any(value & mask for value in values)
    )
    missing_can_arm = sum(not bool(value & _SUPERVISOR_CAN_ARM) for value in values)
    enough_samples = len(values) >= minimum_samples
    passed = enough_samples and missing_can_arm == 0 and not observed_forbidden
    details = {
        "samples": len(values),
        "minimum_samples": minimum_samples,
        "values": list(values),
        "latest": latest,
        "latest_flags": _supervisor_flags(latest),
        "samples_without_can_arm": missing_can_arm,
        "observed_forbidden": observed_forbidden,
    }
    if passed:
        text = "Supervisor reports canArm with no unsafe state bits"
        remediation = ""
    elif not enough_samples:
        text = f"Supervisor produced only {len(values)}/{minimum_samples} required samples"
        remediation = "Check the radio/log connection and rerun the check."
    else:
        reasons = []
        if missing_can_arm:
            reasons.append(f"canArm absent in {missing_can_arm} sample(s)")
        if observed_forbidden:
            reasons.append("unsafe bits: " + ", ".join(observed_forbidden))
        text = "Supervisor is not ready: " + "; ".join(reasons)
        remediation = (
            "Keep the drone disarmed and resolve the reported supervisor state; a LOCKED or "
            "crashed state normally requires the project's explicit recovery/power-cycle procedure."
        )
    return PreflightResult(
        check_id="supervisor_ready",
        passed=passed,
        summary=text,
        details=details,
        remediation=remediation,
        elapsed_s=time.perf_counter() - started,
    )


def check_battery(
    cf,
    *,
    minimum_voltage_v: float = 3.7,
    seconds: float = 1.0,
    period_ms: int = 100,
    minimum_samples: int = 3,
) -> PreflightResult:
    """Sample voltage, reported percentage, and power-management state."""
    _validate_capture_args(seconds=seconds, period_ms=period_ms, minimum_samples=minimum_samples)
    _require_finite_number("minimum_voltage_v", minimum_voltage_v, non_negative=True)

    started = time.perf_counter()
    try:
        rows = capture_log_window(
            cf,
            name="cf_lab_battery",
            variables=_BATTERY_LOG_VARIABLES,
            seconds=seconds,
            period_ms=period_ms,
        )
        voltages = tuple(float(row["pm.vbat"]) for row in rows)
        levels = tuple(_exact_int(row["pm.batteryLevel"]) for row in rows)
        states = tuple(_exact_int(row["pm.state"]) for row in rows)
        if not all(math.isfinite(value) for value in voltages):
            raise ValueError("pm.vbat produced a non-finite value")
    except Exception as exc:  # noqa: BLE001 - an incomplete observation fails closed
        return _failed_operation(
            "battery",
            "Battery observation could not complete",
            exc,
            started,
            remediation="Check the radio link and power-management variables in the log TOC.",
        )

    enough_samples = len(voltages) >= minimum_samples
    minimum_observed = min(voltages) if voltages else None
    passed = (
        enough_samples
        and minimum_observed is not None
        and minimum_observed >= minimum_voltage_v
    )
    details = {
        "samples": len(voltages),
        "minimum_samples": minimum_samples,
        "required_minimum_voltage_v": minimum_voltage_v,
        "voltage_min_v": minimum_observed,
        "voltage_mean_v": sum(voltages) / len(voltages) if voltages else None,
        "voltage_max_v": max(voltages) if voltages else None,
        "battery_level_min_percent": min(levels) if levels else None,
        "battery_level_mean_percent": sum(levels) / len(levels) if levels else None,
        "battery_level_latest_percent": levels[-1] if levels else None,
        "power_state_latest": states[-1] if states else None,
        "power_states_observed": sorted(set(states)),
    }
    if passed:
        text = f"Battery minimum {minimum_observed:.3f} V meets {minimum_voltage_v:.3f} V"
        remediation = ""
    elif not enough_samples:
        text = f"Battery produced only {len(voltages)}/{minimum_samples} required samples"
        remediation = "Check radio/log bandwidth and collect a longer battery window."
    else:
        text = f"Battery minimum {minimum_observed:.3f} V is below {minimum_voltage_v:.3f} V"
        remediation = "Charge or replace the battery before flight."
    return PreflightResult(
        check_id="battery",
        passed=passed,
        summary=text,
        details=details,
        remediation=remediation,
        elapsed_s=time.perf_counter() - started,
    )


def check_expected_parameters(
    cf,
    expected: Mapping[str, str | int | float | bool],
    *,
    numeric_tolerance: float = 0.0,
    tolerances: Mapping[str, float] | None = None,
    timeout_s: float = 3.0,
) -> PreflightResult:
    """Read firmware parameters and compare them with caller-supplied values.

    Numeric expected values are compared after converting the firmware's text
    value to ``float``.  ``numeric_tolerance`` is the default absolute
    tolerance; ``tolerances`` can override it for individual parameter names.
    String expected values use an exact comparison after stripping whitespace.
    """
    expected_values, tolerance_values = _validate_expected_parameters(
        expected,
        numeric_tolerance=numeric_tolerance,
        tolerances=tolerances,
        timeout_s=timeout_s,
    )
    started = time.perf_counter()
    actual: dict[str, str] = {}
    mismatches: list[dict[str, object]] = []
    read_errors: dict[str, str] = {}

    for name, wanted in expected_values.items():
        try:
            raw = read_parameter(cf, name, timeout_s)
            observed = str(raw).strip()
            actual[name] = observed
            tolerance = tolerance_values.get(name, numeric_tolerance)
            matches, reason = _parameter_matches(observed, wanted, tolerance)
            if not matches:
                mismatches.append(
                    {
                        "name": name,
                        "expected": wanted,
                        "actual": observed,
                        "tolerance": tolerance if _is_numeric(wanted) else None,
                        "reason": reason,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - collect every unreadable parameter
            read_errors[name] = _error_text(exc)

    passed = not mismatches and not read_errors
    details = {
        "expected": dict(expected_values),
        "actual": actual,
        "numeric_tolerance": numeric_tolerance,
        "tolerances": tolerance_values,
        "mismatches": mismatches,
        "read_errors": read_errors,
    }
    if passed:
        text = f"All {len(expected_values)} expected firmware parameter(s) match"
        remediation = ""
    else:
        text = (
            f"Expected parameter check found {len(mismatches)} mismatch(es) and "
            f"{len(read_errors)} read error(s)"
        )
        remediation = (
            "Select the intended firmware/configuration and explicitly apply any required "
            "parameter changes before rerunning this read-only check."
        )
    error = None
    if read_errors:
        error = f"Failed to read {len(read_errors)} parameter(s): " + ", ".join(
            f"{name} ({message})" for name, message in read_errors.items()
        )
    return PreflightResult(
        check_id="expected_parameters",
        passed=passed,
        summary=text,
        details=details,
        error=error,
        remediation=remediation,
        elapsed_s=time.perf_counter() - started,
    )


def check_required_decks(
    cf,
    required: Sequence[str],
    *,
    timeout_s: float = 3.0,
) -> PreflightResult:
    """Require selected ``deck.<name>`` firmware parameters to equal one."""
    if (
        not isinstance(required, Sequence)
        or isinstance(required, (str, bytes))
        or not required
    ):
        raise ValueError("required must be a non-empty sequence of deck names")
    _require_finite_number("timeout_s", timeout_s, positive=True)

    parameter_names: list[str] = []
    seen: set[str] = set()
    for raw_name in required:
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("every required deck name must be a non-empty string")
        short_name = raw_name.strip()
        parameter = short_name if short_name.startswith("deck.") else f"deck.{short_name}"
        if parameter not in seen:
            parameter_names.append(parameter)
            seen.add(parameter)

    started = time.perf_counter()
    actual: dict[str, str] = {}
    missing: list[str] = []
    read_errors: dict[str, str] = {}
    for parameter in parameter_names:
        try:
            raw = read_parameter(cf, parameter, timeout_s)
            value = str(raw).strip()
            actual[parameter] = value
            if not _numeric_equal(value, 1.0, 0.0):
                missing.append(parameter)
        except Exception as exc:  # noqa: BLE001 - collect every unreadable deck parameter
            read_errors[parameter] = _error_text(exc)

    passed = not missing and not read_errors
    details = {
        "required": parameter_names,
        "actual": actual,
        "missing": missing,
        "read_errors": read_errors,
    }
    if passed:
        text = f"All {len(parameter_names)} required deck(s) are enumerated"
        remediation = ""
    else:
        text = (
            f"Required deck check found {len(missing)} absent deck(s) and "
            f"{len(read_errors)} read error(s)"
        )
        remediation = (
            "Power down the drone, inspect deck seating/hardware, then cold-boot and rerun "
            "the firmware self-test and deck checks."
        )
    error = None
    if read_errors:
        error = f"Failed to read {len(read_errors)} deck parameter(s): " + ", ".join(
            f"{name} ({message})" for name, message in read_errors.items()
        )
    return PreflightResult(
        check_id="required_decks",
        passed=passed,
        summary=text,
        details=details,
        error=error,
        remediation=remediation,
        elapsed_s=time.perf_counter() - started,
    )


def _validate_capture_args(*, seconds: float, period_ms: int, minimum_samples: int) -> None:
    _require_finite_number("seconds", seconds, positive=True)
    if isinstance(period_ms, bool) or not isinstance(period_ms, Integral) or period_ms <= 0:
        raise ValueError("period_ms must be positive")
    if (
        isinstance(minimum_samples, bool)
        or not isinstance(minimum_samples, Integral)
        or minimum_samples < 1
    ):
        raise ValueError("minimum_samples must be at least 1")


def _validate_expected_parameters(
    expected: Mapping[str, str | int | float | bool],
    *,
    numeric_tolerance: float,
    tolerances: Mapping[str, float] | None,
    timeout_s: float,
) -> tuple[dict[str, str | int | float | bool], dict[str, float]]:
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("expected must be a non-empty mapping")
    _require_finite_number("numeric_tolerance", numeric_tolerance, non_negative=True)
    _require_finite_number("timeout_s", timeout_s, positive=True)

    values: dict[str, str | int | float | bool] = {}
    for name, value in expected.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("every expected parameter name must be a non-empty string")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"expected value for {name} must be str, int, float, or bool")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"expected numeric value for {name} must be finite")
        values[name.strip()] = value

    per_parameter: dict[str, float] = {}
    if tolerances is not None:
        if not isinstance(tolerances, Mapping):
            raise ValueError("tolerances must be a mapping")
        for name, tolerance in tolerances.items():
            if name not in values:
                raise ValueError(f"tolerance supplied for unknown expected parameter {name}")
            _require_finite_number(
                f"tolerance for {name}", tolerance, non_negative=True
            )
            numeric = float(tolerance)
            if not _is_numeric(values[name]):
                raise ValueError(f"tolerance for non-numeric parameter {name} is not meaningful")
            per_parameter[name] = numeric
    return values, per_parameter


def _parameter_matches(actual: str, expected: object, tolerance: float) -> tuple[bool, str]:
    if _is_numeric(expected):
        try:
            matches = _numeric_equal(actual, float(expected), tolerance)
        except (TypeError, ValueError, OverflowError):
            return False, "actual value is not numeric"
        return matches, "numeric value is outside tolerance" if not matches else ""
    matches = actual == str(expected).strip()
    return matches, "string value differs" if not matches else ""


def _numeric_equal(actual: object, expected: float, tolerance: float) -> bool:
    observed = float(actual)
    # The firmware keeps float parameters as float32: 0.1 reads back as 0.10000000149011612.
    as_float32 = struct.unpack("<f", struct.pack("<f", expected))[0]
    return math.isfinite(observed) and any(
        math.isclose(observed, value, rel_tol=0.0, abs_tol=tolerance)
        for value in (expected, as_float32)
    )


def _is_numeric(value: object) -> bool:
    return isinstance(value, (int, float, bool))


def _exact_int(value: object) -> int:
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"expected an integer log value, received {value!r}")
    return int(numeric)


def _supervisor_flags(info: int | None) -> list[str]:
    if info is None:
        return []
    return [name for name, mask in SUPERVISOR_INFO_BITS.items() if info & mask]


def _failed_operation(
    check_id: str,
    summary: str,
    exc: Exception,
    started: float,
    *,
    remediation: str,
) -> PreflightResult:
    return PreflightResult(
        check_id=check_id,
        passed=False,
        summary=summary,
        error=_error_text(exc),
        remediation=remediation,
        elapsed_s=time.perf_counter() - started,
    )


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _require_finite_number(
    name: str,
    value: object,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        qualifier = "positive" if positive else "non-negative" if non_negative else "numeric"
        raise ValueError(f"{name} must be finite and {qualifier}")
    numeric = float(value)
    if not math.isfinite(numeric) or (positive and numeric <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    if non_negative and numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


__all__ = [
    "check_battery",
    "check_expected_parameters",
    "check_firmware_self_test",
    "check_required_decks",
    "check_static_imu",
    "check_supervisor_ready",
]
