"""Pass/fail boundaries of the Crazyflie preflight checks against a faked cflib link."""
import math

import pytest

from cf_lab.preflight.crazyflie import (
    check_battery,
    check_expected_parameters,
    check_firmware_self_test,
    check_required_decks,
    check_static_imu,
    check_supervisor_ready,
)


# --- firmware self-test -----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "passed"),
    [("1", True), (" 1\n", True), ("0", False), ("", False), (KeyError("not in TOC"), False)],
)
def test_self_test_passes_only_on_a_confirmed_one(make_cf, raw, passed):
    cf = make_cf({"system.selftestPassed": raw})

    result = check_firmware_self_test(cf)

    assert result.passed is passed
    if isinstance(raw, Exception):
        assert result.error.startswith("KeyError")


# --- static IMU -------------------------------------------------------------


def test_static_imu_passes_on_a_real_still_window_read_upside_down(make_cf, fakes):
    # Recorded on the bench: acc.z reads -1 g, so the magnitude, not the axis, must be checked.
    rows = fakes.read_csv("imu_still_bench.csv")
    cf = make_cf(series=fakes.imu_series(rows))

    result = check_static_imu(cf)

    assert result.passed, result.summary
    assert result.details["samples"] == len(rows)
    assert math.isclose(result.details["acceleration_magnitude_g"], 1.0, abs_tol=0.01)


@pytest.mark.parametrize(
    "acc",
    [(0.0, 0.0, 1.0), (0.5, 0.0, 0.866), (0.0, -0.707, 0.707), (0.0, 0.0, -1.0)],
    ids=["level", "tilted_30deg_roll", "tilted_45deg_pitch", "inverted"],
)
def test_static_imu_passes_when_still_at_any_attitude(make_cf, fakes, acc):
    cf = make_cf(series=fakes.imu_series(fakes.still_imu_rows(acc=acc)))

    assert check_static_imu(cf).passed


def test_static_imu_fails_on_a_real_in_flight_window(make_cf, fakes):
    cf = make_cf(series=fakes.imu_series(fakes.read_csv("imu_in_flight.csv")))

    result = check_static_imu(cf)

    assert not result.passed
    assert result.summary == "IMU window was not sufficiently still or close to one-g gravity"


@pytest.mark.parametrize(
    "gyro",
    [(1.5, 0.0, 0.0), (0.0, 0.0, -1.2)],
    ids=["gyro_bias_x", "gyro_bias_z"],
)
def test_static_imu_fails_on_a_gyro_bias_above_one_dps(make_cf, fakes, gyro):
    rows = [dict(row, **dict(zip(("gyro.x", "gyro.y", "gyro.z"), gyro))) for row in fakes.still_imu_rows()]
    cf = make_cf(series=fakes.imu_series(rows))

    result = check_static_imu(cf)

    assert not result.passed
    assert result.details["stationary"] is False


def test_static_imu_fails_when_the_drone_is_bumped_during_the_window(make_cf, fakes):
    rows = fakes.still_imu_rows()
    # One sharp knock: a single 1.2 g spike in an otherwise still 75-sample window.
    rows[40] = dict(rows[40], **{"acc.x": 1.2})
    cf = make_cf(series=fakes.imu_series(rows))

    result = check_static_imu(cf)

    assert not result.passed
    assert result.details["acceleration_motion_g"] > 0.05


@pytest.mark.parametrize("axis", ["acc.z", "gyro.x"])
def test_static_imu_fails_on_a_single_nan_reading(make_cf, fakes, axis):
    # NaN compares False against every limit; a check written as "not over the limit" would pass it.
    rows = fakes.still_imu_rows()
    rows[10] = dict(rows[10], **{axis: math.nan})
    cf = make_cf(series=fakes.imu_series(rows))

    assert not check_static_imu(cf).passed


def test_static_imu_fails_when_armed_in_any_sample(make_cf, fakes):
    rows = fakes.still_imu_rows()
    supervisor = [fakes.SUPERVISOR_READY] * len(rows)
    supervisor[-1] = 0x0003  # canArm | ARMED
    cf = make_cf(series=fakes.imu_series(rows, supervisor))

    result = check_static_imu(cf)

    assert not result.passed
    assert result.summary == "Drone was armed in 1 IMU sample(s)"


@pytest.mark.parametrize(
    "error_g", [0.15, 1.0, 5.0], ids=["default", "loosened_to_one_g", "loosened_far"]
)
def test_static_imu_dead_accelerometer_fails_as_not_responding(make_cf, fakes, error_g):
    # An IMU that failed to enumerate logs exactly 0.0 on every axis; it looks perfectly still.
    cf = make_cf(series=fakes.imu_series(fakes.still_imu_rows(acc=(0.0, 0.0, 0.0))))

    result = check_static_imu(cf, max_acceleration_magnitude_error_g=error_g)

    assert not result.passed
    assert result.summary == "Accelerometer reads about 0 g: the IMU is not responding"


def test_static_imu_loosened_tolerances_admit_a_vibrating_window(make_cf, fakes):
    rows = fakes.still_imu_rows(count=74)
    for index, row in enumerate(rows):
        sign = 1.0 if index % 2 else -1.0
        row.update({"acc.x": 0.1 * sign, "gyro.y": 3.0 * sign})
    cf = make_cf(series=fakes.imu_series(rows))

    assert not check_static_imu(cf).passed
    assert check_static_imu(
        cf, max_gyro_motion_dps=5.0, max_acceleration_motion_g=0.2
    ).passed


@pytest.mark.parametrize(("count", "passed"), [(9, False), (10, True)])
def test_static_imu_requires_the_minimum_sample_count(make_cf, fakes, count, passed):
    cf = make_cf(series=fakes.imu_series(fakes.still_imu_rows(count=count)))

    result = check_static_imu(cf)

    assert result.passed is passed
    if not passed:
        assert result.summary == "Static IMU window had only 9/10 required samples"


def test_static_imu_with_no_samples_fails(make_cf, fakes):
    cf = make_cf(series=fakes.imu_series([]))

    assert not check_static_imu(cf).passed


@pytest.mark.parametrize(
    "threshold",
    [
        "max_gyro_rate_dps",
        "max_gyro_motion_dps",
        "max_acceleration_motion_g",
        "max_acceleration_magnitude_error_g",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1])
def test_static_imu_rejects_a_threshold_that_would_disable_its_comparison(
    make_cf, fakes, threshold, value
):
    cf = make_cf(series=fakes.imu_series(fakes.read_csv("imu_in_flight.csv")))

    with pytest.raises(ValueError):
        check_static_imu(cf, **{threshold: value})


# --- supervisor -------------------------------------------------------------


@pytest.mark.parametrize("info", [0x0001, 0x0401, 0x0409])
def test_supervisor_ready_passes_with_can_arm_and_no_unsafe_bits(make_cf, info):
    cf = make_cf(series={"supervisor.info": [info] * 5})

    assert check_supervisor_ready(cf).passed


@pytest.mark.parametrize(
    ("info", "unsafe"),
    [
        (0x0003, "ARMED"),
        (0x0021, "TUMBLED"),
        (0x0041, "LOCKED"),
        (0x0081, "crashed"),
    ],
)
def test_supervisor_ready_fails_on_unsafe_bits_even_with_can_arm(make_cf, info, unsafe):
    cf = make_cf(series={"supervisor.info": [0x0401, 0x0401, info, 0x0401]})

    result = check_supervisor_ready(cf)

    assert not result.passed
    assert result.summary == f"Supervisor is not ready: unsafe bits: {unsafe}"


def test_supervisor_ready_fails_when_can_arm_drops_out_once(make_cf):
    # A self-test failure or a deck fault blocks arming: canArm is simply absent.
    cf = make_cf(series={"supervisor.info": [0x0401, 0x0400, 0x0401]})

    result = check_supervisor_ready(cf)

    assert not result.passed
    assert result.summary == "Supervisor is not ready: canArm absent in 1 sample(s)"


def test_supervisor_ready_fails_without_samples(make_cf):
    cf = make_cf(series={"supervisor.info": []})

    result = check_supervisor_ready(cf)

    assert not result.passed
    assert result.summary == "Supervisor produced only 0/1 required samples"


@pytest.mark.parametrize("value", [1.5, math.nan])
def test_supervisor_ready_fails_closed_on_a_non_integer_log_value(make_cf, value):
    cf = make_cf(series={"supervisor.info": [0x0401, value]})

    result = check_supervisor_ready(cf)

    assert not result.passed
    assert result.error.startswith("ValueError")


def test_supervisor_ready_fails_when_the_log_variable_is_missing(make_cf):
    result = check_supervisor_ready(make_cf(series={}))

    assert not result.passed
    assert result.error.startswith("KeyError")


# --- battery ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("voltages", "passed"),
    [
        ([3.95, 3.94, 3.95], True),
        ([3.7, 3.7, 3.7], True),
        ([3.699, 3.75, 3.75], False),
        ([4.1, 3.6, 4.1], False),
    ],
    ids=["charged", "exactly_minimum", "just_below", "one_sag_below"],
)
def test_battery_fails_on_the_lowest_sample_not_the_mean(make_cf, fakes, voltages, passed):
    cf = make_cf(series=fakes.battery_series(voltages))

    result = check_battery(cf)

    assert result.passed is passed
    assert result.details["voltage_min_v"] == min(voltages)


def test_battery_failure_names_the_voltage(make_cf, fakes):
    cf = make_cf(series=fakes.battery_series([3.5, 3.5, 3.5]))

    assert check_battery(cf).summary == "Battery minimum 3.500 V is below 3.700 V"


def test_battery_fails_with_too_few_samples(make_cf, fakes):
    cf = make_cf(series=fakes.battery_series([4.0, 4.0]))

    result = check_battery(cf)

    assert not result.passed
    assert result.summary == "Battery produced only 2/3 required samples"


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_battery_fails_closed_on_a_non_finite_voltage(make_cf, fakes, bad):
    cf = make_cf(series=fakes.battery_series([4.0, bad, 4.0]))

    result = check_battery(cf)

    assert not result.passed
    assert "non-finite" in result.error


# --- log block cleanup ------------------------------------------------------


@pytest.mark.parametrize(
    "run",
    [
        lambda cf: check_static_imu(cf),
        lambda cf: check_supervisor_ready(cf),
        lambda cf: check_battery(cf),
    ],
    ids=["static_imu", "supervisor_ready", "battery"],
)
@pytest.mark.parametrize("start_error", [None, RuntimeError("log block start failed")])
def test_every_log_block_is_stopped_and_deleted(make_cf, fakes, run, start_error):
    # A leaked block eats the firmware's log-block budget that telemetry needs later.
    series = {
        **fakes.imu_series(fakes.still_imu_rows()),
        **fakes.battery_series([4.0] * 75),
    }
    cf = make_cf(series=series, start_error=start_error)

    result = run(cf)

    assert result.passed is (start_error is None)
    assert cf.log.configs
    assert all(config.stopped and config.deleted for config in cf.log.configs)


# --- expected parameters ----------------------------------------------------


@pytest.mark.parametrize(
    ("expected", "actual", "tolerance", "passed"),
    [
        (1, "1", 0.0, True),
        (1, "1.0", 0.0, True),
        (True, "1", 0.0, True),
        (False, "1", 0.0, False),
        (0.5, "0.5", 0.0, True),
        (0.5, "0.51", 0.0, False),
        (0.5, "0.505", 0.01, True),
        (0.5, "0.52", 0.01, False),
        (2, "abc", 0.0, False),
        (2, "nan", 1e9, False),
        ("PID", "PID", 0.0, True),
        ("PID", "pid", 0.0, False),
        ("1", "1.0", 0.0, False),
    ],
)
def test_expected_parameter_comparison(make_cf, expected, actual, tolerance, passed):
    cf = make_cf({"stabilizer.controller": actual})

    result = check_expected_parameters(
        cf, {"stabilizer.controller": expected}, numeric_tolerance=tolerance
    )

    assert result.passed is passed


def test_float_parameter_matches_the_value_it_was_set_to(make_cf):
    # cflib reports a float param as str() of the unpacked float32: 0.1 reads back like this.
    cf = make_cf({"posCtlPid.xKi": "0.10000000149011612"})

    assert check_expected_parameters(cf, {"posCtlPid.xKi": 0.1}).passed


def test_expected_parameters_per_name_tolerance_overrides_the_default(make_cf):
    cf = make_cf({"a.x": "1.05", "a.y": "1.05"})

    result = check_expected_parameters(
        cf, {"a.x": 1.0, "a.y": 1.0}, numeric_tolerance=0.1, tolerances={"a.y": 0.01}
    )

    assert not result.passed
    assert [mismatch["name"] for mismatch in result.details["mismatches"]] == ["a.y"]


def test_expected_parameters_reports_every_unreadable_name(make_cf):
    cf = make_cf({"a.ok": "1", "a.timeout": TimeoutError("no answer")})

    result = check_expected_parameters(cf, {"a.ok": 1, "a.timeout": 1, "a.missing": 1})

    assert not result.passed
    assert set(result.details["read_errors"]) == {"a.timeout", "a.missing"}
    assert result.error.startswith("Failed to read 2 parameter(s)")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected": {}},
        {"expected": {"a.x": math.nan}},
        {"expected": {"a.x": 1}, "numeric_tolerance": math.nan},
        {"expected": {"a.x": 1}, "tolerances": {"a.other": 0.1}},
        {"expected": {"a.x": "PID"}, "tolerances": {"a.x": 0.1}},
    ],
    ids=["empty", "nan_expected", "nan_tolerance", "tolerance_for_unknown", "tolerance_on_text"],
)
def test_expected_parameters_rejects_meaningless_arguments(make_cf, kwargs):
    with pytest.raises(ValueError):
        check_expected_parameters(make_cf({"a.x": "1"}), **kwargs)


# --- required decks ---------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "passed"),
    [("1", True), (1, True), ("0", False), ("", False), (KeyError("deck.bcAI"), False)],
)
def test_required_deck_must_read_exactly_one(make_cf, value, passed):
    cf = make_cf({"deck.bcAI": value})

    assert check_required_decks(cf, ["bcAI"]).passed is passed


def test_required_decks_accepts_prefixed_names_and_checks_each_once(make_cf):
    cf = make_cf({"deck.bcAI": "1", "deck.bcLighthouse4": "0"})

    result = check_required_decks(cf, ["bcAI", "deck.bcAI", " bcLighthouse4 "])

    assert not result.passed
    assert result.details["required"] == ["deck.bcAI", "deck.bcLighthouse4"]
    assert result.details["missing"] == ["deck.bcLighthouse4"]


@pytest.mark.parametrize("required", ["bcAI", (), ["bcAI", ""]], ids=["bare_string", "empty", "blank"])
def test_required_decks_rejects_a_bare_string_or_empty_names(make_cf, required):
    # A bare string would otherwise be iterated as decks "b", "c", "A", "I".
    with pytest.raises(ValueError):
        check_required_decks(make_cf({"deck.bcAI": "1"}), required)
