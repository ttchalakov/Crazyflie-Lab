"""standard_flight_checks gates a real flight on every baseline check plus the requested decks."""
import pytest

from cf_lab.preflight.core import run_preflight
from cf_lab.preflight.presets import standard_flight_checks


@pytest.fixture
def healthy(fakes):
    """Params and log series of a charged, disarmed drone sitting still with its AI deck."""
    params = {"system.selftestPassed": "1", "deck.bcAI": "1"}
    series = {
        **fakes.imu_series(fakes.read_csv("imu_still_bench.csv")),
        **fakes.battery_series([4.05] * 10),
    }
    return params, series


def _failed_ids(cf, **kwargs):
    report = run_preflight(standard_flight_checks(cf, **kwargs))
    return [result.check_id for result in report.failed], len(report.results)


def test_healthy_drone_passes_the_baseline(make_cf, healthy):
    params, series = healthy

    failed, ran = _failed_ids(make_cf(params, series), required_decks=("bcAI",))

    assert failed == []
    assert ran == 5


@pytest.mark.parametrize(
    ("change", "failing"),
    [
        ({"params": {"system.selftestPassed": "0"}}, ["firmware_self_test"]),
        ({"series": {"supervisor.info": [0x0403] * 75}}, ["supervisor_ready", "static_imu"]),
        ({"series": {"pm.vbat": [3.55] * 10}}, ["battery"]),
        ({"series": {"acc.x": [0.0] * 75, "acc.y": [0.0] * 75, "acc.z": [0.0] * 75}}, ["static_imu"]),
        ({"params": {"deck.bcAI": "0"}}, ["required_decks"]),
    ],
    ids=["self_test", "armed", "low_battery", "dead_imu", "missing_deck"],
)
def test_each_baseline_fault_fails_the_preset(make_cf, healthy, change, failing):
    params, series = healthy
    cf = make_cf({**params, **change.get("params", {})}, {**series, **change.get("series", {})})

    failed, _ = _failed_ids(cf, required_decks=("bcAI",))

    assert failed == failing


def test_deck_check_runs_only_when_decks_are_required(make_cf, healthy):
    params, series = healthy
    cf = make_cf({**params, "deck.bcAI": "0"}, series)

    failed, ran = _failed_ids(cf)

    assert failed == []
    assert ran == 4
