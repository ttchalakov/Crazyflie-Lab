"""run_preflight fails closed and PreflightReport aggregates check outcomes."""
import functools

import pytest

from cf_lab.preflight.core import PreflightReport, PreflightResult, run_preflight


def _passing(check_id="ok"):
    return lambda: PreflightResult(check_id=check_id, passed=True, summary="fine")


def _failing(check_id="bad"):
    return lambda: PreflightResult(
        check_id=check_id, passed=False, summary="broken", remediation="fix it"
    )


class _Recorder:
    def __init__(self, error=None):
        self.events = []
        self.error = error

    def event(self, kind, message="", **data):
        self.events.append((kind, data["check_id"], data["passed"]))
        if self.error is not None:
            raise self.error


def test_one_failed_check_fails_the_whole_report():
    report = run_preflight([_passing("a"), _failing("b"), _passing("c")])

    assert not report.passed
    assert not report
    assert [result.check_id for result in report.failed] == ["b"]
    assert len(report.results) == 3


def test_all_passing_checks_pass_the_report():
    report = run_preflight([_passing("a"), _passing("b")])

    assert report.passed
    assert report.failed == ()


def test_run_with_no_checks_does_not_clear_the_vehicle():
    # A runner whose check list was filtered down to nothing must not read as a pass.
    assert not run_preflight(()).passed


def test_check_that_raises_becomes_a_failed_result_named_after_the_function():
    def check_battery(cf):
        raise TimeoutError("link lost")

    report = run_preflight([functools.partial(check_battery, object()), _passing()])

    failed = report.failed[0]
    assert failed.check_id == "check_battery"
    assert failed.error == "TimeoutError: link lost"
    assert failed.remediation
    # Later checks still run: one broken check must not hide the rest of the report.
    assert report.results[1].passed


def test_check_returning_a_truthy_non_result_fails():
    report = run_preflight([lambda: True, lambda: None, "not callable"])

    assert not report.passed
    assert len(report.failed) == 3
    assert all(result.error.startswith("TypeError") for result in report.failed)


def test_telemetry_receives_every_result_and_a_broken_recorder_cannot_change_them():
    recorder = _Recorder(error=OSError("disk full"))

    report = run_preflight([_passing("a"), _failing("b")], telemetry=recorder)

    assert recorder.events == [
        ("preflight_check", "a", True),
        ("preflight_check", "b", False),
    ]
    assert [result.passed for result in report.results] == [True, False]


def test_format_shows_failures_with_their_remediation():
    report = PreflightReport(
        (
            PreflightResult("supervisor_ready", True, "canArm"),
            PreflightResult("battery", False, "3.500 V is below 3.700 V", remediation="Charge"),
        )
    )

    lines = report.format().splitlines()

    assert lines[0] == "Preflight FAILED: 1/2 checks passed"
    assert lines[1].startswith("[PASS] supervisor_ready: canArm")
    assert lines[2].startswith("[FAIL] battery: 3.500 V is below 3.700 V")
    assert lines[3].strip() == "remediation: Charge"
    assert len(lines) == 4


def test_report_dict_counts_failures():
    report = run_preflight([_passing(), _failing(), _failing()])

    summary = report.as_dict()

    assert summary["passed"] is False
    assert (summary["checks_run"], summary["checks_failed"]) == (3, 2)
