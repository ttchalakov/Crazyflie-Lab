"""Composable, fail-closed preflight check results and execution.

Checks are deliberately plain zero-argument callables.  Hardware-specific
modules can therefore expose independently useful functions while an
experiment chooses which checks to compose and how to treat their results.
This module uses only the standard library and does not import cflib.
"""

from __future__ import annotations

import dataclasses
import functools
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any


@dataclasses.dataclass(frozen=True)
class PreflightResult:
    """The outcome and diagnostic context from one preflight check."""

    check_id: str
    passed: bool
    summary: str
    details: Mapping[str, object] = dataclasses.field(default_factory=dict)
    error: str | None = None
    remediation: str = ""
    elapsed_s: float = 0.0

    def __bool__(self) -> bool:
        return self.passed

    def as_dict(self) -> dict[str, object]:
        """Return a manifest/telemetry-friendly representation."""
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "summary": self.summary,
            "details": dict(self.details),
            "error": self.error,
            "remediation": self.remediation,
            "elapsed_s": self.elapsed_s,
        }


@dataclasses.dataclass(frozen=True)
class PreflightReport:
    """Ordered results from a selected collection of preflight checks."""

    results: tuple[PreflightResult, ...] = ()

    def __post_init__(self) -> None:
        # Accepting any finite sequence at the boundary is convenient, while a
        # tuple keeps the frozen report itself stable after construction.
        object.__setattr__(self, "results", tuple(self.results))
        if not all(isinstance(result, PreflightResult) for result in self.results):
            raise TypeError("PreflightReport results must all be PreflightResult instances")

    @property
    def passed(self) -> bool:
        # A run that checked nothing has not cleared anything.
        return bool(self.results) and all(result.passed for result in self.results)

    @property
    def failed(self) -> tuple[PreflightResult, ...]:
        return tuple(result for result in self.results if not result.passed)

    def __bool__(self) -> bool:
        return self.passed

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks_run": len(self.results),
            "checks_failed": len(self.failed),
            "results": [result.as_dict() for result in self.results],
        }

    def format(self) -> str:
        """Render a compact report suitable for a terminal or run log."""
        passed_count = len(self.results) - len(self.failed)
        state = "PASSED" if self.passed else "FAILED"
        lines = [f"Preflight {state}: {passed_count}/{len(self.results)} checks passed"]
        for result in self.results:
            marker = "PASS" if result.passed else "FAIL"
            line = f"[{marker}] {result.check_id}: {result.summary} ({result.elapsed_s:.3f}s)"
            if result.error:
                line += f"; error: {result.error}"
            lines.append(line)
            if not result.passed and result.remediation:
                lines.append(f"       remediation: {result.remediation}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


Check = Callable[[], PreflightResult]


def run_preflight(
    checks: Iterable[Check],
    *,
    telemetry: Any | None = None,
    stop_on_failure: bool = False,
) -> PreflightReport:
    """Run selected checks in order and return a fail-closed report.

    An exception or an invalid return value becomes a failed result instead of
    allowing the caller to mistake an incomplete check for a pass.  When a
    telemetry recorder is supplied, every executed check emits the same
    ``preflight_check`` event kind; telemetry remains entirely optional and a
    recording failure never changes the check outcome.
    """
    results: list[PreflightResult] = []
    for check in checks:
        check_id = _callable_id(check)
        started = time.perf_counter()
        try:
            if not callable(check):
                raise TypeError(f"preflight check is not callable: {check!r}")
            result = check()
            if not isinstance(result, PreflightResult):
                raise TypeError(
                    f"{check_id} returned {type(result).__name__}; expected PreflightResult"
                )
        except Exception as exc:
            result = PreflightResult(
                check_id=check_id,
                passed=False,
                summary="Check could not complete",
                error=f"{type(exc).__name__}: {exc}",
                remediation="Review the reported error, correct it, and rerun this check.",
            )

        result = dataclasses.replace(result, elapsed_s=time.perf_counter() - started)
        results.append(result)
        _emit_telemetry(telemetry, result)
        if stop_on_failure and not result.passed:
            break

    return PreflightReport(tuple(results))


def _callable_id(check: object) -> str:
    explicit = getattr(check, "check_id", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    if isinstance(check, functools.partial):
        return _callable_id(check.func)
    name = getattr(check, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return type(check).__name__


def _emit_telemetry(telemetry: Any | None, result: PreflightResult) -> None:
    if telemetry is None:
        return
    event = getattr(telemetry, "event", None)
    if not callable(event):
        return
    try:
        event(
            "preflight_check",
            result.summary,
            check_id=result.check_id,
            passed=result.passed,
            details=dict(result.details),
            error=result.error,
            remediation=result.remediation,
            elapsed_s=result.elapsed_s,
        )
    except Exception:
        # Preflight telemetry is supporting evidence, not another condition for
        # flight readiness.  Recorder failures must not rewrite a real result.
        pass


__all__ = ["Check", "PreflightReport", "PreflightResult", "run_preflight"]
