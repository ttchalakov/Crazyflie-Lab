"""Reusable integrity checks for recorded telemetry runs.

The recorder deliberately writes independent asynchronous streams.  This
module checks the recording container and cross-stream invariants without
knowing anything about a particular experiment's controller or dynamics.
Example-specific checks can be appended to the returned ``IntegrityReport``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow.compute as pc

from cf_lab.mocap.relative import OriginZeroer
from cf_lab.mocap.types import PoseSample, Quaternion
from cf_lab.telemetry.sync import RunData, load_run

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

# How far behind the rest of the run a periodic stream may fall before it counts
# as dead rather than merely finishing early.
STREAM_END_TOLERANCE_S = 2.0


@dataclass(frozen=True)
class IntegrityCheck:
    """One named result in an integrity report."""

    status: str
    label: str
    detail: str = ""


@dataclass(frozen=True)
class StreamSummary:
    """Recorded row/drop counts for one declared stream."""

    name: str
    rows: int
    dropped: int


@dataclass
class IntegrityReport:
    """Structured integrity results with a small terminal renderer."""

    run_label: str
    manifest_status: str
    streams: tuple[StreamSummary, ...]
    checks: list[IntegrityCheck] = field(default_factory=list)

    def add(self, ok: bool, label: str, detail: str = "") -> None:
        self.checks.append(IntegrityCheck(PASS if ok else FAIL, label, detail))

    def skip(self, label: str, reason: str) -> None:
        self.checks.append(IntegrityCheck(SKIP, label, reason))

    @property
    def passed(self) -> bool:
        return not any(check.status == FAIL for check in self.checks)

    @property
    def failures(self) -> tuple[IntegrityCheck, ...]:
        return tuple(check for check in self.checks if check.status == FAIL)

    @property
    def skipped(self) -> tuple[IntegrityCheck, ...]:
        return tuple(check for check in self.checks if check.status == SKIP)

    def format(self) -> str:
        width = max((len(stream.name) for stream in self.streams), default=10)
        lines = [
            f"=== Integrity report: {self.run_label} ===",
            f"manifest status={self.manifest_status}",
        ]
        lines.extend(
            f"  {stream.name:{width}s} rows={stream.rows:6d} dropped={stream.dropped}"
            for stream in self.streams
        )
        for check in self.checks:
            detail = f" — {check.detail}" if check.detail else ""
            lines.append(f"  [{check.status}] {check.label}{detail}")

        if self.failures:
            summary = "FAILURES: " + ", ".join(check.label for check in self.failures)
        elif self.skipped:
            summary = f"ALL APPLICABLE CHECKS PASSED ({len(self.skipped)} skipped)"
        else:
            summary = "ALL CHECKS PASSED"
        lines.extend(("", summary))
        return "\n".join(lines)


def validate_run(
    run_or_path: RunData | str | Path,
    *,
    required_streams: Sequence[str] = (),
    skipped_streams: Mapping[str, str] | None = None,
    expected_rows: Mapping[str, float] | None = None,
    minimum_delivery: float = 0.9,
    maximum_rows: Mapping[str, float] | None = None,
    required_events: Sequence[str] = (),
    source_sequence_streams: Sequence[str] = (),
    recreate_extpose: bool = False,
    extpose_tolerance: float = 1e-9,
    clock_rms_limits_ns: Mapping[str, float] | None = None,
    stream_end_tolerance_s: float = STREAM_END_TOLERANCE_S,
) -> tuple[RunData, IntegrityReport]:
    """Load and validate a telemetry run using experiment-supplied expectations.

    The generic checks cover manifest closure, stream presence/delivery, source
    and writer sequence ordering, queue drops, clock-fit residuals, and optional
    replay of the extpose signal from raw mocap plus the recorded origin event.
    Device-specific numerical checks remain the caller's responsibility and can
    be appended with ``report.add(...)``.
    """
    if not 0.0 <= minimum_delivery <= 1.0:
        raise ValueError("minimum_delivery must be between 0 and 1.")
    if extpose_tolerance < 0.0:
        raise ValueError("extpose_tolerance must be non-negative.")
    if stream_end_tolerance_s < 0.0:
        raise ValueError("stream_end_tolerance_s must be non-negative.")

    run, run_label = _coerce_run(run_or_path)
    summaries = tuple(
        StreamSummary(
            name=name,
            rows=run.streams[name].num_rows if name in run.streams else 0,
            dropped=int(entry.get("dropped", 0)),
        )
        for name, entry in sorted(run.manifest.get("streams", {}).items())
    )
    report = IntegrityReport(
        run_label=run_label,
        manifest_status=str(run.manifest.get("status", "missing")),
        streams=summaries,
    )
    skipped = dict(skipped_streams or {})

    report.add(report.manifest_status == "clean", "manifest closed clean")
    for name in required_streams:
        if name in skipped:
            continue
        rows = _stream_rows(run, name)
        report.add(rows > 0, f"{name} stream contains rows", f"{rows} row(s)")
    for name, reason in skipped.items():
        report.skip(f"{name} stream", reason)

    for name, expected in (expected_rows or {}).items():
        if expected <= 0.0:
            raise ValueError(f"expected_rows[{name!r}] must be positive.")
        rows = _stream_rows(run, name)
        delivery = rows / expected
        report.add(
            delivery >= minimum_delivery,
            f"{name} delivery >= {minimum_delivery:.0%}",
            f"{rows}/{expected:.0f} rows ({delivery:.1%})",
        )

    for name, maximum in (maximum_rows or {}).items():
        if maximum < 0.0:
            raise ValueError(f"maximum_rows[{name!r}] must be non-negative.")
        rows = _stream_rows(run, name)
        report.add(
            rows <= maximum,
            f"{name} respects row cap",
            f"{rows} rows vs cap {maximum:.0f}",
        )

    _check_required_events(run, report, required_events)
    for name in source_sequence_streams:
        _check_source_sequence(run, report, name)

    if recreate_extpose:
        try:
            error, rows = extpose_recreation_error(run)
        except (KeyError, ValueError) as exc:
            report.add(False, "extpose recreated from mocap + origin event", str(exc))
        else:
            report.add(
                error <= extpose_tolerance,
                "extpose recreated from mocap + origin event",
                f"max error {error:.2e} across {rows} row(s)",
            )

    for clock, limit_ns in (clock_rms_limits_ns or {}).items():
        fit = run.fits.get(clock)
        if fit is None:
            report.add(False, f"{clock} clock fit available")
        else:
            report.add(
                fit.rms_ns <= limit_ns,
                f"{clock} clock fit rms <= {limit_ns / 1e6:g} ms",
                f"rms {fit.rms_ns / 1e6:.2f} ms over {fit.n} sample(s)",
            )

    unordered = []
    for name, table in run.streams.items():
        if table.num_rows < 2 or "seq" not in table.column_names:
            continue
        values = table.column("seq").to_pylist()
        if any(current <= previous for previous, current in zip(values, values[1:])):
            unordered.append(name)
    report.add(
        not unordered,
        "recorded stream sequences strictly increasing",
        f"unordered: {unordered}" if unordered else "all non-empty streams",
    )

    dropped = {
        summary.name: summary.dropped for summary in summaries if summary.dropped
    }
    report.add(
        not dropped,
        "no writer-queue drops",
        str(dropped) if dropped else "all streams",
    )

    _check_periodic_streams_reach_end(run, report, stream_end_tolerance_s, skipped)
    return run, report


def extpose_recreation_error(run: RunData) -> tuple[float, int]:
    """Recreate extpose from raw mocap and return ``(max_error, rows)``."""
    events = run.streams.get("events")
    if events is None or events.num_rows == 0:
        raise ValueError("events stream is absent or empty")

    origin = None
    for kind, data_json in zip(
        events.column("kind").to_pylist(), events.column("data_json").to_pylist()
    ):
        if kind == "origin_zeroed":
            origin = json.loads(data_json)
            break
    if origin is None:
        raise ValueError("origin_zeroed event is missing")

    mocap = run.streams.get("mocap")
    extpose = run.streams.get("extpose")
    if mocap is None or mocap.num_rows == 0:
        raise ValueError("mocap stream is absent or empty")
    if extpose is None or extpose.num_rows == 0:
        raise ValueError("extpose stream is absent or empty")

    zeroer = OriginZeroer()
    zeroer.set_origin((origin["x"], origin["y"], origin["z"]), origin["yaw_rad"])
    raw = {
        seq: (timestamp, (x, y, z), (qx, qy, qz, qw))
        for seq, timestamp, x, y, z, qx, qy, qz, qw in zip(
            *(mocap.column(column).to_pylist() for column in (
                "source_seq", "source_ts_s", "x", "y", "z", "qx", "qy", "qz", "qw"
            ))
        )
    }

    worst = 0.0
    checked = 0
    for seq, x, y, z, qx, qy, qz, qw in zip(
        *(extpose.column(column).to_pylist() for column in (
            "source_seq", "x", "y", "z", "qx", "qy", "qz", "qw"
        ))
    ):
        if seq not in raw:
            continue
        timestamp, position, quat = raw[seq]
        recreated = zeroer.zero(
            PoseSample(
                timestamp_s=timestamp,
                position_m=position,
                orientation=Quaternion(qx=quat[0], qy=quat[1], qz=quat[2], qw=quat[3]),
                rigid_body="integrity-check",
                source_seq=seq,
            )
        )
        recreated_q = recreated.orientation
        errors = (
            abs(recreated.position_m[0] - x),
            abs(recreated.position_m[1] - y),
            abs(recreated.position_m[2] - z),
            abs(recreated_q.qx - qx),
            abs(recreated_q.qy - qy),
            abs(recreated_q.qz - qz),
            abs(recreated_q.qw - qw),
        )
        worst = max(worst, *errors)
        checked += 1
    if checked == 0:
        raise ValueError("mocap and extpose have no shared source_seq rows")
    return worst, checked


def _coerce_run(run_or_path: RunData | str | Path) -> tuple[RunData, str]:
    if isinstance(run_or_path, RunData):
        return run_or_path, "loaded run"
    path = Path(run_or_path)
    return load_run(path), str(path)


def _stream_rows(run: RunData, name: str) -> int:
    table = run.streams.get(name)
    return table.num_rows if table is not None else 0


def _check_required_events(
    run: RunData, report: IntegrityReport, required_events: Sequence[str]
) -> None:
    if not required_events:
        return
    events = run.streams.get("events")
    kinds = set(events.column("kind").to_pylist()) if events is not None else set()
    for kind in required_events:
        report.add(kind in kinds, f"{kind} event recorded")


def _check_periodic_streams_reach_end(
    run: RunData,
    report: IntegrityReport,
    tolerance_s: float,
    skipped: Mapping[str, str],
) -> None:
    """Fail any periodic stream that stops producing before the run does.

    A log block that dies mid-flight is invisible to every other check here: the
    manifest still closes ``clean`` and ``dropped`` stays 0, because nothing was
    dropped — nothing arrived to drop. Row counts cannot see it either, since the
    recorder's lifetime includes a pre-flight of variable length, so a healthy
    stream covers a varying share of it. What does give it away is the stream
    falling silent while the rest of the run keeps recording, so compare each
    stream's last sample against the newest sample anywhere in the run.
    Only streams declaring a ``period_ms`` are checked: they alone promise to produce
    until the recorder stops. Streams the caller already declared skipped are left
    alone, so a fake backend with no STM32 source does not read as a dead block.
    """
    ends: dict[str, int] = {}
    for name, table in run.streams.items():
        if table.num_rows == 0 or "host_ns" not in table.column_names:
            continue
        ends[name] = pc.max(table.column("host_ns")).as_py()
    if not ends:
        return
    run_end = max(ends.values())

    for name, entry in sorted(run.manifest.get("streams", {}).items()):
        if name in skipped or "period_ms" not in dict(entry.get("meta", {})):
            continue
        label = f"{name} stream records until the run ends"
        end = ends.get(name)
        if end is None:
            report.add(False, label, "stream is empty")
            continue
        silent_s = (run_end - end) / 1e9
        report.add(
            silent_s <= tolerance_s,
            label,
            f"stopped {silent_s:.1f}s before the run's last sample",
        )


def _check_source_sequence(run: RunData, report: IntegrityReport, name: str) -> None:
    table = run.streams.get(name)
    if table is None or table.num_rows == 0 or "source_seq" not in table.column_names:
        report.add(False, f"{name} source_seq strictly increasing", "stream/column missing")
        return
    sequences = table.column("source_seq").to_pylist()
    increasing = all(current > previous for previous, current in zip(sequences, sequences[1:]))
    gaps = sum(
        current - previous - 1
        for previous, current in zip(sequences, sequences[1:])
        if current > previous + 1
    )
    report.add(
        increasing,
        f"{name} source_seq strictly increasing",
        f"{gaps} frame(s) in gaps",
    )
