#!/usr/bin/env python3
"""Parse KBENCH records into machine-readable and Markdown profiling reports."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "artifacts" / "report"


def _value(text: str) -> Any:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(text, 0)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def parse_records(lines: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in lines:
        # The Crazyflie firmware prefixes every CPX console *packet*, not every
        # logical line. Long hardware records span packets, so the prefix can
        # appear in the middle of a value (for example
        # ``repetitions=64CPX: GAP8: |kernel_cycles=...``). Remove the transport
        # decoration before parsing; GVSOC logs never contain it.
        raw = raw.replace("CPX: GAP8: ", "").replace("CPX: GAP8:", "")
        marker = raw.find("KBENCH|")
        if marker < 0:
            continue
        fields = raw[marker:].strip().split("|")
        if len(fields) < 2:
            continue
        record: dict[str, Any] = {"type": fields[1].lower()}
        for field in fields[2:]:
            if "=" not in field:
                continue
            key, value = field.split("=", 1)
            record[key] = _value(value)
        records.append(record)
    return records


def read_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "<F>" in text:
            raise ValueError(
                f"{path}: Crazyflie console overflow marker <F> found; "
                "bytes were dropped, so this profile must be rerun"
            )
        records.extend(parse_records(text.splitlines()))
    return records


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


# Prefer real per-core counter cycles for every comparison and record which
# basis was used, so a kernel-cycle speedup is never silently mixed with an
# FC-elapsed one.
def _basis(row: dict[str, Any]) -> tuple[float | None, str]:
    if row.get("median_kernel_cycles"):
        return float(row["median_kernel_cycles"]), "kernel_cycles"
    if row.get("median_fc_cycles_derived"):
        return float(row["median_fc_cycles_derived"]), "fc_cycles_derived"
    return None, "none"


def _sample_values(
    rows: Iterable[dict[str, Any]], field: str
) -> list[float]:
    """Return normalized numeric samples, ignoring damaged simulator fields.

    GVSOC can occasionally write an asynchronous warning directly into stdout
    while the benchmark is emitting a record. The rest of that run remains
    useful, but the interrupted field is not a number and must not abort the
    complete report. Hardware console overflow is handled separately and is
    always fatal because its ``<F>`` marker means bytes were actually dropped.
    """
    values: list[float] = []
    for row in rows:
        if field not in row:
            continue
        try:
            repetitions = max(1, int(row.get("repetitions", 1)))
            values.append(float(row[field]) / repetitions)
        except (TypeError, ValueError):
            continue
    return values


def _sample_quality(row: dict[str, Any]) -> int:
    score = 0
    for field in (
        "us",
        "cycles",
        "repetitions",
        "kernel_cycles",
        "tcdm_contentions",
        "load_stalls",
    ):
        try:
            float(row[field])
            score += 1
        except (KeyError, TypeError, ValueError):
            pass
    return score


def _attach_speedups(rows: list[dict[str, Any]]) -> None:
    # Adapters that register more than one upstream descriptor qualify the
    # variant name (``pointwise_pulp_nn``), so match the suffix rather than the
    # whole string or those rows silently lose their baseline.
    families = (
        ("speedup_vs_pulp_nn", lambda v: v.endswith("pulp_nn")),
        ("speedup_vs_register4", lambda v: v == "register4"),
        ("speedup_vs_scalar", lambda v: v == "scalar"),
        ("speedup_vs_sequential", lambda v: v == "sequential"),
    )
    for column, matches in families:
        baselines: dict[tuple[Any, Any], float] = {}
        for row in rows:
            measured, _ = _basis(row)
            if measured and matches(str(row.get("variant", ""))):
                baselines[(row.get("case"), row.get("kernel"))] = measured
        for row in rows:
            baseline = baselines.get((row.get("case"), row.get("kernel")))
            measured, basis = _basis(row)
            if baseline and measured:
                row[column] = baseline / measured
                row["speedup_basis"] = basis


def _flag_aliased_baselines(rows: list[dict[str, Any]]) -> None:
    """Catch a REPLACE build where the upstream baseline *is* the candidate.

    conv.mk and add.mk drop the pinned PULP-NN source under
    ``*_REPLACE_PULP_NN=1``, so the baseline descriptor resolves to the
    optimized symbol and would otherwise report a meaningless 1.00x.
    """
    by_case: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[(row.get("case"), row.get("kernel"))].append(row)
    for group in by_case.values():
        baselines = [r for r in group if str(r.get("variant", "")).endswith("pulp_nn")]
        others = [r for r in group if r not in baselines]
        for baseline in baselines:
            measured, _ = _basis(baseline)
            if not measured:
                continue
            for other in others:
                other_measured, _ = _basis(other)
                if (
                    other_measured == measured
                    and other.get("checksum") == baseline.get("checksum")
                    and other.get("operations") == baseline.get("operations")
                ):
                    baseline["baseline_aliased"] = True
                    other["baseline_aliased"] = True
                    other.pop("speedup_vs_pulp_nn", None)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    results: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    cases: dict[Any, dict[str, Any]] = defaultdict(dict)
    for record in records:
        if record["type"] == "case":
            cases[record.get("id")].update(record)
            continue
        if record["type"] not in {"sample", "result"}:
            continue
        key = (record.get("case"), record.get("kernel"), record.get("variant"))
        if record["type"] == "sample":
            samples[key].append(record)
        else:
            results.setdefault(key, {}).update(record)

    rows: list[dict[str, Any]] = []
    for key in sorted(set(samples) | set(results), key=lambda item: tuple(str(v) for v in item)):
        row = dict(results.get(key, {}))
        row.update({"case": key[0], "kernel": key[1], "variant": key[2]})
        case = cases.get(key[0], {})
        row.update(
            {
                "case_name": case.get("name", "-"),
                "batch": case.get("batch", "-"),
                "cores": case.get("cores", "-"),
                "n": case.get("n", "-"),
                "k": case.get("k", "-"),
                "height": case.get("height", "-"),
                "width": case.get("width", "-"),
                "input_channels": case.get("cin", "-"),
                "output_channels": case.get("cout", "-"),
                "kernel_h": case.get("kh", "-"),
                "kernel_w": case.get("kw", "-"),
                "epilogue": case.get("epilogue", "-"),
                "activation": case.get("activation", "-"),
            }
        )
        # GVSOC records are deliberately emitted twice because its own warning
        # stream can splice bytes into semihosted stdout. Keep one sample per
        # run and prefer the copy with the most intact numeric fields.
        samples_by_run: dict[Any, dict[str, Any]] = {}
        for index, sample in enumerate(samples.get(key, [])):
            run = sample.get("run", index)
            previous = samples_by_run.get(run)
            if previous is None or _sample_quality(sample) > _sample_quality(previous):
                samples_by_run[run] = sample
        case_samples = list(samples_by_run.values())
        elapsed = _sample_values(case_samples, "us")
        cycles = _sample_values(case_samples, "cycles")
        kernel_cycles = _sample_values(case_samples, "kernel_cycles")
        tcdm_contentions = _sample_values(case_samples, "tcdm_contentions")
        load_stalls = _sample_values(case_samples, "load_stalls")
        row["samples"] = len(case_samples)
        if elapsed:
            median_us = statistics.median(elapsed)
            row.update(
                {
                    "min_us": min(elapsed),
                    "median_us": median_us,
                    "max_us": max(elapsed),
                }
            )
            # A percentile over a handful of samples is noise dressed as rigour.
            # State the observed dispersion directly instead.
            if median_us:
                row["spread_pct"] = 100.0 * (max(elapsed) - min(elapsed)) / median_us
        if cycles:
            # Not a counter reading: FC elapsed microseconds rescaled by the
            # configured cluster frequency. Named so no report can imply
            # otherwise.
            row["median_fc_cycles_derived"] = statistics.median(cycles)
        if kernel_cycles:
            row["kernel_cycle_samples"] = len(kernel_cycles)
            row["median_kernel_cycles"] = statistics.median(kernel_cycles)
            # Dispersion of the primary metric, kept apart from spread_pct:
            # the first measured run can pay a one-time dispatch cost that FC
            # elapsed time sees and the counters do not. Judge a kernel by
            # this column.
            median_kc = row["median_kernel_cycles"]
            if median_kc:
                row["kernel_spread_pct"] = (
                    100.0 * (max(kernel_cycles) - min(kernel_cycles)) / median_kc
                )
            fc = row.get("median_fc_cycles_derived")
            if fc:
                # How much of the latency is cluster dispatch rather than
                # kernel.
                row["dispatch_overhead_fraction"] = max(
                    0.0, 1.0 - row["median_kernel_cycles"] / float(fc)
                )
        if tcdm_contentions:
            row["median_tcdm_contentions"] = statistics.median(tcdm_contentions)
        if load_stalls:
            row["median_load_stalls"] = statistics.median(load_stalls)
        repetitions = [
            int(sample.get("repetitions", 1)) for sample in case_samples
        ]
        if repetitions:
            row["repetitions_per_sample"] = statistics.median(repetitions)
        operations = row.get("operations")
        if operations and row.get("median_fc_cycles_derived"):
            row["ops_per_cycle"] = float(operations) / float(
                row["median_fc_cycles_derived"]
            )
        if operations and row.get("median_kernel_cycles"):
            row["kernel_ops_per_cycle"] = float(operations) / float(
                row["median_kernel_cycles"]
            )
        executed_operations = row.get("executed_operations")
        if operations and executed_operations:
            row["padding_lane_overhead"] = (
                float(executed_operations) / float(operations) - 1.0
            )
        rows.append(row)

    _attach_speedups(rows)
    _flag_aliased_baselines(rows)

    metadata = [record for record in records if record["type"] == "meta"]
    done = [record for record in records if record["type"] == "done"]
    return {
        "schema_version": 1,
        "metadata": metadata,
        "done": done[-1] if done else None,
        "record_count": len(records),
        "results": rows,
    }


# Eight RI5CY cores x one 4-lane pv.sdotusp.b per cycle. Printing the fraction
# next to every MAC/cycle keeps a 1.6 reading legible as 5% of the machine.
GAP8_PEAK_MAC_PER_CYCLE = 32.0


# Kernels whose `operations` count is multiply-accumulates. For everything else
# -- pooling compares, activation LUT lookups, BatchNorm scales, elementwise
# adds -- `operations` counts logical elements, and dividing it by the 32
# MAC/cycle arithmetic peak produces a number that means nothing.
MAC_KERNELS = (
    "linear_", "conv2d_", "depthwise_conv2d_", "matmul4x2_", "matmul4x3_",
    "matmul4x4_",
)


def _is_mac_kernel(row: dict[str, Any]) -> bool:
    name = str(row.get("kernel", ""))
    return any(name.startswith(prefix) for prefix in MAC_KERNELS)


def _peak_fraction(row: dict[str, Any]) -> str:
    rate = row.get("kernel_ops_per_cycle", row.get("ops_per_cycle"))
    if not isinstance(rate, (int, float)):
        return "-"
    if not _is_mac_kernel(row):
        # Not arithmetic: a fraction of the MAC peak would be meaningless.
        return "n/a"
    return f"{100.0 * float(rate) / GAP8_PEAK_MAC_PER_CYCLE:.1f}%"


def write_report(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = summary["results"]
    fieldnames = sorted({key for row in rows for key in row})
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        if fieldnames:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    lines = ["# kernel_bench profile", ""]
    done = summary.get("done") or {}
    lines.append(f"Run status: `{done.get('status', 'incomplete')}`.  ")
    lines.append(f"Parsed records: {summary['record_count']}.  ")
    lines.append(f"Kernel results: {len(rows)}.")
    lines.append("")
    if rows:
        lines += [
            "| id | case | activation | epilogue | k | n | cores | kernel | variant | status | median us | spread % | kernel spread % | FC cycles (derived) | kernel cycles | dispatch overhead | TCDM cont. | load stalls | kernel ops/cycle | op kind | % of 32 MAC peak | pad lanes | weight rows aligned | vs PULP | vs scalar | vs register4 | vs sequential | basis | aliased | exact |",
            "|---:|---|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|---:|---:|---:|---:|---|:---:|:---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row.get('case', '-')} | {row.get('case_name', '-')} | "
                f"{row.get('activation', '-')} | "
                f"{row.get('epilogue', '-')} | {row.get('k', '-')} | "
                f"{row.get('n', '-')} | "
                f"{row.get('cores', '-')} | {row.get('kernel', '-')} | "
                f"{row.get('variant', '-')} | "
                f"{row.get('status', '-')} | {row.get('median_us', '-')} | "
                f"{row.get('spread_pct', '-')} | "
                f"{row.get('kernel_spread_pct', '-')} | "
                f"{row.get('median_fc_cycles_derived', '-')} | "
                f"{row.get('median_kernel_cycles', '-')} | "
                f"{row.get('dispatch_overhead_fraction', '-')} | "
                f"{row.get('median_tcdm_contentions', '-')} | "
                f"{row.get('median_load_stalls', '-')} | "
                f"{row.get('kernel_ops_per_cycle', row.get('ops_per_cycle', '-'))} | "
                f"{'MAC' if _is_mac_kernel(row) else 'element'} | "
                f"{_peak_fraction(row)} | "
                f"{row.get('padding_lane_overhead', '-')} | "
                f"{row.get('weight_row_aligned', '-')} | "
                f"{row.get('speedup_vs_pulp_nn', '-')} | "
                f"{row.get('speedup_vs_scalar', '-')} | "
                f"{row.get('speedup_vs_register4', '-')} | "
                f"{row.get('speedup_vs_sequential', '-')} | "
                f"{row.get('speedup_basis', '-')} | "
                f"{row.get('baseline_aliased', '-')} | "
                f"{row.get('exact', '-')} |"
            )
    else:
        lines.append("No computational kernel results were reported.")
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        summary = summarize(read_records(args.logs))
    except ValueError as exc:
        parser.error(str(exc))
    write_report(summary, args.out)
    print(args.out / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
