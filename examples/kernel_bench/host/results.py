#!/usr/bin/env python3
"""One table of a kernel_bench run: every case, kernel and variant, GVSOC beside the AI-deck.

    python examples/kernel_bench/host/results.py [artifacts/bench_<stamp>]   # default: the latest run

Writes RESULTS.md into the run folder.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import yaml

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ARTIFACTS = BENCH / "artifacts"

FAMILY_ORDER = {
    "linear": 0,
    "matmul": 1,
    "conv": 2,
    "dw": 3,
    "add": 4,
    "activation": 5,
    "batchnorm": 6,
    "pool": 7,
}
FAMILY_LABEL = {
    "linear": "Linear",
    "matmul": "Matrix multiplication",
    "conv": "Convolution / pointwise",
    "dw": "Depthwise convolution",
    "add": "Residual add",
    "activation": "Activation",
    "batchnorm": "Batch norm",
    "pool": "Pooling",
}


class Configuration(NamedTuple):
    label: str
    hardware_dir: str
    gvsoc_dir: str


def configurations(bench_yaml: Path = HERE / "bench.yaml") -> list[Configuration]:
    """bench.yaml's configurations in its order, labelled `<name> (<make variables>)`,
    with the folders bench.py writes them to (bench.folder)."""
    spec = yaml.safe_load(bench_yaml.read_text(encoding="utf-8"))["configurations"]
    found = []
    for name, configuration in spec.items():
        make = " ".join(configuration.get("make", []))
        label = f"{name} ({make})" if make else name
        found.append(Configuration(label, f"aideck-{name}", f"gvsoc-{name}"))
    return found


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def case_fingerprint(case: dict[str, Any]) -> str:
    return json.dumps(
        {
            key: value
            for key, value in case.items()
            if key not in {"id", "name", "suite"}
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _cycles(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    value = row.get("median_kernel_cycles")
    if value is None:
        value = row.get("median_fc_cycles_derived")
    return float(value) if value is not None else None


def _reference_variant(variants: set[str]) -> str | None:
    pulp = sorted(variant for variant in variants if variant.endswith("pulp_nn"))
    if pulp:
        return pulp[0]
    if len(variants) > 1 and "scalar" in variants:
        return "scalar"
    if len(variants) > 1 and "sequential" in variants:
        return "sequential"
    return None


def _reference_label(reference: str | None, variant: str) -> str:
    if reference is None:
        return "—"
    if reference.endswith("pulp_nn"):
        label = "PULP-NN"
    elif reference == "scalar":
        label = "scalar"
    elif reference == "sequential":
        label = "sequential"
    else:
        label = reference
    return f"{label} (self)" if variant == reference else label


def _number(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:,.1f}"


def _bytes(row: dict[str, Any] | None, key: str) -> str:
    if not row or row.get(key) is None:
        return "—"
    return f"{int(row[key]):,}"


def _gain(reference: float | None, candidate: float | None) -> str:
    if not reference or not candidate:
        return "—"
    ratio = reference / candidate
    reduction = 100.0 * (1.0 - candidate / reference)
    return f"{ratio:.3f}× ({reduction:+.1f}%)"


def _ratio(sim: float | None, deck: float | None) -> str:
    if not sim or not deck:
        return "—"
    return f"{sim / deck:.3f}×"


def _verdict(row: dict[str, Any] | None) -> str:
    if row is None:
        return "missing"
    status = str(row.get("status", "unknown"))
    if status == "pass":
        return "✓" if row.get("exact") == 1 else "not exact"
    return status


def _valid_sim_row(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    if row.get("status") == "skip":
        return True
    if row.get("status") != "pass" or row.get("exact") != 1 or _cycles(row) is None:
        return False
    return int(row.get("kernel_cycle_samples", row.get("samples", 0))) == int(
        row.get("samples", 0)
    )


def _test_label(case: dict[str, Any], row: dict[str, Any]) -> str:
    text = f"{case['name']} · `{row['kernel']}/{row['variant']}`"
    return text.replace("|", "\\|")


def _load_simulations(root: Path) -> dict[str, dict[str, Any]]:
    simulations: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(root.glob("*/manifest.json")):
        suite = manifest_path.parent.name
        summary_path = manifest_path.parent / "report" / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"missing simulator report: {summary_path}")
        manifest = _json(manifest_path)
        summary = _json(summary_path)
        by_fingerprint: dict[str, int] = {}
        for case in manifest["cases"]:
            fingerprint = case_fingerprint(case)
            if fingerprint in by_fingerprint:
                raise ValueError(f"duplicate case fingerprint in {manifest_path}")
            by_fingerprint[fingerprint] = int(case["id"])
        rows = {
            (int(row["case"]), str(row["kernel"]), str(row["variant"])): row
            for row in summary["results"]
        }
        simulations[suite] = {
            "manifest": manifest,
            "summary": summary,
            "case_by_fingerprint": by_fingerprint,
            "rows": rows,
        }
    return simulations


def _configuration_rows(build_root: Path, configuration: Configuration) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hardware_root = build_root / configuration.hardware_dir
    manifest = _json(hardware_root / "manifest.json")
    summary = _json(hardware_root / "report" / "summary.json")
    simulations = _load_simulations(build_root / configuration.gvsoc_dir)

    hardware_cases = {int(case["id"]): case for case in manifest["cases"]}
    hardware_rows = {
        (int(row["case"]), str(row["kernel"]), str(row["variant"])): row
        for row in summary["results"]
    }
    grouped_variants: dict[tuple[int, str], set[str]] = defaultdict(set)
    for case_id, kernel, variant in hardware_rows:
        grouped_variants[(case_id, kernel)].add(variant)

    sim_context: dict[tuple[int, str], tuple[str, dict[str, Any], int]] = {}
    fallbacks: list[str] = []
    for (case_id, kernel), variants in grouped_variants.items():
        case = hardware_cases[case_id]
        fingerprint = case_fingerprint(case)
        preferred_suite = str(case["suite"])
        candidates: list[tuple[bool, bool, str, dict[str, Any], int]] = []
        for suite, sim in simulations.items():
            sim_case_id = sim["case_by_fingerprint"].get(fingerprint)
            if sim_case_id is None:
                continue
            complete = all(
                _valid_sim_row(sim["rows"].get((sim_case_id, kernel, variant)))
                for variant in variants
            )
            candidates.append(
                (complete, suite == preferred_suite, suite, sim, sim_case_id)
            )
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if not candidates or not candidates[0][0]:
            raise ValueError(
                f"{configuration.label}: no complete simulator result group for "
                f"{case['name']}:{kernel}"
            )
        _, _, source_suite, sim, sim_case_id = candidates[0]
        sim_context[(case_id, kernel)] = (source_suite, sim, sim_case_id)
        if source_suite != preferred_suite:
            fallbacks.append(
                f"{preferred_suite}/{case['name']}:{kernel} → {source_suite}"
            )

    output: list[dict[str, Any]] = []
    for (case_id, kernel, variant), hardware_row in hardware_rows.items():
        case = hardware_cases[case_id]
        source_suite, sim, sim_case_id = sim_context[(case_id, kernel)]
        sim_row = sim["rows"].get((sim_case_id, kernel, variant))

        reference = _reference_variant(grouped_variants[(case_id, kernel)])
        hardware_reference = (
            hardware_rows.get((case_id, kernel, reference)) if reference else None
        )
        sim_reference = (
            sim["rows"].get((sim_case_id, kernel, reference)) if reference else None
        )
        output.append(
            {
                "configuration": configuration.label,
                "case": case,
                "hardware": hardware_row,
                "sim": sim_row,
                "reference": reference,
                "hardware_reference": hardware_reference,
                "sim_reference": sim_reference,
                "sim_source": source_suite,
            }
        )

    observed_cases = {int(row["case"]) for row in summary["results"]}
    missing_cases = set(hardware_cases) - observed_cases
    if missing_cases:
        raise ValueError(f"{configuration.label}: missing hardware cases {sorted(missing_cases)}")
    return output, {
        "manifest": manifest,
        "summary": summary,
        "fallbacks": sorted(set(fallbacks)),
    }


def render(build_root: Path) -> str:
    all_rows: list[dict[str, Any]] = []
    configuration_data: dict[str, dict[str, Any]] = {}
    order = configurations()
    for configuration in order:
        # A run holds the configurations its set asked for, on both targets.
        if not ((build_root / configuration.hardware_dir / "manifest.json").is_file()
                and (build_root / configuration.gvsoc_dir).is_dir()):
            continue
        rows, data = _configuration_rows(build_root, configuration)
        all_rows.extend(rows)
        configuration_data[configuration.label] = data

    all_rows.sort(
        key=lambda item: (
            next(i for i, configuration in enumerate(order) if configuration.label == item["configuration"]),
            FAMILY_ORDER.get(item["case"]["family"], 99),
            str(item["case"]["shape"]),
            int(item["case"]["cores"]),
            str(item["case"]["memory"]),
            str(item["case"]["epilogue"]),
            str(item["hardware"]["kernel"]),
            str(item["hardware"]["variant"]),
        )
    )

    if not configuration_data:
        raise SystemExit(f"{build_root}: no configuration ran on both GVSOC and the AI-deck")
    primary = next(iter(configuration_data.values()))
    provenance = primary["manifest"]["provenance"]
    build = provenance.get("build", {})
    pass_rows = sum(
        1
        for item in all_rows
        if item["hardware"].get("status") == "pass"
        and item["hardware"].get("exact") == 1
        and item["sim"]
        and item["sim"].get("status") == "pass"
        and item["sim"].get("exact") == 1
    )
    skip_rows = sum(
        1
        for item in all_rows
        if item["hardware"].get("status") == "skip"
        and item["sim"]
        and item["sim"].get("status") == "skip"
    )
    per_configuration = ", ".join(
        f"{label} {sum(item['configuration'] == label for item in all_rows)}"
        for label in configuration_data)
    fallbacks = sorted(
        {
            fallback
            for data in configuration_data.values()
            for fallback in data.get("fallbacks", [])
        }
    )

    lines = [
        "# kernel_bench results",
        "",
        "One row per measured case, kernel and variant, from both targets; written by "
        "`host/results.py`.",
        "",
        f"- Recorded `{provenance['recorded_utc']}`, commit `{provenance['git']['commit']}`, "
        f"matrix `{provenance['matrix']['sha256_16']}`, SDK image `{provenance['image']['digest']}`.",
        f"- Clocks requested: FC {build.get('fc_freq_hz', '?')} Hz, cluster "
        f"{build.get('cl_freq_hz', '?')} Hz.",
        f"- **{pass_rows} rows exact on both targets**, {skip_rows} skipped on both by "
        f"contract. Rows per configuration: {per_configuration}.",
        "",
        "`cycles` is the median cluster performance-counter value per invocation "
        "(`median_kernel_cycles`), not FC wall time. Improvement is "
        "`reference / candidate`; the percentage is latency reduction "
        "`1 - candidate/reference`. `GVSOC / AI-deck` above 1 means the deck "
        "used fewer cycles. L1/L2 are the GVSOC row's declared arena bytes, "
        "not host RSS or a sum. "
        "a dash means no honest reference exists for that configuration.",
        "",
    ]
    if fallbacks:
        lines += [
            "GVSOC emitted an asynchronous warning into a machine record. That damaged "
            "row is not used here: semantic-fingerprint matching selects the identical "
            f"complete case from an overlapping suite ({'; '.join(fallbacks)}).",
            "",
        ]
    lines += [
        "| Test | Reference | GVSOC ref cycles | GVSOC cycles | GVSOC improvement | AI-deck ref cycles | AI-deck cycles | AI-deck improvement | GVSOC / AI-deck | GVSOC L1 (B) | GVSOC L2 (B) | Exact sim/deck |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]

    current_group: tuple[str, str] | None = None
    for item in all_rows:
        configuration = item["configuration"]
        case = item["case"]
        family = str(case["family"])
        group = (configuration, family)
        if group != current_group:
            label = f"**{configuration} — {FAMILY_LABEL.get(family, family)}**"
            lines.append(f"| {label} |  |  |  |  |  |  |  |  |  |  |  |  |")
            current_group = group

        hardware = item["hardware"]
        sim = item["sim"]
        hardware_reference = item["hardware_reference"]
        sim_reference = item["sim_reference"]
        hardware_cycles = _cycles(hardware)
        sim_cycles = _cycles(sim)
        hardware_ref_cycles = _cycles(hardware_reference)
        sim_ref_cycles = _cycles(sim_reference)
        exact = f"{_verdict(sim)}/{_verdict(hardware)}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _test_label(case, hardware),
                    _reference_label(item["reference"], str(hardware["variant"])),
                    _number(sim_ref_cycles),
                    _number(sim_cycles),
                    _gain(sim_ref_cycles, sim_cycles),
                    _number(hardware_ref_cycles),
                    _number(hardware_cycles),
                    _gain(hardware_ref_cycles, hardware_cycles),
                    _ratio(sim_cycles, hardware_cycles),
                    _bytes(sim, "l1_bytes"),
                    _bytes(sim, "l2_bytes"),
                    exact,
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, nargs="?", help="a bench run folder (default: the latest)")
    args = parser.parse_args()
    run = args.run or max(ARTIFACTS.glob("bench_*"), default=None)
    if run is None:
        parser.error(f"no bench run under {ARTIFACTS}")
    out = run / "RESULTS.md"
    out.write_text(render(run), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
