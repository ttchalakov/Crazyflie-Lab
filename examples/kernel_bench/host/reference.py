#!/usr/bin/env python3
"""Record and check exact-output checksums across runs.

Every result in this bench is an integer-exact comparison against a golden
computed beside it, which catches a kernel that disagrees with its own oracle
but not operands that drift. If the generated operands change -- a scale, a
seed, a bias distribution -- both sides move together and every case still
reports `exact=1` while silently measuring a different problem than the one
its case name describes.

This is the ledger that makes such a drift visible: a committed file of
checksums, checked against a fresh log. It is deliberately small and readable,
and it lives beside the matrix rather than under the ignored build tree, so a
result can be tied to a value under version control.

    python3 host/reference.py --record artifacts/bench_<stamp>/gvsoc-s8bias/*/gvsoc.log
    python3 host/reference.py --check  artifacts/bench_<stamp>/gvsoc-s8bias/*/gvsoc.log
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
DEFAULT_LEDGER = BENCH / "configs" / "reference_results.json"


def collect(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Case name -> kernel/variant -> expected checksum, from KBENCH records."""
    entries: dict[str, dict[str, Any]] = {}
    for path in paths:
        names: dict[str, str] = {}
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            marker = line.find("KBENCH|")
            if marker < 0:
                continue
            fields = line[marker:].strip().split("|")
            kind = fields[1] if len(fields) > 1 else ""
            values = dict(
                field.split("=", 1) for field in fields[2:] if "=" in field
            )
            if kind == "CASE" and "name" in values:
                names[values.get("id", "")] = values["name"]
            elif kind == "RESULT" and "expected" in values:
                case = names.get(values.get("case", ""), values.get("case", ""))
                key = f"{values.get('kernel')}/{values.get('variant')}"
                entries.setdefault(case, {})[key] = {
                    "expected_checksum": int(values["expected"]),
                    "status": values.get("status", ""),
                }
    return entries


def compare(recorded: dict[str, Any], observed: dict[str, Any],
            counts: dict[str, int] | None = None) -> list[str]:
    """Drifts, most specific first. An unrecorded case is not a drift.

    `counts`, if given, receives `compared` and `skipped` tallies. Without them
    the caller cannot tell "nothing moved" from "nothing was checked" -- a case
    whose *name* changed is unrecorded, so it silently compares nothing while
    still reporting success.
    """
    drifts: list[str] = []
    compared = skipped = 0
    for case, kernels in sorted(observed.items()):
        known = recorded.get(case)
        if known is None:
            skipped += len(kernels)
            continue
        for key, value in sorted(kernels.items()):
            before = known.get(key)
            if before is None:
                skipped += 1
                continue
            compared += 1
            if before["expected_checksum"] != value["expected_checksum"]:
                drifts.append(
                    f"{case} {key}: golden changed "
                    f"{before['expected_checksum']} -> "
                    f"{value['expected_checksum']}"
                )
    if counts is not None:
        counts["compared"] = compared
        counts["skipped"] = skipped
    return drifts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", action="store_true",
                      help="merge these logs into the ledger")
    mode.add_argument("--check", action="store_true",
                      help="fail if a recorded golden moved")
    args = parser.parse_args()

    observed = collect(args.logs)
    if args.record:
        ledger: dict[str, Any] = {"schema_version": 1, "cases": {}}
        if args.ledger.exists():
            ledger = json.loads(args.ledger.read_text())
        # Merge per kernel/variant, not per case: recording one variant keeps
        # the case's other variants already in the ledger.
        cases = ledger.setdefault("cases", {})
        for case, kernels in observed.items():
            cases.setdefault(case, {}).update(kernels)
        args.ledger.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"recorded {len(observed)} case(s) -> {args.ledger}")
        return 0

    if not args.ledger.exists():
        print(f"no ledger at {args.ledger}; run --record first",
              file=sys.stderr)
        return 1
    recorded = json.loads(args.ledger.read_text()).get("cases", {})
    counts: dict[str, int] = {}
    drifts = compare(recorded, observed, counts)
    for drift in drifts:
        print(f"DRIFT: {drift}", file=sys.stderr)
    if drifts:
        print(f"{len(drifts)} golden checksum(s) moved", file=sys.stderr)
        return 1
    if counts.get("compared", 0) == 0:
        print(
            f"nothing compared: {counts.get('skipped', 0)} result(s) are not in "
            f"the ledger. A renamed case is unrecorded, so this is a silent "
            f"pass, not a clean one -- re-record if the rename was intended.",
            file=sys.stderr)
        return 1
    message = f"{counts['compared']} result(s) match the ledger"
    if counts.get("skipped"):
        message += f", {counts['skipped']} not recorded"
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
