"""Incremental machine-readable results and a compact human-readable summary."""
from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save(directory: Path, run: dict) -> None:
    write_json(directory / "results.json", run)
    lines = [f"Flash bench: {run['status']}", f"URI: {run['uri']}",
             f"Elapsed: {run.get('elapsed_s', 0):.1f}s",
             f"Passed: {sum(row['status'] == 'pass' for row in run['cases'])}/{run['expected_cases']}",
             "",
             "case                         status       flash(s)  stalls  MD5"]
    for row in run["cases"]:
        flash = row.get("flash", {})
        lines.append(f"{row['name']:28} {row['status']:12} "
                     f"{flash.get('flash_s', 0):8.1f}  "
                     f"{flash.get('stalled_attempts', 0):6}  {flash.get('md5') or '-'}")
        if row.get("error"):
            lines.append(f"  {row['error']}")
        for phase in ("setup", "flash", "persistence"):
            details = row.get(phase, {})
            if details.get("error"):
                lines.append(f"  {phase}: {details['error']}")
            for error in details.get("observation", {}).get("errors", []):
                lines.append(f"  {phase}: {error}")
    if run.get("error"):
        lines += ["", run["error"]]
    if "cleanup" in run:
        lines += ["", f"Final silent-image cleanup: {run['cleanup']['status']}"]
    (directory / "summary.txt").write_text("\n".join(lines) + "\n")
