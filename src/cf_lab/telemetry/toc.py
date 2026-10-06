"""Snapshot the firmware TOCs that cflib downloaded for a telemetry run.

The log and parameter tables of contents are the exact host-visible interface of
the connected firmware.  Saving them beside the sampled streams makes a run
self-describing even when the firmware checkout or cflib cache later changes.
This module records metadata only; it never reads parameter values or writes to
the Crazyflie.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


CRAZYFLIE_TOC_FILE = "crazyflie_toc.json"
CRAZYFLIE_TOC_FORMAT = "cf_lab.crazyflie_toc.v1"


def snapshot_crazyflie_toc(cf) -> dict[str, Any]:
    """Return a JSON-safe snapshot of cflib's downloaded log/parameter TOCs.

    A Crazyflie without a downloaded TOC (such as a test fake) is represented as an
    unavailable table instead of raising.  On a live, fully connected Crazyflie
    both tables should be available and carry their complete entry counts.
    """
    return {
        "format": CRAZYFLIE_TOC_FORMAT,
        "captured_wall_ns": time.time_ns(),
        "log": _snapshot_table(getattr(getattr(cf, "log", None), "toc", None), kind="log"),
        "parameter": _snapshot_table(
            getattr(getattr(cf, "param", None), "toc", None), kind="parameter"
        ),
    }


def save_crazyflie_toc(run_dir: str | Path, cf) -> Path:
    """Atomically save :func:`snapshot_crazyflie_toc` in ``run_dir``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / CRAZYFLIE_TOC_FILE
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(snapshot_crazyflie_toc(cf), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def load_crazyflie_toc(run_dir: str | Path) -> dict[str, Any] | None:
    """Load a saved TOC, or return ``None`` for a run without one."""
    path = Path(run_dir) / CRAZYFLIE_TOC_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_table(container: object, *, kind: str) -> dict[str, Any]:
    groups = getattr(container, "toc", None)
    if not isinstance(groups, dict):
        return {"available": False, "count": 0, "entries": []}

    entries: list[dict[str, Any]] = []
    for group_name in sorted(groups, key=str):
        variables = groups[group_name]
        if not isinstance(variables, dict):
            continue
        for variable_name in sorted(variables, key=str):
            element = variables[variable_name]
            group = str(getattr(element, "group", group_name))
            name = str(getattr(element, "name", variable_name))
            entry: dict[str, Any] = {
                "id": _json_scalar(getattr(element, "ident", None)),
                "group": group,
                "name": name,
                "full_name": f"{group}.{name}",
                "ctype": _json_scalar(getattr(element, "ctype", None)),
                "pytype": _json_scalar(getattr(element, "pytype", None)),
            }
            if kind == "parameter":
                entry["access"] = _json_scalar(element.get_readable_access())
                entry["persistent"] = bool(element.is_persistent())
                entry["extended"] = bool(element.is_extended())
            else:
                # cflib exposes the raw log metadata access flag rather than a
                # stable human-readable enum. Preserve it without interpretation.
                entry["access"] = _json_scalar(getattr(element, "access", None))
            entries.append(entry)
    return {"available": True, "count": len(entries), "entries": entries}


def _json_scalar(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


__all__ = [
    "CRAZYFLIE_TOC_FILE",
    "CRAZYFLIE_TOC_FORMAT",
    "load_crazyflie_toc",
    "save_crazyflie_toc",
    "snapshot_crazyflie_toc",
]
