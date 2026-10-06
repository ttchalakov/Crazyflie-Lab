"""Run-directory builders and the recorded flight shared by the analysis tests."""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cf_lab.telemetry import schema

FLIGHT_RUN = Path(__file__).parent / "data" / "flight_run"
ANCHOR_NS = 5_000_000_000

SPECS = {
    "state_pos": schema.state_pos_spec(100.0),
    "state_att": schema.state_att_spec(100.0),
    "ctrl": schema.ctrl_spec(100.0),
    "mocap": schema.mocap_spec("natnet"),
    "extpose": schema.extpose_spec(),
    "camera": schema.camera_spec(),
    "setpoints": schema.setpoints_spec(),
    "events": schema.events_spec(),
}


def write_run(run_dir: Path, streams: dict, *, status: str = "clean", dropped=None) -> Path:
    """Write a run directory the way RunWriter lays one out.

    ``streams`` maps a stream name from SPECS to its columns ({name: list}); every
    schema column must be given except ``seq`` (1..n when absent). ``host_ns`` is
    absolute; the run's monotonic anchor is ANCHOR_NS.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_streams = {}
    for name, columns in streams.items():
        spec = SPECS[name]
        rows = len(columns["host_ns"])
        columns = {"seq": list(range(1, rows + 1)), **columns}
        fields = [("seq", "int64"), ("host_ns", "int64"), *spec.fields]
        arrow_schema = pa.schema([(field, getattr(pa, kind)()) for field, kind in fields])
        table = pa.table({field: columns[field] for field, _ in fields}, schema=arrow_schema)
        pq.write_table(table, run_dir / f"{name}.parquet")
        manifest_streams[name] = {
            "file": f"{name}.parquet",
            "fields": [field for field, _ in fields],
            "clock": spec.clock,
            "rows": rows,
            "dropped": (dropped or {}).get(name, 0),
            "meta": dict(spec.meta),
        }
    manifest = {
        "format_version": 1,
        "created_wall_ns": 0,
        "created_monotonic_ns": ANCHOR_NS,
        "closed_wall_ns": 0,
        "status": status,
        "error": None,
        "empty_streams": sorted(name for name, c in streams.items() if not c["host_ns"]),
        "streams": manifest_streams,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def host_ns_at(t_s: float) -> int:
    """Absolute host_ns for a time in seconds after the run anchor."""
    return ANCHOR_NS + round(t_s * 1e9)


def events(*rows: tuple[float, str, dict]) -> dict:
    """events stream columns from (t_s, kind, data) rows."""
    return {
        "host_ns": [host_ns_at(t) for t, _, _ in rows],
        "kind": [kind for _, kind, _ in rows],
        "message": ["" for _ in rows],
        "data_json": [json.dumps(data) for _, _, data in rows],
    }


@pytest.fixture(scope="session")
def flight_run():
    """The recorded mocap_waypoint flight, loaded once (read-only)."""
    from cf_lab.telemetry.sync import load_run

    return load_run(FLIGHT_RUN)


@pytest.fixture
def runs():
    """This module's run builders (write_run, host_ns_at, events, ANCHOR_NS)."""
    import sys

    return sys.modules[__name__]
