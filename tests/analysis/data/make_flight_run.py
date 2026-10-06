"""Trims a recorded mocap_waypoint flight into the committed flight_run/ test fixture.

Source: examples/mocap_waypoint/artifacts/run_20260717T171351 (gitignored), a 25 s
NatNet-positioned flight with a circle of waypoints and a land_start. Kept: the
state, control, extpose, setpoint and event streams, and the mocap stream without
its pose. Dropped: camera, sysid streams, the run config (WiFi and network
details) and host details. Rows and timestamps are unchanged.
"""
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "examples/mocap_waypoint/artifacts/run_20260717T171351"
OUT = Path(__file__).resolve().parent / "flight_run"
KEEP = {
    "state_pos": None,
    "state_att": None,
    "ctrl": None,
    "extpose": ["seq", "host_ns", "source_ts_s", "source_seq", "x", "y", "z"],
    "setpoints": None,
    "events": None,
    "mocap": ["seq", "host_ns", "source_ts_s", "source_seq"],
}


def main(source: Path = SOURCE) -> None:
    manifest = json.loads((source / "manifest.json").read_text())
    OUT.mkdir(exist_ok=True)
    streams = {}
    for name, columns in KEEP.items():
        table = pq.read_table(source / f"{name}.parquet", columns=columns)
        pq.write_table(table, OUT / f"{name}.parquet", compression="zstd", compression_level=19)
        info = dict(manifest["streams"][name])
        info["fields"] = table.column_names
        streams[name] = info
    trimmed = {key: manifest[key] for key in
               ("format_version", "created_wall_ns", "created_monotonic_ns", "closed_wall_ns",
                "status", "error", "empty_streams", "example", "git_sha")}
    trimmed["streams"] = streams
    (OUT / "manifest.json").write_text(json.dumps(trimmed, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE)
