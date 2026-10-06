"""Trim a recorded NatNet flight (mocap_waypoint run_20260717T163650) into flight_sample/.

Writes mocap.csv (raw lab-frame poses as the streamer consumed them, with host
arrival ns), extpose.csv (what went up the radio, origin-zeroed and rate-capped)
and origin.json (the origin_zeroed event). Two segments: the first 0.25 s, which
contains the pose the origin locked on, and 2 s mid-flight while the drone yaws.
Run: .venv/bin/python tests/mocap/data/make_flight_sample.py <run_dir>
"""
import csv
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

COLUMNS = ["host_ns", "source_seq", "x", "y", "z", "qx", "qy", "qz", "qw"]
SEGMENTS = {"start": (0.0, 0.25), "flight": (15.0, 17.0)}


def main(run_dir: Path) -> None:
    out = Path(__file__).parent / "flight_sample"
    out.mkdir(exist_ok=True)
    mocap = pq.read_table(run_dir / "mocap.parquet").to_pylist()
    extpose = pq.read_table(run_dir / "extpose.parquet").to_pylist()
    t0 = mocap[0]["host_ns"]
    for name, rows in (("mocap", mocap), ("extpose", extpose)):
        with open(out / f"{name}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["segment", *COLUMNS])
            for row in rows:
                t = (row["host_ns"] - t0) / 1e9
                for segment, (lo, hi) in SEGMENTS.items():
                    if lo <= t < hi:
                        writer.writerow([segment, *(repr(row[c]) for c in COLUMNS)])
    events = pq.read_table(run_dir / "events.parquet").to_pylist()
    origin = next(json.loads(e["data_json"]) for e in events if e["kind"] == "origin_zeroed")
    (out / "origin.json").write_text(json.dumps(origin, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
