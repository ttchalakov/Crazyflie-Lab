"""Trim a real telemetry_bench recording (2 kHz mocap + 100 Hz log blocks) to a short window."""
import json
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "examples/telemetry_bench/artifacts/run_20261001T165053"
)
OUT = Path(__file__).parent / "bench_run"
STREAMS = ("state_pos", "tba", "mocap", "extpose", "setpoints", "events")
BEFORE_ORIGIN_S, AFTER_ORIGIN_S = 0.1, 2.5

manifest = json.loads((SOURCE / "manifest.json").read_text())
events = pq.read_table(SOURCE / "events.parquet").to_pylist()
origin_ns = next(e["host_ns"] for e in events if e["kind"] == "origin_zeroed")
lo, hi = origin_ns - int(BEFORE_ORIGIN_S * 1e9), origin_ns + int(AFTER_ORIGIN_S * 1e9)

OUT.mkdir(exist_ok=True)
streams = {}
for name in STREAMS:
    table = pq.read_table(SOURCE / f"{name}.parquet")
    host = table.column("host_ns")
    if name != "events":
        table = table.filter(pc.and_(pc.greater_equal(host, lo), pc.less_equal(host, hi)))
    else:
        table = table.filter(pc.less_equal(host, hi))
    pq.write_table(table, OUT / f"{name}.parquet", compression="zstd")
    streams[name] = dict(manifest["streams"][name], rows=table.num_rows)

# No config (it carries WiFi credentials and IPs), no TOC, no git metadata.
trimmed = {key: manifest[key] for key in (
    "format_version", "created_wall_ns", "created_monotonic_ns", "closed_wall_ns",
    "status", "error", "empty_streams", "example",
)}
trimmed["streams"] = streams
(OUT / "manifest.json").write_text(json.dumps(trimmed, indent=2))
