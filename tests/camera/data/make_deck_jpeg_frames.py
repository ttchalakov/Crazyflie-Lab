"""Trim 30 real QVGA JPEG frames (with their camera_stream extension) out of a recorded run.

Source: examples/mocap_waypoint/artifacts/run_20260717T171534 (gitignored, local only),
rows 900..929. Run from the repo root:
    .venv/bin/python tests/camera/data/make_deck_jpeg_frames.py RUN_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

START, COUNT = 900, 30

run = Path(sys.argv[1] if len(sys.argv) > 1 else
           "examples/mocap_waypoint/artifacts/run_20260717T171534")
rows = pq.read_table(run / "camera.parquet").slice(START, COUNT).to_pylist()
blob = (run / "camera_frames.bin").read_bytes()
payloads = [blob[r["blob_offset"]: r["blob_offset"] + r["size"]] for r in rows]
np.savez(
    Path(__file__).with_name("deck_jpeg_frames.npz"),
    payload=np.frombuffer(b"".join(payloads), np.uint8),
    size=np.array([len(p) for p in payloads], np.int64),
    ext=np.array([list(r["ext"]) for r in rows], np.uint8),
    host_ns=np.array([r["host_ns"] for r in rows], np.int64),
    frame_id=np.array([r["frame_id"] for r in rows], np.int64),
    deck_ms=np.array([r["deck_ms"] for r in rows], np.int64),
    width=rows[0]["width"],
    height=rows[0]["height"],
)
