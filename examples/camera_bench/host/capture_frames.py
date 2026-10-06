#!/usr/bin/env python3
"""Save a spread of frames from the AI-deck stream, plus a montage, to inspect a build.

Run it while the deck streams (after a bench case, or any camera_stream build),
with this computer on the deck's network:

  python examples/camera_bench/host/capture_frames.py --count 8 --out-dir /tmp/frames
"""

from __future__ import annotations

import argparse
import pathlib
import time

import cv2
import numpy as np

from cf_lab.camera.feed import (
    DEFAULT_READ_TIMEOUT_S,
    close_stream,
    frames as cf_frames,
    open_stream,
)


def _write_montage(frames, out_path: pathlib.Path, *, scale: int, columns: int) -> None:
    if not frames:
        return

    labeled = []
    for image, frame_id in frames:
        tile = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            tile,
            f"frame {frame_id}",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        labeled.append(tile)

    height, width = labeled[0].shape[:2]
    rows = (len(labeled) + columns - 1) // columns
    montage = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(labeled):
        row, col = divmod(index, columns)
        montage[row * height:(row + 1) * height, col * width:(col + 1) * width] = tile

    cv2.imwrite(str(out_path), montage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-ip", default="192.168.4.1")
    parser.add_argument("--deck-port", type=int, default=5000)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT_S)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("/tmp/camera_stream_frames"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--min-frame-step", type=int, default=20)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sock, recv = open_stream(
        args.deck_ip,
        args.deck_port,
        connect_timeout_s=args.connect_timeout,
        read_timeout_s=args.read_timeout,
        log=lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True),
    )

    saved = []
    last_saved_frame_id: int | None = None
    try:
        for frame in cf_frames(recv):
            ext = frame.ext
            frame_id = ext.frame_id if ext is not None else len(saved)
            if (
                last_saved_frame_id is not None
                and frame_id - last_saved_frame_id < args.min_frame_step
            ):
                continue
            image = frame.decode()
            if image is None:
                continue

            frame_path = args.out_dir / f"frame_{len(saved):02d}_id_{frame_id}.png"
            cv2.imwrite(str(frame_path), image)
            timings = (
                f", deck {ext.encode_ms} ms, cap {ext.capture_ms} ms, "
                f"pre {ext.preprocess_ms} ms, jpg {ext.jpeg_ms} ms"
                if ext is not None else ""
            )
            print(
                f"saved {frame_path} ({frame.width}x{frame.height}, "
                f"{frame.header.size} B{timings})",
                flush=True,
            )
            saved.append((image.copy(), frame_id))
            last_saved_frame_id = frame_id
            if len(saved) >= args.count:
                break
    finally:
        close_stream(sock)

    montage_path = args.out_dir / "montage.png"
    _write_montage(saved, montage_path, scale=args.scale, columns=args.columns)
    print(f"wrote {montage_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
