#!/usr/bin/env python3
"""nn_bench on the deck: build a prepared point, flash it, and record its report.

  python examples/nn_bench/host/run.py examples/nn_bench/artifacts/points/mlp-12to256to256to4-l3 \
      --runner lib_kernels

The point is a directory point.py prepared. The app is built for it (io=uart),
flashed over the radio with the drone power-cycled, and the radio console is
recorded until the app prints `NN| DONE` or --seconds pass. Everything lands in
<point>/aideck/<runner>/: build.log, nn_bench.img, console.log and report.json,
the verdict as point.py gives GVSOC's. The radio URI comes from configs/drone.yaml.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from cf_lab.bench import RadioConsole, flash_and_boot
from cf_lab.config import load_config
from cf_lab.firmware import build_gap8, gap8_artifact

sys.path.insert(0, str(Path(__file__).resolve().parent))
import point as points  # noqa: E402

APP_DIR = points.BENCH_DIR / "gap8"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def capture(uri: str, path: Path, seconds: float) -> str:
    """The radio console from now until the app reports NN| DONE."""
    console = RadioConsole(uri, path)
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline and "NN| DONE" not in path.read_text(errors="replace"):
            time.sleep(1.0)
    finally:
        console.close()
    return path.read_text(errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("point", type=Path, help="a point directory point.py prepared")
    parser.add_argument("--runner", choices=points.RUNNERS, default="pulp_nn")
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--seconds", type=float, default=180, help="longest wait for NN| DONE after power-up")
    parser.add_argument("--config", type=Path, default=points.REPO / "configs" / "drone.yaml")
    args = parser.parse_args(argv)

    point = args.point.resolve()
    host = json.loads((point / "host.json").read_text())
    out = point / "aideck" / args.runner
    out.mkdir(parents=True, exist_ok=True)
    try:
        log(f"build {point.name} {args.runner}")
        with (out / "build.log").open("w") as build_log:
            build_gap8(APP_DIR, f"clean build image io=uart NNBENCH_RUNS={args.runs} {points.make_vars(point, args.runner)}",
                       build_image=False, output=build_log)
        image = shutil.copy2(gap8_artifact(APP_DIR), out / "nn_bench.img")
        uri = load_config(args.config).radio.uri
        log(f"flash {Path(image).stat().st_size} B and power-cycle")
        flash_and_boot(Path(image), uri, log=lambda message: log(f"  {message}"))
        log(f"record the radio console (up to {args.seconds:.0f} s)")
        verdict = points.judge(points.parse(capture(uri, out / "console.log", args.seconds)),
                               host["layers"], args.runs)
    except Exception as exc:  # a failed build or flash must not end an unattended queue
        verdict = {"result": "fail", "detail": f"{type(exc).__name__}: {exc}", "median_us": None, "layer_us": []}
    (out / "report.json").write_text(json.dumps(verdict) + "\n")
    log(f"{verdict['result']}: {verdict['detail']}, {verdict['median_us']} us")
    return 0 if verdict["result"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
