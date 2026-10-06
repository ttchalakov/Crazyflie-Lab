#!/usr/bin/env python3
"""jpeg_cluster_bench on the AI-deck: build one case, flash it, and record its rounds.

  python examples/jpeg_cluster_bench/host/run.py --out examples/jpeg_cluster_bench/artifacts/320x240
  python examples/jpeg_cluster_bench/host/run.py --out <dir> --window 316x236
  python examples/jpeg_cluster_bench/host/run.py --out <dir> --defines=-DJPEG_CL_EARLY_EOB=0
  python examples/jpeg_cluster_bench/host/run.py --out <dir> --seconds 900     # a soak
  python examples/jpeg_cluster_bench/host/run.py --out <dir> --listen          # capture only

The app encodes a window of the real frame on the FC and on the cluster, and
reports every round of 100 frames on the radio console, forever: the time of
each path, whether the two JPEGs were byte-identical, their hash, and the
overflow count. This builds it with the window and defines as BENCH_DEFINES,
flashes it over the radio, power-cycles the drone and records the console for
--seconds.

--out gets build.log, jpeg_cluster_bench.img, console.log and report.json. A run passes
when the deck ran the requested window, every round's JPEGs were identical and
the same as the first round's, nothing overflowed, and rounds were still
arriving at the end (a wedged cluster stops them). The exit status is 0 for a
pass. host/bench.py also compares the hash with the host's.

The radio URI comes from configs/drone.yaml.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import statistics
import sys
import time
from pathlib import Path

from cf_lab.bench import RadioConsole, flash_and_boot
from cf_lab.config import load_config

HOST_DIR = Path(__file__).resolve().parent
APP_DIR = HOST_DIR.parent / "gap8"
ROOT = HOST_DIR.parents[2]
# Rounds take about 3.3 s; a gap this long means the deck stopped.
STALL_S = 20.0

HEADER = re.compile(r"== jpeg_cluster_bench: (\d+)x(\d+) gray")
ROUND = re.compile(r"#ROUND (\d+) frames (\d+) \| fc (\d+) us \| cluster (\d+) us \([\d.]+x\) \| "
                   r"hash ([0-9a-f]{8})/([0-9a-f]{8}) \| identical (YES|NO) \| overflows (\d+) \| "
                   r"peak_seg (\d+)/(\d+)")
POLICY = re.compile(r"#POL keep_all (\d+) us \(L1 (\d+)\) \| keep_tables (\d+) us \(L1 (\d+)\) \| "
                    r"release_all (\d+) us \(L1 (\d+)")
STACK = re.compile(r"#STACK bench task: (\d+) B total, (\d+) B min free, (\d+) B peak used")
BUFFERS = re.compile(r"#BUF out (\d+)/(\d+) B seg (\d+)/(\d+) B -> overflow (\S+)")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build(defines: str, build_log: Path) -> Path:
    """Build the bench for the deck; return the flash image."""
    from cf_lab.firmware import build_gap8, gap8_artifact

    with build_log.open("w") as out:
        build_gap8(APP_DIR, "clean build image " + shlex.quote(f"BENCH_DEFINES={defines}"),
                   build_image=False, output=out)
    return gap8_artifact(APP_DIR)


def capture(uri: str, path: Path, seconds: float) -> float:
    """Record the radio console for `seconds`; return how long before the end the last round came."""
    console = RadioConsole(uri, path)
    started = last_round = time.monotonic()
    rounds = 0
    next_note = started + 60.0
    try:
        while time.monotonic() - started < seconds:
            time.sleep(1.0)
            count = path.read_text(encoding="utf-8", errors="replace").count("#ROUND")
            if count > rounds:
                rounds, last_round = count, time.monotonic()
            if time.monotonic() >= next_note:
                next_note += 60.0
                log(f"  {time.monotonic() - started:.0f} s, {rounds} rounds")
    finally:
        console.close()
    return time.monotonic() - last_round


def judge(raw: str, window: str, stalled_s: float) -> dict:
    # The STM32 prefixes every CPX console packet, so a long line can carry the
    # prefix mid-line.
    text = raw.replace("CPX: GAP8: ", "").replace("CPX: GAP8:", "")
    rounds = [m.groups() for m in ROUND.finditer(text)]
    policies = [tuple(map(int, m.groups())) for m in POLICY.finditer(text)]
    stacks = [int(m.group(3)) for m in STACK.finditer(text)]
    header = HEADER.search(text)
    buffers = BUFFERS.search(text)
    hashes = sorted({h for r in rounds for h in (r[4], r[5])})
    report = {
        "window": f"{header.group(1)}x{header.group(2)}" if header else None,
        "rounds": len(rounds),
        "frames": int(rounds[-1][1]) if rounds else 0,
        "fc_us": statistics.median(int(r[2]) for r in rounds) if rounds else None,
        "cluster_us": statistics.median(int(r[3]) for r in rounds) if rounds else None,
        "hash": hashes[0] if len(hashes) == 1 else None,
        "hashes": hashes,
        "identical": bool(rounds) and all(r[6] == "YES" for r in rounds),
        "overflows": int(rounds[-1][7]) if rounds else None,
        "peak_segment": max((int(r[8]) for r in rounds), default=None),
        "segment_cap": int(rounds[-1][9]) if rounds else None,
        "overflow_possible": buffers.group(5) != "IMPOSSIBLE" if buffers else None,
        "policy_us": {name: statistics.median(p[2 * i] for p in policies)
                      for i, name in enumerate(("keep_all", "keep_tables", "release_all"))} if policies else {},
        "stack_peak_bytes": max(stacks, default=None),
        "stalled_s": round(stalled_s, 1),
    }
    failure = re.search(r"CPX: GAP8: (FAIL: [^\n]*)", raw)
    if failure:
        detail = failure.group(1).strip()
    elif not rounds:
        detail = "no rounds (see console.log)"
    elif report["window"] != window:
        detail = f"the deck ran {report['window']}, not {window}"
    elif not report["identical"]:
        detail = "the FC and cluster JPEGs differed"
    elif len(hashes) != 1:
        detail = f"the JPEG changed between rounds: {', '.join(hashes)}"
    elif report["overflows"]:
        detail = f"{report['overflows']} overflows"
    elif stalled_s > STALL_S:
        detail = f"no round for the last {stalled_s:.0f} s: the deck stopped"
    else:
        detail = None
    report["result"] = "fail" if detail else "pass"
    report["detail"] = detail or (f"{len(rounds)} rounds, {report['frames']} frames, "
                                  f"FC {report['fc_us']:.0f} µs, cluster {report['cluster_us']:.0f} µs "
                                  f"({report['fc_us'] / report['cluster_us']:.1f}×)")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", default="320x240", help="WxH, within the 320x240 frame")
    parser.add_argument("--defines", default="", help="extra -D flags for the encoder (--defines=-DX=0)")
    parser.add_argument("--seconds", type=float, default=60.0, help="how long to record")
    parser.add_argument("--listen", action="store_true", help="capture only: no build, no flash")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "drone.yaml",
                        help="the drone config; only radio.uri is used")
    args = parser.parse_args(argv)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    uri = load_config(args.config).radio.uri
    width, height = args.window.split("x")
    defines = f"-DIMG_W={width} -DIMG_H={height} {args.defines}".strip()

    if not args.listen:
        log(f"build {defines} (log: {out / 'build.log'})")
        try:
            image = build(defines, out / "build.log")
        except Exception as exc:
            log(f"build failed: {exc}")
            return 2
        pinned = out / "jpeg_cluster_bench.img"
        shutil.copy2(image, pinned)
        log(f"flash {pinned.stat().st_size} B and power-cycle")
        try:
            flash_and_boot(pinned, uri, log=lambda m: log(f"  {m}"))
        except Exception as exc:  # a failed flash must not end an unattended queue
            log(f"flash failed: {exc}")
            return 2

    console = out / "console.log"
    console.unlink(missing_ok=True)
    log(f"record the radio console for {args.seconds:.0f} s")
    stalled_s = capture(uri, console, args.seconds)
    report = judge(console.read_text(encoding="utf-8", errors="replace"), args.window, stalled_s)
    report.update(defines=defines, seconds=args.seconds)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    log(f"{report['result']}: {report['detail']}")
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
