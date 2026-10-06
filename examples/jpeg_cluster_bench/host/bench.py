#!/usr/bin/env python3
"""jpeg_cluster_bench: the GAP8 JPEG encoder held to the host's bytes, in GVSOC and on the AI-deck.

  python examples/jpeg_cluster_bench/host/bench.py                    # the common set (~3 min)
  python examples/jpeg_cluster_bench/host/bench.py --set exhaustive   # every case, the soak among them (~25 min)
  python examples/jpeg_cluster_bench/host/bench.py --target gvsoc     # the simulator only; no drone
  python examples/jpeg_cluster_bench/host/bench.py --list

First, on this computer (gcc): the reference encoder for each window
(host_bench.c: the reciprocal quantizer is exact, the N-way merge is
byte-identical to the serial encode, the JPEG decodes), and the frame corpus in
frames/ (profile_frames.c: the merge holds on every frame, and the worst frame
and segment fit gap8/main.c's buffers). Then each case of the set
(host/bench.yaml) in GVSOC, several containers at once, and on the deck one at
a time (host/run.py), after the idle controller is put on the STM32
(--keep-controller skips that). Every JPEG must equal the host's.

Everything lands in examples/jpeg_cluster_bench/artifacts/bench_<stamp>/: run.json,
results.jsonl (one line per case and platform, and the corpus), summary.md,
logs/<job>.log, host/, gvsoc/ and aideck/<case>/. The exit status is 0 when
every result passed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from cf_lab.bench import Job, idle_controller_job, run_jobs

HOST_DIR = Path(__file__).resolve().parent
BENCH_DIR = HOST_DIR.parent
ROOT = BENCH_DIR.parents[1]
APP_DIR = BENCH_DIR / "gap8"
JPEG = ROOT / "lib" / "gap8" / "jpeg"
GAP8_IMAGE = os.environ.get("AIDECK_IMAGE", "cf-lab/aideck-gap8:latest")
PY = sys.executable
MIN_PSNR_DB = 28.0  # well below what quality 50 gives; a floor for a broken bitstream
GVSOC_TIMEOUT_S = 900


def defines(case: dict) -> str:
    width, height = case["window"].split("x")
    return f"-DIMG_W={width} -DIMG_H={height} {case.get('defines', '')}".strip()


def last_line(text: str) -> str:
    return (text.strip().splitlines() or ["no output"])[-1]


def gcc(out: Path, sources: list[Path], includes: list[Path]) -> None:
    subprocess.run(["gcc", "-O2", "-std=c11", "-Wall", "-o", str(out),
                    *[f"-I{path}" for path in includes], *map(str, sources)], check=True)


def psnr_db(pgm: Path, jpeg: Path) -> float:
    source = cv2.imread(str(pgm), cv2.IMREAD_GRAYSCALE).astype(np.float64)
    decoded = cv2.imread(str(jpeg), cv2.IMREAD_GRAYSCALE).astype(np.float64)
    mse = float(np.mean((source - decoded) ** 2)) if source.shape == decoded.shape else math.nan
    return math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def host_reference(window: str, binary: Path, out: Path) -> dict:
    """Encode the window on the host; the JPEG every platform must reproduce."""
    out.mkdir(parents=True, exist_ok=True)
    width, height = window.split("x")
    proc = subprocess.run([str(binary), "--width", width, "--height", height,
                           "--pgm", "window.pgm", "--serial", "serial.jpg", "--merged", "merged.jpg"],
                          cwd=out, capture_output=True, text=True)
    (out / "host_bench.log").write_text(proc.stdout + proc.stderr)
    found = re.search(r"serial\s+:.*\((\d+) bytes\) hash ([0-9a-f]{8})", proc.stdout)
    row = {"platform": "host", "window": window, "log": str(out / "host_bench.log")}
    if proc.returncode or not found:
        return {**row, "result": "fail", "detail": last_line(proc.stdout + proc.stderr)}
    psnr = psnr_db(out / "window.pgm", out / "serial.jpg")
    detail = f"{found.group(1)} B, hash {found.group(2)}, {psnr:.2f} dB"
    return {**row, "result": "pass" if psnr >= MIN_PSNR_DB else "fail", "detail": detail,
            "bytes": int(found.group(1)), "hash": found.group(2), "psnr_db": round(psnr, 2),
            "jpeg": str(out / "serial.jpg")}


def corpus(binary: Path, out: Path) -> dict:
    """The frame corpus: the merge holds on every frame, and the worst case fits the app's buffers."""
    frames = sorted((BENCH_DIR / "frames").glob("*.pgm"))
    proc = subprocess.run([str(binary), *map(str, frames)], capture_output=True, text=True)
    (out / "corpus.log").write_text(proc.stdout + proc.stderr)
    main_c = (APP_DIR / "main.c").read_text()
    caps = {name: int(a) * int(b) for name, a, b in
            re.findall(r"#define (OUT_CAP|SEG_CAP) \((\d+) \* (\d+)\)", main_c)}
    frame = re.search(r"worst frame : (\S+) at (\d+) bytes", proc.stdout)
    segment = re.search(r"worst seg\s+: (\d+) bytes", proc.stdout)
    row = {"platform": "host", "case": "corpus", "frames": len(frames), "log": str(out / "corpus.log")}
    if proc.returncode or not (frame and segment and len(caps) == 2):
        return {**row, "result": "fail", "detail": last_line(proc.stdout + proc.stderr)}
    worst, seg = int(frame.group(2)), int(segment.group(1))
    detail = (f"{len(frames)} frames merge byte-identical; worst frame {frame.group(1)} {worst:,} B "
              f"of {caps['OUT_CAP']:,} ({caps['OUT_CAP'] / worst:.2f}×), worst segment {seg:,} B "
              f"of {caps['SEG_CAP']:,} ({caps['SEG_CAP'] / seg:.2f}×)")
    fits = worst < caps["OUT_CAP"] and seg < caps["SEG_CAP"]
    return {**row, "result": "pass" if fits else "fail", "detail": detail,
            "worst_frame_bytes": worst, "worst_segment_bytes": seg, **caps}


def gvsoc_job(name: str, case: dict, out: Path) -> Job:
    """One frame in GVSOC. The app hex-dumps its JPEG and prints #DONE; reading
    stops there, because the simulated ROM reboots the chip when the app exits.
    The container is named and removed: gvsoc keeps writing after the client
    detaches, and a leftover's output would mix into the next run's."""
    rel = lambda path: "/module/" + str(path.relative_to(ROOT))  # noqa: E731
    container = f"jpeg_cluster_bench_gvsoc_{os.getpid()}_{name}"
    inner = ("source /gap_sdk/configs/ai_deck.sh >/dev/null && "
             # Line-buffered, or a piped simulator stalls before #DONE.
             f"timeout {GVSOC_TIMEOUT_S} stdbuf -oL -eL make clean all run platform=gvsoc io=host "
             f"{shlex.quote('BENCH_DEFINES=' + defines(case))} "
             f"BUILDDIR={rel(out / 'gvsoc' / name / 'build')} 2>&1 | sed -u '/^#DONE/q'")
    docker = ["docker", "run", "--rm", "--name", container, "--platform", "linux/amd64",
              "-v", f"{ROOT}:/module", "-w", rel(APP_DIR), GAP8_IMAGE, "/bin/bash", "-lc", inner]
    return Job(f"gvsoc-{name}", [["bash", "-c", f"{shlex.join(docker)}; status=$?; "
                                               f"docker rm -f {container} >/dev/null 2>&1; exit $status"]])


def judge_gvsoc(row: dict, out: Path, reference: dict) -> dict:
    path = out / "logs" / f"gvsoc-{row['case']}.log"
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    stats = dict(re.findall(r"^#STAT (\w+) (\S+)$", text, re.M))
    dump = re.search(r"#JPEG-BEGIN (\d+)\n(.*?)#JPEG-END", text, re.S)
    data = bytes.fromhex(re.sub(r"[^0-9a-fA-F]", "", dump.group(2))) if dump else b""
    if data:
        (out / "gvsoc" / row["case"] / "gvsoc.jpg").write_bytes(data)
    if "#DONE" not in text:
        detail = "never reached #DONE (see the job log)"
    elif stats.get("identical") != "1":
        detail = "the FC and cluster JPEGs differed"
    elif not dump or len(data) != int(dump.group(1)):
        detail = "no complete JPEG in the output"
    elif data != Path(reference["jpeg"]).read_bytes():
        detail = f"hash {stats.get('hash')}, not the host's {reference['hash']}"
    else:
        return {**row, "result": "pass", "detail": f"{len(data)} B, identical to the host's",
                "bytes": len(data), "hash": stats.get("hash")}
    return {**row, "result": "fail", "detail": detail}


def judge_aideck(row: dict, out: Path, reference: dict) -> dict:
    path = out / "aideck" / row["case"] / "report.json"
    if not path.is_file():
        return {**row, "result": "fail", "detail": "no report (see the job log)"}
    report = json.loads(path.read_text())
    keep = {key: report[key] for key in ("rounds", "frames", "fc_us", "cluster_us", "hash", "policy_us",
                                         "stack_peak_bytes")}
    if report["result"] == "pass" and report["hash"] != reference["hash"]:
        return {**row, **keep, "result": "fail", "detail": f"hash {report['hash']}, not the host's {reference['hash']}"}
    return {**row, **keep, "result": report["result"], "detail": report["detail"]}


def plan(spec: dict, cases: dict, out: Path, target: str, keep_controller: bool) -> tuple[list[Job], dict]:
    """The set's jobs, and for each job the rows it produces."""
    jobs, rows = [], {}
    if target in ("gvsoc", "both"):
        for name in spec.get("gvsoc", []):
            job = gvsoc_job(name, cases[name], out)
            jobs.append(job)
            rows[job.name] = [{"platform": "gvsoc", "case": name}]
    if target in ("aideck", "both") and spec.get("aideck"):
        if not keep_controller:
            jobs.append(idle_controller_job())
        for name in spec["aideck"]:
            case = cases[name]
            job = Job(f"aideck-{name}", [[PY, str(HOST_DIR / "run.py"), "--out", str(out / "aideck" / name),
                                          "--window", case["window"], f"--defines={case.get('defines', '')}",
                                          "--seconds", str(case.get("seconds", 60))]], radio=True)
            jobs.append(job)
            rows[job.name] = [{"platform": "aideck", "case": name}]
    return jobs, rows


def write_summary(rows: list[dict], cases: dict, path: Path) -> None:
    by: dict[str, dict] = {}
    for row in rows:
        by.setdefault(row["case"], {})[row["platform"]] = row

    def cell(row: dict | None) -> str:
        return "" if row is None else f"{row['result']}: {row['detail']}"

    lines = ["| case | build | host | GVSOC | AI-deck |", "| --- | --- | --- | --- | --- |"]
    for name, cells in by.items():
        if name != "corpus":
            lines.append(f"| {name} | {defines(cases[name])} | {cell(cells.get('host'))} | "
                         f"{cell(cells.get('gvsoc'))} | {cell(cells.get('aideck'))} |")
    corpus_row = by.get("corpus", {}).get("host")
    corpus_line = f"\n\nFrame corpus: {cell(corpus_row)}\n" if corpus_row else "\n"
    path.write_text("# jpeg_cluster_bench\n\n" + "\n".join(lines) + corpus_line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", type=Path, default=HOST_DIR / "bench.yaml")
    parser.add_argument("--set", default="common", help="common (the default) or exhaustive")
    parser.add_argument("--target", choices=("gvsoc", "aideck", "both"), default="both")
    parser.add_argument("--list", action="store_true", help="list the set's jobs, then exit")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                        help="GVSOC containers at once")
    parser.add_argument("--keep-controller", action="store_true",
                        help="skip flashing the idle controller to the STM32 first")
    args = parser.parse_args(argv)

    bench = yaml.safe_load(args.bench.read_text())
    cases, sets = bench["cases"], bench["sets"]
    if args.set not in sets:
        parser.error(f"no set {args.set!r}; choose from {sorted(sets)}")
    out = BENCH_DIR / "artifacts" / f"bench_{datetime.datetime.now():%Y%m%dT%H%M%S}"
    jobs, planned = plan(sets[args.set], cases, out, args.target, args.keep_controller)
    if args.list:
        for job in jobs:
            print(f"{'radio' if job.radio else 'gvsoc':6} {job.name}")
        print(f"{len(jobs)} jobs, plus the host reference and the frame corpus")
        return 0

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    out.mkdir(parents=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=ROOT).stdout.strip()
    (out / "run.json").write_text(json.dumps({
        "set": args.set, "target": args.target, "jobs": [job.name for job in jobs], "git": git,
        "started": datetime.datetime.now().isoformat(timespec="seconds")}, indent=1) + "\n")
    log(f"{len(jobs)} jobs -> {out}")
    rows: list[dict] = []

    def record(row: dict) -> None:
        rows.append(row)
        with (out / "results.jsonl").open("a") as results:
            results.write(json.dumps(row) + "\n")
        log(f"  {row['result'].upper():4}  {row['platform']:6} {row['case']}: {row['detail']}")

    host = out / "host"
    host.mkdir()
    # Made here, so the containers do not make them as root.
    for job_rows in planned.values():
        for row in job_rows:
            (out / row["platform"] / row["case"]).mkdir(parents=True, exist_ok=True)
    gcc(host / "host_bench", [HOST_DIR / "host_bench.c", JPEG / "src" / "jpeg_gray.c", JPEG / "src" / "jpeg_dct.c",
                              APP_DIR / "realimg.c"], [JPEG / "inc", APP_DIR])
    gcc(host / "profile_frames", [HOST_DIR / "profile_frames.c", JPEG / "src" / "jpeg_gray.c",
                                  JPEG / "src" / "jpeg_dct.c"], [JPEG / "inc"])
    record(corpus(host / "profile_frames", host))
    used = {row["case"] for job_rows in planned.values() for row in job_rows}
    references = {}
    for name in sorted(used):
        window = cases[name]["window"]
        if window not in references:
            references[window] = host_reference(window, host / "host_bench", host / window)
        record({**references[window], "case": name})

    def done(job: Job) -> None:
        if job.name not in planned:
            log(f"  {'done' if job.ok else 'FAILED'}  {job.name} ({job.minutes:.1f} min)")
            return
        for row in planned[job.name]:
            reference = references[cases[row["case"]]["window"]]
            judge = judge_gvsoc if row["platform"] == "gvsoc" else judge_aideck
            judged = judge(row, out, reference) if reference["result"] == "pass" else \
                {**row, "result": "fail", "detail": "no host reference to compare with"}
            record({**judged, "job": job.name, "minutes": round(job.minutes, 1)})

    try:
        run_jobs(jobs, out / "logs", args.jobs, ROOT, done)
    except KeyboardInterrupt:
        log("Interrupted: GVSOC containers may still be running (docker ps).")
    write_summary(rows, cases, out / "summary.md")
    failed = sum(row["result"] != "pass" for row in rows)
    log(f"{len(rows) - failed} passed, {failed} not. Summary: {out / 'summary.md'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
