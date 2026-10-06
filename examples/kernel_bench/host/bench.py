#!/usr/bin/env python3
"""kernel_bench: every kernel suite in GVSOC and on the AI-deck.

  python examples/kernel_bench/host/bench.py                    # the common set (~5 min)
  python examples/kernel_bench/host/bench.py --set exhaustive   # every suite (~1 h)
  python examples/kernel_bench/host/bench.py --target gvsoc     # the simulator only; no drone
  python examples/kernel_bench/host/bench.py --list

A set (host/bench.yaml) gives each configuration its suites. In GVSOC each suite
runs in its own Docker container, several at once; on
the deck each configuration's suites are built into one image and captured over
the radio, one image at a time, after the idle controller is put on the STM32
(--keep-controller skips that).

Everything lands in examples/kernel_bench/artifacts/bench_<stamp>/: run.json,
results.jsonl (one line per suite or image), summary.md, RESULTS.md
(every case, kernel and variant, GVSOC beside the deck; host/results.py),
logs/<job>.log, and the raw output under gvsoc-<configuration>/ and
aideck-<configuration>/. The exit status is 0 when every suite passed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

HOST_DIR = Path(__file__).resolve().parent
BENCH_DIR = HOST_DIR.parent
ROOT = BENCH_DIR.parents[1]
sys.path.insert(0, str(HOST_DIR))

import sweep  # noqa: E402
from cf_lab.bench import Job, idle_controller_job, run_jobs  # noqa: E402

PY = sys.executable


def folder(platform: str, configuration: str) -> str:
    """gvsoc-s8bias, aideck-i32bias, ...: the layout host/results.py reads."""
    return f"{platform}-{configuration}"


def plan(bench: dict, spec: dict, out: Path, target: str, keep_controller: bool) -> tuple[list[Job], dict]:
    """The set's jobs, and what each job's result row is about."""
    matrix = sweep.load_config()
    jobs, rows = [], {}
    for name, suites in spec.items():
        configuration = bench["configurations"][name]
        for platform in ("gvsoc", "aideck"):
            if target not in (platform, "both") or not suites.get(platform):
                continue
            make = list(configuration.get("make", []))
            make_args = [arg for var in make for arg in ("--make-var", var)]
            results = out / folder(platform, name)
            if platform == "gvsoc":
                for suite in sweep.resolve_suites(matrix, suites["gvsoc"]):
                    job = Job(f"{folder('gvsoc', name)}-{suite}", [
                        [PY, str(HOST_DIR / "sweep.py"), "--suite", suite, "--gvsoc",
                         "--out", str(results), *make_args],
                        [PY, str(HOST_DIR / "report.py"), str(results / suite / "gvsoc.log"),
                         "--out", str(results / suite / "report")]])
                    jobs.append(job)
                    rows[job.name] = {"kind": "suite", "platform": "gvsoc", "configuration": name,
                                      "name": suite, "report": results / suite / "report" / "summary.json"}
            else:
                job = Job(folder("aideck", name), [
                    [PY, str(HOST_DIR / "run.py"), "--suite", ",".join(suites["aideck"]),
                     "--seconds", "1800", "--out", str(results), *make_args]], radio=True)
                jobs.append(job)
                rows[job.name] = {"kind": "image", "platform": "aideck", "configuration": name,
                                  "name": ",".join(suites["aideck"]),
                                  "report": results / "report" / "summary.json"}
    if not keep_controller and any(job.radio for job in jobs):
        jobs.insert(0, idle_controller_job())
    return jobs, rows


def judge_report(row: dict) -> dict:
    """A suite's or an image's guest verdict: pass, or incomplete when a kernel
    declined cases it does not support (a skip) and every other result was exact."""
    report = row.pop("report")
    if not report.is_file():
        return {**row, "result": "fail", "detail": "no report (see the job log)"}
    summary = json.loads(report.read_text())
    results = summary.get("results", [])
    failed = [r for r in results if r.get("status") not in ("pass", "skip") or r.get("exact") == 0]
    status = summary.get("done", {}).get("status", "fail")
    skipped = sum(r.get("status") == "skip" for r in results)
    return {**row, "result": "fail" if failed else status,
            "detail": f"{len(results) - skipped} results" + (f", {skipped} skipped" if skipped else "")
            + (f", {len(failed)} failed" if failed else "")}


def write_summary(rows: list[dict], path: Path) -> None:
    lines = ["# kernel_bench", "",
             "| what | configuration / target | GVSOC or AI-deck | result | detail |",
             "| --- | --- | --- | --- | --- |"]
    for row in rows:
        where = row.get("configuration") or row.get("target", "")
        lines.append(f"| {row['name']} | {where} | {row['platform']} | {row['result']} | {row['detail']} |")
    path.write_text("\n".join(lines) + "\n")


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
    if args.set not in bench["sets"]:
        parser.error(f"no set {args.set!r}; choose from {sorted(bench['sets'])}")
    out = BENCH_DIR / "artifacts" / f"bench_{datetime.datetime.now():%Y%m%dT%H%M%S}"
    jobs, planned = plan(bench, bench["sets"][args.set], out, args.target, args.keep_controller)
    if args.list:
        for job in jobs:
            print(f"{'radio' if job.radio else 'gvsoc':6} {job.name}")
        print(f"{len(jobs)} jobs")
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

    def done(job: Job) -> None:
        if job.name not in planned:
            log(f"  {'done' if job.ok else 'FAILED'}  {job.name} ({job.minutes:.1f} min)")
            return
        row = dict(planned[job.name], job=job.name)
        found = [judge_report(row)]
        for row in found:
            row["passed"] = row["result"] in ("pass", "incomplete")
            rows.append(row)
            with (out / "results.jsonl").open("a") as results:
                results.write(json.dumps(row) + "\n")
            where = row.get("configuration") or row.get("target", "")
            log(f"  {'PASS' if row['passed'] else 'FAIL'}  {row['platform']:6} {row['name']} "
                f"({where}): {row['result']}, {row['detail']}")

    try:
        run_jobs(jobs, out / "logs", args.jobs, ROOT, done)
    except KeyboardInterrupt:
        log("Interrupted: GVSOC containers may still be running (docker ps).")
    write_summary(rows, out / "summary.md")
    if any(r["platform"] == "aideck" for r in rows) and any(r["platform"] == "gvsoc" for r in rows):
        subprocess.run([PY, str(HOST_DIR / "results.py"), str(out)], cwd=ROOT, capture_output=True)
    failed = sum(not row["passed"] for row in rows)
    log(f"{len(rows) - failed} passed, {failed} failed. Summary: {out / 'summary.md'}")
    return 0 if rows and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
