#!/usr/bin/env python3
"""nn_bench: networks through DORY, checked layer by layer, in GVSOC and on the AI-deck.

  python examples/nn_bench/host/bench.py                    # the common set
  python examples/nn_bench/host/bench.py --set exhaustive   # every network, memory model and runner
  python examples/nn_bench/host/bench.py --target gvsoc     # the simulator only; no drone
  python examples/nn_bench/host/bench.py --list

A set (host/bench.yaml) names networks, DORY memory models and runners for GVSOC
and for the deck. Every point (a network built for one memory model) is generated
and code-generated once, then GVSOC simulates each point's runners, several
points at once, while the deck runs one at a time over the radio, after the idle
controller is put on the STM32 (--keep-controller skips that).

Everything lands in examples/nn_bench/artifacts/bench_<stamp>/: run.json,
results.jsonl (one line per point, runner and platform), summary.md, logs/, and
the points under points/. The exit status is 0 when every result passed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

HOST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HOST_DIR))

import networks  # noqa: E402
import point as points  # noqa: E402
from cf_lab.bench import Job, idle_controller_job, run_jobs  # noqa: E402

PY = sys.executable
ROOT = points.REPO


def expand(spec: dict, platform: str) -> list[dict]:
    """One row per platform, network, memory model and runner, in set order."""
    rows = {}
    for where in ("gvsoc", "aideck"):
        if platform not in (where, "both"):
            continue
        for entry in spec.get(where, []):
            for network in networks.names(entry["networks"], entry.get("inputs", networks.INPUTS)):
                for memory in entry["memory"]:
                    for runner in entry.get("runners", points.RUNNERS):
                        rows.setdefault((where, network, memory, runner), {
                            "platform": where, "network": network, "memory": memory, "runner": runner,
                            "timeout_s": entry.get("timeout_s", 1800), "seconds": entry.get("seconds", 180)})
    return list(rows.values())


def plan(rows: list[dict], out: Path, keep_controller: bool) -> tuple[list[Job], dict[str, list[dict]]]:
    """The jobs, and the rows each job settles."""
    jobs, settles = [], {}
    gvsoc: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["platform"] == "gvsoc":
            gvsoc.setdefault((row["network"], row["memory"]), []).append(row)
    for (network, memory), group in gvsoc.items():
        job = Job(f"gvsoc-{network}-{memory}", [[
            PY, str(HOST_DIR / "point.py"), network, "--memory", memory, "--out", str(row_dir(out, group[0])),
            "--prepared", "--gvsoc", "--runner", *[row["runner"] for row in group],
            "--timeout", str(group[0]["timeout_s"])]])
        jobs.append(job)
        settles[job.name] = group
    deck = [row for row in rows if row["platform"] == "aideck"]
    if deck and not keep_controller:
        jobs.append(idle_controller_job())
    for row in deck:
        job = Job(f"aideck-{row['network']}-{row['memory']}-{row['runner']}", [[
            PY, str(HOST_DIR / "run.py"), str(row_dir(out, row)), "--runner", row["runner"],
            "--seconds", str(row["seconds"])]], radio=True)
        jobs.append(job)
        settles[job.name] = [row]
    return jobs, settles


def row_dir(out: Path, row: dict) -> Path:
    return out / "points" / f"{row['network']}-{row['memory']}"


def verdict(out: Path, row: dict) -> dict:
    where = row_dir(out, row)
    path = where / "gvsoc" / f"{row['runner']}.json" if row["platform"] == "gvsoc" \
        else where / "aideck" / row["runner"] / "report.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"result": "fail", "detail": "no verdict (see the job log)", "median_us": None}


def write_summary(rows: list[dict], hosts: dict, path: Path) -> None:
    columns = [(p, r) for p in ("gvsoc", "aideck") for r in points.RUNNERS
               if any(row["platform"] == p and row["runner"] == r for row in rows)]
    cells = {(row["network"], row["memory"], row["platform"], row["runner"]): row for row in rows}

    def cell(network: str, memory: str, platform: str, runner: str) -> str:
        row = cells.get((network, memory, platform, runner))
        if row is None:
            return ""
        if row["result"] != "pass":
            return f"**{row['result']}**: {row['detail']}"
        text = f"{row['median_us']:,.0f}"
        base = cells.get((network, memory, platform, "pulp_nn"))
        if runner != "pulp_nn" and base and base["result"] == "pass" and row["median_us"]:
            text += f" ({base['median_us'] / row['median_us']:.2f}x)"
        return text

    head = " | ".join(f"{'GVSOC' if p == 'gvsoc' else 'deck'} {r} µs" for p, r in columns)
    lines = [f"| network | memory | kMAC | weights KB | {head} |", "| --- | --- | ---: | ---: |" + " ---: |" * len(columns)]
    for (network, memory), host in hosts.items():
        lines.append(f"| {network} | {memory} | {host['macs'] / 1e3:,.0f} | {host['weight_bytes'] / 1024:,.0f} | "
                     + " | ".join(cell(network, memory, p, r) for p, r in columns) + " |")
    path.write_text("# nn_bench\n\nMedian inference time; (Nx) is the speedup over pulp_nn.\n\n"
                    + "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", default="common", help="common (the default) or exhaustive")
    parser.add_argument("--target", choices=("gvsoc", "aideck", "both"), default="both")
    parser.add_argument("--list", action="store_true", help="list the set's jobs, then exit")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="GVSOC containers at once")
    parser.add_argument("--keep-controller", action="store_true", help="skip putting the idle controller on the STM32")
    args = parser.parse_args(argv)

    sets = yaml.safe_load((HOST_DIR / "bench.yaml").read_text())["sets"]
    if args.set not in sets:
        parser.error(f"no set {args.set!r}; choose from {sorted(sets)}")
    out = points.BENCH_DIR / "artifacts" / f"bench_{datetime.datetime.now():%Y%m%dT%H%M%S}"
    rows = expand(sets[args.set], args.target)
    if args.list:
        for job in plan(rows, out, args.keep_controller)[0]:
            print(f"{'radio' if job.radio else 'gvsoc':6} {job.name}")
        return 0

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    out.mkdir(parents=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    (out / "run.json").write_text(json.dumps({"set": args.set, "target": args.target, "git": git,
                                              "started": datetime.datetime.now().isoformat(timespec="seconds")}) + "\n")
    hosts, results = {}, []

    def settle(row: dict, found: dict) -> None:
        row = {key: row[key] for key in ("platform", "network", "memory", "runner")} | found
        results.append(row)
        with (out / "results.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
        us = f", {row['median_us']:,.0f} µs" if row.get("median_us") else ""
        log(f"  {row['result'].upper():4}  {row['platform']:6} {row['network']} {row['memory']} "
            f"{row['runner']}: {row['detail']}{us}")

    for network, memory in dict.fromkeys((row["network"], row["memory"]) for row in rows):
        log(f"prepare {network} {memory}")
        hosts[network, memory] = points.prepare(network, memory, out / "points" / f"{network}-{memory}")
    # A fused row only where an executor runs that network on that memory model.
    rows = [row for row in rows if row["runner"] != "fused" or hosts[row["network"], row["memory"]].get("fused")]
    ready = [row for row in rows if not hosts[row["network"], row["memory"]]["failures"]]
    for row in rows:
        if row not in ready:
            settle(row, {"result": "fail", "detail": "; ".join(hosts[row["network"], row["memory"]]["failures"]),
                         "median_us": None})
    jobs, settles = plan(ready, out, args.keep_controller)
    log(f"{len(jobs)} jobs -> {out}")

    def done(job: Job) -> None:
        for row in settles.get(job.name, []):
            settle(row, verdict(out, row))
        if job.name not in settles:
            log(f"  {'done' if job.ok else 'FAILED'}  {job.name} ({job.minutes:.1f} min)")

    try:
        run_jobs(jobs, out / "logs", args.jobs, ROOT, done)
    except KeyboardInterrupt:
        log("Interrupted: GVSOC containers may still be running (docker ps).")
    write_summary(results, hosts, out / "summary.md")
    failed = sum(row["result"] != "pass" for row in results)
    log(f"{len(results) - failed} passed, {failed} failed. Summary: {out / 'summary.md'}")
    return 0 if results and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
