#!/usr/bin/env python3
"""hyperbus_bench: the AI-deck's HyperFlash, flash filesystem, HyperRAM and L3 weight load, checked on the deck.

  python examples/hyperbus_bench/host/bench.py                         # ~6 min
  python examples/hyperbus_bench/host/bench.py --plan 128:burst:150 --iterations 20
  python examples/hyperbus_bench/host/bench.py --no-load                # the unit tests only (~2 min)
  python examples/hyperbus_bench/host/bench.py --keep-controller

A run puts the idle controller on the STM32 (--keep-controller skips that),
writes the load file into the app's readfs, builds the GAP8 app, power-cycles
the drone (a JTAG session leaves radio flashing slow until one), flashes it,
power-cycles again, and records the radio console until the app's report is
complete. The unit tests pass when the HyperFlash reads inside the image return
its bytes as flashed, the filesystem mounts and hb_probe.txt reads back intact,
and every HyperRAM pattern is exact. The load test (host/bench.yaml) passes when
every step's loads complete and every byte in HyperRAM is right.

The GAP8 app never restarts itself: if the chip resets alone, the Bitcraze
bootloader waits for a jump command the ESP32 sends only when it starts. So a
GAP8 that falls silent before its report is complete has reset (or hung); the
report names the last line it printed.

Everything lands in examples/hyperbus_bench/artifacts/bench_<stamp>/: run.json,
results.jsonl, summary.md, build.log, the image, console.log and report.json.
The exit status is 0 for a pass. The radio URI comes from configs/drone.yaml.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import yaml

from cf_lab.aideck.flash import repair_gap8_image
from cf_lab.backends.crazyflie import power_cycle
from cf_lab.bench import RadioConsole, flash_and_boot, flash_idle_controller
from cf_lab.config import load_config

HOST_DIR = Path(__file__).resolve().parent
BENCH_DIR = HOST_DIR.parent
APP_DIR = BENCH_DIR / "gap8"
ROOT = BENCH_DIR.parents[1]
PROBE = APP_DIR / "readfs" / "hb_probe.txt"
LOAD_FILE = APP_DIR / "readfs" / "hb_load.bin"
IMAGE_AT = 0x40000  # where the radio flasher writes the image
DONE = "HYPERBUS BENCH done"
SILENT_S = 20.0  # no GAP8 line for this long, before the report is done: it reset or hung
LOAD = re.compile(r"LOAD step=(\d+) chunk=(\d+) spi=(\w+) fc=(\d+) it=(\d+) load=(-?\d+) load_us=(\d+) "
                  r"verify_us=(\d+) bad=(\d+) flash_bad=(\d+) first=(-?\d+)")
STUCK = re.compile(r"LOAD step=(\d+) STUCK it=(\d+) at=(\d+)")
TILE = re.compile(r"TILE step=(\d+) cl=(\d+) spi=(\w+) it=(\d+) us=(\d+) bad=(-?\d+) first=(-?\d+)")


def write_load_file(kb: int, seed: int) -> None:
    """The load file: a xorshift32 stream, which the GAP8 regenerates to check every byte."""
    x, words = seed, []
    for _ in range(kb * 256):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        words.append(x)
    LOAD_FILE.write_bytes(struct.pack(f"<{len(words)}I", *words))


def build(make: str, build_log: Path) -> Path:
    from cf_lab.firmware import build_gap8, gap8_artifact

    with build_log.open("w") as out:
        build_gap8(APP_DIR, f"clean build image {make}", build_image=False, output=out)
    return gap8_artifact(APP_DIR)


def capture(uri: str, path: Path, timeout_s: float, log) -> tuple[str, bool]:
    """Record the radio console until the report is complete; return it and whether the GAP8 fell silent."""
    console = RadioConsole(uri, path)
    started = time.monotonic()
    last_gap8, seen, silent = started, 0, False
    try:
        while time.monotonic() - started < timeout_s:
            time.sleep(1.0)
            text = path.read_text(errors="replace")
            if DONE in text:
                break
            count = text.count("GAP8:")
            if count > seen:
                seen, last_gap8 = count, time.monotonic()
            elif seen and time.monotonic() - last_gap8 > SILENT_S:
                silent = True
                break
            if int(time.monotonic() - started) % 30 == 0:
                log(f"  {time.monotonic() - started:.0f} s, {len(LOAD.findall(text))} loads reported")
    finally:
        console.close()
    return path.read_text(errors="replace"), silent


def report_lines(raw: str) -> list[str]:
    """The report, in first-seen order: the app prints it once, then forever."""
    # The STM32 prefixes every CPX console packet, so a line can carry the
    # prefix mid-line.
    text = raw.replace("CPX: GAP8: ", "").replace("CPX: GAP8:", "")
    seen: dict[str, None] = {}
    for line in text.splitlines():
        found = re.search(r"HB[:|] (.*)$", line)
        if found:
            seen.setdefault(found.group(1).strip())
    return list(seen)


def judge_unit(lines: list[str], flashed: bytes) -> tuple[list[dict], str | None]:
    """The unit tests against the image as flashed and the probe file; (reads, failure)."""
    reads = []
    for line in lines:
        found = re.fullmatch(r"FLASH @0x([0-9a-f]+): ((?:[0-9a-f]{2} ?){16})", line)
        if found:
            address, data = int(found.group(1), 16), bytes.fromhex(found.group(2))
            offset = address - IMAGE_AT
            expected = flashed[offset:offset + 16] if 0 <= offset <= len(flashed) - 16 else None
            reads.append({"address": hex(address), "read": data.hex(" "),
                          "image": expected.hex(" ") if expected is not None else None,
                          "match": None if expected is None else data == expected})
    probe = PROBE.read_bytes()
    size = next((int(m.group(1)) for line in lines
                 if (m := re.fullmatch(r"OPEN hb_probe\.txt: OK \(size=(\d+)\)", line))), None)
    text = next((m.group(1) for line in lines if (m := re.fullmatch(r"READ 32B: n=\d+ '(.*)'", line))), None)
    failures = [line for line in lines if "FAIL" in line]
    wrong = [read for read in reads if read["match"] is False]
    if failures:
        return reads, failures[0]
    if wrong or not any(read["match"] for read in reads):
        return reads, (f"flash @{wrong[0]['address']} read {wrong[0]['read']}, the image has {wrong[0]['image']}"
                       if wrong else "no flash read inside the image")
    if size != len(probe) or text != probe[:32].decode():
        return reads, f"hb_probe.txt read back as {size} B {text!r}"
    return reads, None


def judge_load(raw: str, plan: list[str], iterations: int) -> tuple[list[dict], str | None]:
    """Per-step load results; (steps, failure)."""
    loads = [dict(zip(("step", "chunk", "spi", "fc", "it", "load", "load_us", "verify_us", "bad", "flash_bad",
                       "first"), map(lambda v: v if v.isalpha() else int(v), m.groups())))
             for m in LOAD.finditer(raw.replace("CPX: GAP8: ", ""))]
    stuck = {int(m.group(1)): (int(m.group(2)), int(m.group(3))) for m in STUCK.finditer(raw.replace("CPX: GAP8: ", ""))}
    steps, failure = [], None
    for index, spec in enumerate(plan):
        runs = [row for row in loads if row["step"] == index]
        bad = sum(row["bad"] + (row["load"] != 0) for row in runs)
        if index in stuck and failure is None:
            it, at = stuck[index]
            failure = f"step {index} ({spec}): load {it} hung at byte {at} after {len(runs)} good loads"
        steps.append({"step": index, "spec": spec, "loads": len(runs), "bad": bad,
                      "flash_bad": sum(row["flash_bad"] for row in runs),
                      "first_bad": next((row["first"] for row in runs if row["bad"]), None),
                      "load_us": sorted(row["load_us"] for row in runs)[len(runs) // 2] if runs else None,
                      "verify_us": sorted(row["verify_us"] for row in runs)[len(runs) // 2] if runs else None})
        if failure is None and (bad or len(runs) < iterations):
            failure = (f"step {index} ({spec}): {bad} bad of {len(runs)} loads" if bad
                       else f"step {index} ({spec}): {len(runs)} of {iterations} loads")
    return steps, failure


def judge_tiles(raw: str, plan: list[str], iterations: int) -> tuple[list[dict], str | None]:
    """Per-step tiled-layer results; (steps, failure)."""
    passes = [dict(zip(("step", "cl", "spi", "it", "us", "bad", "first"),
                       map(lambda v: v if v.isalpha() else int(v), m.groups())))
              for m in TILE.finditer(raw.replace("CPX: GAP8: ", ""))]
    steps, failure = [], None
    for index, spec in enumerate(plan):
        runs = [row for row in passes if row["step"] == index]
        bad = sum(abs(row["bad"]) for row in runs)
        steps.append({"step": index, "spec": spec, "passes": len(runs), "bad": bad,
                      "us": sorted(row["us"] for row in runs)[len(runs) // 2] if runs else None})
        if failure is None and (bad or len(runs) < iterations):
            failure = (f"tiles step {index} ({spec}): {bad} bad tiles in {len(runs)} passes" if bad
                       else f"tiles step {index} ({spec}): {len(runs)} of {iterations} passes")
    return steps, failure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", type=Path, default=HOST_DIR / "bench.yaml")
    parser.add_argument("--plan", help="load steps, comma-separated chunk:spi:fc_mhz (default: bench.yaml)")
    parser.add_argument("--iterations", type=int, help="loads per step (default: bench.yaml)")
    parser.add_argument("--tile-plan", help="tiled-layer steps, comma-separated cl_mhz:spi (default: bench.yaml)")
    parser.add_argument("--no-load", action="store_true", help="the unit tests only")
    parser.add_argument("--keep-controller", action="store_true",
                        help="skip flashing the idle controller to the STM32 first")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "drone.yaml",
                        help="the drone config; only radio.uri is used")
    args = parser.parse_args(argv)

    settings = yaml.safe_load(args.bench.read_text())
    load, tiles = settings["load"], settings["tiles"]
    kb = 0 if args.no_load else load["kb"]
    plan = args.plan.split(",") if args.plan else load["plan"]
    iterations = args.iterations or load["iterations"]
    tile_plan = [] if not kb else args.tile_plan.split(",") if args.tile_plan else tiles["plan"]

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    out = BENCH_DIR / "artifacts" / f"bench_{datetime.datetime.now():%Y%m%dT%H%M%S}"
    out.mkdir(parents=True)
    uri = load_config(args.config).radio.uri
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=ROOT).stdout.strip()
    (out / "run.json").write_text(json.dumps({
        "uri": uri, "git": git, "load_kb": kb, "plan": plan, "iterations": iterations, "tile_plan": tile_plan,
        "started": datetime.datetime.now().isoformat(timespec="seconds")}, indent=1) + "\n")
    log(f"-> {out}")

    row: dict = {"load_kb": kb, "plan": plan if kb else []}
    try:
        if not args.keep_controller:
            with (out / "controller.log").open("w") as controller_log:
                flash_idle_controller(uri, controller_log, log)
        make = f"HB_LOAD_KB={kb}"
        if kb:
            write_load_file(kb, load["seed"])
            make += (f" HB_LOAD_SEED={load['seed']:#x} HB_LOAD_ITERS={iterations} "
                     + shlex.quote("HB_PLAN=" + ",".join(plan)) + f" HB_TILE_ITERS={tiles['iterations']} "
                     + shlex.quote("HB_TILE_PLAN=" + ",".join(tile_plan)))
        log(f"build {make}")
        image = out / "hyperbus_bench.img"
        shutil.copy2(build(make, out / "build.log"), image)
        flashed, dropped = repair_gap8_image(image.read_bytes())
        row["empty_areas_dropped"] = dropped
        # A JTAG session leaves GAP8 flashing ~10x slower until a power cycle, and the
        # flasher's first connect times out if the STM32 has not finished booting.
        power_cycle(uri, log=log)
        time.sleep(6)
        log(f"flash {image.stat().st_size} B and power-cycle")
        flash_and_boot(image, uri, log=lambda m: log(f"  {m}"))
        timeout = 60 + (8 * iterations * len(plan) + 2 * tiles["iterations"] * len(tile_plan) if kb else 0)
        raw, silent = capture(uri, out / "console.log", timeout, log)
        lines = report_lines(raw)
        reads, unit_failure = judge_unit(lines, flashed)
        steps, load_failure = judge_load(raw, plan, iterations) if kb else ([], None)
        tile_steps, tile_failure = judge_tiles(raw, tile_plan, tiles["iterations"]) if tile_plan else ([], None)
        row.update(flash_reads=reads, steps=steps, tile_steps=tile_steps)
        if silent or DONE not in "\n".join(lines):
            detail = f"the GAP8 fell silent (reset or hang) after: {lines[-1] if lines else 'nothing'}"
        else:
            detail = unit_failure or load_failure or tile_failure
        row["result"] = "fail" if detail else "pass"
        row["detail"] = detail or "flash reads match, mounted, probe intact, HyperRAM exact" + (
            f"; {sum(s['loads'] for s in steps)} loads exact" if kb else "") + (
            f"; {sum(s['passes'] for s in tile_steps)} tiled passes exact" if tile_plan else "")
        (out / "report.json").write_text(json.dumps({**row, "lines": lines}, indent=1) + "\n")
    except Exception as exc:
        row.update(result="fail", detail=f"{type(exc).__name__}: {exc}")
    (out / "results.jsonl").write_text(json.dumps(row) + "\n")
    table = ["| step | chunk:spi:fc_mhz | loads | bad chunks | of them flash | median load µs | median verify µs |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    table += [f"| {s['step']} | {s['spec']} | {s['loads']} | {s['bad']} | {s['flash_bad']} | {s['load_us']} | "
              f"{s['verify_us']} |" for s in row.get("steps", [])]
    (out / "summary.md").write_text(
        f"# hyperbus_bench\n\n{row['result']}: {row['detail']}\n\n"
        f"Empty areas the flasher dropped from the image: {row.get('empty_areas_dropped', '')}\n\n"
        + ("\n".join(table) + "\n" if row.get("steps") else ""))
    if row.get("tile_steps"):
        tiles_table = ["| step | cl_mhz:spi | passes | bad tiles | median pass µs |", "| ---: | --- | ---: | ---: | ---: |"]
        tiles_table += [f"| {s['step']} | {s['spec']} | {s['passes']} | {s['bad']} | {s['us']} |" for s in row["tile_steps"]]
        with (out / "summary.md").open("a") as summary:
            summary.write("\n" + "\n".join(tiles_table) + "\n")
    log(f"{row['result'].upper()}: {row['detail']}")
    for s in row.get("steps", []):
        log(f"  step {s['step']} {s['spec']:16} {s['loads']} loads, {s['bad']} bad, "
            f"load {s['load_us']} µs, verify {s['verify_us']} µs")
    for s in row.get("tile_steps", []):
        log(f"  tiles {s['step']} {s['spec']:12} {s['passes']} passes, {s['bad']} bad, {s['us']} µs a pass")
    return 0 if row["result"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
