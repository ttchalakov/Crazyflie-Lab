#!/usr/bin/env python3
"""Profile and verify production GAP8 flashing on a real AI-deck."""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from cf_lab.config import radio_uri

# Allow both `python path/to/bench.py` and module invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "examples.flash_bench.host"

from .cases import CASES, FIXTURES, SETS, select  # noqa: E402
from .firmware import BENCH, append_empty_area, build, describe_image  # noqa: E402
from .results import save, write_json  # noqa: E402

HOST = Path(__file__).resolve().parent
ROOT = BENCH.parents[1]
# The STM32 runs the lab's idle app, so nothing else talks to the deck.
IDLE_CONTROLLER = ROOT / "examples" / "idle" / "controller"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    # Let production's finally blocks reboot the deck, or finish powering the
    # STM32 back up if cancellation landed during a cold boot.
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def worker(request: dict, directory: Path, timeout_s: float) -> dict:
    directory.mkdir(parents=True)
    request_path = directory / "request.json"
    write_json(request_path, request)
    started = time.monotonic()
    timed_out = False
    with (directory / "worker.log").open("w") as output:
        process = subprocess.Popen([sys.executable, str(HOST / "device.py"), str(request_path)],
                                   cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                try:
                    process.wait(timeout=min(30, max(0.01, timeout_s -
                                                    (time.monotonic() - started))))
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started >= timeout_s:
                        timed_out = True
                        stop(process)
                        break
                    log(f"{directory.name}: active {time.monotonic() - started:.0f}s")
        finally:
            stop(process)
    result_path = directory / "result.json"
    if result_path.exists() and not timed_out:
        result = json.loads(result_path.read_text())
        if process.returncode != 0:
            result.update(passed=False, status="failed")
        return result
    result = {"passed": False, "status": "timeout" if timed_out else "worker-error",
              "elapsed_s": round(time.monotonic() - started, 3),
              "error": f"worker {'exceeded its deadline' if timed_out else 'exited without a result'}; "
                       f"see {directory / 'worker.log'}"}
    write_json(result_path, result)
    return result


def provenance() -> dict:
    def command(*args):
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=20)
        return result.stdout.strip() if result.returncode == 0 else None
    import cflib
    from cf_lab.aideck import flash
    return {"commit": command("git", "rev-parse", "HEAD"),
            "working_tree": command("git", "status", "--short"),
            "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for base, patterns in ((BENCH, ("host/*.py", "gap8/*.c", "gap8/Makefile")),
                                       (IDLE_CONTROLLER, ("Makefile", "app-config", "src/*.c")))
                for pattern in patterns
                for p in sorted(base.glob(pattern))
            },
            "submodules": command("git", "submodule", "status"),
            "sdk_image": command("docker", "image", "inspect", "cf-lab/aideck-gap8:latest",
                                 "--format", "{{.Id}}"),
            "python": sys.version, "cflib": getattr(cflib, "__version__", "unknown"),
            "flasher": "cf_lab.aideck.flash.flash_gap8",
            "default_attempts": flash.DEFAULT_ATTEMPTS,
            "stall_timeout_s": flash.DEFAULT_STALL_TIMEOUT_S}


def run_benchmark(args, cases, output: Path) -> int:
    started = time.monotonic()
    run = {"schema": 1, "status": "running", "uri": args.uri,
           "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "selection": [dataclasses.asdict(case) for case in cases],
           "expected_cases": len(cases) * args.repeat,
           "repeat": args.repeat, "controller": "kept" if args.keep_controller else "idle",
           "provenance": provenance(), "cases": []}
    save(output, run)
    images = {}
    state = None
    touched_hardware = False

    def flash_image(image, directory, **options):
        # A large image is mostly radio transfer time. Budget below assumes at
        # least 1 kB/s, two full attempts, and connection/verification overhead.
        timeout = args.flash_timeout or (180 + 2 * image["bytes"] / 1000)
        return worker({"action": "flash", "fixture": image, "uri": args.uri, **options},
                      directory, timeout)

    try:
        needed = {"silent"} | {name for case in cases for name in (case.source, case.target)}
        for name in sorted(needed):
            log(f"Building fixture {name}")
            images[name] = build(FIXTURES[name], output / "images" / name)
        if any(case.repair for case in cases):
            image = output / "images" / "a" / "empty-area.img"
            image.write_bytes(append_empty_area(Path(images['a']['path']).read_bytes()))
            images["repair"] = {**images["a"], **describe_image(image)}
        write_json(output / "images.json", images)
        if not args.keep_controller:
            log("Building and flashing the idle STM32 controller")
            touched_hardware = True
            run["controller_result"] = worker({"action": "controller", "uri": args.uri,
                                               "app": str(IDLE_CONTROLLER)},
                                              output / "controller", 600)
            if not run["controller_result"]["passed"]:
                raise RuntimeError("controller preparation failed; see controller/result.json")
        for repetition in range(1, args.repeat + 1):
            for index, case in enumerate(cases, 1):
                name = f"{repetition:02d}-{index:02d}-{case.name}"
                directory = output / name
                row = {"name": name, "status": "running", "case": dataclasses.asdict(case)}
                run["cases"].append(row)
                save(output, run)
                log(f"[{repetition}/{args.repeat} {index}/{len(cases)}] {case.name}")
                touched_hardware = True
                if state != case.source:
                    row["setup"] = flash_image(images[case.source], directory / "setup")
                    if not row["setup"]["passed"]:
                        row.update(status="failed", error="source-image setup failed")
                        raise RuntimeError(f"{name}: cannot establish the source fixture")
                    state = case.source
                image = images["repair" if case.repair else case.target]
                row["flash"] = flash_image(image, directory / "flash",
                                           cold_boot=case.cold_boot, attempts=case.attempts)
                passed = row["flash"]["passed"]
                state = case.target if passed else None
                if passed and case.power_cycle_after:
                    row["persistence"] = worker({"action": "power-cycle", "uri": args.uri,
                                                 "fixture": image}, directory / "persistence", 90)
                    passed = row["persistence"]["passed"]
                    if not passed:
                        state = None
                row["status"] = "pass" if passed else "failed"
                log(f"{case.name}: {row['status']}")
                run["elapsed_s"] = time.monotonic() - started
                save(output, run)
                if not passed:
                    # Do not let recovery erase evidence or assume the next
                    # transition began in the source state declared by its case.
                    raise RuntimeError(f"{name}: failed; benchmark run stopped, remaining cases unrun")
        run["status"] = "pass"
    except KeyboardInterrupt:
        run.update(status="interrupted", error="interrupted by operator")
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if touched_hardware and "silent" in images:
            log("Cleanup: flash and verify the silent GAP8 fixture")
            try:
                run["cleanup"] = flash_image(images["silent"], output / "cleanup")
                if not run["cleanup"]["passed"]:
                    run["status"] = "failed"
            except (Exception, KeyboardInterrupt) as exc:
                run["cleanup"] = {"status": "failed", "error": str(exc)}
                run["status"] = "failed"
        for row in run["cases"]:
            if row["status"] == "running":
                row["status"] = "interrupted" if run["status"] == "interrupted" else "failed"
        run["elapsed_s"] = round(time.monotonic() - started, 3)
        run["passed_cases"] = sum(row["status"] == "pass" for row in run["cases"])
        run["attempted_cases"] = len(run["cases"])
        run["unrun_cases"] = len(cases) * args.repeat - len(run["cases"])
        run["expected_cases"] = len(cases) * args.repeat
        save(output, run)
    print((output / "summary.txt").read_text(), end="")
    log(f"Results: {output}")
    return 0 if run["status"] == "pass" else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/drone.yaml")
    parser.add_argument("--uri", help="override radio.uri")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--set", choices=SETS, default="common")
    group.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--keep-controller", action="store_true",
                        help="use existing STM32 firmware; missing MD5 verification fails")
    parser.add_argument("--output", type=Path, help="new directory for this benchmark run")
    parser.add_argument("--flash-timeout", type=float,
                        help="worker deadline in seconds; default scales with image size")
    args = parser.parse_args(argv)
    if args.repeat < 1 or (args.flash_timeout is not None and args.flash_timeout <= 0):
        parser.error("repeat and timeout must be positive")
    cases = select(args.set, args.case)
    if args.list:
        for case in cases:
            flags = [key for key in ("cold_boot", "repair", "power_cycle_after")
                     if getattr(case, key)]
            if case.attempts:
                flags.append(f"attempts={case.attempts}")
            print(f"{case.name:24} {case.source:6} -> {case.target:6} {' '.join(flags)}")
        print(f"{len(cases)} cases per repetition; sets: {', '.join(SETS)}")
        return 0
    args.uri = radio_uri(args.config, args.uri)
    if not args.uri.startswith("radio://"):
        parser.error("this benchmark run uses Crazyradio (including power-cycle cases)")
    output = (args.output or BENCH / "artifacts" /
              datetime.datetime.now().strftime("bench_%Y%m%dT%H%M%S_%f")).resolve()
    # SDK mounts cover the workspace; readfs payloads must be inside that mount.
    if not output.is_relative_to(ROOT):
        parser.error("output must be inside the repository for Docker to mount fixture payloads")
    lock_dir = BENCH / "artifacts"
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / ".bench.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another flash_bench benchmark run is already running")
        output.mkdir(parents=True, exist_ok=False)
        return run_benchmark(args, cases, output)


if __name__ == "__main__":
    raise SystemExit(main())
