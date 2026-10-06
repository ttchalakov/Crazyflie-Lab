#!/usr/bin/env python3
"""camera_bench: build, flash and measure camera_stream builds on the drone.

  python examples/camera_bench/host/bench.py                     # the common set
  python examples/camera_bench/host/bench.py --set exhaustive    # every combination (~5 h)
  python examples/camera_bench/host/bench.py --case qvga60-native-jpegcl-preview-tcp
  python examples/camera_bench/host/bench.py --list
  python examples/camera_bench/host/bench.py --record-baseline   # save results as the baseline

A run first puts the idle controller on the STM32 (--keep-controller skips
that), so nothing else talks to the deck. Then, for each case in host/bench.yaml,
it builds the bench GAP8 app with the case's make variables, flashes it over the
radio, power-cycles the drone, joins the deck's access point, checks that the
build the deck reports is the case's, and measures. Cases that share a build run
back to back on one flash.

Each run gets its own folder, examples/camera_bench/artifacts/bench_<stamp>/:
run.json (what ran), results.jsonl (one line per case), and a folder per case
with its build log, the radio console from power-up, and the measured frames
(frames.csv) and deck status lines (deck.jsonl). host/plot.py draws a run. The
exit status is 0 when no case failed; a build that does not fit in L2 is
reported as unsupported, not failed.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml

from cf_lab.bench import AccessPoint, BenchError, RadioConsole, flash_and_boot, flash_idle_controller
from cf_lab.camera.feed import CameraFeed
from cf_lab.config import load_camera_config, load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import Acceptance, Collector, judge, summarize  # noqa: E402

HOST_DIR = Path(__file__).resolve().parent
BENCH_DIR = HOST_DIR.parent
APP_DIR = BENCH_DIR / "gap8"
BASELINE = HOST_DIR / "baseline.json"
KINDS = ("measure", "soak", "reconnect", "lifecycle")
# The bench app's radio-console line when a build does not fit in L2.
OUT_OF_L2 = "out of L2 memory"
# camera_stream extension fields saved per frame (cf_lab.camera.ext.CameraStreamExt).
FRAME_FIELDS = ("frame_id", "deck_ms", "encode_ms", "capture_ms", "preprocess_ms", "jpeg_ms",
                "camera_frame_id", "camera_done_ms", "dropped_capture", "dropped_ready",
                "dropped_streaming")
# How long a reconnect may take to deliver its first frames.
RECONNECT_TIMEOUT_S = 15.0


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    make: str
    transport: str = "tcp"
    kind: str = "measure"

    @property
    def build(self) -> tuple:
        """What determines the image: cases with the same build share a flash."""
        return tuple(sorted(shlex.split(self.make))), self.transport

    def expected(self) -> dict:
        """The config-line fields this case's build must report."""
        make = dict(token.split("=", 1) for token in shlex.split(self.make))
        want: dict = {"app": "camera_bench", "transport": self.transport}
        if "SENSOR" in make:
            want["sensor"] = make["SENSOR"].upper()
        for var, key in (("FORMAT", "format"), ("JPEG", "jpeg"), ("POLICY", "policy"),
                         ("SIZE", "out"), ("FIT", "fit")):
            if var in make:
                want[key] = make[var].lower()
        for var, key in (("FPS", "fps"), ("QUALITY", "quality"), ("BUFFERS", "buffers"),
                         ("LIFECYCLE", "lifecycle"), ("TRACE", "trace")):
            if var in make:
                want[key] = int(make[var])
        return want


def matrix_cases(matrix: dict) -> dict[str, Case]:
    """Every combination of the matrix axes, named sensor-size-format-policy-transport."""
    cases = {}
    native_only = set(matrix.get("native_only", ()))
    for sensor, sensor_make in matrix["sensor"].items():
        for size, size_make in matrix["size"].items():
            for fmt, fmt_make in matrix["format"].items():
                for policy, policy_make in matrix["policy"].items():
                    if size_make and policy in native_only:
                        continue
                    for transport in matrix["transport"]:
                        name = f"{sensor}-{size}-{fmt}-{policy}-{transport}"
                        make = " ".join(part for part in (sensor_make, size_make, fmt_make,
                                                          policy_make) if part)
                        cases[name] = Case(name=name, make=make, transport=transport)
    return cases


@dataclasses.dataclass
class Bench:
    ssid: str
    key: str
    channel: int
    deck_ip: str
    port: int
    profile: str
    warmup_s: float
    measure_s: float
    soak_s: float
    reconnects: int
    boot_timeout_s: float
    acceptance: Acceptance
    cases: dict[str, Case]
    sets: dict[str, list[str]]

    @classmethod
    def load(cls, path: Path) -> Bench:
        data = yaml.safe_load(path.read_text())
        wifi, timing = data["wifi"], data["timing"]
        cases = matrix_cases(data["matrix"])
        exhaustive = list(cases)
        for name, spec in data.get("special", {}).items():
            base = cases[spec["like"]]
            make = " ".join(part for part in (base.make, spec.get("make", "")) if part)
            cases[name] = Case(name=name, make=make, transport=base.transport, kind=spec["kind"])
            if spec["kind"] not in KINDS:
                raise ValueError(f"special case {name}: kind must be one of {KINDS}")
            if spec["kind"] != "soak":  # right after its base case, to share its flash
                exhaustive.insert(exhaustive.index(base.name) + 1, name)
        sets = {name: list(names) for name, names in data.get("sets", {}).items()}
        sets["exhaustive"] = exhaustive
        for set_name, names in sets.items():
            unknown = [name for name in names if name not in cases]
            if unknown:
                raise ValueError(f"set {set_name} names unknown cases: {unknown}")
        return cls(
            ssid=wifi["ssid"], key=wifi.get("key", ""), channel=int(wifi.get("channel", 0)),
            deck_ip=wifi["deck_ip"], port=int(wifi["port"]), profile=wifi["profile"],
            warmup_s=float(timing["warmup_s"]), measure_s=float(timing["measure_s"]),
            soak_s=float(timing["soak_s"]), reconnects=int(timing["reconnects"]),
            boot_timeout_s=float(timing["boot_timeout_s"]),
            acceptance=Acceptance(**data.get("acceptance", {})),
            cases=cases, sets=sets,
        )

    @property
    def access_point(self) -> AccessPoint:
        return AccessPoint(self.profile, self.ssid, self.key)


# -- the drone -------------------------------------------------------------------

def build(case: Case, bench: Bench, log_file) -> Path:
    from cf_lab.firmware import build_gap8, gap8_artifact

    wifi = load_camera_config(None, mode="access_point", ssid=bench.ssid, key=bench.key,
                              transport=case.transport, channel=bench.channel)
    build_gap8(APP_DIR, case.make, build_image=False, wifi=wifi, output=log_file)
    return gap8_artifact(APP_DIR)


# -- one case ---------------------------------------------------------------------

def open_feed(bench: Bench, case: Case, collector: Collector, *, timeout_s: float,
              want_frames: bool = True) -> CameraFeed:
    """Join the deck, connect, and wait for its config line (and a frame)."""
    deadline = time.monotonic() + timeout_s
    bench.access_point.up(deadline)
    feed = CameraFeed(ip=bench.deck_ip, port=bench.port, transport=case.transport,
                      connect_timeout_s=max(deadline - time.monotonic(), 1.0),
                      read_timeout_s=10.0)
    feed.subscribe(collector)
    try:
        feed.start()
    except OSError as exc:
        raise BenchError(f"cannot connect to the deck: {exc}") from exc
    if feed.wait_for_config(max(deadline - time.monotonic(), 1.0)) is None:
        error = feed.error
        feed.stop()
        raise BenchError(f"the deck sent no config line ({error or 'timeout'}); "
                         f"its radio console log says why")
    if want_frames and not feed.wait_for_frame(max(deadline - time.monotonic(), 1.0)):
        error = feed.error
        feed.stop()
        raise BenchError(f"no frames ({error or 'timeout'})")
    return feed


def config_mismatch(case: Case, config: dict) -> dict:
    return {key: {"want": want, "deck": config.get(key)}
            for key, want in case.expected().items() if config.get(key) != want}


def measure(bench: Bench, case: Case, collector: Collector, feed: CameraFeed,
            duration_s: float, baseline: dict | None) -> dict:
    time.sleep(bench.warmup_s)
    collector.start()
    end = time.monotonic() + duration_s
    while time.monotonic() < end and feed.error is None:
        time.sleep(0.2)
    collector.stop()
    summary = summarize(collector)
    checks = judge(summary, config=feed.deck_config or {}, transport=case.transport,
                   baseline=baseline, acceptance=bench.acceptance)
    if feed.error is not None:
        checks.append({"name": "stream", "passed": False, "measured": str(feed.error),
                       "want": "no error"})
    return {"summary": summary, "checks": checks}


def reconnect(bench: Bench, case: Case, feed: CameraFeed) -> dict:
    """Disconnect and reconnect repeatedly; each connection must deliver frames."""
    feed.stop()
    times, checks = [], []
    for cycle in range(bench.reconnects):
        collector = Collector()
        start = time.monotonic()
        try:
            new = open_feed(bench, case, collector, timeout_s=RECONNECT_TIMEOUT_S)
            collector.start()
            while len(collector.frames) < 5 and time.monotonic() - start < RECONNECT_TIMEOUT_S:
                time.sleep(0.02)
            new.stop()
            passed = len(collector.frames) >= 5
        except BenchError as exc:
            passed = False
            checks.append({"name": f"cycle_{cycle + 1}", "passed": False, "measured": str(exc),
                           "want": "5 frames"})
            continue
        times.append(time.monotonic() - start)
        if not passed:
            checks.append({"name": f"cycle_{cycle + 1}", "passed": False,
                           "measured": len(collector.frames), "want": "5 frames"})
    checks.insert(0, {"name": "reconnects", "passed": len(times) == bench.reconnects,
                      "measured": len(times), "want": bench.reconnects})
    summary = {"cycles": bench.reconnects,
               "connect_to_5_frames_s": {"max": max(times, default=0.0),
                                         "avg": sum(times) / len(times) if times else 0.0}}
    return {"summary": summary, "checks": checks}


def lifecycle(collector: Collector, timeout_s: float = 10.0) -> dict:
    end = time.monotonic() + timeout_s
    result = None
    while result is None and time.monotonic() < end:
        result = next((line.text for _, line in collector.console
                       if line.text.startswith("lifecycle ")), None)
        time.sleep(0.1)
    passed = result is not None and "result=pass" in result
    return {"summary": {"lifecycle": result},
            "checks": [{"name": "lifecycle", "passed": passed, "measured": result,
                        "want": "result=pass"}]}


def save_samples(case_dir: Path, collector: Collector) -> None:
    """The measured frames and deck lines, for plotting: frames.csv and deck.jsonl."""
    frames = list(collector.frames)
    if not frames:
        return
    start = frames[0][0]
    with (case_dir / "frames.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("t_s", "bytes", *FRAME_FIELDS))
        for arrival, size, ext in frames:
            values = [getattr(ext, name) for name in FRAME_FIELDS] if ext else [""] * len(FRAME_FIELDS)
            writer.writerow((f"{arrival - start:.6f}", size, *values))
    with (case_dir / "deck.jsonl").open("w") as handle:
        for kind, lines in (("status", collector.status), ("bench", collector.bench)):
            for arrival, fields in lines:
                handle.write(json.dumps({"t_s": round(arrival - start, 3), "kind": kind, **fields})
                             + "\n")


def run_case(bench: Bench, case: Case, uri: str, state: dict, case_dir: Path,
             baseline: dict | None, log) -> dict:
    row: dict = {"case": case.name, "make": case.make, "transport": case.transport,
                 "kind": case.kind, "started": datetime.datetime.now().isoformat(timespec="seconds")}
    case_dir.mkdir(parents=True, exist_ok=True)
    if state.get("build") != case.build:
        with (case_dir / "build.log").open("w") as build_log:
            log("  build")
            image = build(case, bench, build_log)
        log("  flash")
        bench.access_point.down()  # the deck stops streaming; the next join starts clean
        state["build"] = None
        flash_and_boot(image, uri, log=lambda message: log(f"  {message}"))
        state["build"] = case.build
    console_log = case_dir / "console.log"
    console = RadioConsole(uri, console_log)
    collector = Collector()
    feed = None
    try:
        log("  connect")
        feed = open_feed(bench, case, collector, timeout_s=bench.boot_timeout_s,
                         want_frames=case.kind != "lifecycle")
        console.close()
        row["deck_config"] = feed.deck_config
        mismatch = config_mismatch(case, feed.deck_config or {})
        if mismatch:
            raise BenchError(f"the deck runs a different build: {mismatch}")
        log(f"  {case.kind}")
        if case.kind in ("measure", "soak"):
            duration = bench.measure_s if case.kind == "measure" else bench.soak_s
            row.update(measure(bench, case, collector, feed, duration, baseline))
            save_samples(case_dir, collector)
        elif case.kind == "reconnect":
            row.update(reconnect(bench, case, feed))
        else:
            row.update(lifecycle(collector))
    except BenchError as exc:
        row["checks"] = [{"name": "run", "passed": False, "measured": str(exc), "want": "measured"}]
    finally:
        if feed is not None:
            feed.stop()
        console.close()
    checks = row.get("checks", [])
    row["passed"] = bool(checks) and all(c["passed"] or c.get("advisory") for c in checks)
    # A build that does not fit in L2 is a property of the configuration, not a
    # regression: the app reports it on the radio console at boot.
    if not row["passed"] and console_log.is_file() and OUT_OF_L2 in console_log.read_text():
        row["unsupported"] = True
    row["result"] = "pass" if row["passed"] else "unsupported" if row.get("unsupported") else "fail"
    return row


# -- main --------------------------------------------------------------------------

def _describe(row: dict) -> str:
    summary = row.get("summary", {})
    parts = []
    if "fps_avg" in summary:
        parts.append(f"rx {summary['fps_avg']:.1f} fps")
    if "camera_fps_avg" in summary:
        parts.append(f"cam {summary['camera_fps_avg']:.1f} fps")
    if "bitrate_kbps_avg" in summary:
        parts.append(f"{summary['bitrate_kbps_avg'] / 1000:.2f} Mbit/s")
    if "connect_to_5_frames_s" in summary:
        parts.append(f"reconnect max {summary['connect_to_5_frames_s']['max']:.1f} s")
    if summary.get("lifecycle"):
        parts.append(summary["lifecycle"])
    failed = [f"{c['name']}={c['measured']} (want {c['want']})"
              for c in row.get("checks", []) if not c["passed"] and not c.get("advisory")]
    notes = [f"{c['name']}={c['measured']} (want {c['want']})"
             for c in row.get("checks", []) if not c["passed"] and c.get("advisory")]
    return ("  ".join(parts) + (f"  FAILED: {'; '.join(failed)}" if failed else "")
            + (f"  note: {'; '.join(notes)}" if notes else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", type=Path, default=HOST_DIR / "bench.yaml")
    parser.add_argument("--config", type=Path, default=Path("configs/drone.yaml"),
                        help="the drone config; only radio.uri is used")
    choose = parser.add_mutually_exclusive_group()
    choose.add_argument("--set", default="common",
                        help="common (the default), exhaustive, or another set in bench.yaml")
    choose.add_argument("--case", action="append", help="run this case (repeatable)")
    parser.add_argument("--list", action="store_true", help="list the cases and sets, then exit")
    parser.add_argument("--record-baseline", action="store_true",
                        help="save this run's passing measurements to baseline.json")
    parser.add_argument("--channel", type=int, metavar="1..13",
                        help="the deck's access point channel, overriding bench.yaml")
    parser.add_argument("--keep-controller", action="store_true",
                        help="skip flashing the idle controller to the STM32 first")
    args = parser.parse_args(argv)

    bench = Bench.load(args.bench)
    if args.channel is not None:
        bench.channel = args.channel
    if args.list:
        for name, case in bench.cases.items():
            print(f"{name:32} {case.kind:9} {case.transport}  {case.make}")
        for name, names in bench.sets.items():
            print(f"set {name}: {len(names)} cases")
        return 0
    names = args.case or bench.sets.get(args.set)
    if names is None:
        parser.error(f"no set {args.set!r}; choose from {sorted(bench.sets)}")
    unknown = [name for name in names if name not in bench.cases]
    if unknown:
        parser.error(f"unknown cases: {unknown}")

    uri = load_config(args.config).radio.uri
    out = BENCH_DIR / "artifacts" / f"bench_{datetime.datetime.now():%Y%m%dT%H%M%S}"
    out.mkdir(parents=True)
    baselines = json.loads(BASELINE.read_text()).get("cases", {}) if BASELINE.is_file() else {}
    (out / "run.json").write_text(json.dumps({
        "set": None if args.case else args.set, "cases": names, "channel": bench.channel,
        "ssid": bench.ssid, "uri": uri, "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }, indent=1) + "\n")

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    bench.access_point.ensure(log)
    if not args.keep_controller:
        with (out / "controller.log").open("w") as controller_log:
            flash_idle_controller(uri, controller_log, log)
    log(f"{len(names)} case(s) -> {out}")
    rows, state = [], {}
    try:
        for index, name in enumerate(names, 1):
            case = bench.cases[name]
            log(f"[{index}/{len(names)}] {name}: {case.make} ({case.transport.upper()}, {case.kind})")
            try:
                row = run_case(bench, case, uri, state, out / name, baselines.get(name), log)
            except Exception as exc:  # a build or flash failure fails this case, not the run
                state["build"] = None
                row = {"case": name, "passed": False, "result": "fail",
                       "checks": [{"name": "setup", "passed": False,
                                   "measured": f"{type(exc).__name__}: {exc}", "want": "ok"}]}
            rows.append(row)
            with (out / "results.jsonl").open("a") as results:
                results.write(json.dumps(row, sort_keys=True) + "\n")
            log(f"  {row['result'].upper()}  {_describe(row)}")
    except KeyboardInterrupt:
        log("Interrupted.")
    finally:
        bench.access_point.down()

    counts = {result: sum(row["result"] == result for row in rows)
              for result in ("pass", "fail", "unsupported")}
    log(f"{counts['pass']} passed, {counts['fail']} failed, {counts['unsupported']} unsupported "
        f"(out of L2 memory). Results: {out / 'results.jsonl'}")
    if args.record_baseline:
        record_baseline(rows, log)
    return 0 if len(rows) == len(names) and counts["fail"] == 0 else 1


def record_baseline(rows: list[dict], log) -> None:
    """Merge this run's passing measurements into baseline.json."""
    data = json.loads(BASELINE.read_text()) if BASELINE.is_file() else {"cases": {}}
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=BENCH_DIR).stdout.strip()
    recorded = 0
    for row in rows:
        if row["passed"] and row.get("kind") == "measure":
            data["cases"][row["case"]] = {**row["summary"], "recorded": row["started"],
                                          "git": git}
            recorded += 1
    BASELINE.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    log(f"Baseline: {recorded} case(s) written to {BASELINE}")


if __name__ == "__main__":
    raise SystemExit(main())
