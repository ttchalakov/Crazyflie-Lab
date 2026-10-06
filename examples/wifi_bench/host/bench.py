#!/usr/bin/env python3
"""wifi_bench: synthetic GAP8 <-> ESP32 <-> host traffic over TCP and UDP.

  python examples/wifi_bench/host/bench.py                   # the common set (both transports)
  python examples/wifi_bench/host/bench.py --set long        # one-hour soaks
  python examples/wifi_bench/host/bench.py --case reconnect-tcp
  python examples/wifi_bench/host/bench.py --list
  python examples/wifi_bench/host/bench.py --record-baseline # save results as the baseline

A run first puts the idle controller on the STM32 (--keep-controller skips
that). For each transport it builds the bench GAP8 app, flashes it over the
radio, power-cycles the drone, joins the deck's access point, checks that the
deck runs that transport, and runs the transport's cases. The radio console
stays open throughout, because the ESP32's heap and error lines are part of
what is measured; it costs some WiFi throughput (--no-console closes it).

Each run gets its own folder, examples/wifi_bench/artifacts/bench_<stamp>/:
run.json (what ran), results.jsonl (one line per case), per transport the build
log and the radio console, and a folder per case (stream samples, telemetry).
The exit status is 0 when no case failed.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import re
import secrets
import struct
import subprocess
import sys
import time
from pathlib import Path

import yaml

from cf_lab.bench import AccessPoint, BenchError, RadioConsole, flash_and_boot, flash_idle_controller
from cf_lab.config import LabConfig, load_camera_config, load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite  # noqa: E402
from link import STATS, Link  # noqa: E402

HOST_DIR = Path(__file__).resolve().parent
BENCH_DIR = HOST_DIR.parent
APP_DIR = BENCH_DIR / "gap8"
BASELINE = HOST_DIR / "baseline.json"
TRANSPORT_CODES = {"tcp": 0, "udp": 1}  # CPX_WIFI_TRANSPORT_* in the GAP8 app's STATS reply


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    test: str  # a key of suite.TESTS, or the kind for special cases
    transport: str
    kind: str = "test"  # test, soak, integration


@dataclasses.dataclass
class Bench:
    ssid: str
    key: str
    channel: int
    deck_ip: str
    port: int
    profile: str
    timing: dict
    acceptance: dict
    load: dict
    telemetry: dict
    cases: dict[str, Case]
    sets: dict[str, list[str]]

    @classmethod
    def load_file(cls, path: Path) -> Bench:
        data = yaml.safe_load(path.read_text())
        wifi = data["wifi"]
        cases = {}
        for transport in ("tcp", "udp"):  # one flash per transport
            for test, (_, transports, _) in suite.TESTS.items():
                if transport in transports:
                    cases[f"{test}-{transport}"] = Case(f"{test}-{transport}", test, transport)
        common = list(cases)
        for name, spec in data.get("special", {}).items():
            cases[name] = Case(name, spec["kind"], spec["transport"], spec["kind"])
        sets = {"common": common, **{name: list(names) for name, names in data.get("sets", {}).items()}}
        for set_name, names in sets.items():
            unknown = [name for name in names if name not in cases]
            if unknown:
                raise ValueError(f"set {set_name} names unknown cases: {unknown}")
        return cls(ssid=wifi["ssid"], key=wifi.get("key", ""), channel=int(wifi["channel"]),
                   deck_ip=wifi["deck_ip"], port=int(wifi["port"]), profile=wifi["profile"],
                   timing=data["timing"], acceptance=data.get("acceptance", {}),
                   load=data["load"], telemetry=data.get("telemetry", {}), cases=cases, sets=sets)

    @property
    def access_point(self) -> AccessPoint:
        return AccessPoint(self.profile, self.ssid, self.key)


# -- the deck ----------------------------------------------------------------------

def build(bench: Bench, transport: str, log_file) -> Path:
    from cf_lab.firmware import build_gap8, gap8_artifact

    wifi = load_camera_config(None, mode="access_point", ssid=bench.ssid, key=bench.key,
                              transport=transport, channel=bench.channel)
    build_gap8(APP_DIR, "", build_image=False, wifi=wifi, output=log_file)
    return gap8_artifact(APP_DIR)


def deck_stats(bench: Bench, transport: str, deadline: float) -> tuple:
    """The GAP8's STATS reply, retrying until the deck serves or the deadline passes."""
    error = None
    while time.monotonic() < deadline:
        try:
            with Link(bench.deck_ip, bench.port, transport, timeout=3) as link:
                return struct.unpack("<10I", link.request(STATS, secrets.randbits(32)).body)
        except (OSError, TimeoutError, ValueError) as exc:
            error = exc
            time.sleep(1)
    raise BenchError(f"the deck did not answer over {transport.upper()} ({error})")


def start_transport(bench: Bench, transport: str, uri: str, out: Path, console: bool, log):
    """Build, flash and boot the transport's app; join; check the deck runs it."""
    with (out / f"{transport}-build.log").open("w") as build_log:
        log(f"{transport.upper()}: build")
        image = build(bench, transport, build_log)
    log(f"{transport.upper()}: flash")
    bench.access_point.down()
    flash_and_boot(image, uri, log=lambda message: log(f"  {message}"))
    radio = RadioConsole(uri, out / f"{transport}-console.log") if console else None
    try:
        deadline = time.monotonic() + float(bench.timing["boot_timeout_s"])
        bench.access_point.up(deadline)
        stats = deck_stats(bench, transport, deadline)
        if (stats[8], stats[9]) != (TRANSPORT_CODES[transport], bench.channel):
            raise BenchError(f"the deck runs transport code {stats[8]} on channel {stats[9]}, "
                             f"not {transport} on {bench.channel}")
    except BaseException:
        if radio:
            radio.close()
        raise
    return radio


# -- one case ----------------------------------------------------------------------

ESP_EVENTS = {"send_timeouts": "Send timeout", "send_stalls": "Send stalled",
              "send_errors": "Error occurred during sending", "disconnects": "Client disconnected"}


def esp_diagnostics(text: str) -> dict:
    """The ESP32's heap and send trouble in a stretch of its console."""
    heaps = [(int(free), int(low)) for free, low in re.findall(r"heap (\d+) \(min (\d+)\)", text)]
    return {"heap_free_min": min((free for free, _ in heaps), default=None),
            "heap_watermark": min((low for _, low in heaps), default=None),
            **{key: text.count(marker) for key, marker in ESP_EVENTS.items()}}


def integration(bench: Bench, ctx: suite.Context, uri: str, console_path: Path | None) -> dict:
    """The WiFi load while cflib records STM32 telemetry; the recording must check out."""
    from cf_lab.backends.crazyflie import CrazyflieRadioBackend
    from cf_lab.telemetry import TelemetryRecorder, validate_run

    config = LabConfig.from_mapping({"telemetry": bench.telemetry}).telemetry
    sink = console_path.open("a", buffering=1) if console_path else None
    try:
        with CrazyflieRadioBackend(uri, console_sink=(lambda line: sink.write(
                f"{time.strftime('%H:%M:%S')} {line}\n")) if sink else None) as backend:
            recorder = TelemetryRecorder(config, run_root=ctx.directory, example="wifi_bench")
            recorder.start()
            try:
                recorder.attach_stm32(backend.crazyflie)
                row = suite.stream(ctx, bench.load["size"], bench.load["rate"],
                                   float(bench.timing["integration_s"]))
            finally:
                recorder.detach_stm32()
                recorder.close()
    finally:
        if sink:
            sink.close()
    _, report = validate_run(recorder.run_dir, required_streams=("state_pos", "state_att", "ctrl"))
    (ctx.directory / "telemetry_integrity.txt").write_text(report.format() + "\n")
    row["checks"].append(suite.check("telemetry_integrity", report.passed,
                                     "pass" if report.passed else "fail", "pass"))
    return row


def judge_baseline(row: dict, baseline: dict | None, acceptance: dict) -> list[dict]:
    summary, checks = row.get("summary", {}), []
    if not baseline:
        return checks
    if "mbps" in summary and baseline.get("mbps"):
        want = baseline["mbps"] * acceptance.get("mbps_ratio", 0.9)
        checks.append(suite.check("mbps_vs_baseline", summary["mbps"] >= want, summary["mbps"],
                                  f">= {want:.2f}", advisory=True))
    if "rtt_median_ms" in summary and baseline.get("rtt_median_ms"):
        want = baseline["rtt_median_ms"] * acceptance.get("rtt_ratio", 1.5) + 2
        checks.append(suite.check("rtt_vs_baseline", summary["rtt_median_ms"] <= want,
                                  summary["rtt_median_ms"], f"<= {want:.1f}", advisory=True))
    return checks


def run_case(bench: Bench, case: Case, uri: str, out: Path, console_path: Path | None,
             baseline: dict | None, log) -> dict:
    row: dict = {"case": case.name, "test": case.test, "transport": case.transport,
                 "kind": case.kind, "started": datetime.datetime.now().isoformat(timespec="seconds")}
    directory = out / case.name
    directory.mkdir(parents=True, exist_ok=True)
    ctx = suite.Context(ip=bench.deck_ip, port=bench.port, transport=case.transport,
                        directory=directory, console=console_path, access_point=bench.access_point,
                        log=log, options=bench.acceptance)
    offset = console_path.stat().st_size if console_path and console_path.exists() else 0
    try:
        if case.kind == "soak":
            row.update(suite.stream(ctx, bench.load["size"], bench.load["rate"],
                                    float(bench.timing["soak_s"])))
            row["checks"].append(suite.recovers(ctx))
        elif case.kind == "integration":
            row.update(integration(bench, ctx, uri, console_path))
        else:
            function, _, arguments = suite.TESTS[case.test]
            if function is suite.stream:
                arguments = {**arguments, "duration_s": float(bench.timing["stream_s"])}
            row.update(function(ctx, **arguments))
    except Exception as exc:
        row.setdefault("checks", []).append(suite.check("run", False, f"{type(exc).__name__}: {exc}",
                                                        "completed"))
    if console_path and console_path.exists():
        with console_path.open("rb") as handle:
            handle.seek(offset)
            text = handle.read().decode("utf-8", "replace")
        row.setdefault("summary", {})["esp"] = esp = esp_diagnostics(text)
        if esp["heap_free_min"] is not None:
            limit = bench.acceptance.get("esp_heap_min_bytes", 4096)
            row["checks"].append(suite.check("esp_heap", esp["heap_free_min"] >= limit,
                                             esp["heap_free_min"], f">= {limit}", advisory=True))
    row["checks"] = row.get("checks", []) + judge_baseline(row, baseline, bench.acceptance)
    checks = row["checks"]
    if row.get("skipped"):
        row["result"], row["passed"] = "skipped", True
    else:
        row["passed"] = bool(checks) and all(c["passed"] or c.get("advisory") for c in checks)
        row["result"] = "pass" if row["passed"] else "fail"
    return row


# -- main --------------------------------------------------------------------------

def _describe(row: dict) -> str:
    summary = row.get("summary", {})
    parts = []
    if "mbps" in summary:
        parts.append(f"{summary['mbps']:.2f} Mbit/s, {summary['received']}/{summary['gap8_accepted']}"
                     " received")
    if "rtt_median_ms" in summary:
        parts.append(f"RTT {summary['rtt_median_ms']:.1f} ms")
    if summary.get("esp", {}).get("heap_free_min") is not None:
        parts.append(f"ESP heap >= {summary['esp']['heap_free_min']} B")
    if row.get("skipped"):
        parts.append(f"skipped: {row['skipped']}")
    failed = [f"{c['name']}={c['measured']} (want {c['want']})"
              for c in row.get("checks", []) if not c["passed"] and not c.get("advisory")]
    notes = [f"{c['name']}={c['measured']} (want {c['want']})"
             for c in row.get("checks", []) if not c["passed"] and c.get("advisory")]
    return ("  ".join(parts) + (f"  FAILED: {'; '.join(failed)}" if failed else "")
            + (f"  note: {'; '.join(notes)}" if notes else ""))


def record_baseline(rows: list[dict], log) -> None:
    """Merge this run's passing stream and echo measurements into baseline.json."""
    data = json.loads(BASELINE.read_text()) if BASELINE.is_file() else {"cases": {}}
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=BENCH_DIR).stdout.strip()
    recorded = 0
    for row in rows:
        summary = row.get("summary", {})
        kept = {key: summary[key] for key in ("mbps", "rtt_median_ms") if key in summary}
        if row["passed"] and kept:
            data["cases"][row["case"]] = {**kept, "recorded": row["started"], "git": git}
            recorded += 1
    BASELINE.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    log(f"Baseline: {recorded} case(s) written to {BASELINE}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", type=Path, default=HOST_DIR / "bench.yaml")
    parser.add_argument("--config", type=Path, default=Path("configs/drone.yaml"),
                        help="the drone config; only radio.uri is used")
    choose = parser.add_mutually_exclusive_group()
    choose.add_argument("--set", default="common", help="common (the default), long or integration")
    choose.add_argument("--case", action="append", help="run this case (repeatable)")
    parser.add_argument("--list", action="store_true", help="list the cases and sets, then exit")
    parser.add_argument("--record-baseline", action="store_true",
                        help="save this run's passing measurements to baseline.json")
    parser.add_argument("--channel", type=int, metavar="1..13",
                        help="the deck's access point channel, overriding bench.yaml")
    parser.add_argument("--keep-controller", action="store_true",
                        help="skip flashing the idle controller to the STM32 first")
    parser.add_argument("--no-console", action="store_true",
                        help="close the radio console (no ESP32 diagnostics or SPI rejection test)")
    args = parser.parse_args(argv)

    bench = Bench.load_file(args.bench)
    if args.channel is not None:
        bench.channel = args.channel
    if args.list:
        for name, case in bench.cases.items():
            print(f"{name:28} {case.kind:12} {case.transport}")
        for name, names in bench.sets.items():
            print(f"set {name}: {len(names)} cases")
        return 0
    names = args.case or bench.sets.get(args.set)
    if names is None:
        parser.error(f"no set {args.set!r}; choose from {sorted(bench.sets)}")
    unknown = [name for name in names if name not in bench.cases]
    if unknown:
        parser.error(f"unknown cases: {unknown}")
    if args.no_console and any(bench.cases[name].kind == "integration" for name in names):
        parser.error("the integration cases need the radio")
    names.sort(key=lambda name: bench.cases[name].transport != "tcp")  # one flash per transport

    uri = load_config(args.config).radio.uri
    out = BENCH_DIR / "artifacts" / f"bench_{datetime.datetime.now():%Y%m%dT%H%M%S}"
    out.mkdir(parents=True)
    baselines = json.loads(BASELINE.read_text()).get("cases", {}) if BASELINE.is_file() else {}
    esp = subprocess.run(["git", "-C", str(BENCH_DIR.parents[1] / "external/aideck-esp-firmware"),
                          "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "run.json").write_text(json.dumps({
        "set": None if args.case else args.set, "cases": names, "channel": bench.channel,
        "ssid": bench.ssid, "uri": uri, "console": not args.no_console,
        "esp_checkout": esp, "esp_note": "the checkout; the deck's ESP32 firmware is not read back",
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }, indent=1) + "\n")

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    bench.access_point.ensure(log)
    if not args.keep_controller:
        with (out / "controller.log").open("w") as controller_log:
            flash_idle_controller(uri, controller_log, log)
    log(f"{len(names)} case(s) -> {out}")
    rows, transport, radio = [], None, None
    broken: dict[str, str] = {}  # transport -> why its build, flash or boot failed
    try:
        for index, name in enumerate(names, 1):
            case = bench.cases[name]
            console_path = None if args.no_console else out / f"{case.transport}-console.log"
            try:
                if case.transport in broken:
                    raise BenchError(broken[case.transport])
                if case.transport != transport:
                    if radio:
                        radio.close()
                    transport = None
                    radio = start_transport(bench, case.transport, uri, out, not args.no_console, log)
                    transport = case.transport
                log(f"[{index}/{len(names)}] {name}")
                # The host drops the deck's access point when it misses its beacons
                # (an ESP32 short of heap stops sending them), and NetworkManager
                # does not rejoin by itself. Rejoin, so one drop fails one case.
                rejoined = not bench.access_point.is_up()
                if rejoined:
                    log("  the host had lost the deck's access point; rejoining")
                    bench.access_point.up(time.monotonic() + float(bench.timing["boot_timeout_s"]))
                if case.kind == "integration" and radio:
                    radio.close()  # cflib takes the radio and records the console itself
                row = run_case(bench, case, uri, out, console_path, baselines.get(name), log)
                row["host_rejoined_ap"] = rejoined
                if case.kind == "integration" and not args.no_console:
                    radio = RadioConsole(uri, console_path)
            except Exception as exc:  # a build, flash or boot failure fails the transport's cases
                transport = None
                broken.setdefault(case.transport, f"{type(exc).__name__}: {exc}")
                row = {"case": name, "passed": False, "result": "fail",
                       "checks": [suite.check("setup", False, broken[case.transport], "ok")]}
            rows.append(row)
            with (out / "results.jsonl").open("a") as results:
                results.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            log(f"  {row['result'].upper()}  {_describe(row)}")
    except KeyboardInterrupt:
        log("Interrupted.")
    finally:
        if radio:
            radio.close()
        bench.access_point.down()

    counts = {result: sum(row["result"] == result for row in rows)
              for result in ("pass", "fail", "skipped")}
    log(f"{counts['pass']} passed, {counts['fail']} failed, {counts['skipped']} skipped. "
        f"Results: {out / 'results.jsonl'}")
    if args.record_baseline:
        record_baseline(rows, log)
    return 0 if len(rows) == len(names) and counts["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
