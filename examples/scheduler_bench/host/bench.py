"""scheduler_bench: lib/gap8/sched's sequential and pipelined schedulers on the AI-deck.

  python examples/scheduler_bench/host/bench.py                     # the common set
  python examples/scheduler_bench/host/bench.py --set mlp_smoke
  python examples/scheduler_bench/host/bench.py --case mlp-12to64to64to4-l2-qqvga120-jpegcl-100hz

Per case: generate the network (host/sched_net.py), build one image holding both
schedulers, flash it, join the deck's access point, receive its video and read its
SB| report lines from the radio console, and compare the two schedulers. Results land in
artifacts/bench_<stamp>/: results.jsonl, summary.md, and per case the build log,
the radio console and the report lines.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shlex
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from cf_lab.bench import AccessPoint, BenchError, RadioConsole, flash_and_boot
from cf_lab.camera.ext import SchedFrameExt
from cf_lab.camera.feed import CameraFeed
from cf_lab.config import load_camera_config, load_config

import sched_net

HOST_DIR = Path(__file__).resolve().parent
BENCH_DIR = HOST_DIR.parent
APP_DIR = BENCH_DIR / "gap8"
ROOT = BENCH_DIR.parents[1]
KINDS = {0: "seq", 1: "pipe"}


@dataclass(frozen=True)
class Case:
    name: str
    network: str
    camera: str
    rate_hz: int


def cases_of(data: dict) -> dict[str, Case]:
    """State networks at each request rate; image networks at the frame rate."""
    cases = {}
    for network, spec in data["networks"].items():
        rates = data["rates"] if spec["family"] == "mlp" else [0]
        for camera in data["cameras"]:
            for rate in rates:
                name = f"{network}-{camera}" + (f"-{rate}hz" if rate else "")
                cases[name] = Case(name, network, camera, rate)
    return cases


def frame_size(camera: dict) -> tuple[int, int]:
    """The camera_stream frame of a SENSOR mode."""
    return (160, 120) if "QQVGA" in camera["SENSOR"].upper() else (320, 240)


# Every CHECK_EVERY-th frame of each scheduler is checked (jpeg_ok).
CHECK_EVERY = 10
RESTART = re.compile(rb"\xff[\xd0-\xd7]")


def jpeg_ok(payload: bytes, width: int, height: int) -> bool:
    """Whole, with the restart markers its DRI implies in order, decodable to the frame.

    The deck encodes a frame in slices of restart segments; a decoder may accept a
    wrong segment with only a warning, so the markers are counted here.
    """
    import cv2
    import numpy as np

    sos = payload.find(b"\xff\xda")
    if sos < 0 or not payload.endswith(b"\xff\xd9"):
        return False
    dri = payload.find(b"\xff\xdd", 0, sos)
    if dri >= 0:
        interval = int.from_bytes(payload[dri + 4:dri + 6], "big")
        mcus = -(-width // 8) * -(-height // 8)
        markers = [m.group()[1] for m in RESTART.finditer(payload, sos)]
        if len(markers) != -(-mcus // interval) - 1:
            return False
        if any(m != 0xD0 + i % 8 for i, m in enumerate(markers)):
            return False
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
    return image is not None and image.shape == (height, width)


class Collector:
    """Frames per scheduler, from each frame's extension, and a sample checked."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frames: dict[str, list[tuple[int, int, int]]] = {"seq": [], "pipe": []}
        self.checked = {"seq": 0, "pipe": 0}
        self.bad = {"seq": 0, "pipe": 0}

    def offer(self, frame) -> None:
        ext = frame.ext  # lib/gap8/sched's sched_frame_ext_t, decoded by the feed
        if not isinstance(ext, SchedFrameExt) or ext.kind not in KINDS:
            return
        name = KINDS[ext.kind]
        with self.lock:
            self.frames[name].append((frame.host_ns, ext.frame_id, len(frame.payload)))
            check = len(self.frames[name]) % CHECK_EVERY == 1
        if check:
            ok = (jpeg_ok(frame.payload, frame.width, frame.height) if frame.is_jpeg
                  else len(frame.payload) == frame.width * frame.height)
            with self.lock:
                self.checked[name] += 1
                self.bad[name] += 0 if ok else 1


def parse_reports(lines: list[str]) -> dict[str, dict]:
    """SB|<tag> k=<kind> a=1 b=2 ... into {kind: {tag_a: 1, ...}}."""
    phases: dict[str, dict] = {}
    for line in lines:
        tag, _, rest = line[3:].partition(" ")
        fields = dict(word.split("=", 1) for word in rest.split() if "=" in word)
        kind = fields.pop("k", None)
        if kind not in ("seq", "pipe") or tag not in ("phase", "lat", "wait", "inf", "res"):
            continue
        row = phases.setdefault(kind, {})
        for key, value in fields.items():
            if "/" in value:  # vf=failures/runs
                failures, runs = value.split("/")
                row["verify_failures"], row["verify_runs"] = int(failures), int(runs)
            else:
                row[key if tag == "phase" else f"{tag}_{key}"] = int(value)
    return phases


def host_fps(frames: list[tuple[int, int, int]]) -> dict:
    if len(frames) < 2:
        return {"host_frames": len(frames)}
    span_s = (frames[-1][0] - frames[0][0]) / 1e9
    ids = [f[1] for f in frames]
    return {"host_frames": len(frames), "host_fps": round((len(frames) - 1) / span_s, 1),
            "host_kb_per_frame": round(sum(f[2] for f in frames) / len(frames) / 1024, 1),
            "camera_frame_span": ids[-1] - ids[0]}


def build(case: Case, data: dict, net_dir: Path, wifi, log_file) -> Path:
    from cf_lab.firmware import build_gap8, gap8_artifact

    camera = data["cameras"][case.camera]
    timing = data["timing"]
    make = {"NET_DIR": str(net_dir), **camera, "PHASE_MS": timing["phase_ms"],
            "SETTLE_MS": timing["settle_ms"]}
    if case.rate_hz:
        make["STATE_RATE_HZ"] = case.rate_hz
    args = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in make.items())
    build_gap8(APP_DIR, args + " clean build image", build_image=False, wifi=wifi, output=log_file)
    return gap8_artifact(APP_DIR)


def static_l2(elf: Path) -> int:
    """Bytes of L2 the image itself occupies (code, data, linked weights)."""
    import subprocess
    out = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{ROOT}:{ROOT}", "cf-lab/aideck-gap8:latest", "bash", "-lc",
         f"source /gap_sdk/configs/ai_deck.sh >/dev/null; riscv32-unknown-elf-size -A {elf}"],
        capture_output=True, text=True).stdout
    total = 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            if 0x1C000000 <= int(parts[2]) < 0x1C080000:
                total += int(parts[1])
    return total


def run_case(case: Case, data: dict, uri: str, out: Path, log) -> dict:
    case_dir = out / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    row: dict = {"case": case.name, "network": case.network, "camera": case.camera,
                 "rate_hz": case.rate_hz, "started": datetime.datetime.now().isoformat(timespec="seconds")}
    wifi_cfg = data["wifi"]
    spec = sched_net.parse_spec(case.network, data["networks"][case.network],
                               frame_size(data["cameras"][case.camera]))
    log("  network")
    net_dir = sched_net.prepare(spec, BENCH_DIR / "artifacts" / "networks")
    wifi = load_camera_config(None, mode="access_point", ssid=wifi_cfg["ssid"], key=wifi_cfg["key"],
                              transport="udp", channel=int(wifi_cfg["channel"]))
    log("  build")
    with (case_dir / "build.log").open("w") as build_log:
        image = build(case, data, net_dir, wifi, build_log)
    row["static_l2"] = static_l2(APP_DIR / "BUILD" / "GAP8_V2" / "GCC_RISCV_FREERTOS" / "scheduler_bench")
    access_point = AccessPoint(wifi_cfg["profile"], wifi_cfg["ssid"], wifi_cfg["key"])
    access_point.ensure(log)
    access_point.down()
    log("  flash")
    flash_and_boot(image, uri, log=lambda message: log(f"  {message}"))

    console_log = case_dir / "console.log"
    console = RadioConsole(uri, console_log)
    collector = Collector()
    feed = None
    timing = data["timing"]
    try:
        deadline = time.monotonic() + float(timing["boot_timeout_s"])
        log("  connect")
        access_point.up(deadline)
        feed = CameraFeed(ip=wifi_cfg["deck_ip"], port=int(wifi_cfg["port"]), transport="udp",
                          connect_timeout_s=max(deadline - time.monotonic(), 1.0), read_timeout_s=10.0)
        feed.subscribe(collector)
        feed.start()
        run_s = (timing["settle_ms"] + 2 * timing["phase_ms"]) / 1000 + 30
        log(f"  measure (up to {run_s:.0f} s)")
        end = time.monotonic() + run_s
        while time.monotonic() < end:
            radio = console_log.read_text(errors="replace") if console_log.is_file() else ""
            if "SB|fail" in radio:
                raise BenchError(next(line for line in radio.splitlines() if "SB|fail" in line))
            if "SB|done" in radio:
                break
            time.sleep(1)
        else:
            raise BenchError("no SB|done line from the deck")
    except (BenchError, OSError) as exc:  # OSError: no video ever arrived (socket timeout)
        row["error"] = f"{type(exc).__name__}: {exc}" if not isinstance(exc, BenchError) else str(exc)
    finally:
        if feed is not None:
            feed.stop()
        console.close()

    radio_lines = [line.split("GAP8: ", 1)[1].strip() for line in
                   (console_log.read_text(errors="replace").splitlines() if console_log.is_file() else [])
                   if "GAP8: SB|" in line]
    lines = radio_lines
    (case_dir / "reports.txt").write_text("\n".join(lines) + "\n")
    for line in radio_lines:
        if line.startswith(("SB|open", "SB|mem", "SB|clock", "SB|l2")):
            row.update({f"{line[3:].split()[0]}_{k}": v for k, v in
                        (w.split("=", 1) for w in line.split()[1:] if "=" in w)})
    phases = parse_reports(lines)
    for kind in ("seq", "pipe"):
        phases.setdefault(kind, {}).update(host_fps(collector.frames[kind]))
        phases[kind].update(host_checked=collector.checked[kind], host_bad=collector.bad[kind])
    row["phases"] = phases
    if "error" not in row and any(p.get("sent") and not p["host_frames"] for p in phases.values()):
        row["error"] = "the deck sent frames but none reached the host"
    row["result"] = "error" if "error" in row else "measured"
    return row


def summarize(rows: list[dict], path: Path) -> None:
    head = ("| case | sched | inf/s | lat p50 | lat p99 | lat max | wait p99 | period p99 | sent/s "
            "| host fps | bad frames | enc p50 | send max | cluster % | verify |")
    lines = ["# scheduler_bench", "", head, "| --- " * 15 + "|"]
    for row in rows:
        if row.get("error"):
            lines.append(f"| {row['case']} | error: {row['error']} |" + " |" * 13)
            continue
        for kind in ("seq", "pipe"):
            p = row["phases"].get(kind, {})
            s = max(p.get("ms", 0), 1) / 1000
            lines.append(
                f"| {row['case']} | {kind} | {p.get('inf', 0) / s:.0f} | {p.get('lat_p50', '')} | "
                f"{p.get('lat_p99', '')} | {p.get('lat_max', '')} | {p.get('wait_p99', '')} | "
                f"{p.get('inf_per99', '')} | {p.get('sent', 0) / s:.1f} | {p.get('host_fps', '')} | "
                f"{p.get('host_bad', '')}/{p.get('host_checked', '')} | "
                f"{p.get('inf_enc50', '')} | {p.get('wait_sendmax', '')} | {p.get('res_clpm', 0) / 10:.1f} | {p.get('verify_failures', '?')}/{p.get('verify_runs', '?')} |")
    lines += ["", "Times in us. lat: input appeared to inference done; wait: input appeared to "
              "inference start; period: between inferences; send: queueing one frame for the "
              "link (FC). sent/s: frames the deck queued; host fps: frames that arrived; bad "
              "frames: of every 10th that arrived, those not whole (JPEG: restart markers, "
              "decode, size)."]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench", type=Path, default=HOST_DIR / "bench.yaml")
    parser.add_argument("--set", default="common")
    parser.add_argument("--case", action="append", help="one case by name (repeatable)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "drone.yaml")
    args = parser.parse_args(argv)

    data = yaml.safe_load(args.bench.read_text())
    cases = cases_of(data)
    names = args.case or data["sets"][args.set]
    unknown = [name for name in names if name not in cases]
    if unknown:
        parser.error(f"unknown cases {unknown}; choose from {sorted(cases)}")
    uri = load_config(args.config).radio.uri
    out = BENCH_DIR / "artifacts" / f"bench_{datetime.datetime.now():%Y%m%dT%H%M%S}"
    out.mkdir(parents=True)

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    log(f"{len(names)} cases -> {out}")
    rows = []
    for name in names:
        log(name)
        row = run_case(cases[name], data, uri, out, log)
        rows.append(row)
        with (out / "results.jsonl").open("a") as results:
            results.write(json.dumps(row) + "\n")
        summarize(rows, out / "summary.md")
        log(f"  {row['result']}" + (f": {row['error']}" if row.get("error") else ""))
    log(f"summary: {out / 'summary.md'}")
    return 0 if all(row["result"] == "measured" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
