"""Measure a running camera stream and judge it against the case and a baseline.

``Collector`` is a sink on a ``CameraFeed``: it keeps every frame's arrival time,
size and camera_stream extension, and every status line the bench app sends, once
``start()`` is called. ``summarize`` turns that into the numbers a bench row
records; ``judge`` checks them.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field

# Status fields that count trouble on the deck; each must not grow while measured.
DECK_FAULT_COUNTERS = ("watchdog_restarts", "encode_failures", "spi_hard")


@dataclass
class Collector:
    """Frames and status lines, from ``start()`` on."""

    frames: list = field(default_factory=list)  # (arrival_s, size, ext)
    status: list = field(default_factory=list)  # (arrival_s, fields)
    bench: list = field(default_factory=list)  # (arrival_s, fields)
    console: list = field(default_factory=list)  # (arrival_s, DeckLine)
    collecting: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> None:
        with self._lock:
            self.frames.clear()
            self.status.clear()
            self.bench.clear()
            self.collecting = True

    def stop(self) -> None:
        self.collecting = False

    def offer(self, frame) -> None:
        if self.collecting:
            with self._lock:
                self.frames.append((frame.host_ns / 1e9, frame.header.size, frame.ext))

    def offer_console(self, line) -> None:
        now = time.monotonic()
        self.console.append((now, line))
        if not self.collecting:
            return
        with self._lock:
            if line.kind == "status":
                self.status.append((now, line.fields))
            elif line.text.startswith("bench "):
                self.bench.append((now, _fields(line.text[len("bench "):])))


def _fields(text: str) -> dict:
    out = {}
    for word in text.split():
        key, sep, value = word.partition("=")
        if sep:
            out[key] = int(value) if value.isdigit() else value
    return out


def _series(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        index = (len(ordered) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return {"min": ordered[0], "avg": statistics.fmean(ordered), "p50": pct(0.5),
            "p95": pct(0.95), "max": ordered[-1]}


def _window_fps(times: list[float], window_s: float = 1.0) -> list[float]:
    """Frame rate over each trailing ``window_s`` second window."""
    rates = []
    start = 0
    for end, t in enumerate(times):
        while times[start] < t - window_s:
            start += 1
        if t - times[start] >= window_s * 0.5:
            rates.append((end - start) / (t - times[start]))
    return rates


def summarize(collector: Collector) -> dict:
    """The numbers one bench row records."""
    with collector._lock:
        frames = list(collector.frames)
        status = list(collector.status)
        bench = list(collector.bench)
    if len(frames) < 2:
        return {"frames": len(frames)}

    times = [t for t, _, _ in frames]
    sizes = [size for _, size, _ in frames]
    exts = [ext for _, _, ext in frames if ext is not None]
    span = times[-1] - times[0]
    summary: dict = {
        "frames": len(frames),
        "duration_s": span,
        "fps_avg": (len(frames) - 1) / span,
        "fps_1s": _series(_window_fps(times)),
        "bitrate_kbps_avg": sum(sizes[1:]) * 8 / span / 1000,
        "frame_bytes": _series([float(size) for size in sizes]),
        "interarrival_ms": _series([(b - a) * 1000 for a, b in zip(times, times[1:])]),
    }
    if len(exts) >= 2:
        first, last = exts[0], exts[-1]
        captured_s = (last.camera_done_ms - first.camera_done_ms) / 1000
        offsets = [t * 1000 - ext.deck_ms for (t, _, ext) in frames if ext is not None]
        floor = min(offsets)
        summary.update(
            camera_fps_avg=(last.camera_frame_id - first.camera_frame_id) / captured_s
            if captured_s > 0 else 0.0,
            # Frames the deck sent (its frame_id counts sends) that never arrived.
            frames_lost_in_transit=(last.frame_id - first.frame_id + 1) - len(exts),
            deck_capture_ms=_series([float(ext.capture_ms) for ext in exts]),
            deck_preprocess_ms=_series([float(ext.preprocess_ms) for ext in exts]),
            deck_jpeg_ms=_series([float(ext.jpeg_ms) for ext in exts]),
            deck_total_ms=_series([float(ext.encode_ms) for ext in exts]),
            link_jitter_ms=_series([offset - floor for offset in offsets]),
            drops={
                "capture": last.dropped_capture - first.dropped_capture,
                "ready": last.dropped_ready - first.dropped_ready,
                "while_sending": last.dropped_streaming - first.dropped_streaming,
            },
        )
    if len(status) >= 2:
        first, last = status[0][1], status[-1][1]
        summary["deck_faults"] = {
            key: int(last.get(key, 0)) - int(first.get(key, 0)) for key in DECK_FAULT_COUNTERS
        }
        summary["deck_faulted"] = int(last.get("faulted", 0))
        summary["tx_free_min"] = min(int(fields.get("tx_free", 0)) for _, fields in status)
    if bench:
        last = bench[-1][1]
        summary["l2_free_min"] = int(last.get("l2_free_min", 0))
        summary["l2_largest_min"] = min(int(fields.get("l2_largest", 0)) for _, fields in bench)
        for task in ("stream", "camera", "status"):
            summary[f"stack_{task}_free_min"] = min(
                int(fields.get(f"stack_{task}", 0)) for _, fields in bench)
    return summary


@dataclass(frozen=True)
class Acceptance:
    camera_fps_tolerance: float = 0.02
    received_fps_ratio: float = 0.9
    latency_tolerance: float = 0.1
    latency_floor_ms: float = 1.0


def judge(summary: dict, *, config: dict, transport: str, baseline: dict | None,
          acceptance: Acceptance) -> list[dict]:
    """Checks for one measured case; each is {name, passed, measured, want}.

    The received rate depends on the WiFi around the bench as much as on the
    build, so its check is advisory: reported, never failing the case.
    """
    checks: list[dict] = []

    def check(name: str, passed: bool, measured, want, advisory: bool = False) -> None:
        checks.append({"name": name, "passed": bool(passed), "measured": measured, "want": want,
                       **({"advisory": True} if advisory else {})})

    check("frames", summary.get("frames", 0) >= 2, summary.get("frames", 0), ">= 2")
    if summary.get("frames", 0) < 2:
        return checks

    sensor_fps = float(config.get("sensor_fps", 0))
    if "camera_fps_avg" in summary and sensor_fps:
        low, high = (sensor_fps * (1 - acceptance.camera_fps_tolerance),
                     sensor_fps * (1 + acceptance.camera_fps_tolerance))
        measured = round(summary["camera_fps_avg"], 2)
        check("camera_fps", low <= measured <= high, measured, f"{low:.1f}..{high:.1f}")
    if transport == "tcp" and "frames_lost_in_transit" in summary:
        lost = summary["frames_lost_in_transit"]
        check("tcp_frames_lost", lost == 0, lost, "== 0")
    for key, value in summary.get("deck_faults", {}).items():
        check(f"deck_{key}", value == 0, value, "== 0")
    if "deck_faulted" in summary:
        check("camera_faulted", summary["deck_faulted"] == 0, summary["deck_faulted"], "== 0")

    if baseline:
        want = baseline.get("fps_avg", 0.0) * acceptance.received_fps_ratio
        measured = round(summary["fps_avg"], 2)
        check("received_fps", measured >= want, measured, f">= {want:.1f}", advisory=True)
        for key in ("deck_preprocess_ms", "deck_jpeg_ms"):
            if key not in summary or key not in baseline:
                continue
            base = baseline[key]["avg"]
            limit = base + max(base * acceptance.latency_tolerance, acceptance.latency_floor_ms)
            measured = round(summary[key]["avg"], 2)
            check(key, measured <= limit, measured, f"<= {limit:.1f}")
    return checks
