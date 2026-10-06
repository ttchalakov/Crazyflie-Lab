"""Live stream metrics: frame rates, bitrate, deck timings, drops.

``FeedStats`` is a frame sink, so it counts every frame the feed delivers, not
only the ones a viewer gets round to painting. ``offer`` runs on the receive
thread and only appends to two short windows; readers take a ``snapshot()``.

Latency is two honest numbers rather than one dishonest one. The deck's own
capture and encode times are absolute. The link figure is *relative*: host and
deck clocks share no origin, so the running minimum of ``arrival - deck_time``
is subtracted, which cancels the unknown offset and the fixed pipeline floor and
leaves the jitter above it. An absolute link latency is not measurable without a
synchronized clock, so none is reported.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, replace

from cf_lab.camera.feed import CapturedFrame


@dataclass(frozen=True)
class StatsSnapshot:
    frames: int = 0
    fps: float = 0.0
    camera_fps: float = 0.0
    bitrate_kbps: float = 0.0
    last_size: int = 0
    width: int = 0
    height: int = 0
    is_jpeg: bool = False
    last_arrival_s: float = 0.0
    # From the camera_stream extension; zero when the app sends none.
    has_ext: bool = False
    frame_id: int = 0
    camera_frame_id: int = 0
    capture_ms: int = 0
    preprocess_ms: int = 0
    jpeg_ms: int = 0
    deck_ms: int = 0
    link_jitter_ms: float = 0.0
    dropped_capture: int = 0
    dropped_ready: int = 0
    dropped_streaming: int = 0


class FeedStats:
    """Rolling metrics over the last ``window_s`` seconds of frames."""

    def __init__(self, window_s: float = 1.0) -> None:
        self._window_s = window_s
        self._lock = threading.Lock()
        self._arrivals: deque[tuple[float, int]] = deque()  # (arrival_s, bytes)
        self._captures: deque[tuple[float, int]] = deque()  # (deck_done_s, camera_frame_id)
        self._min_offset_ms: float | None = None
        self._last_deck_ms: int | None = None
        self._last = StatsSnapshot()

    def offer(self, frame: CapturedFrame) -> None:
        arrival_s = frame.host_ns / 1e9
        with self._lock:
            self._arrivals.append((arrival_s, frame.header.size))
            while self._arrivals[0][0] < arrival_s - self._window_s:
                self._arrivals.popleft()
            values = dict(
                frames=self._last.frames + 1,
                last_size=frame.header.size,
                width=frame.width,
                height=frame.height,
                is_jpeg=frame.is_jpeg,
                last_arrival_s=arrival_s,
            )
            ext = frame.ext
            if ext is not None:
                # A rebooted deck restarts its clock (a UDP feed outlives the reboot):
                # measure from the new clock, not against the old one. More than a
                # window back, so one late UDP frame does not count as a reboot.
                if self._last_deck_ms is not None and (
                    ext.deck_ms < self._last_deck_ms - self._window_s * 1000.0
                ):
                    self._captures.clear()
                    self._min_offset_ms = None
                self._last_deck_ms = ext.deck_ms
                offset_ms = arrival_s * 1000.0 - float(ext.deck_ms)
                if self._min_offset_ms is None or offset_ms < self._min_offset_ms:
                    self._min_offset_ms = offset_ms
                camera_frame_id = getattr(ext, "camera_frame_id", 0)
                camera_done_ms = getattr(ext, "camera_done_ms", 0)
                if camera_frame_id > 0 and camera_done_ms > 0:
                    done_s = camera_done_ms / 1000.0
                    self._captures.append((done_s, camera_frame_id))
                    while self._captures[0][0] < done_s - self._window_s:
                        self._captures.popleft()
                values.update(
                    has_ext=True,
                    frame_id=ext.frame_id,
                    camera_frame_id=camera_frame_id,
                    capture_ms=getattr(ext, "capture_ms", 0),
                    preprocess_ms=getattr(ext, "preprocess_ms", 0),
                    jpeg_ms=getattr(ext, "jpeg_ms", 0),
                    deck_ms=getattr(ext, "encode_ms", 0),
                    link_jitter_ms=offset_ms - self._min_offset_ms,
                    dropped_capture=getattr(ext, "dropped_capture", 0),
                    dropped_ready=getattr(ext, "dropped_ready", 0),
                    dropped_streaming=getattr(ext, "dropped_streaming", 0),
                )
            # Bytes after the frame that opens the window, over the window's span.
            window_bytes = sum(n for _, n in self._arrivals) - self._arrivals[0][1]
            self._last = replace(
                self._last,
                **values,
                fps=_rate(self._arrivals, count=len(self._arrivals) - 1),
                camera_fps=_rate(self._captures, count=_id_span(self._captures)),
                bitrate_kbps=_rate(self._arrivals, count=window_bytes) * 8 / 1000.0,
            )

    def snapshot(self) -> StatsSnapshot:
        with self._lock:
            return self._last


def _rate(window: deque, *, count: float) -> float:
    span = window[-1][0] - window[0][0] if len(window) >= 2 else 0.0
    return count / span if span > 0 else 0.0


def _id_span(window: deque) -> int:
    return window[-1][1] - window[0][1] if len(window) >= 2 else 0


def stats_line(stats: StatsSnapshot) -> str:
    """One-line summary for a terminal."""
    line = (
        f"rx {stats.fps:5.1f} fps  cam {stats.camera_fps:5.1f} fps  "
        f"{stats.bitrate_kbps:7.0f} kbps  {stats.width}x{stats.height} "
        f"{'JPEG' if stats.is_jpeg else 'raw'} {stats.last_size} B"
    )
    if stats.has_ext:
        line += (
            f"  deck {stats.deck_ms} ms (cap {stats.capture_ms} resize {stats.preprocess_ms} "
            f"jpeg {stats.jpeg_ms})  jitter +{stats.link_jitter_ms:.1f} ms  "
            f"drops {stats.dropped_capture}/{stats.dropped_ready}/{stats.dropped_streaming}"
        )
    return line
