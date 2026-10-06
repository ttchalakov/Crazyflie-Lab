"""Clock-alignment diagnostics for one recorded run.

``load_run`` times every device-clocked sample with one line per clock domain,
fitted along the fastest arrivals (see ``cf_lab.telemetry.sync``). These
helpers show, for the streams you name, how that line sits against the samples:

  analyze_time_sync(run_dir, ["mocap", "camera"])   # numbers per stream
  plot_time_sync(run_dir, ["mocap", "camera"])      # <run_dir>/time_sync.png

Each stream gets one row of two panels. Left: host arrival minus device time at
the clock's nominal rate, with the fitted line. The line's slope is the drift a
nominal-rate clock would accumulate, and it should touch the fastest samples.
Right: each sample's delay above the line. Its floor should be flat; a floor
that bends means the clock's rate or the link's minimum delay changed during
the run, and one line follows neither. Samples below zero arrived before the
line says they were measured: the fit dropped them as clock glitches (for
anchored camera frames, the anchor mapping is off). Every sample is drawn (as SVG: kaleido cannot render WebGL traces);
thinning them would hide the rare fastest arrivals that the line rests on.

The arrival is the one the fit used: for NatNet mocap that is the host arrival
minus Motive's own latency, when it was recorded. STM32 streams share one fit,
so a stream whose delays never reach zero lags the stream that set the line.
Camera frames carrying STM32 clock anchors are timed through the STM32 clock
rather than their own WiFi fit; their floor sits at the WiFi link's minimum
delay, the time the anchors keep out of the camera's timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cf_lab.telemetry.schema import (
    CLOCK_GAP8_MS,
    CLOCK_NATNET_FRAME,
    CLOCK_NONE,
    CLOCK_ROS_STAMP_S,
    CLOCK_STM32_MS,
)
from cf_lab.telemetry.sync import GAP8_STM32_FIT, RunData, arrival_ns, device_axis, load_run

# Host ns per device unit at each clock's nominal rate. NatNet counts frames, so
# its nominal rate is the mocap frame rate, rounded from the fit.
NOMINAL_NS = {CLOCK_STM32_MS: 1e6, CLOCK_GAP8_MS: 1e6, CLOCK_ROS_STAMP_S: 1e9}

# Image size: WIDTH x ROW_HEIGHT layout pixels per stream, written SCALE times larger.
WIDTH = 1500
ROW_HEIGHT = 380
SCALE = 3

# One colour per device, from the reference categorical palette's first three slots.
COLORS = {CLOCK_STM32_MS: "#2a78d6", CLOCK_GAP8_MS: "#eb6834",
          CLOCK_NATNET_FRAME: "#1baf7a", CLOCK_ROS_STAMP_S: "#1baf7a"}


@dataclass(frozen=True)
class StreamSync:
    """One stream's samples against the line that times them."""

    stream: str
    mapping: str  # the fit that times this stream, e.g. "stm32_ms" or "gap8_ms -> stm32_ms"
    clock: str
    t_s: np.ndarray  # fitted measurement time, seconds since the run started
    line_ms: np.ndarray  # the line minus device time at the nominal rate; 0 at the first sample
    delay_ms: np.ndarray  # arrival above the line
    drift_ppm: float  # device clock rate against the host's, minus nominal

    @property
    def offset_ms(self) -> np.ndarray:
        """Arrival minus device time at the nominal rate: what the left panel draws."""
        return self.line_ms + self.delay_ms

    def summary(self) -> dict[str, float | int | str]:
        delay = self.delay_ms
        median = float(np.median(delay))
        return {
            "stream": self.stream,
            "mapping": self.mapping,
            "samples": int(delay.size),
            "duration_s": float(self.t_s.max() - self.t_s.min()),
            "drift_ppm": self.drift_ppm,
            "drift_ms": float(abs(self.line_ms[-1] - self.line_ms[0])),
            "min_delay_ms": float(delay.min()),
            "median_delay_ms": median,
            "p99_delay_ms": float(np.percentile(delay, 99)),
            "jitter_ms": float(np.sqrt(np.mean((delay - median) ** 2))),
            "below_line": int(np.count_nonzero(delay < 0)),
        }


def analyze_time_sync(
    run: str | Path | RunData, streams: Sequence[str] | None = None
) -> list[StreamSync]:
    """Each named stream's samples against its clock line.

    By default, every device-clocked stream with at least two samples to align:
    one that recorded nothing, such as a camera that never connected, is left out.
    """
    if not isinstance(run, RunData):
        run = load_run(run)
    clocks = {name: info["clock"] for name, info in run.manifest["streams"].items()
              if name in run.streams}
    if streams is None:
        streams = [
            name for name, clock in clocks.items()
            if clock != CLOCK_NONE
            and sum(value is not None for value in device_axis(run.streams[name], clock)) > 1
        ]
    for name in streams:
        if name not in clocks:
            raise KeyError(f"Run has no stream {name!r}; it has {sorted(clocks)}.")
        if clocks[name] == CLOCK_NONE:
            raise ValueError(f"Stream {name!r} is stamped on the host; it has no clock to align.")
    return [_stream_sync(run, name, clocks[name]) for name in streams]


def _stream_sync(run: RunData, name: str, clock: str) -> StreamSync:
    if name == "camera" and GAP8_STM32_FIT in run.fits:
        mapping = f"{CLOCK_GAP8_MS} -> {CLOCK_STM32_MS}"  # see sync._gap8_via_stm32
    elif clock in run.fits:
        mapping = clock
    else:
        raise ValueError(f"Stream {name!r}: its {clock} clock had too few samples to fit.")

    table = run.streams[name]
    axis = device_axis(table, clock)
    keep = np.array([value is not None for value in axis])
    device = np.array([value for value in axis if value is not None], dtype=float)
    anchor_ns = run.manifest["created_monotonic_ns"]
    arrival_s = (np.array(arrival_ns(table), dtype=np.int64)[keep] - anchor_ns) / 1e9
    t_s = table.column("t_s").to_numpy()[keep]
    order = np.argsort(device, kind="stable")
    device, arrival_s, t_s = device[order], arrival_s[order], t_s[order]

    # The line is affine in device time, so its two ends give its slope.
    slope_ns = (t_s[-1] - t_s[0]) * 1e9 / (device[-1] - device[0])
    nominal_ns = NOMINAL_NS.get(clock) or 1e9 / round(1e9 / slope_ns)
    nominal_s = t_s[0] + (device - device[0]) * nominal_ns / 1e9
    return StreamSync(
        stream=name,
        mapping=mapping,
        clock=clock,
        t_s=t_s,
        line_ms=(t_s - nominal_s) * 1e3,
        delay_ms=(arrival_s - t_s) * 1e3,
        drift_ppm=(slope_ns / nominal_ns - 1.0) * 1e6,
    )


def plot_time_sync(
    run_dir: str | Path,
    streams: Sequence[str] | None = None,
    path: str | Path | None = None,
) -> Path:
    """Write the clock-alignment figure for the named streams; returns the PNG path.

    ``path`` defaults to ``<run_dir>/time_sync.png``. Also prints each stream's
    summary numbers.
    """
    run_dir = Path(run_dir)
    syncs = analyze_time_sync(run_dir, streams)
    figure = time_sync_figure(syncs, title=f"Clock alignment: {run_dir.name}")
    path = Path(path) if path is not None else run_dir / "time_sync.png"
    figure.write_image(path, width=figure.layout.width, height=figure.layout.height, scale=SCALE)
    for sync in syncs:
        print(_summary_line(sync.summary()))
    print(f"Wrote {path}")
    return path


def time_sync_figure(syncs: Sequence[StreamSync], *, title: str = "Clock alignment") -> go.Figure:
    """One row per stream: offset with the fitted line, then delay above the line."""
    stats = [sync.summary() for sync in syncs]
    titles = []
    for s in stats:
        titles += [f"<b>{s['stream']}</b> ({s['mapping']}): drift {s['drift_ppm']:+.0f} ppm, "
                   f"{s['drift_ms']:.1f} ms over {s['duration_s']:.0f} s",
                   f"delay above the line: median {s['median_delay_ms']:.2f}, "
                   f"p99 {s['p99_delay_ms']:.1f}, jitter {s['jitter_ms']:.2f} ms"]
    figure = make_subplots(rows=len(syncs), cols=2, column_widths=[0.6, 0.4],
                           subplot_titles=titles, horizontal_spacing=0.07,
                           vertical_spacing=min(0.12, 0.35 / len(syncs)))
    for row, (sync, s) in enumerate(zip(syncs, stats), start=1):
        marker = {"size": 2.5, "color": COLORS[sync.clock], "opacity": 0.35}
        offset = sync.offset_ms
        figure.add_trace(go.Scatter(x=sync.t_s, y=offset, mode="markers", marker=marker,
                                    showlegend=False), row=row, col=1)
        ends = [0, -1]
        figure.add_trace(go.Scatter(x=sync.t_s[ends], y=sync.line_ms[ends], mode="lines",
                                    line={"color": "#1f1f1f", "width": 2.5}, showlegend=False),
                         row=row, col=1)
        low, high = np.percentile(offset, [0.1, 99.7])
        low, high = min(low, sync.line_ms.min()), max(high, sync.line_ms.max())
        pad = 0.1 * (high - low)
        figure.update_yaxes(range=[low - pad, high + pad], title_text="arrival − device clock (ms)",
                            row=row, col=1)

        delay = sync.delay_ms
        figure.add_trace(go.Scatter(x=sync.t_s, y=delay, mode="markers", marker=marker,
                                    showlegend=False), row=row, col=2)
        figure.add_hline(y=0, line={"color": "#1f1f1f", "width": 2}, row=row, col=2)
        figure.add_hline(y=s["median_delay_ms"], row=row, col=2,
                         line={"color": "#1f1f1f", "width": 1.2, "dash": "dash"})
        high = np.percentile(delay, 99.5)
        low = min(0.0, s["min_delay_ms"])
        pad = 0.1 * (high - low)
        figure.update_yaxes(range=[low - pad, high + pad], title_text="delay (ms)", row=row, col=2)
        for col in (1, 2):
            figure.update_xaxes(range=[sync.t_s[0], sync.t_s[-1]], row=row, col=col)

    for col in (1, 2):
        figure.update_xaxes(title_text="time into run (s)", row=len(syncs), col=col)
    figure.update_annotations(font={"size": 14})  # the subplot titles
    figure.update_layout(title={"text": title, "x": 0.5}, template="plotly_white",
                         font={"size": 14}, width=WIDTH, height=120 + ROW_HEIGHT * len(syncs),
                         margin={"l": 80, "r": 30, "t": 90, "b": 60})
    return figure


def _summary_line(stats: dict) -> str:
    return (f"{stats['stream']:>22} ({stats['mapping']}): {stats['samples']:>6,} samples, "
            f"drift {stats['drift_ppm']:+7.1f} ppm ({stats['drift_ms']:.1f} ms over "
            f"{stats['duration_s']:.0f} s), delay min {stats['min_delay_ms']:.2f} / median "
            f"{stats['median_delay_ms']:.2f} / p99 {stats['p99_delay_ms']:.1f} ms, "
            f"jitter {stats['jitter_ms']:.2f} ms, {stats['below_line']} below the line")
