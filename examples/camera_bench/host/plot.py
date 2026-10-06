#!/usr/bin/env python3
"""Draw one camera_bench run as high-resolution PNGs and a results table, in its folder.

  python examples/camera_bench/host/plot.py examples/camera_bench/artifacts/bench_<stamp>

Writes <run>/plots/: results.md, the table of every case, and a PNG per plot:
which rate each case received against what the camera captured (rate.png), how
steady that rate was (timeline.png), where the deck spends its time per frame
(deck_time.png), how late frames arrive (jitter.png), and how much L2 each
build leaves free (l2.png). A small run gets a bar per case; a large one (the
exhaustive set) gets grids of sensor and size against format and policy, one
panel per transport, with the value in each cell.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Rate-over-time panels are drawn for at most this many cases (an exhaustive run
# has ~200): the least steady ones when there are more.
MAX_TIMELINES = 12

# From this many measured cases on, the per-case bars become grids.
GRID_FROM = 30

# Image size: WIDTH layout pixels, written SCALE times larger.
WIDTH = 1100
SCALE = 3

# Series colours by role: the reference categorical palette's first three slots,
# validated as a set, and a recessive gray for context.
COLORS = {"received": "#2a78d6", "capture": "#2a78d6", "resize": "#eb6834",
          "jpeg": "#1baf7a", "context": "#c3c2b7"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                          "#e1e0d9", "#c3c2b7")
# Grid cells: the reference sequential blue, steps 100 to 700 (more is darker);
# a case that did not fit in L2 is gray, a failed one the critical status red.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
UNSUPPORTED_CELL, FAILED_CELL = "#e1e0d9", "#d03b3b"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

_AXIS = dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, ticks="",
             tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED)))
TEMPLATE = go.layout.Template(layout=dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family=FONT, size=12, color=INK),
    xaxis=_AXIS, yaxis=_AXIS,
))


def load_run(run_dir: Path):
    info = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").is_file() else {}
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line]
    return info, rows


def load_frames(case_dir: Path) -> list[dict]:
    path = case_dir / "frames.csv"
    if not path.is_file():
        return []
    with path.open() as handle:
        return [{key: float(value) if value else None for key, value in row.items()}
                for row in csv.DictReader(handle)]


def window_fps(times: list[float], window_s: float = 1.0) -> tuple[list[float], list[float]]:
    """Received rate over each trailing ``window_s`` window, sampled every 0.25 s."""
    xs, ys = [], []
    start, t = 0, window_s
    while times and t <= times[-1]:
        while start < len(times) and times[start] < t - window_s:
            start += 1
        end = start
        while end < len(times) and times[end] <= t:
            end += 1
        xs.append(round(t, 3))
        ys.append((end - start) / window_s)
        t += 0.25
    return xs, ys


def _bar_height(count: int, bars_per_case: int = 1) -> int:
    return 130 + (18 + 14 * bars_per_case) * count


def _base_layout(title: str, height: int, xtitle: str, *, per_case: bool = True,
                 legend: bool = True) -> dict:
    """Shared chrome; ``per_case`` charts list the cases down a categorical y axis."""
    layout = dict(
        template=TEMPLATE,
        # The title sits at the top left of the figure and the legend on its own
        # row just above the plot's right end, clear of any panel titles.
        title=dict(text=title, x=0, xanchor="left", xref="container", y=1, yref="container",
                   yanchor="top", pad=dict(t=10, l=10), font=dict(size=15)),
        width=WIDTH, height=height - (0 if legend else 28),
        margin=dict(l=10, r=40, t=84 if legend else 56, b=40),
        xaxis=dict(title=xtitle, zeroline=False, showline=True),
        legend=dict(orientation="h", x=1, xanchor="right", y=1, yanchor="bottom", title=None,
                    traceorder="normal"),
    )
    if per_case:
        layout.update(barmode="group", bargap=0.25, bargroupgap=0.1,
                      yaxis=dict(automargin=True, autorange="reversed", ticklabelstandoff=8))
    return layout


def _value_labels(values: list[float], fmt: str) -> dict:
    """Numbers at the bar ends, in text ink: a PNG has no hover to read them from."""
    return dict(text=[format(value, fmt) for value in values], textposition="outside",
                textfont=dict(color=INK_2, size=11), cliponaxis=False)


def rate_chart(measured: list[dict]) -> go.Figure:
    names = [row["case"] for row in measured]
    captured = [row["summary"].get("camera_fps_avg", 0.0) for row in measured]
    received = [row["summary"].get("fps_avg", 0.0) for row in measured]
    figure = go.Figure()
    figure.add_bar(y=names, x=captured, orientation="h", name="Captured by the camera",
                   marker=dict(color=COLORS["context"], cornerradius=4))
    figure.add_bar(y=names, x=received, orientation="h", name="Received by the host",
                   marker=dict(color=COLORS["received"], cornerradius=4),
                   **_value_labels(received, ".1f"))
    figure.update_layout(**_base_layout("Frame rate: received against captured",
                                        _bar_height(len(names), 2), "frames per second"))
    return figure


def _steadiness(row: dict) -> float:
    """The worst second's rate as a share of the average rate."""
    s = row["summary"]
    return s["fps_1s"]["min"] / s["fps_avg"] if s.get("fps_avg") else 0.0


def timeline_cases(measured: list[dict], run_dir: Path) -> list[dict]:
    shown = [row for row in measured if (run_dir / row["case"] / "frames.csv").is_file()]
    if len(shown) > MAX_TIMELINES:
        shown = sorted(shown, key=_steadiness)[:MAX_TIMELINES]
    return shown


def timeline_chart(measured: list[dict], run_dir: Path) -> go.Figure | None:
    shown = timeline_cases(measured, run_dir)
    if not shown:
        return None
    figure = make_subplots(rows=len(shown), cols=1, shared_xaxes=True,
                           vertical_spacing=min(0.12, 60 / (170 * len(shown))),
                           subplot_titles=[row["case"] for row in shown])
    for index, row in enumerate(shown, 1):
        frames = load_frames(run_dir / row["case"])
        xs, ys = window_fps([frame["t_s"] for frame in frames])
        captured = row["summary"].get("camera_fps_avg")
        if captured:
            figure.add_scatter(x=[xs[0], xs[-1]] if xs else [0, 1], y=[captured, captured],
                               mode="lines", line=dict(color=COLORS["context"], width=2),
                               name="Captured", showlegend=index == 1, row=index, col=1)
        figure.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=COLORS["received"], width=2),
                           name="Received, 1 s window", showlegend=index == 1, row=index, col=1)
        figure.update_yaxes(rangemode="tozero", automargin=True, ticklabelstandoff=6, row=index, col=1)
    title = "Received rate over the measurement, frames per second"
    if len(shown) < len([row for row in measured if (run_dir / row["case"] / "frames.csv").is_file()]):
        title += f": the {len(shown)} least steady cases, by their worst second"
    figure.update_layout(**_base_layout(title, 130 + 170 * len(shown), "", per_case=False))
    figure.update_xaxes(title_text="seconds into the measurement", row=len(shown), col=1)
    figure.update_annotations(font=dict(size=12, color=INK), x=0, xanchor="left", yanchor="bottom")
    return figure


def deck_time_chart(measured: list[dict]) -> go.Figure:
    names = [row["case"] for row in measured]
    figure = go.Figure()
    for role, key, label in (("capture", "deck_capture_ms", "Capture"),
                             ("resize", "deck_preprocess_ms", "Resize"),
                             ("jpeg", "deck_jpeg_ms", "JPEG")):
        values = [row["summary"].get(key, {}).get("avg", 0.0) for row in measured]
        # A 2 px surface-coloured outline keeps the stacked segments apart.
        figure.add_bar(y=names, x=values, orientation="h", name=label,
                       marker=dict(color=COLORS[role], cornerradius=4,
                                   line=dict(color=SURFACE, width=2)))
    layout = _base_layout("Deck time per frame (average)", _bar_height(len(names)), "milliseconds")
    layout["barmode"] = "stack"
    figure.update_layout(**layout)
    return figure


def jitter_chart(measured: list[dict], run_dir: Path) -> go.Figure | None:
    figure = go.Figure()
    for row in measured:
        frames = [f for f in load_frames(run_dir / row["case"]) if f.get("deck_ms") is not None]
        if not frames:
            continue
        offsets = [f["t_s"] * 1000 - f["deck_ms"] for f in frames]
        floor = min(offsets)
        figure.add_box(x=[offset - floor for offset in offsets], y=[row["case"]] * len(offsets),
                       orientation="h", name=row["case"], boxpoints=False,
                       line=dict(color=COLORS["received"], width=1.5),
                       fillcolor="rgba(42, 120, 214, 0.2)", showlegend=False)
    if not figure.data:
        return None
    figure.update_layout(**_base_layout("How late frames arrive, above the fastest one",
                                        _bar_height(len(figure.data)),
                                        "milliseconds (relative; the clocks share no origin)",
                                        legend=False))
    return figure


def l2_chart(measured: list[dict]) -> go.Figure | None:
    rows = [row for row in measured if row["summary"].get("l2_free_min")]
    if not rows:
        return None
    free = [row["summary"]["l2_free_min"] / 1024 for row in rows]
    figure = go.Figure()
    figure.add_bar(y=[row["case"] for row in rows], x=free, orientation="h", showlegend=False,
                   marker=dict(color=COLORS["received"], cornerradius=4),
                   **_value_labels(free, ".0f"))
    figure.update_layout(**_base_layout("Lowest free L2 while streaming", _bar_height(len(rows)),
                                        "KiB", legend=False))
    return figure


# -- grids, for large runs --------------------------------------------------------

def _grid_key(row: dict) -> tuple[str, str, str] | None:
    """(row label, column label, transport) of a matrix case, else None."""
    parts = row["case"].split("-")
    if len(parts) != 5:
        return None
    sensor, size, fmt, policy, transport = parts
    return f"{sensor} {size}", f"{fmt}<br>{policy}", transport


def _grid_axes(rows: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """Row, column and transport labels in run order, which follows the matrix."""
    ys, xs, transports = [], [], []
    for row in rows:
        key = _grid_key(row)
        if key is None:
            continue
        for value, seen in zip(key, (ys, xs, transports)):
            if value not in seen:
                seen.append(value)
    return ys, xs, transports


def grid_chart(rows: list[dict], title: str, metrics: list[tuple[str, callable, callable]],
               fmt: str) -> go.Figure | None:
    """One grid per metric (a figure row) and transport (a column).

    Each metric is (label, value, shade): value(row) is printed in the cell,
    shade(row) sets its colour, on a scale shared by the metric's panels. A case
    that did not fit in L2 or failed is marked instead; a combination the run
    did not include stays blank.
    """
    ys, xs, transports = _grid_axes(rows)
    if not ys:
        return None
    by_key = {_grid_key(row): row for row in rows if _grid_key(row)}
    labels = [f"{label}, {transport.upper()}" if len(transports) > 1 else label
              for label, _, _ in metrics for transport in transports]
    # Each panel: its rows plus its title and column labels, which sit above it.
    panel, gap, top = 24 * len(ys), 90, 110
    height = top + panel * len(metrics) + gap * (len(metrics) - 1) + 20
    figure = make_subplots(rows=len(metrics), cols=len(transports), shared_yaxes=True,
                           horizontal_spacing=0.03,
                           vertical_spacing=gap / (height - top - 20),
                           subplot_titles=labels)
    for m, (_, value, shade) in enumerate(metrics, 1):
        shades = [shade(row) for row in rows if row.get("result") == "pass"]
        zmax = max((v for v in shades if v is not None), default=1.0) or 1.0
        for t, transport in enumerate(transports, 1):
            z, text, marks, mark_text = [], [], [], []
            for y in ys:
                z_row, text_row, mark_row, mark_text_row = [], [], [], []
                for x in xs:
                    row = by_key.get((y, x, transport))
                    result = row.get("result") if row else None
                    v = value(row) if result == "pass" else None
                    z_row.append(shade(row) if v is not None else None)
                    text_row.append(format(v, fmt) if v is not None else "")
                    mark = {"unsupported": 0, "fail": 1}.get(result)
                    mark_row.append(mark)
                    mark_text_row.append({0: "L2", 1: "fail"}.get(mark, ""))
                z.append(z_row), text.append(text_row)
                marks.append(mark_row), mark_text.append(mark_text_row)
            figure.add_heatmap(z=marks, x=xs, y=ys, text=mark_text, texttemplate="%{text}",
                               colorscale=[[0, UNSUPPORTED_CELL], [1, FAILED_CELL]], zmin=0, zmax=1,
                               showscale=False, xgap=2, ygap=2, row=m, col=t)
            figure.add_heatmap(z=z, x=xs, y=ys, text=text, texttemplate="%{text}",
                               colorscale=[[i / (len(SEQUENTIAL) - 1), c]
                                           for i, c in enumerate(SEQUENTIAL)],
                               zmin=0, zmax=zmax, showscale=False, xgap=2, ygap=2,
                               textfont=dict(size=11), row=m, col=t)
    figure.update_yaxes(autorange="reversed", showgrid=False, automargin=True, ticklabelstandoff=6)
    figure.update_xaxes(showgrid=False, side="top", tickfont=dict(size=10))
    layout = _base_layout(title, height, "", per_case=False, legend=False)
    layout.pop("xaxis")
    layout["height"] = height
    layout["margin"] = dict(l=10, r=20, t=top, b=20)
    figure.update_layout(**layout)
    # Panel titles go above the column labels, left-aligned with the panel.
    figure.update_annotations(font=dict(size=12, color=INK), xanchor="left", yanchor="bottom",
                              yshift=34)
    for annotation, axis in zip(figure.layout.annotations,
                                ("xaxis" + (str(i) if i > 1 else "") for i in range(1, len(labels) + 1))):
        annotation.x = figure.layout[axis].domain[0]
    return figure


def _summary(key: str, stat: str = "avg"):
    return lambda row: row["summary"].get(key, {}).get(stat)


def _sensor_share(row: dict) -> float | None:
    sensor_fps = (row.get("deck_config") or {}).get("sensor_fps")
    return row["summary"]["fps_avg"] / sensor_fps if sensor_fps else None


def _unless(token: str, getter):
    """The value, except for cases whose name has ``token`` (no resize, no JPEG)."""
    return lambda row: None if token in row["case"].split("-") else getter(row)


def grid_figures(rows: list[dict]) -> dict[str, go.Figure | None]:
    received = lambda row: row["summary"].get("fps_avg")
    free_l2 = lambda row: row["summary"]["l2_free_min"] / 1024 if row["summary"].get("l2_free_min") else None
    return {
        "rate": grid_chart(rows, "Received frames per second; shade: the share of the sensor's rate",
                           [("Received FPS", received, _sensor_share)], ".1f"),
        "deck_time": grid_chart(rows, "Deck time per frame, milliseconds (average)", [
            ("Capture", _summary("deck_capture_ms"), _summary("deck_capture_ms")),
            ("Resize", _unless("native", _summary("deck_preprocess_ms")),
             _unless("native", _summary("deck_preprocess_ms"))),
            ("JPEG", _unless("raw", _summary("deck_jpeg_ms")), _unless("raw", _summary("deck_jpeg_ms"))),
        ], ".1f"),
        "jitter": grid_chart(rows, "How late frames arrive, milliseconds", [
            ("Late, 95th percentile", _summary("link_jitter_ms", "p95"), _summary("link_jitter_ms", "p95")),
            ("Longest gap between frames", _summary("interarrival_ms", "max"),
             _summary("interarrival_ms", "max")),
        ], ".0f"),
        "l2": grid_chart(rows, "Lowest free L2 while streaming, KiB", [("Free L2", free_l2, free_l2)],
                         ".0f"),
    }


def _notes(row: dict) -> str:
    """What a special case measured, then any failed check."""
    s = row.get("summary", {})
    notes = []
    if "connect_to_5_frames_s" in s:
        notes.append(f"{s.get('cycles', 0)} reconnects, 5 frames within "
                     f"{s['connect_to_5_frames_s']['max']:.2f} s at worst")
    if "lifecycle" in s:
        notes.append(s["lifecycle"])
    if row.get("result") == "unsupported":
        return "out of L2 memory"
    notes += [f"{c['name']}={c['measured']} (want {c['want']})"
              for c in row.get("checks", []) if not c["passed"]]
    return "; ".join(notes)


def results_table(rows: list[dict]) -> str:
    head = ("Case", "Result", "Received FPS", "Captured FPS", "Mbit/s", "Frame KiB",
            "JPEG ms", "Jitter p95 ms", "Longest gap ms", "Free L2 KiB", "Notes")
    lines = ["| " + " | ".join(head) + " |",
             "| --- | --- | " + " | ".join("---:" for _ in head[2:-1]) + " | --- |"]
    for row in rows:
        s = row.get("summary", {})
        raw = (row.get("deck_config") or {}).get("format") == "raw"

        def num(value, fmt):
            return format(value, fmt) if isinstance(value, (int, float)) else ""

        cells = (
            row["case"], row.get("result", "pass" if row.get("passed") else "fail"),
            num(s.get("fps_avg"), ".1f"), num(s.get("camera_fps_avg"), ".1f"),
            num(s["bitrate_kbps_avg"] / 1000 if "bitrate_kbps_avg" in s else None, ".2f"),
            num(s["frame_bytes"]["avg"] / 1024 if "frame_bytes" in s else None, ".1f"),
            num(None if raw else s.get("deck_jpeg_ms", {}).get("avg"), ".1f"),
            num(s.get("link_jitter_ms", {}).get("p95"), ".0f"),
            num(s.get("interarrival_ms", {}).get("max"), ".0f"),
            num(s["l2_free_min"] / 1024 if s.get("l2_free_min") else None, ".0f"),
            _notes(row),
        )
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="a camera_bench run folder, bench_<stamp>")
    args = parser.parse_args(argv)

    info, rows = load_run(args.run_dir)
    measured = [row for row in rows if row.get("kind") in ("measure", "soak")
                and row.get("summary", {}).get("frames", 0) >= 2]
    if len(measured) >= GRID_FROM:
        grids = grid_figures([row for row in rows if row.get("kind") == "measure"])
        charts = {"rate": grids["rate"], "timeline": timeline_chart(measured, args.run_dir),
                  "deck_time": grids["deck_time"], "jitter": grids["jitter"], "l2": grids["l2"]}
    else:
        charts = {"rate": rate_chart(measured) if measured else None,
                  "timeline": timeline_chart(measured, args.run_dir),
                  "deck_time": deck_time_chart(measured) if measured else None,
                  "jitter": jitter_chart(measured, args.run_dir),
                  "l2": l2_chart(measured)}
    figures = {name: figure for name, figure in charts.items() if figure is not None}

    counts = {result: sum(row.get("result") == result for row in rows)
              for result in ("pass", "fail", "unsupported")}
    meta = (f"{len(rows)} cases{f' (set {info['set']})' if info.get('set') else ''}: "
            f"{counts['pass']} passed, {counts['fail']} failed, {counts['unsupported']} unsupported. "
            f"Access point {info.get('ssid', '?')} on channel {info.get('channel', '?')}; "
            f"started {info.get('started', '?')}.")
    if len([r for r in measured if (args.run_dir / r["case"] / "frames.csv").is_file()]) > MAX_TIMELINES:
        meta += f" Rate over time is drawn for the {MAX_TIMELINES} least steady cases."

    out = args.run_dir / "plots"
    out.mkdir(exist_ok=True)
    stamp = args.run_dir.name.removeprefix("bench_")
    table = out / "results.md"
    table.write_text(f"# camera_bench run {stamp}\n\n{meta}\n\n{results_table(rows)}\n")
    print(f"Wrote {table}")
    if figures:
        paths = [out / f"{name}.png" for name in figures]
        # One call renders every figure in one browser session; it needs each
        # size spelled out, or every image comes out at the 700x500 default.
        pio.write_images(list(figures.values()), paths, scale=SCALE,
                         width=[figure.layout.width for figure in figures.values()],
                         height=[figure.layout.height for figure in figures.values()])
        for path in paths:
            print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
