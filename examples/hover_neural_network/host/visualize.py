#!/usr/bin/env python3
"""Visualize a recorded hover_neural_network run with Plotly."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cf_lab.analysis import load_dataset
from cf_lab.telemetry import load_run


HOVER_HEIGHT_M = 0.8
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
OUTPUT_DIR = ARTIFACTS_DIR / "visualizations"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--png", action="store_true", help="Also write high-resolution PNGs")
    args = parser.parse_args(argv)

    run_dir = args.run_dir or latest_run(ARTIFACTS_DIR)
    if run_dir is None:
        parser.error(f"no recorded runs found below {ARTIFACTS_DIR}")
    run = load_run(run_dir)
    data = load_dataset(
        run,
        state="full_12d",
        include_mocap_position=True,
        include_setpoints=True,
    )
    figures = {
        "trajectory_3d": trajectory_figure(data, run_dir.name),
        "state_error": state_figure(data, run_dir.name),
    }
    controls = control_figure(run, run_dir.name)
    if controls is not None:
        figures["neural_control"] = controls
    link = link_figure(run, run_dir.name)
    if link is not None:
        figures["cpx_timing"] = link

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        html = args.output_dir / f"{run_dir.name}_{name}.html"
        figure.write_html(html, include_plotlyjs="cdn")
        print(html)
        if args.png:
            png = html.with_suffix(".png")
            figure.write_image(png, width=figure.layout.width or 1500, scale=2)
            print(png)
        if args.show:
            figure.show()
    return 0


def latest_run(root: Path) -> Path | None:
    manifests = tuple(root.rglob("manifest.json")) if root.is_dir() else ()
    return max((path.parent for path in manifests), key=lambda path: path.stat().st_mtime, default=None)


def trajectory_figure(data, source: str):
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=data["x"],
            y=data["y"],
            z=data["z"],
            mode="lines",
            line={"color": data.t_s, "colorscale": "Viridis", "width": 7},
            name="EKF trajectory",
        )
    )
    if all(f"mocap_{axis}" in data.signals for axis in "xyz"):
        figure.add_trace(
            go.Scatter3d(
                x=data["mocap_x"],
                y=data["mocap_y"],
                z=data["mocap_z"],
                mode="lines",
                line={"color": "rgba(230,80,70,0.55)", "width": 3},
                name="Mocap/extpose",
            )
        )
    figure.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[HOVER_HEIGHT_M],
            mode="markers",
            marker={"size": 7, "symbol": "diamond", "color": "black"},
            name="Hover equilibrium",
        )
    )
    figure.update_layout(
        title=f"{source}: isometric hover trajectory",
        width=1100,
        height=850,
        scene={
            "xaxis_title": "x (m)",
            "yaxis_title": "y (m)",
            "zaxis_title": "z (m)",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.05}},
        },
    )
    return figure


def state_figure(data, source: str):
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Position error", "Velocity", "Attitude", "Body rates"),
    )
    position = {
        "x": data["x"],
        "y": data["y"],
        "z": tuple(value - HOVER_HEIGHT_M for value in data["z"]),
    }
    for name, values in position.items():
        figure.add_trace(go.Scatter(x=data.t_s, y=values, name=f"e_{name}"), row=1, col=1)
    for name in ("vx", "vy", "vz"):
        figure.add_trace(go.Scatter(x=data.t_s, y=data[name], name=name), row=1, col=2)
    for name in ("roll", "pitch", "yaw"):
        values = tuple(math.degrees(value) for value in data[name])
        figure.add_trace(go.Scatter(x=data.t_s, y=values, name=name), row=2, col=1)
    for name in ("p", "q", "r"):
        values = tuple(math.degrees(value) for value in data[name])
        figure.add_trace(go.Scatter(x=data.t_s, y=values, name=name), row=2, col=2)
    figure.update_yaxes(title_text="m", row=1, col=1)
    figure.update_yaxes(title_text="m/s", row=1, col=2)
    figure.update_yaxes(title_text="deg", row=2, col=1)
    figure.update_yaxes(title_text="deg/s", row=2, col=2)
    figure.update_xaxes(title_text="time (s)", row=2, col=1)
    figure.update_xaxes(title_text="time (s)", row=2, col=2)
    figure.update_layout(title=f"{source}: neural equilibrium state", width=1500, height=900)
    return figure


def control_figure(run, source: str):
    table = run.streams.get("hnn_control")
    if table is None or table.num_rows == 0:
        return None
    t_s = host_time_s(table)
    active = column(table, "hnn.active")

    def while_active(values):
        return [value if enabled else None for value, enabled in zip(values, active)]

    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, subplot_titles=("Thrust", "Body rates")
    )
    figure.add_trace(
        go.Scatter(
            x=t_s,
            y=while_active(column(table, "hnn.thrustN")),
            name="applied neural thrust",
        ),
        row=1,
        col=1,
    )
    for name in ("wx", "wy", "wz"):
        figure.add_trace(
            go.Scatter(
                x=t_s,
                y=while_active(column(table, f"hnn.{name}")),
                name=name,
            ),
            row=2,
            col=1,
        )
    figure.update_yaxes(title_text="N", row=1, col=1)
    figure.update_yaxes(title_text="rad/s", row=2, col=1)
    figure.update_xaxes(title_text="host time (s)", row=2, col=1)
    figure.update_layout(
        title=f"{source}: applied CTBR setpoint", width=1500, height=800
    )
    return figure


def link_figure(run, source: str):
    """Policy replies per second, differentiated from the accepted-reply counter.

    The firmware keeps one counter rather than per-reply timings, so the rate is
    derived here. It is the honest health signal for this link: the STM32 only
    counts a reply it accepted as well-formed and in-range, so a sustained 100/s
    means the whole loop -- state out, inference, command back, validation -- is
    running at rate. A dip is a stall, wherever it came from.
    """
    table = run.streams.get("hnn_link")
    if table is None or table.num_rows < 2:
        return None
    t_s = host_time_s(table)
    counts = column(table, "hnn.rxCount")
    rate_t, rate_hz = [], []
    for (ta, ca), (tb, cb) in zip(zip(t_s, counts), zip(t_s[1:], counts[1:])):
        if tb > ta:
            rate_t.append(tb)
            rate_hz.append((cb - ca) / (tb - ta))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=rate_t, y=rate_hz, name="accepted replies"))
    figure.add_hline(
        y=100.0, line={"dash": "dot", "color": "grey"}, annotation_text="100 Hz request rate"
    )
    figure.update_yaxes(title_text="replies/s", rangemode="tozero")
    figure.update_xaxes(title_text="host time (s)")
    figure.update_layout(title=f"{source}: control-link reply rate", width=1500, height=500)
    return figure


def host_time_s(table) -> list[float]:
    values = column(table, "host_ns")
    start = values[0]
    return [(value - start) * 1e-9 for value in values]


def column(table, name: str) -> list:
    return table.column(name).to_pylist()


if __name__ == "__main__":
    raise SystemExit(main())
