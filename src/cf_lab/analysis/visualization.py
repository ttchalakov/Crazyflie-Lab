"""Small Plotly visualizations shared by telemetry examples and notebooks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cf_lab.analysis.dataset import TelemetryDataset


def plot_timeseries(
    data: TelemetryDataset,
    groups: Mapping[str, Sequence[str]],
    *,
    title: str = "Telemetry",
) -> go.Figure:
    """Plot named signal groups on vertically stacked, shared time axes.

    ``groups`` maps each subplot title to the signals it should contain. Units
    are read from the dataset and displayed when every signal in a group shares
    one unit. The returned figure is not shown or written automatically.
    """
    normalized = [(str(label), tuple(names)) for label, names in groups.items()]
    if not normalized:
        raise ValueError("plot_timeseries requires at least one signal group.")
    for label, names in normalized:
        if not names:
            raise ValueError(f"Plot group {label!r} contains no signals.")
        _require_signals(data, names)

    figure = make_subplots(
        rows=len(normalized),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.08, 0.25 / len(normalized)),
        subplot_titles=tuple(label for label, _names in normalized),
    )
    for row, (_label, names) in enumerate(normalized, start=1):
        for name in names:
            figure.add_trace(
                go.Scatter(
                    x=data.t_s,
                    y=data[name],
                    name=name,
                    mode="lines",
                    line={"width": 2},
                    legendgroup=name,
                ),
                row=row,
                col=1,
            )
        units = {data.units.get(name, "") for name in names}
        units.discard("")
        if len(units) == 1:
            figure.update_yaxes(title_text=next(iter(units)), row=row, col=1)

    figure.update_xaxes(title_text="Time (s)", row=len(normalized), col=1)
    figure.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        height=max(340, 235 * len(normalized)),
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )
    return figure


def plot_xy_trajectory(
    data: TelemetryDataset,
    traces: Mapping[str, tuple[str, str]] | None = None,
    *,
    title: str = "XY trajectory",
) -> go.Figure:
    """Plot one or more x/y signal pairs with equal spatial axis scaling.

    With no explicit ``traces``, the function includes whichever canonical EKF,
    zeroed-mocap, and waypoint pairs are present in the dataset.
    """
    selected = dict(traces) if traces is not None else _default_xy_traces(data)
    if not selected:
        raise ValueError(
            "plot_xy_trajectory found no complete x/y pair; provide traces explicitly."
        )

    figure = go.Figure()
    x_values: list[float] = []
    y_values: list[float] = []
    for label, pair in selected.items():
        if len(pair) != 2:
            raise ValueError(f"XY trace {label!r} must contain exactly two signal names.")
        x_name, y_name = pair
        _require_signals(data, pair)
        x_values.extend(float(value) for value in data[x_name])
        y_values.extend(float(value) for value in data[y_name])
        is_setpoint = "setpoint" in label.lower() or "setpoint" in x_name.lower()
        figure.add_trace(
            go.Scatter(
                x=data[x_name],
                y=data[y_name],
                name=str(label),
                mode="lines+markers" if is_setpoint else "lines",
                line={"width": 2, "shape": "hv" if is_setpoint else "linear"},
                marker={"size": 5},
                hovertemplate=(
                    f"{x_name}=%{{x:.3f}}<br>{y_name}=%{{y:.3f}}<extra>{label}</extra>"
                ),
            )
        )

    x_range, y_range, x_ticks, y_ticks = _square_xy_axes(x_values, y_values)
    figure.update_xaxes(
        title_text="x (m)",
        range=x_range,
        tickmode="array",
        tickvals=x_ticks,
        constrain="domain",
    )
    figure.update_yaxes(
        title_text="y (m)",
        range=y_range,
        tickmode="array",
        tickvals=y_ticks,
        scaleanchor="x",
        scaleratio=1,
        constrain="domain",
    )
    figure.update_layout(
        title=title,
        template="plotly_white",
        # With these margins, the 800x840 canvas leaves a 700x700 plot domain.
        width=800,
        height=840,
        autosize=False,
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )
    return figure


def _square_xy_axes(
    x_values: Sequence[float], y_values: Sequence[float]
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Return equal-span ranges and equally spaced/count X/Y grid ticks."""
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    data_span = max(x_max - x_min, y_max - y_min)
    tick_step = _nice_tick_step(data_span if data_span > 0.0 else 1.0)

    def enclosing_ticks(low: float, high: float) -> list[float]:
        first = math.floor(low / tick_step) * tick_step
        last = math.ceil(high / tick_step) * tick_step
        count = round((last - first) / tick_step) + 1
        return [first + index * tick_step for index in range(count)]

    x_ticks = enclosing_ticks(x_min, x_max)
    y_ticks = enclosing_ticks(y_min, y_max)
    tick_count = max(len(x_ticks), len(y_ticks))

    def expand_ticks(ticks: list[float], center: float) -> tuple[float, ...]:
        while len(ticks) < tick_count:
            lower_center = 0.5 * (ticks[0] - tick_step + ticks[-1])
            upper_center = 0.5 * (ticks[0] + ticks[-1] + tick_step)
            if abs(upper_center - center) <= abs(lower_center - center):
                ticks.append(ticks[-1] + tick_step)
            else:
                ticks.insert(0, ticks[0] - tick_step)
        return tuple(round(value, 12) for value in ticks)

    x_ticks_out = expand_ticks(x_ticks, 0.5 * (x_min + x_max))
    y_ticks_out = expand_ticks(y_ticks, 0.5 * (y_min + y_max))
    half_tick = 0.5 * tick_step
    return (
        (x_ticks_out[0] - half_tick, x_ticks_out[-1] + half_tick),
        (y_ticks_out[0] - half_tick, y_ticks_out[-1] + half_tick),
        x_ticks_out,
        y_ticks_out,
    )


def _nice_tick_step(span: float, target_intervals: int = 10) -> float:
    """Choose a shared 1/2/5-decade grid step near the target density."""
    rough_step = span / target_intervals
    magnitude = 10.0 ** math.floor(math.log10(rough_step))
    candidates = (magnitude, 2.0 * magnitude, 5.0 * magnitude, 10.0 * magnitude)
    return min(candidates, key=lambda candidate: abs(math.log(candidate / rough_step)))


def _default_xy_traces(data: TelemetryDataset) -> dict[str, tuple[str, str]]:
    candidates = (
        ("EKF", ("x", "y")),
        ("Zeroed mocap", ("mocap_x", "mocap_y")),
        ("Setpoint", ("setpoint_x", "setpoint_y")),
    )
    return {
        label: pair
        for label, pair in candidates
        if all(name in data.signals for name in pair)
    }


def _require_signals(data: TelemetryDataset, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in data.signals]
    if missing:
        raise KeyError(
            f"Telemetry dataset is missing signal(s) {missing}; "
            f"available signals: {sorted(data.signals)}."
        )
