"""Aligned, unit-aware telemetry datasets for offline analysis.

``load_dataset`` turns the asynchronous Arrow tables returned by
``cf_lab.telemetry.load_run`` into one shared numeric timeline.

The canonical full state is ``[x, y, z, vx, vy, vz, roll, pitch, yaw, p, q,
r]`` in SI units.  Crazyflie's legacy estimator pitch sign is converted to the
lab's FLU convention (``pitch = -stateEstimate.pitch``); ``q`` is ``gyro.y``
with its sign unchanged.  Select the raw columns with explicit ``SignalSpec``
entries when firmware-native signs or units are desired instead.

Stock ``controller.cmd_*`` values are intentionally labelled ``raw``.  Their
physical meaning depends on the selected firmware controller/control mode, so
they must not be interpreted as newtons or newton-metres without calibration.
"""

from __future__ import annotations

import csv
import math
import statistics
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_DEG_TO_RAD = math.pi / 180.0
_INTERPOLATION_MODES = ("linear", "previous")


def _normalize_interpolation(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in _INTERPOLATION_MODES:
        raise ValueError(f"Unknown interpolation mode {mode!r}; use 'linear' or 'previous'.")
    return mode


@dataclass(frozen=True)
class SignalSpec:
    """Map one recorded stream column onto a named analysis signal."""

    stream: str
    column: str
    unit: str = ""
    scale: float = 1.0
    interpolation: str = "linear"

    def __post_init__(self) -> None:
        if not self.stream:
            raise ValueError("SignalSpec.stream must not be empty.")
        if not self.column:
            raise ValueError("SignalSpec.column must not be empty.")
        scale = float(self.scale)
        if not math.isfinite(scale):
            raise ValueError("SignalSpec scale must be finite.")
        interpolation = _normalize_interpolation(self.interpolation)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "interpolation", interpolation)


@dataclass
class TelemetryDataset:
    """Signals sampled on one strictly increasing timeline.

    Values are stored as tuples to make accidental in-place edits difficult.
    ``state_names`` and ``input_names`` list the state and control signals in
    order; other signals remain available by name.
    """

    t_s: Sequence[float]
    signals: Mapping[str, Sequence[float]]
    state_names: Sequence[str] = ()
    input_names: Sequence[str] = ()
    units: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        times = _finite_values(self.t_s, "t_s")
        if not times:
            raise ValueError("TelemetryDataset requires at least one sample.")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("TelemetryDataset.t_s must be strictly increasing.")

        normalized: dict[str, tuple[float, ...]] = {}
        for name, values in self.signals.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Telemetry signal names must be non-empty strings.")
            series = _finite_values(values, f"signal {name!r}")
            if len(series) != len(times):
                raise ValueError(
                    f"Signal {name!r} has {len(series)} samples; expected {len(times)} "
                    "to match t_s."
                )
            normalized[name] = series
        if not normalized:
            raise ValueError("TelemetryDataset requires at least one signal.")

        state_names = tuple(self.state_names)
        input_names = tuple(self.input_names)
        _validate_named_group("state_names", state_names, normalized)
        _validate_named_group("input_names", input_names, normalized)
        overlap = set(state_names) & set(input_names)
        if overlap:
            raise ValueError(f"Signals cannot be both state and input: {sorted(overlap)}.")

        units = {str(name): str(unit) for name, unit in self.units.items()}
        unknown_units = set(units) - set(normalized)
        if unknown_units:
            raise ValueError(f"Units supplied for unknown signals: {sorted(unknown_units)}.")

        self.t_s = times
        self.signals = normalized
        self.state_names = state_names
        self.input_names = input_names
        self.units = units
        self.metadata = dict(self.metadata)

    def __len__(self) -> int:
        return len(self.t_s)

    def __getitem__(self, name: str) -> tuple[float, ...]:
        if not isinstance(name, str):
            raise TypeError("TelemetryDataset indices are signal names, for example data['x'].")
        return self.signals[name]

    @property
    def dt_s(self) -> float | None:
        """Median sample interval, or ``None`` for a one-row dataset."""
        if len(self) < 2:
            return None
        return statistics.median(b - a for a, b in zip(self.t_s, self.t_s[1:]))

    @property
    def sample_hz(self) -> float | None:
        dt = self.dt_s
        return None if dt is None else 1.0 / dt

    @property
    def duration_s(self) -> float:
        return self.t_s[-1] - self.t_s[0]

    def select(self, *names: str | Iterable[str]) -> "TelemetryDataset":
        """Return a dataset containing only the requested signals."""
        selected = _selection_names(names)
        if not selected:
            raise ValueError("select() needs at least one signal name.")
        missing = [name for name in selected if name not in self.signals]
        if missing:
            raise KeyError(f"Unknown telemetry signals: {missing}.")
        metadata = _metadata_for_selection(self.metadata, selected)
        return TelemetryDataset(
            self.t_s,
            {name: self.signals[name] for name in selected},
            state_names=tuple(name for name in self.state_names if name in selected),
            input_names=tuple(name for name in self.input_names if name in selected),
            units={name: self.units[name] for name in selected if name in self.units},
            metadata=metadata,
        )

    def window(
        self, start_s: float | None = None, end_s: float | None = None
    ) -> "TelemetryDataset":
        """Select an inclusive time window without rebasing its timeline."""
        start = self.t_s[0] if start_s is None else float(start_s)
        end = self.t_s[-1] if end_s is None else float(end_s)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Window bounds must be finite.")
        if end < start:
            raise ValueError(f"end_s ({end}) must be >= start_s ({start}).")
        first = bisect_left(self.t_s, start)
        stop = bisect_right(self.t_s, end)
        if first == stop:
            raise ValueError(
                f"Window [{start}, {end}] contains no samples; dataset spans "
                f"[{self.t_s[0]}, {self.t_s[-1]}]."
            )
        metadata = dict(self.metadata)
        metadata["window_s"] = (start, end)
        return TelemetryDataset(
            self.t_s[first:stop],
            {name: values[first:stop] for name, values in self.signals.items()},
            state_names=self.state_names,
            input_names=self.input_names,
            units=self.units,
            metadata=metadata,
        )

    def resample(
        self,
        sample_hz: float,
        *,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> "TelemetryDataset":
        """Resample onto a uniform grid using each signal's interpolation mode."""
        rate = _positive_rate(sample_hz)
        requested_start = self.t_s[0] if start_s is None else float(start_s)
        requested_end = self.t_s[-1] if end_s is None else float(end_s)
        if not math.isfinite(requested_start) or not math.isfinite(requested_end):
            raise ValueError("Resampling bounds must be finite.")
        start = max(requested_start, self.t_s[0])
        end = min(requested_end, self.t_s[-1])
        if end < start:
            raise ValueError("Requested resampling window does not overlap the dataset.")
        grid = _uniform_grid(start, end, rate)
        interpolation = dict(self.metadata.get("interpolation", {}))
        angles = set(self.metadata.get("angle_signals", ()))
        output = {}
        for name, values in self.signals.items():
            source = _unwrap_angles(values) if name in angles else values
            output[name] = _interpolate(
                self.t_s,
                source,
                grid,
                _normalize_interpolation(interpolation.get(name, "linear")),
            )
        metadata = dict(self.metadata)
        metadata["sample_hz"] = rate
        metadata["interpolation"] = {
            name: _normalize_interpolation(interpolation.get(name, "linear"))
            for name in self.signals
        }
        return TelemetryDataset(
            grid,
            output,
            state_names=self.state_names,
            input_names=self.input_names,
            units=self.units,
            metadata=metadata,
        )

    @classmethod
    def from_csv(
        cls,
        path: str | PathLike[str],
        *,
        time_column: str = "t_s",
        state_names: Sequence[str] = (),
        input_names: Sequence[str] = (),
        units: Mapping[str, str] | None = None,
        interpolation: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TelemetryDataset":
        """Load a numeric, headered CSV file with one column per signal."""
        csv_path = Path(path)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames
            if not headers:
                raise ValueError(f"CSV file {csv_path} has no header row.")
            if time_column not in headers:
                raise ValueError(
                    f"CSV file {csv_path} is missing time column {time_column!r}; "
                    f"available columns: {headers}."
                )
            signal_names = [name for name in headers if name != time_column]
            if not signal_names:
                raise ValueError(f"CSV file {csv_path} contains no signal columns.")
            times: list[float] = []
            values: dict[str, list[float]] = {name: [] for name in signal_names}
            for row_number, row in enumerate(reader, start=2):
                times.append(_csv_float(row.get(time_column), csv_path, row_number, time_column))
                for name in signal_names:
                    values[name].append(_csv_float(row.get(name), csv_path, row_number, name))

        if not times:
            raise ValueError(f"CSV file {csv_path} contains no data rows.")
        interpolation_map = {
            name: _normalize_interpolation(mode)
            for name, mode in (interpolation or {}).items()
        }
        unknown = set(interpolation_map) - set(signal_names)
        if unknown:
            raise ValueError(f"Interpolation supplied for unknown CSV signals: {sorted(unknown)}.")
        combined_metadata = dict(metadata or {})
        combined_metadata.update(
            {
                "source": "csv",
                "source_path": str(csv_path),
                "interpolation": interpolation_map,
            }
        )
        return cls(
            times,
            values,
            state_names=state_names,
            input_names=input_names,
            units=units or {},
            metadata=combined_metadata,
        )


_STATE_SPECS: dict[str, SignalSpec] = {
    "x": SignalSpec("state_pos", "stateEstimate.x", "m"),
    "y": SignalSpec("state_pos", "stateEstimate.y", "m"),
    "z": SignalSpec("state_pos", "stateEstimate.z", "m"),
    "vx": SignalSpec("state_pos", "stateEstimate.vx", "m/s"),
    "vy": SignalSpec("state_pos", "stateEstimate.vy", "m/s"),
    "vz": SignalSpec("state_pos", "stateEstimate.vz", "m/s"),
    "roll": SignalSpec("state_att", "stateEstimate.roll", "rad", _DEG_TO_RAD),
    # Firmware stateEstimate.pitch uses the legacy sign; canonical analysis is FLU.
    "pitch": SignalSpec("state_att", "stateEstimate.pitch", "rad", -_DEG_TO_RAD),
    "yaw": SignalSpec("state_att", "stateEstimate.yaw", "rad", _DEG_TO_RAD),
    "p": SignalSpec("state_att", "gyro.x", "rad/s", _DEG_TO_RAD),
    "q": SignalSpec("state_att", "gyro.y", "rad/s", _DEG_TO_RAD),
    "r": SignalSpec("state_att", "gyro.z", "rad/s", _DEG_TO_RAD),
}

STATE_LAYOUTS: dict[str, tuple[str, ...]] = {
    "pose_6d": ("x", "y", "z", "roll", "pitch", "yaw"),
    "translation_6d": ("x", "y", "z", "vx", "vy", "vz"),
    "rotation_6d": ("roll", "pitch", "yaw", "p", "q", "r"),
    "full_12d": ("x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "p", "q", "r"),
}

_CONTROL_SPECS: dict[str, SignalSpec] = {
    "thrust_cmd": SignalSpec(
        "ctrl", "controller.cmd_thrust", "raw", interpolation="previous"
    ),
    "roll_cmd": SignalSpec("ctrl", "controller.cmd_roll", "raw", interpolation="previous"),
    "pitch_cmd": SignalSpec(
        "ctrl", "controller.cmd_pitch", "raw", interpolation="previous"
    ),
    "yaw_cmd": SignalSpec("ctrl", "controller.cmd_yaw", "raw", interpolation="previous"),
}

_MOCAP_POSITION_SPECS: dict[str, SignalSpec] = {
    # Use the post-zeroing extpose stream so these positions share the EKF and
    # waypoint frame. The raw pre-zeroing 6D mocap stream remains in the run.
    "mocap_x": SignalSpec("extpose", "x", "m"),
    "mocap_y": SignalSpec("extpose", "y", "m"),
    "mocap_z": SignalSpec("extpose", "z", "m"),
}

_SETPOINT_SPECS: dict[str, SignalSpec] = {
    "setpoint_x": SignalSpec("setpoints", "x", "m", interpolation="previous"),
    "setpoint_y": SignalSpec("setpoints", "y", "m", interpolation="previous"),
    "setpoint_z": SignalSpec("setpoints", "z", "m", interpolation="previous"),
    "setpoint_yaw": SignalSpec(
        "setpoints", "yaw_deg", "rad", _DEG_TO_RAD, interpolation="previous"
    ),
}

_ANGLE_SIGNALS = ("roll", "pitch", "yaw", "setpoint_yaw")


def load_dataset(
    run_or_path: Any,
    state: str = "full_12d",
    sample_hz: float | None = None,
    include_controls: bool = True,
    include_mocap_position: bool = False,
    include_setpoints: bool = False,
    extra_signals: Mapping[str, SignalSpec] | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
) -> TelemetryDataset:
    """Load selected run signals and align them on a common overlap timeline.

    With ``sample_hz=None`` the grid runs at the slowest state signal's rate, and
    the other signals are sampled onto it; either way, the result uses a uniform
    grid. Continuous signals are linearly interpolated and control commands use
    previous-value (zero-order-hold) interpolation. Controls, the mocap position
    and the setpoints are each included when asked for and recorded (their stream
    present and non-empty); the metadata says whether they were.
    ``include_mocap_position`` adds the origin-zeroed mocap position actually
    supplied to the EKF as ``mocap_x/y/z``; raw mocap orientation stays recorded
    but is intentionally not treated as a baseline here.
    """
    if state not in STATE_LAYOUTS:
        raise ValueError(f"Unknown state layout {state!r}; choose one of {sorted(STATE_LAYOUTS)}.")
    if sample_hz is not None:
        sample_hz = _positive_rate(sample_hz)
    if start_s is not None and not math.isfinite(float(start_s)):
        raise ValueError("start_s must be finite.")
    if end_s is not None and not math.isfinite(float(end_s)):
        raise ValueError("end_s must be finite.")
    if start_s is not None and end_s is not None and float(end_s) < float(start_s):
        raise ValueError("end_s must be >= start_s.")

    run = _coerce_run(run_or_path)
    state_names = STATE_LAYOUTS[state]
    requested: dict[str, SignalSpec] = {name: _STATE_SPECS[name] for name in state_names}
    input_names: tuple[str, ...] = ()
    if include_controls and _stream_has_rows(run, "ctrl"):
        requested.update(_CONTROL_SPECS)
        input_names = tuple(_CONTROL_SPECS)
    include_mocap_position = include_mocap_position and _stream_has_rows(run, "extpose")
    if include_mocap_position:
        requested.update(_MOCAP_POSITION_SPECS)
    include_setpoints = include_setpoints and _stream_has_rows(run, "setpoints")
    if include_setpoints:
        requested.update(_SETPOINT_SPECS)
    for alias, spec in (extra_signals or {}).items():
        if not isinstance(alias, str) or not alias:
            raise ValueError("extra_signals aliases must be non-empty strings.")
        if alias in requested:
            raise ValueError(f"extra_signals alias {alias!r} collides with a canonical signal.")
        if not isinstance(spec, SignalSpec):
            raise TypeError(f"extra_signals[{alias!r}] must be a SignalSpec.")
        requested[alias] = spec

    series: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
    used_streams: list[str] = []
    for alias, spec in requested.items():
        times, raw_values = _read_signal(run, alias, spec)
        transformed = tuple(value * spec.scale for value in raw_values)
        if alias in _ANGLE_SIGNALS:
            transformed = _unwrap_angles(transformed)
        series[alias] = (times, transformed)
        if spec.stream not in used_streams:
            used_streams.append(spec.stream)

    overlap_start = max(times[0] for times, _ in series.values())
    overlap_end = min(times[-1] for times, _ in series.values())
    if start_s is not None:
        overlap_start = max(overlap_start, float(start_s))
    if end_s is not None:
        overlap_end = min(overlap_end, float(end_s))
    if overlap_end < overlap_start:
        spans = {name: (times[0], times[-1]) for name, (times, _) in series.items()}
        raise ValueError(
            "Selected telemetry signals have no common time overlap after applying the "
            f"requested window; spans: {spans}."
        )

    # Held signals such as the setpoints change in steps, so they must not slow the grid.
    state_sample_hz = _infer_sample_hz([series[name][0] for name in state_names])
    effective_sample_hz = sample_hz or state_sample_hz
    grid = _uniform_grid(overlap_start, overlap_end, effective_sample_hz)

    aligned = {
        alias: _interpolate(times, values, grid, requested[alias].interpolation)
        for alias, (times, values) in series.items()
    }
    interpolation = {alias: spec.interpolation for alias, spec in requested.items()}
    units = {alias: spec.unit for alias, spec in requested.items() if spec.unit}
    manifest = getattr(run, "manifest", {})
    metadata = {
        "source": "telemetry_run",
        "state_layout": state,
        "sample_hz": effective_sample_hz,
        "sample_hz_inferred": sample_hz is None,
        "interpolation": interpolation,
        "angle_signals": tuple(name for name in _ANGLE_SIGNALS if name in aligned),
        "streams": tuple(used_streams),
        "manifest_status": manifest.get("status") if isinstance(manifest, Mapping) else None,
        "mocap_position_included": include_mocap_position,
        "setpoints_included": include_setpoints,
    }
    return TelemetryDataset(
        grid,
        aligned,
        state_names=state_names,
        input_names=input_names,
        units=units,
        metadata=metadata,
    )


def _coerce_run(run_or_path: Any) -> Any:
    if isinstance(run_or_path, (str, PathLike)):
        # Keep pyarrow optional until a recorded run is actually requested.
        from cf_lab.telemetry.sync import load_run

        return load_run(run_or_path)
    if hasattr(run_or_path, "streams") and hasattr(run_or_path, "manifest"):
        return run_or_path
    raise TypeError("load_dataset expects a telemetry RunData object or a run-directory path.")


def _read_signal(
    run: Any, alias: str, spec: SignalSpec
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    streams = run.streams
    if spec.stream not in streams:
        raise ValueError(
            f"Signal {alias!r} requires telemetry stream {spec.stream!r}, but the run only "
            f"contains {sorted(streams)}. Enable that stream when recording or omit the signal."
        )
    table = streams[spec.stream]
    columns = _column_names(table)
    missing = [name for name in ("t_s", spec.column) if name not in columns]
    if missing:
        raise ValueError(
            f"Signal {alias!r} requires column(s) {missing} in stream {spec.stream!r}; "
            f"available columns: {columns}."
        )
    times = _column_values(table, "t_s")
    values = _column_values(table, spec.column)
    if len(times) != len(values):
        raise ValueError(
            f"Stream {spec.stream!r} columns 't_s' and {spec.column!r} have different lengths."
        )
    if not times:
        raise ValueError(f"Telemetry stream {spec.stream!r} contains no samples for {alias!r}.")

    pairs = sorted(zip(times, values), key=lambda item: item[0])
    deduplicated: list[tuple[float, float]] = []
    for time_value, signal_value in pairs:
        t = _finite_float(time_value, f"{spec.stream}.t_s")
        value = _finite_float(signal_value, f"{spec.stream}.{spec.column}")
        if deduplicated and t == deduplicated[-1][0]:
            deduplicated[-1] = (t, value)  # last sample wins at a duplicate timestamp
        else:
            deduplicated.append((t, value))
    result_times, result_values = zip(*deduplicated)
    return tuple(result_times), tuple(result_values)


def _column_names(table: Any) -> list[str]:
    if isinstance(table, Mapping):
        return list(table)
    names = getattr(table, "column_names", None)
    if names is None:
        raise TypeError(f"Unsupported telemetry table type {type(table).__name__}.")
    return list(names)


def _column_values(table: Any, name: str) -> list[Any]:
    column = table[name] if isinstance(table, Mapping) else table.column(name)
    if hasattr(column, "to_pylist"):
        return column.to_pylist()
    return list(column)


def _stream_has_rows(run: Any, stream: str) -> bool:
    if stream not in run.streams:
        return False
    table = run.streams[stream]
    num_rows = getattr(table, "num_rows", None)
    if num_rows is not None:
        return int(num_rows) > 0
    columns = _column_names(table)
    if not columns:
        return False
    return bool(_column_values(table, columns[0]))


def _infer_sample_hz(timelines: Sequence[Sequence[float]]) -> float:
    """The slowest of these timelines' rates, for a uniform output grid."""
    rates = []
    for times in timelines:
        deltas = [right - left for left, right in zip(times, times[1:]) if right > left]
        if deltas:
            rates.append(1.0 / statistics.median(deltas))
    if not rates:
        raise ValueError(
            "Cannot infer sample_hz from state streams with fewer than two distinct "
            "timestamps; pass sample_hz explicitly."
        )
    # Do not manufacture bandwidth by upsampling a slower state signal.
    return min(rates)


def _interpolate(
    source_t: Sequence[float],
    source_values: Sequence[float],
    target_t: Sequence[float],
    mode: str,
) -> tuple[float, ...]:
    mode = _normalize_interpolation(mode)
    output: list[float] = []
    for target in target_t:
        if mode == "previous":
            index = bisect_right(source_t, target) - 1
            index = max(0, min(index, len(source_t) - 1))
            output.append(source_values[index])
            continue
        right = bisect_left(source_t, target)
        if right == 0:
            output.append(source_values[0])
        elif right == len(source_t):
            output.append(source_values[-1])
        elif source_t[right] == target:
            output.append(source_values[right])
        else:
            left = right - 1
            fraction = (target - source_t[left]) / (source_t[right] - source_t[left])
            output.append(source_values[left] + fraction * (source_values[right] - source_values[left]))
    return tuple(output)


def _unwrap_angles(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        return ()
    output = [float(values[0])]
    adjustment = 0.0
    previous = float(values[0])
    for raw_value in values[1:]:
        value = float(raw_value)
        delta = value - previous
        if delta > math.pi:
            adjustment -= 2.0 * math.pi
        elif delta < -math.pi:
            adjustment += 2.0 * math.pi
        output.append(value + adjustment)
        previous = value
    return tuple(output)


def _uniform_grid(start: float, end: float, sample_hz: float) -> tuple[float, ...]:
    if end < start:
        raise ValueError("Cannot create a grid with end before start.")
    count = math.floor((end - start) * sample_hz + 1e-9) + 1
    return tuple(start + index / sample_hz for index in range(max(1, count)))


def _positive_rate(value: float) -> float:
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"sample_hz must be finite and positive, got {value!r}.")
    return rate


def _finite_values(values: Iterable[float], label: str) -> tuple[float, ...]:
    return tuple(_finite_float(value, label) for value in values)


def _finite_float(value: Any, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains non-numeric value {value!r}.") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{label} contains non-finite value {value!r}.")
    return converted


def _validate_named_group(
    label: str, names: tuple[str, ...], signals: Mapping[str, Sequence[float]]
) -> None:
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate names: {names}.")
    missing = [name for name in names if name not in signals]
    if missing:
        raise ValueError(f"{label} references unknown signals: {missing}.")


def _selection_names(names: Sequence[str | Iterable[str]]) -> tuple[str, ...]:
    if len(names) == 1 and not isinstance(names[0], str):
        selected = tuple(names[0])
    else:
        selected = tuple(names)  # type: ignore[arg-type]
    if any(not isinstance(name, str) for name in selected):
        raise TypeError("select() signal names must be strings.")
    return selected


def _metadata_for_selection(metadata: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    result = dict(metadata)
    if isinstance(result.get("interpolation"), Mapping):
        result["interpolation"] = {
            name: result["interpolation"][name]
            for name in names
            if name in result["interpolation"]
        }
    if "angle_signals" in result:
        result["angle_signals"] = tuple(
            name for name in result["angle_signals"] if name in names
        )
    return result


def _csv_float(
    value: str | None, path: Path, row_number: int, column: str
) -> float:
    if value is None or not value.strip():
        raise ValueError(f"CSV {path} row {row_number}, column {column!r} is empty.")
    try:
        converted = float(value)
    except ValueError as exc:
        raise ValueError(
            f"CSV {path} row {row_number}, column {column!r} is not numeric: {value!r}."
        ) from exc
    if not math.isfinite(converted):
        raise ValueError(
            f"CSV {path} row {row_number}, column {column!r} is not finite: {value!r}."
        )
    return converted
