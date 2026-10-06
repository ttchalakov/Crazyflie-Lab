"""Stream schemas for the telemetry recorder.

Each stream is described by a ``StreamSpec``: its column names/types plus which
device clock (if any) its source timestamp column belongs to. The writer adds
two implicit leading columns to every stream — ``seq`` (per-stream counter that
increments even when a record is dropped, so drops show up as gaps) and
``host_ns`` (``time.monotonic_ns()`` at enqueue, the one shared host clock).

Arrow types are named by string here and resolved in the writer.
"""

from __future__ import annotations

from dataclasses import dataclass

from cf_lab.config import CustomStreamConfig, TelemetryConfig


# cflib log variable type -> (size in the log packet, arrow type name).
LOG_VAR_TYPES: dict[str, tuple[int, str]] = {
    "float": (4, "float32"),
    "FP16": (2, "float32"),
    "uint8_t": (1, "uint8"),
    "int8_t": (1, "int8"),
    "uint16_t": (2, "uint16"),
    "int16_t": (2, "int16"),
    "uint32_t": (4, "uint32"),
    "int32_t": (4, "int32"),
}

# Max payload of one CRTP log block (cflib LogConfig.MAX_LEN).
LOG_BLOCK_MAX_BYTES = 26

# Onboard log-task period bounds: 10 ms granularity, one unsigned byte of ticks.
LOG_PERIOD_MS_MIN = 10
LOG_PERIOD_MS_MAX = 2540

# Clock domains a stream's source timestamp can belong to (see telemetry.sync).
CLOCK_NONE = "none"
CLOCK_STM32_MS = "stm32_ms"
CLOCK_NATNET_FRAME = "natnet_frame"
CLOCK_ROS_STAMP_S = "ros_stamp_s"
CLOCK_GAP8_MS = "gap8_ms"

RESERVED_STREAM_NAMES = frozenset(
    {"state_pos", "state_att", "ctrl", "mocap", "extpose", "setpoints", "events", "camera"}
)

# Sidecar holding the frame payloads that camera.parquet indexes.
CAMERA_BLOB_FILE = "camera_frames.bin"


@dataclass(frozen=True)
class StreamSpec:
    """One recorded stream: schema columns (excluding seq/host_ns) + clock tag."""

    name: str
    fields: tuple[tuple[str, str], ...]  # (column name, arrow type name)
    clock: str = CLOCK_NONE
    # Free-form notes that land in the manifest (log variables, period, ...),
    # one entry per key.
    meta: tuple[tuple[str, str | tuple[str, ...]], ...] = ()

    @property
    def log_variables(self) -> tuple[tuple[str, str], ...]:
        """(toc name, cflib type) pairs for STM32 streams; empty otherwise."""
        return tuple(tuple(entry.split(":", 1)) for entry in dict(self.meta).get("log_vars", ()))


def state_pos_spec(rate_hz: float) -> StreamSpec:
    return _stm32_spec(
        "state_pos",
        rate_hz,
        (
            ("stateEstimate.x", "float"),
            ("stateEstimate.y", "float"),
            ("stateEstimate.z", "float"),
            ("stateEstimate.vx", "float"),
            ("stateEstimate.vy", "float"),
            ("stateEstimate.vz", "float"),
        ),
    )


def state_att_spec(rate_hz: float) -> StreamSpec:
    # gyro.* are filtered sensor body rates (deg/s) used directly by the rate
    # PID; they are paired with EKF attitude for the canonical 12D state because
    # stateEstimate has no angular-rate entries.
    return _stm32_spec(
        "state_att",
        rate_hz,
        (
            ("stateEstimate.roll", "float"),
            ("stateEstimate.pitch", "float"),
            ("stateEstimate.yaw", "float"),
            ("gyro.x", "float"),
            ("gyro.y", "float"),
            ("gyro.z", "float"),
        ),
    )


def ctrl_spec(rate_hz: float) -> StreamSpec:
    spec = _stm32_spec(
        "ctrl",
        rate_hz,
        (
            ("controller.cmd_thrust", "float"),
            ("controller.cmd_roll", "float"),
            ("controller.cmd_pitch", "float"),
            ("controller.cmd_yaw", "float"),
        ),
    )
    return StreamSpec(
        name=spec.name,
        fields=spec.fields,
        clock=spec.clock,
        meta=spec.meta
        + (
            ("control_stage", "pre_mixer"),
            ("control_mode", "legacy"),
            ("units", "firmware mixer commands"),
        ),
    )


def custom_spec(cfg: CustomStreamConfig) -> StreamSpec:
    """Build (and fully validate) a spec for one user-configured log stream."""
    if cfg.name in RESERVED_STREAM_NAMES:
        raise ValueError(f"telemetry.custom stream name {cfg.name!r} is reserved.")
    if not cfg.variables:
        raise ValueError(f"telemetry.custom.{cfg.name} lists no variables.")
    if not LOG_PERIOD_MS_MIN <= cfg.period_ms <= LOG_PERIOD_MS_MAX or cfg.period_ms % 10:
        raise ValueError(
            f"telemetry.custom.{cfg.name}.period_ms must be a multiple of 10 in "
            f"[{LOG_PERIOD_MS_MIN}, {LOG_PERIOD_MS_MAX}], got {cfg.period_ms}."
        )
    variables = []
    total_bytes = 0
    for entry in cfg.variables:
        name, _, cflib_type = entry.partition(":")
        if cflib_type not in LOG_VAR_TYPES:
            raise ValueError(
                f"telemetry.custom.{cfg.name} variable {entry!r} has unknown log "
                f"type {cflib_type!r}; expected one of {sorted(LOG_VAR_TYPES)}."
            )
        total_bytes += LOG_VAR_TYPES[cflib_type][0]
        variables.append((name, cflib_type))
    if total_bytes > LOG_BLOCK_MAX_BYTES:
        raise ValueError(
            f"telemetry.custom.{cfg.name} variables need {total_bytes} bytes but one "
            f"log block carries {LOG_BLOCK_MAX_BYTES}; split into two streams."
        )
    return _stm32_spec(cfg.name, 1000.0 / cfg.period_ms, tuple(variables))


def _stm32_spec(
    name: str, rate_hz: float, variables: tuple[tuple[str, str], ...]
) -> StreamSpec:
    return StreamSpec(
        name=name,
        fields=(("stm32_ms", "int64"),)
        + tuple((var, LOG_VAR_TYPES[cflib_type][1]) for var, cflib_type in variables),
        clock=CLOCK_STM32_MS,
        meta=(
            ("log_vars", tuple(f"{var}:{cflib_type}" for var, cflib_type in variables)),
            ("period_ms", str(log_period_ms(rate_hz))),
        ),
    )


def log_period_ms(rate_hz: float) -> int:
    """Snap a rate to the log task's 10 ms grid (clamped to the legal range)."""
    period = round(1000.0 / rate_hz / 10.0) * 10
    return max(LOG_PERIOD_MS_MIN, min(LOG_PERIOD_MS_MAX, period))


def mocap_spec(transport: str) -> StreamSpec:
    return StreamSpec(
        name="mocap",
        fields=(
            ("source_ts_s", "float64"),
            ("source_seq", "int64"),  # NatNet frame number; -1 when absent
            ("x", "float64"),
            ("y", "float64"),
            ("z", "float64"),
            ("qx", "float64"),
            ("qy", "float64"),
            ("qz", "float64"),
            ("qw", "float64"),
            # Motive's exposure-to-transmit latency; -1 when not reported.
            ("latency_us", "float64"),
        ),
        clock=CLOCK_NATNET_FRAME if transport == "natnet" else CLOCK_ROS_STAMP_S,
        meta=(
            ("transport", transport),
            ("frame", "mocap_global"),
            ("signal_stage", "raw source pose before rate-cap or runner transform"),
        ),
    )


def extpose_spec(frame: str = "start_origin_yaw_zeroed") -> StreamSpec:
    """The pose samples actually sent up the radio in ``frame``.

    ``source_seq`` cross-references rows back to the raw ``mocap`` stream, so
    the uplink can be recreated exactly: this stream *is* the signal the
    estimator received (modulo the backend's position-only switch, recorded as
    an ``extpose_config`` event).
    """
    return StreamSpec(
        name="extpose",
        fields=(
            ("source_ts_s", "float64"),
            ("source_seq", "int64"),
            ("x", "float64"),
            ("y", "float64"),
            ("z", "float64"),
            ("qx", "float64"),
            ("qy", "float64"),
            ("qz", "float64"),
            ("qw", "float64"),
        ),
        meta=(
            ("frame", frame),
            ("signal_stage", "as sent to the estimator"),
        ),
    )


def camera_spec() -> StreamSpec:
    """Index rows for the AI-deck image stream; pixels live in a sidecar blob.

    Frames are *not* stored in this stream. Parquet finalization reads a whole
    table into memory, and raw QVGA at 60 FPS is ~276 MB/minute, so payloads go
    to an append-only ``camera_frames.bin`` and each row here points at one with
    ``blob_offset``/``size``. Rows are written only after their bytes are on
    disk, so an index row always points at bytes that exist.

    ``frame_id`` and ``deck_ms`` come from the app extension (-1 when the app
    sends none). ``deck_ms`` is when the frame was complete in its buffer, on
    the deck's clock: the frame's timestamp. ``seq`` counts what the host
    enqueued, so a gap in ``frame_id`` is the only evidence of a frame the deck
    captured but never sent. ``stm32_anchor_ms``/``deck_anchor_ms`` are the
    newest STM32-stamped message the deck had received and when it arrived
    (-1 when the app sends none); telemetry.sync uses them to put frames on the
    STM32 clock. ``ext`` keeps the app's extension verbatim, so timings this
    schema does not name survive to analysis.
    """
    return StreamSpec(
        name="camera",
        fields=(
            ("frame_id", "int64"),
            ("deck_ms", "int64"),
            ("width", "uint16"),
            ("height", "uint16"),
            ("pixel_format", "uint8"),  # cf_img type: 0 raw gray, 1 jpeg
            ("depth", "uint8"),
            ("size", "int64"),
            ("blob_offset", "int64"),
            ("ext", "binary"),
            ("stm32_anchor_ms", "int64"),
            ("deck_anchor_ms", "int64"),
        ),
        clock=CLOCK_GAP8_MS,
        meta=(
            ("blob", CAMERA_BLOB_FILE),
            ("frame", "exactly as received; never resized or re-encoded"),
        ),
    )


def setpoints_spec() -> StreamSpec:
    return StreamSpec(
        name="setpoints",
        fields=(
            ("x", "float64"),
            ("y", "float64"),
            ("z", "float64"),
            ("yaw_deg", "float64"),
        ),
    )


def events_spec() -> StreamSpec:
    return StreamSpec(
        name="events",
        fields=(("kind", "string"), ("message", "string"), ("data_json", "string")),
    )


def build_specs(
    cfg: TelemetryConfig,
    *,
    mocap_transport: str,
    extpose_frame: str = "start_origin_yaw_zeroed",
) -> list[StreamSpec]:
    """All stream specs enabled by a telemetry config (events always included)."""
    specs: list[StreamSpec] = []
    if cfg.state_hz > 0:
        specs += [state_pos_spec(cfg.state_hz), state_att_spec(cfg.state_hz)]
    if cfg.controls_hz > 0:
        specs.append(ctrl_spec(cfg.controls_hz))
    specs += [custom_spec(custom) for custom in cfg.custom]
    if cfg.mocap:
        specs.append(mocap_spec(mocap_transport))
    if cfg.extpose:
        specs.append(extpose_spec(extpose_frame))
    if cfg.setpoints:
        specs.append(setpoints_spec())
    if cfg.camera:
        specs.append(camera_spec())
    specs.append(events_spec())
    return specs
