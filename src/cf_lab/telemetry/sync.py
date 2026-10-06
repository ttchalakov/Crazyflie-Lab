"""Post-hoc clock alignment and run loading.

Every device (STM32, GAP8, mocap system) runs its own free clock; every record
also carries ``host_ns``, the host-monotonic arrival time. After the flight we
fit the affine map ``host_ns ~ slope * device_ts + offset`` per clock domain:
slope captures drift, offset the epoch difference.

The fit runs along the *fastest* arrivals (the lower envelope of the points),
not through their middle. A sample can only arrive after it was measured, and
every link adds a delay that varies from packet to packet; a line through the
middle would place each clock later by its average delay, which differs
between the radio, WiFi and the mocap network. Along the envelope each device
time maps to the host time of its quickest possible arrival.

Device axes: STM32 streams share the onboard ms counter (24-bit, unwrapped
here; pooled across all STM32 streams for one fit). NatNet mocap uses the frame
number; Motive's own exposure-to-transmit latency, when recorded, is taken off
the arrival first, so mocap lands at mid-exposure. VRPN mocap uses the ROS
header stamp. Camera frames are stamped on the GAP8 when the frame is complete
in its buffer; when the frames also carry STM32 clock anchors, the GAP8 clock
is mapped onto the STM32 clock onboard and then through the STM32 fit, which
skips the WiFi delay entirely. Host-native streams (setpoints, events) need no
fit. Pure Python: no numpy needed at the sizes involved.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from cf_lab.telemetry.schema import (
    CLOCK_GAP8_MS,
    CLOCK_NATNET_FRAME,
    CLOCK_NONE,
    CLOCK_ROS_STAMP_S,
    CLOCK_STM32_MS,
)
from cf_lab.telemetry.toc import load_crazyflie_toc

STM32_MS_WRAP = 1 << 24  # cflib log timestamps are a 24-bit ms counter


@dataclass(frozen=True)
class ClockFit:
    slope: float  # host ns per device unit
    offset_ns: float
    rms_ns: float
    n: int
    clock: str = ""

    def to_host_ns(self, device_ts: float) -> float:
        return self.slope * device_ts + self.offset_ns


def unwrap_wrapping_counter(values: Iterable[int], wrap: int) -> list[int]:
    """Undo counter wraparound: any backward jump > wrap/2 adds one period."""
    unwrapped: list[int] = []
    base = 0
    previous: int | None = None
    for value in values:
        if previous is not None and previous - value > wrap // 2:
            base += wrap
        previous = value
        unwrapped.append(value + base)
    return unwrapped


def fit_affine(
    device_ts: Sequence[float], host_ns: Sequence[int], *, clock: str = "", slack: float = 1e6
) -> ClockFit:
    """The line under every point, through the fastest arrivals (see above).

    Of all lines below every point, this one is closest to the points on
    average: the lower convex hull's edge spanning the mean device time. A
    clock glitch can make a single sample look impossibly early, so samples far
    below a plain least-squares fit are dropped first (``slack``, in the
    arrival's unit, keeps a very quiet link from losing good samples).
    ``rms_ns`` is the spread of the arrival delays about their median: the
    link's jitter.
    """
    if len(device_ts) != len(host_ns):
        raise ValueError("device_ts and host_ns must be the same length.")
    if len(device_ts) < 2:
        raise ValueError(f"Need at least 2 samples to fit a clock, got {len(device_ts)}.")
    slope, offset = _least_squares(device_ts, host_ns)
    residuals = [y - (slope * x + offset) for x, y in zip(device_ts, host_ns)]
    med = statistics.median(residuals)
    mad = statistics.median(abs(r - med) for r in residuals)
    floor = med - 10.0 * mad - slack
    points = [(x, y) for x, y, r in zip(device_ts, host_ns, residuals) if r >= floor]
    slope, offset = _lower_envelope(points)

    delays = [y - (slope * x + offset) for x, y in points]
    med = statistics.median(delays)
    rms = (sum((d - med) ** 2 for d in delays) / len(delays)) ** 0.5
    return ClockFit(slope=slope, offset_ns=offset, rms_ns=rms, n=len(points), clock=clock)


def _lower_envelope(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Slope and offset of the lower hull edge spanning the mean x."""
    earliest: dict[float, float] = {}
    for x, y in points:  # only the earliest arrival per device time can touch it
        if x not in earliest or y < earliest[x]:
            earliest[x] = y
    pts = sorted(earliest.items())
    hull: list[tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()  # hull[-1] lies on or above the chord hull[-2] -> p
            else:
                break
        hull.append(p)
    if len(hull) < 2:
        raise ValueError("Device timestamps are all identical; cannot fit a clock.")
    mean_x = sum(x for x, _ in points) / len(points)
    for (x1, y1), (x2, y2) in zip(hull, hull[1:]):
        if x2 >= mean_x or (x2, y2) == hull[-1]:
            slope = (y2 - y1) / (x2 - x1)
            return slope, y1 - slope * x1
    raise AssertionError("unreachable")


def _least_squares(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var = sum((x - mean_x) ** 2 for x in xs)
    if var == 0:
        raise ValueError("Device timestamps are all identical; cannot fit a clock.")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov / var
    return slope, mean_y - slope * mean_x


@dataclass
class RunData:
    manifest: dict[str, Any]
    streams: dict[str, Any]  # name -> pyarrow.Table, with t_host_s and t_s added
    fits: dict[str, ClockFit] = field(default_factory=dict)  # keyed by clock domain
    crazyflie_toc: dict[str, Any] | None = None


def load_run(run_dir: str | Path) -> RunData:
    """Load a run directory and put every stream on one aligned timeline.

    Adds two float columns to every table: ``t_host_s`` (arrival time, seconds
    since the run's monotonic anchor) and ``t_s`` (measurement time via the
    clock fit; equals ``t_host_s`` for host-native streams or when a domain
    had too few samples to fit). Crashed runs fall back to reading the
    ``.arrow`` files up to their last complete batch.
    """
    import pyarrow as pa

    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    anchor_ns = manifest["created_monotonic_ns"]

    streams: dict[str, Any] = {}
    for name in manifest["streams"]:
        table = read_stream(run_dir, name)
        if table is not None:
            streams[name] = table

    # One (device, host) pair pool per clock domain, then one fit per domain.
    pools: dict[str, tuple[list[float], list[int]]] = {}
    device_axes: dict[str, list[float | None]] = {}
    for name, table in streams.items():
        clock = manifest["streams"][name]["clock"]
        if clock == CLOCK_NONE:
            continue
        axis = device_axis(table, clock)
        device_axes[name] = axis
        device, host = pools.setdefault(clock, ([], []))
        for value, host_ns in zip(axis, arrival_ns(table)):
            if value is not None:
                device.append(value)
                host.append(host_ns)

    fits: dict[str, ClockFit] = {}
    for clock, (device, host) in pools.items():
        try:
            fits[clock] = fit_affine(device, host, clock=clock)
        except ValueError:
            pass  # too few samples: streams keep t_s == t_host_s

    # Camera frames carrying STM32 clock anchors go through the STM32 fit.
    via_stm32 = _gap8_via_stm32(streams.get("camera"), pools.get(CLOCK_STM32_MS), fits)

    for name, table in streams.items():
        host_ns = table.column("host_ns").to_pylist()
        t_host_s = [(h - anchor_ns) / 1e9 for h in host_ns]
        clock = manifest["streams"][name]["clock"]
        fit = fits.get(clock)
        if name == "camera" and via_stm32 is not None:
            t_s = [
                t_host if value is None else (via_stm32(value) - anchor_ns) / 1e9
                for value, t_host in zip(device_axes[name], t_host_s)
            ]
        elif fit is None:
            t_s = t_host_s
        else:
            axis = device_axes[name]
            t_s = [
                (fit.to_host_ns(value) - anchor_ns) / 1e9 if value is not None else t_host
                for value, t_host in zip(axis, t_host_s)
            ]
        table = table.append_column("t_host_s", pa.array(t_host_s, pa.float64()))
        streams[name] = table.append_column("t_s", pa.array(t_s, pa.float64()))

    return RunData(
        manifest=manifest,
        streams=streams,
        fits=fits,
        crazyflie_toc=load_crazyflie_toc(run_dir),
    )


def arrival_ns(table) -> list[int]:
    """Host arrival per row; for mocap, minus Motive's own latency when recorded,
    so the fit places a frame at its mid-exposure rather than at transmit."""
    host = table.column("host_ns").to_pylist()
    if "latency_us" not in table.column_names:
        return host
    latency = table.column("latency_us").to_pylist()
    return [h - int(us * 1000) if us is not None and us >= 0 else h for h, us in zip(host, latency)]


# Fewer distinct anchors than this, and frames keep their own GAP8 fit.
MIN_CLOCK_ANCHORS = 20
GAP8_STM32_FIT = "gap8_ms_from_stm32_ms"


def _gap8_via_stm32(camera, stm32_pool, fits: dict[str, ClockFit]):
    """deck_ms -> host_ns through the STM32 clock, or None.

    Each frame carries the newest (STM32 timestamp, GAP8 arrival time) pair the
    deck received: an STM32 message can only arrive after it was stamped, so
    the lower envelope of those pairs maps STM32 time to GAP8 time over the
    short onboard link. The fit is recorded in ``fits`` for inspection.
    """
    stm32_fit = fits.get(CLOCK_STM32_MS)
    if camera is None or stm32_fit is None or stm32_pool is None or not stm32_pool[0]:
        return None
    if "stm32_anchor_ms" not in camera.column_names:
        return None
    reference = statistics.median(stm32_pool[0])  # the unwrapped log axis
    pairs = set()
    for remote, local in zip(camera.column("stm32_anchor_ms").to_pylist(),
                             camera.column("deck_anchor_ms").to_pylist()):
        if remote is None or remote < 0 or local is None or local < 0:
            continue
        low = remote % STM32_MS_WRAP  # log stamps are the tick's low 24 bits
        pairs.add((low + round((reference - low) / STM32_MS_WRAP) * STM32_MS_WRAP, local))
    if len({remote for remote, _ in pairs}) < MIN_CLOCK_ANCHORS:
        return None
    remote, local = zip(*sorted(pairs))
    try:
        anchor = fit_affine(remote, local, clock=GAP8_STM32_FIT, slack=1.0)
    except ValueError:
        return None
    fits[GAP8_STM32_FIT] = anchor  # in ms, not ns: slope GAP8 ms per STM32 ms

    def to_host_ns(deck_ms: float | None) -> float | None:
        if deck_ms is None:
            return None
        return stm32_fit.to_host_ns((deck_ms - anchor.offset_ns) / anchor.slope)

    return to_host_ns


def device_axis(table, clock: str) -> list[float | None]:
    """Per-row device-clock value for a stream, None where the row has none."""
    if clock == CLOCK_STM32_MS:
        unwrapped = unwrap_wrapping_counter(table.column("stm32_ms").to_pylist(), STM32_MS_WRAP)
        return [float(v) for v in unwrapped]
    if clock == CLOCK_NATNET_FRAME:
        return [None if v < 0 else float(v) for v in table.column("source_seq").to_pylist()]
    if clock == CLOCK_ROS_STAMP_S:
        return [float(v) for v in table.column("source_ts_s").to_pylist()]
    if clock == CLOCK_GAP8_MS:
        # -1 marks a frame whose app sent no extension: no deck clock to fit, so
        # the row keeps its host arrival time like any unfittable stream.
        return [None if v < 0 else float(v) for v in table.column("deck_ms").to_pylist()]
    raise ValueError(f"Unknown clock domain {clock!r}.")


def read_stream(run_dir: Path, name: str):
    """One stream's table, from its .parquet or a crashed run's .arrow; None if absent."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_path = run_dir / f"{name}.parquet"
    if parquet_path.exists():
        return pq.read_table(parquet_path)
    arrow_path = run_dir / f"{name}.arrow"
    if not arrow_path.exists():
        return None
    # Crashed-run path: collect every complete batch, ignore the torn tail.
    batches = []
    with arrow_path.open("rb") as handle:
        reader = pa.ipc.open_stream(handle)
        schema = reader.schema
        while True:
            try:
                batches.append(reader.read_next_batch())
            except StopIteration:
                break
            except Exception:
                break
    return pa.Table.from_batches(batches, schema=schema)
