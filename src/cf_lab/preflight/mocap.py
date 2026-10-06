"""Composable, non-owning mocap preflight checks."""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections.abc import Sequence

from cf_lab.mocap.stream import ExtposeStreamer, MocapObservation
from cf_lab.mocap.types import PoseSample
from cf_lab.preflight.core import PreflightResult


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def _orientation_step_deg(left: PoseSample, right: PoseSample) -> float:
    a = left.orientation
    b = right.orientation
    av = (a.qx, a.qy, a.qz, a.qw)
    bv = (b.qx, b.qy, b.qz, b.qw)
    an = math.sqrt(sum(value * value for value in av))
    bn = math.sqrt(sum(value * value for value in bv))
    if an == 0.0 or bn == 0.0:
        return math.inf
    dot = abs(sum(x * y for x, y in zip(av, bv)) / (an * bn))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


DEFAULT_DURATION_S = 1.0  # length of the observed window
DEFAULT_MAX_FRESHNESS_MS = 50.0  # oldest mocap pose the quality check accepts
DEFAULT_MAX_POSITION_SPREAD_M = 0.05  # stationary drift allowed over the window
DEFAULT_MAX_POSITION_JUMP_M = 0.05  # largest step between consecutive poses
DEFAULT_MAX_ORIENTATION_JUMP_DEG = 20.0  # largest rotation between consecutive poses


def check_mocap_quality(
    streamer: ExtposeStreamer,
    *,
    duration_s: float = DEFAULT_DURATION_S,
    expected_rate_hz: float | None = None,
    minimum_rate_hz: float | None = None,
    max_freshness_ms: float = DEFAULT_MAX_FRESHNESS_MS,
    minimum_samples: int = 2,
    expected_rigid_body: str | None = None,
    quaternion_norm_tolerance: float = 0.05,
    max_position_spread_m: float | None = DEFAULT_MAX_POSITION_SPREAD_M,
    max_position_jump_m: float | None = DEFAULT_MAX_POSITION_JUMP_M,
    max_orientation_jump_deg: float | None = DEFAULT_MAX_ORIENTATION_JUMP_DEG,
) -> PreflightResult:
    """Measure the already-running extpose stream over a bounded window.

    This function attaches to :class:`ExtposeStreamer`'s observer path; it
    never calls the mocap source and therefore cannot steal samples from the
    single live consumer. Source timestamps are retained for reporting, but
    freshness and rate use host-monotonic receipt/forward timestamps.

    The default ``max_position_spread_m`` expects the vehicle to sit still on
    its pad; pass ``None`` for a moving rigid body. ``max_position_jump_m``
    and ``max_orientation_jump_deg`` instead guard consecutive-pose
    discontinuity and also hold for a moving rigid body.
    """

    started_s = time.monotonic()
    invalid: list[str] = []
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        invalid.append("duration_s must be finite and > 0")
    if minimum_samples < 1:
        invalid.append("minimum_samples must be >= 1")
    for name, value in (
        ("expected_rate_hz", expected_rate_hz),
        ("minimum_rate_hz", minimum_rate_hz),
        ("max_freshness_ms", max_freshness_ms),
        ("quaternion_norm_tolerance", quaternion_norm_tolerance),
        ("max_position_spread_m", max_position_spread_m),
        ("max_position_jump_m", max_position_jump_m),
        ("max_orientation_jump_deg", max_orientation_jump_deg),
    ):
        if value is not None and (not math.isfinite(value) or value < 0.0):
            invalid.append(f"{name} must be finite and >= 0")
    if invalid:
        return PreflightResult(
            check_id="mocap_quality",
            passed=False,
            summary="Invalid mocap-quality check configuration.",
            details={"validation_errors": tuple(invalid)},
            error="; ".join(invalid),
            remediation="Correct the preflight thresholds before using this check.",
            elapsed_s=time.monotonic() - started_s,
        )

    if minimum_rate_hz is None and expected_rate_hz is not None:
        minimum_rate_hz = expected_rate_hz * 0.8

    observations: list[MocapObservation] = []
    observations_lock = threading.Lock()

    def observe(observation: MocapObservation) -> None:
        with observations_lock:
            observations.append(observation)

    try:
        unsubscribe = streamer.add_observer(observe)
    except Exception as exc:
        return PreflightResult(
            check_id="mocap_quality",
            passed=False,
            summary="Could not observe the running mocap stream.",
            details={},
            error=f"{type(exc).__name__}: {exc}",
            remediation="Pass the active ExtposeStreamer used by the experiment.",
            elapsed_s=time.monotonic() - started_s,
        )

    sample_started_s = time.monotonic()
    try:
        threading.Event().wait(duration_s)
    finally:
        unsubscribe()
    sample_ended_s = time.monotonic()
    with observations_lock:
        samples = tuple(observations)

    poses = [observation.pose for observation in samples]
    receive_times = [observation.received_monotonic_s for observation in samples]
    forwarded_samples = [
        observation for observation in samples if observation.forwarded_monotonic_s is not None
    ]
    forward_times = [
        float(observation.forwarded_monotonic_s) for observation in forwarded_samples
    ]
    measured_window_s = max(sample_ended_s - sample_started_s, 1e-12)
    received_rate_hz = len(samples) / measured_window_s
    observed_rate_hz = len(forwarded_samples) / measured_window_s
    receipt_intervals_ms = [
        (right - left) * 1000.0 for left, right in zip(receive_times, receive_times[1:])
    ]
    intervals_ms = [
        (right - left) * 1000.0 for left, right in zip(forward_times, forward_times[1:])
    ]
    forward_latency_ms = [
        max(
            0.0,
            (
                float(observation.forwarded_monotonic_s)
                - observation.received_monotonic_s
            )
            * 1000.0,
        )
        for observation in forwarded_samples
    ]

    if receive_times:
        receipt_freshness_ms = max(0.0, (sample_ended_s - receive_times[-1]) * 1000.0)
    elif streamer.latest_received_monotonic_s is not None:
        receipt_freshness_ms = max(
            0.0,
            (sample_ended_s - streamer.latest_received_monotonic_s) * 1000.0,
        )
    else:
        receipt_freshness_ms = math.inf
    if forward_times:
        forward_freshness_ms = max(0.0, (sample_ended_s - forward_times[-1]) * 1000.0)
    elif streamer.latest_forwarded_monotonic_s is not None:
        forward_freshness_ms = max(
            0.0,
            (sample_ended_s - streamer.latest_forwarded_monotonic_s) * 1000.0,
        )
    else:
        forward_freshness_ms = math.inf

    invalid_position_samples = 0
    invalid_quaternion_samples = 0
    quaternion_norm_errors: list[float] = []
    for pose in poses:
        if not all(math.isfinite(value) for value in pose.position_m):
            invalid_position_samples += 1
        q = pose.orientation
        q_values = (q.qx, q.qy, q.qz, q.qw)
        if not all(math.isfinite(value) for value in q_values):
            invalid_quaternion_samples += 1
            continue
        norm = math.sqrt(sum(value * value for value in q_values))
        quaternion_norm_errors.append(abs(norm - 1.0))
        if norm == 0.0 or abs(norm - 1.0) > quaternion_norm_tolerance:
            invalid_quaternion_samples += 1

    finite_positions = [
        pose.position_m
        for pose in poses
        if all(math.isfinite(value) for value in pose.position_m)
    ]
    if finite_positions:
        axis_ranges = tuple(
            max(position[axis] for position in finite_positions)
            - min(position[axis] for position in finite_positions)
            for axis in range(3)
        )
        position_spread_m = math.sqrt(sum(value * value for value in axis_ranges))
    else:
        axis_ranges = (math.inf, math.inf, math.inf)
        position_spread_m = math.inf

    position_jumps = [
        _distance(left.position_m, right.position_m)
        for left, right in zip(poses, poses[1:])
        if all(math.isfinite(value) for value in (*left.position_m, *right.position_m))
    ]
    orientation_jumps = [
        _orientation_step_deg(left, right) for left, right in zip(poses, poses[1:])
    ]
    max_position_jump_observed_m = max(position_jumps, default=0.0)
    max_orientation_jump_observed_deg = max(orientation_jumps, default=0.0)

    seqs = [pose.source_seq for pose in poses if pose.source_seq is not None]
    source_seq_regressions = 0
    source_seq_duplicates = 0
    source_seq_gaps = 0
    for previous, current in zip(seqs, seqs[1:]):
        if current < previous:
            source_seq_regressions += 1
        elif current == previous:
            source_seq_duplicates += 1
        else:
            source_seq_gaps += max(0, current - previous - 1)

    rigid_bodies = sorted({pose.rigid_body for pose in poses})
    failures: list[str] = []
    if len(forwarded_samples) < minimum_samples:
        failures.append(
            f"only {len(forwarded_samples)} forwarded sample(s), "
            f"need at least {minimum_samples}"
        )
    if minimum_rate_hz is not None and observed_rate_hz < minimum_rate_hz:
        failures.append(
            f"forwarded rate {observed_rate_hz:.1f} Hz is below {minimum_rate_hz:.1f} Hz"
        )
    if receipt_freshness_ms > max_freshness_ms:
        failures.append(
            f"latest received pose is {receipt_freshness_ms:.1f} ms old "
            f"(limit {max_freshness_ms:.1f} ms)"
        )
    if forward_freshness_ms > max_freshness_ms:
        failures.append(
            f"latest forwarded pose is {forward_freshness_ms:.1f} ms old "
            f"(limit {max_freshness_ms:.1f} ms)"
        )
    if invalid_position_samples:
        failures.append(f"{invalid_position_samples} pose(s) have non-finite position")
    if invalid_quaternion_samples:
        failures.append(f"{invalid_quaternion_samples} pose(s) have invalid quaternion")
    if source_seq_regressions or source_seq_duplicates:
        failures.append(
            "source sequence is not strictly increasing "
            f"({source_seq_regressions} regression(s), {source_seq_duplicates} duplicate(s))"
        )
    if expected_rigid_body is not None and rigid_bodies != [expected_rigid_body]:
        failures.append(
            f"observed rigid body/bodies {rigid_bodies!r}, expected {expected_rigid_body!r}"
        )
    if max_position_spread_m is not None and position_spread_m > max_position_spread_m:
        failures.append(
            f"stationary position spread {position_spread_m:.4f} m exceeds "
            f"{max_position_spread_m:.4f} m"
        )
    if (
        max_position_jump_m is not None
        and max_position_jump_observed_m > max_position_jump_m
    ):
        failures.append(
            f"position jump {max_position_jump_observed_m:.4f} m exceeds "
            f"{max_position_jump_m:.4f} m"
        )
    if (
        max_orientation_jump_deg is not None
        and max_orientation_jump_observed_deg > max_orientation_jump_deg
    ):
        failures.append(
            f"orientation jump {max_orientation_jump_observed_deg:.2f} deg exceeds "
            f"{max_orientation_jump_deg:.2f} deg"
        )
    if streamer.error is not None:
        failures.append(f"streamer stopped with {type(streamer.error).__name__}: {streamer.error}")

    details: dict[str, object] = {
        "sample_count": len(samples),
        "forwarded_sample_count": len(forwarded_samples),
        "window_s": measured_window_s,
        "received_rate_hz": received_rate_hz,
        "observed_rate_hz": observed_rate_hz,
        "expected_rate_hz": expected_rate_hz,
        "minimum_rate_hz": minimum_rate_hz,
        # ``freshness_ms`` remains the operational uplink freshness for simple
        # callers; the source and uplink values are also reported separately.
        "freshness_ms": _finite_or_none(forward_freshness_ms),
        "receipt_freshness_ms": _finite_or_none(receipt_freshness_ms),
        "forward_freshness_ms": _finite_or_none(forward_freshness_ms),
        "receipt_interval_mean_ms": (
            statistics.fmean(receipt_intervals_ms) if receipt_intervals_ms else None
        ),
        "receipt_interval_jitter_ms": (
            statistics.pstdev(receipt_intervals_ms)
            if len(receipt_intervals_ms) > 1
            else 0.0
        ),
        "receipt_longest_gap_ms": max(receipt_intervals_ms, default=None),
        "interval_mean_ms": statistics.fmean(intervals_ms) if intervals_ms else None,
        "interval_jitter_ms": statistics.pstdev(intervals_ms) if len(intervals_ms) > 1 else 0.0,
        "longest_gap_ms": max(intervals_ms, default=None),
        "forward_latency_max_ms": max(forward_latency_ms, default=None),
        "invalid_position_samples": invalid_position_samples,
        "invalid_quaternion_samples": invalid_quaternion_samples,
        "quaternion_norm_error_max": max(quaternion_norm_errors, default=None),
        "position_axis_ranges_m": tuple(_finite_or_none(value) for value in axis_ranges),
        "position_spread_m": _finite_or_none(position_spread_m),
        "max_position_jump_m": _finite_or_none(max_position_jump_observed_m),
        "max_orientation_jump_deg": _finite_or_none(
            max_orientation_jump_observed_deg
        ),
        "source_seq_regressions": source_seq_regressions,
        "source_seq_duplicates": source_seq_duplicates,
        "source_seq_gaps": source_seq_gaps,
        "rigid_bodies": tuple(rigid_bodies),
        "streamer_sent_total": streamer.sent,
        "streamer_dropped_total": streamer.dropped,
        "observer_errors_total": streamer.observer_errors,
        "failures": tuple(failures),
    }
    passed = not failures
    return PreflightResult(
        check_id="mocap_quality",
        passed=passed,
        summary=(
            f"Mocap quality passed: {len(samples)} received / "
            f"{len(forwarded_samples)} forwarded sample(s) at "
            f"{received_rate_hz:.1f}/{observed_rate_hz:.1f} Hz."
            if passed
            else "Mocap quality failed: " + "; ".join(failures)
        ),
        details=details,
        error=(
            f"{type(streamer.error).__name__}: {streamer.error}"
            if streamer.error is not None
            else None
        ),
        remediation=(
            "Check rigid-body selection, mocap visibility/network health, and keep the "
            "vehicle motionless."
            if not passed
            else ""
        ),
        elapsed_s=time.monotonic() - started_s,
    )


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None
