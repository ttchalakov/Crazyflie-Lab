"""Composable camera preflight checks built around the one-feed contract."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

from cf_lab.camera.feed import CameraFeed, CapturedFrame
from cf_lab.camera.preflight import RouteCheck, check_routes
from cf_lab.config import LabConfig
from cf_lab.preflight.core import PreflightResult


def check_camera_mocap_coexistence(cfg: LabConfig) -> PreflightResult:
    """Report the camera/mocap route gate without raising.

    Station mode may legitimately use the same interface. Access-point mode
    requires a distinct mocap path unless the explicit configuration override
    is set.
    """

    started_s = time.monotonic()
    try:
        route = check_routes(cfg)
    except Exception as exc:
        return PreflightResult(
            check_id="camera_mocap_coexistence",
            passed=False,
            summary="Could not inspect the camera and mocap routes.",
            details={},
            error=f"{type(exc).__name__}: {exc}",
            remediation="Check `cf-lab find-deck`, the mocap address and host networking.",
            elapsed_s=time.monotonic() - started_s,
        )

    details: dict[str, object] = {
        "camera_enabled": cfg.camera.enabled,
        "camera_mode": cfg.camera.mode,
        **_route_details(route),
    }
    if route is None:
        return PreflightResult(
            check_id="camera_mocap_coexistence",
            passed=True,
            summary="Camera/mocap route coexistence is not applicable.",
            details=details,
            elapsed_s=time.monotonic() - started_s,
        )

    if cfg.camera.mode == "access_point" and route.shared_interface:
        return PreflightResult(
            check_id="camera_mocap_coexistence",
            passed=False,
            summary=(
                "Camera AP and mocap resolve through the same host interface; "
                "joining the camera AP would remove the mocap path."
            ),
            details=details,
            remediation="Use station mode, or join the camera AP on a second interface.",
            elapsed_s=time.monotonic() - started_s,
        )

    return PreflightResult(
        check_id="camera_mocap_coexistence",
        passed=True,
        summary="Camera and mocap routes can coexist.",
        details=details,
        elapsed_s=time.monotonic() - started_s,
    )


def _route_details(route: RouteCheck | None) -> dict[str, object]:
    if route is None:
        return {
            "camera_ip": None,
            "camera_source": None,
            "mocap_ip": None,
            "mocap_source": None,
            "shared_interface": None,
        }
    return {
        "camera_ip": route.camera_ip,
        "camera_source": route.camera_source,
        "mocap_ip": route.mocap_ip,
        "mocap_source": route.mocap_source,
        "shared_interface": route.shared_interface,
    }


@dataclass
class _FrameCollector:
    frames: list[CapturedFrame] = field(default_factory=list)
    received_ns: list[int] = field(default_factory=list)
    condition: threading.Condition = field(default_factory=threading.Condition)

    def offer(self, frame: CapturedFrame) -> None:
        with self.condition:
            self.frames.append(frame)
            self.received_ns.append(time.monotonic_ns())
            self.condition.notify_all()

    def wait_for(self, count: int, timeout_s: float) -> None:
        deadline_s = time.monotonic() + timeout_s
        with self.condition:
            while len(self.frames) < count:
                remaining_s = deadline_s - time.monotonic()
                if remaining_s <= 0.0:
                    return
                self.condition.wait(remaining_s)


def _counter_progress(values: list[int]) -> dict[str, int]:
    regressions = 0
    duplicates = 0
    gaps = 0
    for previous, current in zip(values, values[1:]):
        delta = (current - previous) & 0xFFFFFFFF
        if delta == 0:
            duplicates += 1
        elif delta >= 0x80000000:
            regressions += 1
        else:
            gaps += max(0, delta - 1)
    return {"regressions": regressions, "duplicates": duplicates, "gaps": gaps}


def _cumulative_delta(values: list[int]) -> tuple[int, int]:
    """Return (increase, regressions), accepting normal uint32 wrap."""

    increase = 0
    regressions = 0
    for previous, current in zip(values, values[1:]):
        delta = (current - previous) & 0xFFFFFFFF
        if delta >= 0x80000000:
            regressions += 1
        else:
            increase += delta
    return increase, regressions


def check_camera_stream(
    cfg: LabConfig | None = None,
    *,
    feed: CameraFeed | None = None,
    timeout_s: float = 8.0,
    sample_frames: int = 8,
    minimum_fps: float | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_format: str | None = None,
    require_extension: bool = False,
    max_new_drops: int | None = None,
    analyze_pixels: bool = False,
    dark_level: int = 5,
    clipped_level: int = 250,
    minimum_mean: float | None = None,
    maximum_mean: float | None = None,
    maximum_dark_fraction: float | None = None,
    maximum_clipped_fraction: float | None = None,
) -> PreflightResult:
    """Check several frames from either a new or the experiment's live feed.

    Passing ``feed`` is the safe choice once an experiment has opened the
    camera: this attaches a temporary observer and never starts, stops, or
    replaces that connection. If ``feed`` is omitted, the check owns a short
    connection created from ``cfg`` and always closes it before returning.

    Drop counters are reported whenever the camera-stream extension is
    available. They are not failures by default because preview and overwrite
    policies intentionally skip frames; set ``max_new_drops`` when a particular
    experiment has a stricter policy. Pixel analysis is likewise opt-in because
    scene brightness is normally advisory rather than a flight-safety invariant.
    """

    started_s = time.monotonic()
    expected_kind = expected_format.lower().replace("-", "_") if expected_format else None
    if expected_kind in ("raw_grayscale", "grayscale"):
        expected_kind = "raw"

    validation_errors: list[str] = []
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        validation_errors.append("timeout_s must be finite and > 0")
    if sample_frames < 2:
        validation_errors.append("sample_frames must be >= 2")
    if minimum_fps is not None and (not math.isfinite(minimum_fps) or minimum_fps < 0.0):
        validation_errors.append("minimum_fps must be finite and >= 0")
    if expected_width is not None and expected_width <= 0:
        validation_errors.append("expected_width must be > 0")
    if expected_height is not None and expected_height <= 0:
        validation_errors.append("expected_height must be > 0")
    if expected_kind not in (None, "raw", "jpeg"):
        validation_errors.append("expected_format must be 'raw' or 'jpeg'")
    if max_new_drops is not None and max_new_drops < 0:
        validation_errors.append("max_new_drops must be >= 0")
    if not 0 <= dark_level <= 255 or not 0 <= clipped_level <= 255:
        validation_errors.append("dark_level and clipped_level must be in [0, 255]")
    if dark_level >= clipped_level:
        validation_errors.append("dark_level must be below clipped_level")
    for name, value in (
        ("minimum_mean", minimum_mean),
        ("maximum_mean", maximum_mean),
    ):
        if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 255.0):
            validation_errors.append(f"{name} must be finite and in [0, 255]")
    for name, value in (
        ("maximum_dark_fraction", maximum_dark_fraction),
        ("maximum_clipped_fraction", maximum_clipped_fraction),
    ):
        if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
            validation_errors.append(f"{name} must be finite and in [0, 1]")
    if feed is None and cfg is None:
        validation_errors.append("cfg is required when feed is not supplied")
    if validation_errors:
        return PreflightResult(
            check_id="camera_stream",
            passed=False,
            summary="Invalid camera-stream check configuration.",
            details={"validation_errors": tuple(validation_errors)},
            error="; ".join(validation_errors),
            remediation="Correct the preflight thresholds or supply the active feed.",
            elapsed_s=time.monotonic() - started_s,
        )

    owned_feed = feed is None
    if owned_feed:
        assert cfg is not None
        if not cfg.camera.enabled:
            return PreflightResult(
                check_id="camera_stream",
                passed=False,
                summary="Camera stream is disabled in the supplied configuration.",
                details={"owned_feed": True},
                remediation=(
                    "Set the selected mode's SSID under camera: or omit this check "
                    "for a camera-free run."
                ),
                elapsed_s=time.monotonic() - started_s,
            )
        try:
            feed = CameraFeed.from_config(
                cfg,
                connect_timeout_s=timeout_s,
                read_timeout_s=timeout_s,
            )
        except OSError as exc:  # the deck did not answer its name
            return PreflightResult(
                check_id="camera_stream",
                passed=False,
                summary="Could not find the deck on the network.",
                details={"owned_feed": True},
                error=f"{type(exc).__name__}: {exc}",
                remediation="Check `cf-lab find-deck` and the deck's WiFi settings.",
                elapsed_s=time.monotonic() - started_s,
            )

    assert feed is not None
    collector = _FrameCollector()
    try:
        unsubscribe = feed.add_observer(collector)
    except Exception as exc:
        return PreflightResult(
            check_id="camera_stream",
            passed=False,
            summary="Could not observe the camera feed.",
            details={"owned_feed": owned_feed},
            error=f"{type(exc).__name__}: {exc}",
            remediation="Pass the active CameraFeed or a valid camera configuration.",
            elapsed_s=time.monotonic() - started_s,
        )

    operation_error: Exception | None = None
    observed_feed_error: BaseException | None = None
    sample_started_s = time.monotonic()
    try:
        if owned_feed:
            feed.start()
        collector.wait_for(sample_frames, timeout_s)
    except Exception as exc:
        operation_error = exc
    finally:
        unsubscribe()
        # Closing an owned socket intentionally wakes its receive thread with
        # an exception. Capture real stream errors before that shutdown so the
        # check does not mistake its own cleanup for a camera failure.
        observed_feed_error = feed.error
        if owned_feed:
            feed.stop()
    sample_ended_s = time.monotonic()

    with collector.condition:
        frames = collector.frames[:sample_frames]
        receipt_ns = collector.received_ns[:sample_frames]
    if len(receipt_ns) >= 2 and receipt_ns[-1] > receipt_ns[0]:
        observed_fps = (len(receipt_ns) - 1) * 1e9 / (receipt_ns[-1] - receipt_ns[0])
    else:
        observed_fps = 0.0

    ext_frames = [frame for frame in frames if frame.ext is not None]
    app_ids = [int(frame.ext.frame_id) for frame in ext_frames]
    camera_ids = [
        int(getattr(frame.ext, "camera_frame_id"))
        for frame in ext_frames
        if hasattr(frame.ext, "camera_frame_id")
    ]
    app_progress = _counter_progress(app_ids)
    camera_progress = _counter_progress(camera_ids)

    drop_fields = ("dropped_capture", "dropped_ready", "dropped_streaming")
    drop_deltas: dict[str, int | None] = {}
    drop_regressions = 0
    for field_name in drop_fields:
        values = [
            int(getattr(frame.ext, field_name))
            for frame in ext_frames
            if hasattr(frame.ext, field_name)
        ]
        if len(values) == len(ext_frames) and len(values) >= 1:
            delta, regressions = _cumulative_delta(values)
            drop_deltas[field_name] = delta
            drop_regressions += regressions
        else:
            drop_deltas[field_name] = None
    new_drops = sum(value for value in drop_deltas.values() if value is not None)

    failures: list[str] = []
    if operation_error is not None:
        failures.append(f"{type(operation_error).__name__}: {operation_error}")
    elif observed_feed_error is not None:
        failures.append(
            f"{type(observed_feed_error).__name__}: {observed_feed_error}"
        )
    if len(frames) < sample_frames:
        failures.append(f"received {len(frames)} frame(s), need {sample_frames}")
    if minimum_fps is not None and observed_fps < minimum_fps:
        failures.append(f"received FPS {observed_fps:.1f} is below {minimum_fps:.1f}")
    widths = sorted({frame.width for frame in frames})
    heights = sorted({frame.height for frame in frames})
    kinds = sorted({"jpeg" if frame.is_jpeg else "raw" for frame in frames})
    if expected_width is not None and widths != [expected_width]:
        failures.append(f"width(s) {widths!r}, expected {expected_width}")
    if expected_height is not None and heights != [expected_height]:
        failures.append(f"height(s) {heights!r}, expected {expected_height}")
    if expected_kind is not None and kinds != [expected_kind]:
        failures.append(f"format(s) {kinds!r}, expected {expected_kind}")
    if require_extension and len(ext_frames) != len(frames):
        failures.append(
            f"camera extension decoded for {len(ext_frames)}/{len(frames)} frame(s)"
        )
    if app_progress["regressions"] or app_progress["duplicates"]:
        failures.append("application frame IDs are not strictly advancing")
    if camera_progress["regressions"] or camera_progress["duplicates"]:
        failures.append("camera frame IDs are not strictly advancing")
    if drop_regressions:
        failures.append(f"camera drop counters regressed {drop_regressions} time(s)")
    if max_new_drops is not None and new_drops > max_new_drops:
        failures.append(f"drop counters increased by {new_drops} (limit {max_new_drops})")

    pixel_details: dict[str, object] = {}
    pixel_requested = analyze_pixels or any(
        threshold is not None
        for threshold in (
            minimum_mean,
            maximum_mean,
            maximum_dark_fraction,
            maximum_clipped_fraction,
        )
    )
    if pixel_requested and frames:
        try:
            pixel_details = _pixel_metrics(frames, dark_level, clipped_level)
        except Exception as exc:
            failures.append(f"pixel analysis failed: {type(exc).__name__}: {exc}")
        else:
            mean = float(pixel_details["pixel_mean"])
            dark_fraction = float(pixel_details["dark_fraction"])
            clipped_fraction = float(pixel_details["clipped_fraction"])
            if minimum_mean is not None and mean < minimum_mean:
                failures.append(f"pixel mean {mean:.1f} is below {minimum_mean:.1f}")
            if maximum_mean is not None and mean > maximum_mean:
                failures.append(f"pixel mean {mean:.1f} exceeds {maximum_mean:.1f}")
            if maximum_dark_fraction is not None and dark_fraction > maximum_dark_fraction:
                failures.append(
                    f"dark fraction {dark_fraction:.3f} exceeds {maximum_dark_fraction:.3f}"
                )
            if (
                maximum_clipped_fraction is not None
                and clipped_fraction > maximum_clipped_fraction
            ):
                failures.append(
                    f"clipped fraction {clipped_fraction:.3f} exceeds "
                    f"{maximum_clipped_fraction:.3f}"
                )

    details: dict[str, object] = {
        "owned_feed": owned_feed,
        "sample_count": len(frames),
        "requested_sample_count": sample_frames,
        "sample_window_s": sample_ended_s - sample_started_s,
        "observed_fps": observed_fps,
        "widths": tuple(widths),
        "heights": tuple(heights),
        "formats": tuple(kinds),
        "extension_frames": len(ext_frames),
        "application_frame_ids": tuple(app_ids),
        "application_id_regressions": app_progress["regressions"],
        "application_id_duplicates": app_progress["duplicates"],
        "application_id_gaps": app_progress["gaps"],
        "camera_frame_ids": tuple(camera_ids),
        "camera_id_regressions": camera_progress["regressions"],
        "camera_id_duplicates": camera_progress["duplicates"],
        "camera_id_gaps": camera_progress["gaps"],
        "drop_deltas": drop_deltas,
        "new_drops": new_drops,
        "feed_frames_received_total": feed.frames_received,
        "feed_sink_errors_total": feed.sink_errors,
        "feed_observer_errors_total": feed.observer_errors,
        "failures": tuple(failures),
        **pixel_details,
    }
    passed = not failures
    return PreflightResult(
        check_id="camera_stream",
        passed=passed,
        summary=(
            f"Camera stream passed: {len(frames)} frame(s), {observed_fps:.1f} FPS, "
            f"{widths[0]}x{heights[0]} {kinds[0]}."
            if passed and widths and heights and kinds
            else "Camera stream failed: " + "; ".join(failures)
        ),
        details=details,
        error=(
            f"{type(operation_error).__name__}: {operation_error}"
            if operation_error is not None
            else (
                f"{type(observed_feed_error).__name__}: {observed_feed_error}"
                if observed_feed_error is not None
                else None
            )
        ),
        remediation=(
            "Confirm the host has the deck's only connection, the GAP8 build matches "
            "the configured transport/profile, and the camera scene is usable."
            if not passed
            else ""
        ),
        elapsed_s=time.monotonic() - started_s,
    )


def _pixel_metrics(
    frames: list[CapturedFrame], dark_level: int, clipped_level: int
) -> dict[str, object]:
    import numpy as np

    histogram = np.zeros(256, dtype=np.int64)
    decoded_frames = 0
    for frame in frames:
        image = frame.decode()
        if image is None:
            raise ValueError(
                f"could not decode a {frame.width}x{frame.height} "
                f"{'JPEG' if frame.is_jpeg else 'raw'} frame"
            )
        flat = np.asarray(image, dtype=np.uint8).reshape(-1)
        histogram += np.bincount(flat, minlength=256)
        decoded_frames += 1
    pixel_count = int(histogram.sum())
    if pixel_count == 0:
        raise ValueError("decoded frames contain no pixels")
    values = np.arange(256, dtype=np.float64)
    cumulative = np.cumsum(histogram)

    def percentile(q: float) -> int:
        return int(np.searchsorted(cumulative, math.ceil(q * pixel_count), side="left"))

    return {
        "pixel_frames": decoded_frames,
        "pixel_count": pixel_count,
        "pixel_mean": float((histogram * values).sum() / pixel_count),
        "pixel_min": int(np.flatnonzero(histogram)[0]),
        "pixel_max": int(np.flatnonzero(histogram)[-1]),
        "pixel_p10": percentile(0.10),
        "pixel_median": percentile(0.50),
        "pixel_p90": percentile(0.90),
        "dark_level": dark_level,
        "dark_fraction": float(histogram[: dark_level + 1].sum() / pixel_count),
        "clipped_level": clipped_level,
        "clipped_fraction": float(histogram[clipped_level:].sum() / pixel_count),
    }
