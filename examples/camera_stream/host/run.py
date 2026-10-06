#!/usr/bin/env python3
"""camera_stream: watch and record the AI-deck camera, with mocap when configured.

Flash the GAP8 app (see the README), put this computer on the deck's network,
then:

  python examples/camera_stream/host/run.py                  # window + recording
  python examples/camera_stream/host/run.py --no-display     # metrics in the terminal
  python examples/camera_stream/host/run.py --duration 30    # stop after 30 s

Everything comes from --config (default configs/drone.yaml): radio.uri names the
deck to look up, its camera: section says how to talk to it, and its telemetry:
section what to record. Nothing is recorded until the run has what it needs, and
it stops instead if:

  - no frames arrive from the deck, or
  - telemetry.mocap is on and no poses arrive for mocap.rigid_body.

With telemetry.mocap off it records the camera alone. At the end it renders the
recorded frames to camera.mp4 in the run directory, at the frame rate the deck
was built for (--no-render skips that).

Ctrl-C (or closing the window) stops cleanly: the socket closes at a frame
boundary, which leaves the deck ready for the next run.
"""

from __future__ import annotations

import argparse
import dataclasses
import signal
import threading
import time
from functools import partial
from pathlib import Path

from cf_lab.camera.feed import DEFAULT_READ_TIMEOUT_S, CameraFeed
from cf_lab.camera.find_deck import resolve_deck
from cf_lab.camera.stats import FeedStats, stats_line
from cf_lab.config import LabConfig, load_config

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
# How long the startup checks wait for frames and for a mocap pose.
CAMERA_CHECK_S = 10.0
MOCAP_CHECK_S = 5.0


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _describe(config: dict | None) -> str:
    if not config:
        return "the deck did not describe its build"
    return (f"{config.get('sensor')} -> {config.get('out')} {config.get('format')}, "
            f"{config.get('policy')}, {config.get('fps')} fps, {str(config.get('transport')).upper()}")


def _check_camera(feed: CameraFeed) -> bool:
    from cf_lab.preflight.camera import check_camera_stream

    result = check_camera_stream(feed=feed, timeout_s=CAMERA_CHECK_S, sample_frames=5)
    if not result:
        _log(f"No camera frames: {result.error or result.summary}")
        return False
    _log(f"Camera: {_describe(feed.deck_config)}")
    return True


def _check_mocap(cfg: LabConfig):
    """Connect to mocap and wait for a pose of the rigid body; the source, or None."""
    from cf_lab.mocap import build_live_pose_source
    from cf_lab.mocap.stream import first_pose

    body = cfg.mocap.rigid_body
    try:
        source = build_live_pose_source(cfg=cfg)
        pose = first_pose(source, timeout_s=MOCAP_CHECK_S)
    except Exception as exc:
        _log(f"Mocap: cannot connect ({type(exc).__name__}: {exc})")
        return None
    if pose is None:
        _log(f"Mocap: no pose for rigid body {body!r} within {MOCAP_CHECK_S:.0f} s. "
             f"Is it tracked in Motive?")
        return None
    x, y, z = pose.position_m
    _log(f"Mocap: {body!r} at ({x:.2f}, {y:.2f}, {z:.2f}) m")
    return source


def _start_recorder(cfg: LabConfig, run_root: Path, feed: CameraFeed, mocap_source):
    """Record what telemetry: selects. Only the camera and raw mocap apply here:
    the STM32, extpose and setpoint streams need a flight loop (see mocap_waypoint)."""
    from cf_lab.telemetry.recorder import TelemetryRecorder

    record = dataclasses.replace(
        cfg.telemetry, state_hz=0.0, controls_hz=0.0, setpoints=False, extpose=False, custom=(),
    )
    recorder = TelemetryRecorder(record, run_root=run_root, lab_config=cfg, example="camera_stream")
    recorder.start()
    recorder.attach_camera(feed)
    if mocap_source is not None:
        recorder.attach_mocap(mocap_source)
    return recorder


def _render(run_dir: Path) -> None:
    from cf_lab.camera.video import render_video

    out = run_dir / "camera.mp4"
    try:
        stats = render_video(run_dir, out)
    except Exception as exc:
        _log(f"Video not rendered: {exc}")
        return
    _log(f"Video: {out} ({stats.source_frames} frames at {stats.fps:g} fps, "
         f"{stats.duration_s:.1f} s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/drone.yaml"))
    parser.add_argument("--no-display", action="store_true",
                        help="no window; print metrics once a second")
    parser.add_argument("--no-render", action="store_true",
                        help="do not render the recorded frames to camera.mp4")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="stop after this many seconds (0: until Ctrl-C)")
    parser.add_argument("--connect-timeout", type=float, default=30.0,
                        help="seconds to keep retrying the deck; it takes ~15 s to boot")
    parser.add_argument("--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT_S,
                        help="seconds without data before the stream counts as lost")
    parser.add_argument("--record-root", type=Path, default=ARTIFACTS_DIR,
                        help="where run_<timestamp> directories go")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    telemetry = cfg.telemetry
    need_mocap = telemetry.enabled and telemetry.mocap
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    _log(f"Looking up the deck of {cfg.radio.uri} ...")
    try:
        ip = resolve_deck(cfg.radio.uri, args.connect_timeout)
    except OSError as exc:
        _log(f"Cannot find the deck: {exc}")
        return 1
    feed = CameraFeed(ip=ip, port=cfg.camera.deck_port, transport=cfg.camera.transport,
                      connect_timeout_s=args.connect_timeout,
                      read_timeout_s=args.read_timeout,
                      look_up=partial(resolve_deck, cfg.radio.uri), log=_log)
    stats = FeedStats()
    feed.subscribe(stats)
    viewer = None
    if not args.no_display:
        from cf_lab.camera.viewer import Viewer

        viewer = Viewer(title="camera_stream", snapshot_dir=args.record_root)
        feed.subscribe(viewer)

    address = f"{ip}:{cfg.camera.deck_port} ({cfg.camera.transport.upper()})"
    _log(f"Connecting to the deck at {address} ...")
    try:
        feed.start()
    except OSError as exc:
        _log(f"Cannot reach the deck: {exc}. Is the GAP8 app flashed, and is this "
             f"computer on the deck's network?")
        return 1

    recorder = None
    status = "clean"

    def recording_status() -> str:
        if recorder is None:
            return ""
        parts = [f"REC {recorder.run_dir.name}"]
        if recorder.camera_stats:
            parts.append(f"{recorder.camera_stats['written']} frames")
        if recorder.mocap_stats:
            parts.append(f"{recorder.mocap_stats['recorded']} poses")
        return "  ".join(parts)

    try:
        if not _check_camera(feed):
            _log("Not starting.")
            return 1
        mocap_source = None
        if need_mocap:
            mocap_source = _check_mocap(cfg)
            if mocap_source is None:
                _log("Not starting. Set telemetry.mocap: false to record the camera alone.")
                return 1

        if telemetry.enabled and (telemetry.camera or telemetry.mocap):
            recorder = _start_recorder(cfg, args.record_root, feed, mocap_source)
            _log(f"Recording to {recorder.run_dir}")
        else:
            _log("Not recording (telemetry.enabled, .camera and .mocap are off).")

        deadline = time.monotonic() + args.duration if args.duration > 0 else None

        def should_stop() -> bool:
            return (stop.is_set() or feed.error is not None
                    or (deadline is not None and time.monotonic() >= deadline))

        _log("Streaming. Ctrl-C to stop.")
        if viewer is not None:
            viewer.status = recording_status
            if recorder is not None:
                viewer.snapshot_dir = recorder.run_dir
            viewer.run(should_stop)
        else:
            while not should_stop():
                stop.wait(1.0)
                _log(stats_line(stats.snapshot()))
        if feed.error is not None:
            _log(f"Stream lost: {type(feed.error).__name__}: {feed.error}")
            status = "error"
    finally:
        feed.stop()
        if recorder is not None:
            run_dir = recorder.run_dir
            recorder.close(status=status)
            counts = [f"{recorder.camera_stats['written']} frames" if recorder.camera_stats else "",
                      f"{recorder.mocap_stats['recorded']} poses" if recorder.mocap_stats else ""]
            _log(f"Recorded {', '.join(c for c in counts if c) or 'nothing'} -> {run_dir}")
            if telemetry.camera and not args.no_render and recorder.camera_stats.get("written"):
                _render(run_dir)
    _log(f"Stopped after {feed.frames_received} frames.")
    return 0 if status == "clean" else 1


if __name__ == "__main__":
    raise SystemExit(main())
