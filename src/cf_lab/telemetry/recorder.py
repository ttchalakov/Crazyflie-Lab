"""TelemetryRecorder: the one object an example runner touches.

A runner builds it from ``cfg.telemetry``, calls ``start()``, wraps its mocap
source, attaches the STM32 log taps once the radio is up, records setpoints
and event marks where they happen, and calls ``close()`` in a ``finally``.
When telemetry is disabled every method is a no-op.
"""

from __future__ import annotations

import dataclasses
import datetime
import platform
import subprocess
import threading
from pathlib import Path
from typing import Any

from cf_lab.config import LabConfig, TelemetryConfig
from cf_lab.telemetry import schema
from cf_lab.telemetry.camera import CameraRecorder
from cf_lab.telemetry.mocap import TappedExtposeBackend, TappedPoseSource
from cf_lab.telemetry.stm32 import Stm32LogTap
from cf_lab.telemetry.toc import CRAZYFLIE_TOC_FILE, save_crazyflie_toc
from cf_lab.telemetry.writer import RunWriter

# The time step attach_mocap's pump asks of its source. Only a source that makes
# up its own poses (the fakes) uses it; a live one hands over each pose as it
# arrives, so every pose is recorded.
_MOCAP_PUMP_DT_S = 0.01


class TelemetryRecorder:
    def __init__(
        self,
        cfg: TelemetryConfig,
        *,
        run_root: str | Path,
        lab_config: LabConfig | None = None,
        example: str | None = None,
        extpose_frame: str = "start_origin_yaw_zeroed",
    ) -> None:
        self._cfg = cfg
        self._run_root = Path(run_root)
        self._lab_config = lab_config
        self._example = example
        self._extpose_frame = extpose_frame
        self._writer: RunWriter | None = None
        self._stm32_tap: Stm32LogTap | None = None
        self._camera: CameraRecorder | None = None
        self._camera_feed = None
        self._mocap_pump: "_MocapRecordPump | None" = None
        self._run_dir: Path | None = None
        self._crazyflie_toc_path: Path | None = None

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    @property
    def run_dir(self) -> Path | None:
        """The run directory, once started; None when disabled."""
        return self._run_dir

    def start(self) -> None:
        if not self._cfg.enabled:
            return
        transport = self._lab_config.mocap.transport if self._lab_config else "unknown"
        specs = schema.build_specs(
            self._cfg,
            mocap_transport=transport,
            extpose_frame=self._extpose_frame,
        )
        self._run_dir = _new_run_dir(self._run_root)
        self._writer = RunWriter(self._run_dir, specs, manifest_extra=self._manifest_extra())
        self._writer.start()
        self._stm32_tap = Stm32LogTap(self._writer, specs)
        if self._cfg.camera:
            self._camera = CameraRecorder(self._writer)
            self._camera.start()

    def wrap_pose_source(self, source):
        """Tap a mocap source for recording; pass-through when not recording."""
        if self._writer is None or not self._cfg.mocap:
            return source
        return TappedPoseSource(source, self._writer)

    def wrap_extpose_backend(self, backend):
        """Tap a backend so the extpose uplink is recorded; pass-through otherwise."""
        if self._writer is None or not self._cfg.extpose:
            return backend
        return TappedExtposeBackend(backend, self._writer)

    def attach_stm32(self, cf) -> None:
        """Archive the firmware TOCs, then start its pushed log streams.

        TOC capture happens even when no STM32 streams were selected: the tables
        describe the exact connected firmware and are useful run configuration
        metadata in their own right. It is read-only and uses the TOCs cflib has
        already downloaded while opening the link.
        """
        self.capture_crazyflie_toc(cf)
        if self._stm32_tap is not None:
            self._stm32_tap.start(cf)

    def capture_crazyflie_toc(self, cf) -> Path | None:
        """Archive cflib's downloaded TOCs without starting any log block.

        Runners may call this as soon as the radio link opens, which preserves
        the firmware interface even when a later preflight check refuses the
        flight. :meth:`attach_stm32` calls it too, so a runner that only attaches
        the STM32 streams still gets the TOCs.
        """
        if self._writer is None:
            return None
        if self._crazyflie_toc_path is not None:
            return self._crazyflie_toc_path
        path = save_crazyflie_toc(self._writer.run_dir, cf)
        self._crazyflie_toc_path = path
        self.event(
            "crazyflie_toc_saved",
            file=CRAZYFLIE_TOC_FILE,
        )
        return path

    @property
    def crazyflie_toc_path(self) -> Path | None:
        """Saved TOC sidecar path, once a connected Crazyflie was captured."""
        return self._crazyflie_toc_path

    def detach_stm32(self) -> None:
        if self._stm32_tap is not None:
            self._stm32_tap.stop()

    def attach_camera(self, feed) -> None:
        """Record every frame from a ``CameraFeed``; no-op when not recording.

        The feed may already be running. It stays the single owner of the
        deck connection -- a viewer subscribes to the same feed rather than
        opening a second one, which the deck's one-client server would refuse.
        Use this when the runner owns the feed (e.g. to also show a viewer);
        otherwise ``attach_camera_from_config`` lets the recorder own it.
        """
        if self._camera is not None:
            feed.subscribe(self._camera)
            # The deck describes its build once per connection; a recorder that
            # joins a running feed missed that line, so it is recorded here.
            line = getattr(feed, "deck_config_line", None)
            if line is not None:
                self._camera.offer_console(line)

    def attach_camera_from_config(self, *, connect_timeout_s: float = 8.0, log=print) -> bool:
        """Open and record the AI-deck feed described by ``camera:`` in the config.

        The recorder owns the socket so a runner needs only one line: the deck
        is looked up by ``lab_config.radio.uri`` and the port and transport come
        from ``lab_config.camera``. Returns True once frames are flowing.

        Soft-fail, exactly like mocap: a deck that will not connect is a warning,
        never a reason to ground a flight that never needed images. The camera
        stream is then simply empty, which the empty-stream warning reports.
        """
        if self._camera is None or self._lab_config is None:
            return False
        if not self._lab_config.camera.enabled:
            return False
        from cf_lab.camera import CameraFeed

        try:
            feed = CameraFeed.from_config(
                self._lab_config, connect_timeout_s=connect_timeout_s, log=log
            )
            feed.subscribe(self._camera)
            feed.start()
        except Exception as exc:
            log(f"camera feed not recording ({type(exc).__name__}: {exc}); flight continues")
            return False
        self._camera_feed = feed
        return True

    @property
    def camera_stats(self) -> dict[str, int]:
        """{written, dropped} for the image stream; empty when not recording."""
        if self._camera is None:
            return {}
        return {"written": self._camera.frames_written, "dropped": self._camera.dropped}

    @property
    def camera_feed(self):
        """The recorder-owned live feed, or ``None`` before a successful attach."""
        return self._camera_feed

    def attach_mocap_from_config(self, *, log=print) -> bool:
        """Record raw global mocap poses without a flight loop to drive them.

        Symmetric to ``attach_camera_from_config``: builds the live source from
        ``lab_config.mocap`` and records it with :meth:`attach_mocap`.

        Soft-fail like the camera feed: an absent or unconfigured mocap system
        is a warning, never a reason to abort; the mocap stream is then empty.
        """
        if self._writer is None or not self._cfg.mocap or self._lab_config is None:
            return False
        from cf_lab.mocap import build_live_pose_source

        try:
            source = build_live_pose_source(cfg=self._lab_config)
        except Exception as exc:
            log(f"mocap not recording ({type(exc).__name__}: {exc}); continuing")
            return False
        return self.attach_mocap(source)

    def attach_mocap(self, source) -> bool:
        """Record every raw global pose from ``source`` in the ``mocap`` stream.

        The recorder taps the source and owns a pump thread that drains it, so
        every pose lands in the raw (un-zeroed) stream. That is what lets a viewer
        -- or carrying the drone by hand -- log global position beside the camera
        frames. Returns False when mocap is not being recorded.

        Use this ONLY when the runner does not consume the pose source itself. A
        flight runner drives the source through its extpose streamer (see
        ``wrap_pose_source``); doing both would put two consumers on one source
        and split the stream between them.
        """
        if self._writer is None or not self._cfg.mocap:
            return False
        pump = _MocapRecordPump(self.wrap_pose_source(source), self._writer)
        pump.start()
        self._mocap_pump = pump
        return True

    @property
    def mocap_stats(self) -> dict[str, int]:
        """{recorded} raw poses from the built-in pump; empty when it is unused."""
        if self._mocap_pump is None:
            return {}
        return {"recorded": self._mocap_pump.recorded}

    def record_setpoint(self, x: float, y: float, z: float, yaw_deg: float) -> None:
        if self._writer is not None and self._cfg.setpoints:
            self._writer.append("setpoints", (x, y, z, yaw_deg))

    def event(self, kind: str, message: str = "", **data: Any) -> None:
        if self._writer is not None:
            self._writer.event(kind, message, data)

    def stats(self) -> dict[str, dict[str, int]]:
        """Live per-stream {rows, dropped, queued} counters; empty when disabled."""
        return self._writer.stats() if self._writer is not None else {}

    def close(self, *, status: str = "clean") -> None:
        """Finalize parquet + manifest; idempotent, safe in finally blocks."""
        # Stop everything that still appends before the writer closes: the
        # STM32 log blocks, the camera feed (stops the socket, so no frame
        # arrives after its sink is gone), the mocap pump (stops pulling poses
        # into the tap), then the camera sink. The pump and the camera sink each
        # record a closing event, which has to reach the events stream while the
        # writer is open.
        self.detach_stm32()
        if self._camera_feed is not None:
            self._camera_feed.stop()
            self._camera_feed = None
        if self._mocap_pump is not None:
            self._mocap_pump.stop()  # idempotent; the pump stays for mocap_stats
        if self._camera is not None:
            self._camera.stop()
        if self._writer is not None:
            self._writer.close(status=status)

    def _manifest_extra(self) -> dict[str, Any]:
        config = dataclasses.asdict(self._lab_config) if self._lab_config else None
        if config is not None:
            del config["camera"]["key"]  # the WiFi password stays out of run folders
        return {
            "example": self._example,
            "git_sha": _git(["rev-parse", "HEAD"]),
            "git_dirty": bool(_git(["status", "--porcelain"])),
            "host": {"platform": platform.platform(), "python": platform.python_version()},
            "config": config,
        }


class _MocapRecordPump(threading.Thread):
    """Drive raw mocap recording when the runner has no consumer of its own.

    The mocap tap records a pose only as a side effect of it being *read*, so a
    viewer or a hand-carry position+camera capture -- which, unlike a flight
    loop, never pulls poses -- would otherwise record nothing. This thread does
    only that pulling: it drains the tapped source so every raw global pose
    lands in ``mocap.parquet``, and exits when the source is closed or stopped.
    """

    def __init__(self, tapped_source, writer: RunWriter) -> None:
        super().__init__(daemon=True, name="mocap-record-pump")
        self._source = tapped_source
        self._writer = writer
        self._stop = threading.Event()
        self.recorded = 0
        self.error: Exception | None = None

    def run(self) -> None:
        from cf_lab.mocap.stream import iter_pose_samples

        try:
            for _pose in iter_pose_samples(self._source, start_s=0.0, dt_s=_MOCAP_PUMP_DT_S):
                self.recorded += 1
                if self._stop.is_set():
                    return
        except Exception as exc:  # recording stops; nothing else is affected
            self.error = exc

    def stop(self, *, timeout_s: float = 2.0) -> None:
        """Stop pulling, close the source so a blocked read can unblock, and
        record the outcome as a ``mocap_recorded`` event. Idempotent."""
        if self._stop.is_set():
            return
        self._stop.set()
        close = getattr(self._source, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
        self.join(timeout=timeout_s)
        self._writer.event(
            "mocap_recorded",
            f"{self.recorded} pose(s)",
            {"poses": self.recorded, "error": str(self.error) if self.error else None},
        )


def _new_run_dir(root: Path) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = root / f"run_{stamp}"
    suffix = 2
    while run_dir.exists():
        run_dir = root / f"run_{stamp}_{suffix}"
        suffix += 1
    return run_dir


def _git(args: list[str]) -> str:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except Exception:
        return ""  # not a repo / git missing: manifest just goes without it
