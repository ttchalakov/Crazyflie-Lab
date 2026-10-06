#!/usr/bin/env python3
"""Record configured telemetry without arming or sending control commands."""
from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path

from cf_lab.backends.crazyflie import CrazyflieRadioBackend
from cf_lab.config import load_config
from cf_lab.runner import make_run_logger
from cf_lab.telemetry import TelemetryRecorder

EXAMPLE_DIR = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(EXAMPLE_DIR / "config.yaml"))
    parser.add_argument("--duration-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("--duration-s must be positive and finite")
    cfg = load_config(args.config)
    if not cfg.telemetry.enabled:
        print("Telemetry is disabled in the config; nothing to record.")
        return 0

    recorder = TelemetryRecorder(
        cfg.telemetry, run_root=EXAMPLE_DIR / "artifacts",
        lab_config=cfg, example="telemetry_demo",
    )
    recorder.start()
    log, console_sink, close_log = make_run_logger(recorder.run_dir, echo_console=False)
    status = "clean"
    try:
        with CrazyflieRadioBackend(cfg.radio.uri, console_sink=console_sink) as backend:
            lost = threading.Event()
            backend.crazyflie.connection_lost.add_callback(lambda *_: lost.set())
            try:
                recorder.attach_stm32(backend.crazyflie)
                recorder.attach_camera_from_config(log=log)
                recorder.attach_mocap_from_config(log=log)
                log(f"Recording for {args.duration_s:g}s -> {recorder.run_dir}")
                deadline = time.monotonic() + args.duration_s
                while (remaining := deadline - time.monotonic()) > 0:
                    if lost.wait(remaining):
                        raise RuntimeError("Radio link lost")
            finally:
                recorder.detach_stm32()
    except KeyboardInterrupt:
        log("Stopped; saving recording.")
    except Exception:
        status = "error"
        raise
    finally:
        try:
            recorder.close(status=status)
        finally:
            close_log()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
