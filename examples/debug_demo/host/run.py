#!/usr/bin/env python3
"""debug_demo — capture the tagged debug/console logging from real hardware.

The paired STM32 controller and GAP8 app each print one line per second. This
host connects over the radio, captures the merged firmware console, and tags every
line by source with a shared `[t=..]` time tag: `[HOST]` (this script), `[STM32]`
(the controller's DEBUG_PRINT), `[GAP8]` (the GAP8 app, forwarded by the STM32).
All tiers can be printed to the terminal and are saved to the log file.
The captured file is printed at the end so you can see all tiers interleaved.

Flash the controller + GAP8 app first (see README), then run this against the drone.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cf_lab.backends.crazyflie import CrazyflieRadioBackend
from cf_lab.config import load_config
from cf_lab.runner import make_run_logger

DEBUG_LOG_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "debug_log"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/drone.yaml")
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Seconds to capture the firmware console",
    )
    args = parser.parse_args(argv)

    log, console_sink, close = make_run_logger(DEBUG_LOG_DIR, echo_console=False)
    try:
        uri = load_config(args.config).radio.uri
        log(f"Connecting to {uri}; capturing firmware console for {args.duration:.0f}s.")
        with CrazyflieRadioBackend(uri, console_sink=console_sink):
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline:
                log("host heartbeat")
                time.sleep(1.0)
        log("Disconnected.")
    finally:
        close()

    path = DEBUG_LOG_DIR / "debug_log.txt"
    print(f"\n--- captured log: {path} ---")
    print(path.read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
