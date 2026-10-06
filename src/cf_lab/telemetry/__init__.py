"""Per-run flight telemetry: multi-source streams recorded to Parquet.

``TelemetryRecorder`` is the runner-facing entry point; ``load_run`` (in
``cf_lab.telemetry.sync``) is the analysis-side loader with post-hoc clock
alignment.
"""

from cf_lab.telemetry.recorder import TelemetryRecorder
from cf_lab.telemetry.integrity import IntegrityCheck, IntegrityReport, validate_run
from cf_lab.telemetry.sync import ClockFit, RunData, load_run
from cf_lab.telemetry.toc import (
    CRAZYFLIE_TOC_FILE,
    CRAZYFLIE_TOC_FORMAT,
    load_crazyflie_toc,
    save_crazyflie_toc,
    snapshot_crazyflie_toc,
)

__all__ = [
    "ClockFit",
    "CRAZYFLIE_TOC_FILE",
    "CRAZYFLIE_TOC_FORMAT",
    "IntegrityCheck",
    "IntegrityReport",
    "RunData",
    "TelemetryRecorder",
    "load_run",
    "load_crazyflie_toc",
    "save_crazyflie_toc",
    "snapshot_crazyflie_toc",
    "validate_run",
]
