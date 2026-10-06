"""Composable, read-only preflight checks and reports."""

from cf_lab.preflight.camera import (
    check_camera_mocap_coexistence,
    check_camera_stream,
)
from cf_lab.preflight.configuration import check_configuration
from cf_lab.preflight.core import Check, PreflightReport, PreflightResult, run_preflight
from cf_lab.preflight.crazyflie import (
    check_battery,
    check_expected_parameters,
    check_firmware_self_test,
    check_required_decks,
    check_static_imu,
    check_supervisor_ready,
)
from cf_lab.preflight.mocap import check_mocap_quality
from cf_lab.preflight.presets import standard_flight_checks

__all__ = [
    "Check",
    "PreflightReport",
    "PreflightResult",
    "check_battery",
    "check_camera_mocap_coexistence",
    "check_camera_stream",
    "check_configuration",
    "check_expected_parameters",
    "check_firmware_self_test",
    "check_mocap_quality",
    "check_required_decks",
    "check_static_imu",
    "check_supervisor_ready",
    "run_preflight",
    "standard_flight_checks",
]
