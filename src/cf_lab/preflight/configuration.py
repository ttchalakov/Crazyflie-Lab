"""Read-only validation for a fully constructed lab configuration."""

from __future__ import annotations

import math
from typing import Any

from cf_lab.config import LabConfig
from cf_lab.preflight.core import PreflightResult
from cf_lab.telemetry.schema import build_specs


_MOCAP_TRANSPORTS = ("vrpn", "natnet")
_VRPN_ORIENTATION_FORMATS = ("quaternion", "euler-rad", "euler-deg")
_VALIDATED_SECTIONS = ("radio", "mocap", "camera", "telemetry")


def check_configuration(cfg: LabConfig) -> PreflightResult:
    """Validate static lab settings without opening hardware or changing state.

    ``load_config`` already rejects the values it can check on its own as it
    reads the file. This covers the rest: empty addresses, port ranges, the mocap
    settings and the cross-field constraints. All problems are returned together
    to avoid a fix/rerun cycle for every individual field.
    """
    issues: list[dict[str, str]] = []

    def issue(field: str, problem: str, fix: str) -> None:
        issues.append({"field": field, "problem": problem, "fix": fix})

    _validate_radio(cfg, issue)
    _validate_mocap(cfg, issue)
    _validate_camera(cfg, issue)
    specs = _validate_telemetry(cfg, issue)

    if issues:
        remediation = "Correct every listed field, then rerun preflight: " + "; ".join(
            f"{entry['field']}: {entry['fix']}" for entry in issues
        )
        return PreflightResult(
            check_id="configuration",
            passed=False,
            summary=f"Configuration has {len(issues)} issue{'s' if len(issues) != 1 else ''}",
            details={
                "issue_count": len(issues),
                "issues": tuple(issues),
                "validated_sections": _VALIDATED_SECTIONS,
            },
            remediation=remediation,
        )

    camera_state = (
        f"{cfg.camera.mode}/{cfg.camera.transport}" if cfg.camera.enabled else "disabled"
    )
    return PreflightResult(
        check_id="configuration",
        passed=True,
        summary=(
            "Configuration is internally consistent "
            f"(mocap={cfg.mocap.transport}, camera={camera_state})"
        ),
        details={
            "validated_sections": _VALIDATED_SECTIONS,
            "mocap_transport": cfg.mocap.transport,
            "camera_enabled": cfg.camera.enabled,
            "camera_mode": cfg.camera.mode if cfg.camera.enabled else None,
            "camera_transport": cfg.camera.transport if cfg.camera.enabled else None,
            "telemetry_enabled": cfg.telemetry.enabled,
            "telemetry_streams": tuple(spec.name for spec in specs),
        },
    )


def _validate_radio(cfg: LabConfig, issue: Any) -> None:
    if not isinstance(cfg.radio.uri, str) or not cfg.radio.uri.strip():
        issue(
            "radio.uri",
            "radio URI is empty",
            "set a nonempty Crazyradio URI such as radio://0/80/2M/E7E7E7E7E7",
        )


def _validate_mocap(cfg: LabConfig, issue: Any) -> None:
    mocap = cfg.mocap
    if mocap.transport not in _MOCAP_TRANSPORTS:
        issue(
            "mocap.transport",
            f"unsupported transport {mocap.transport!r}",
            f"choose one of {_MOCAP_TRANSPORTS}",
        )
    if not _finite_positive(mocap.stream_rate_hz):
        issue(
            "mocap.stream_rate_hz",
            f"must be finite and greater than zero, got {mocap.stream_rate_hz!r}",
            "set the desired positive pose uplink rate",
        )

    if mocap.transport == "vrpn":
        if not isinstance(mocap.vrpn.ros_host, str) or not mocap.vrpn.ros_host.strip():
            issue(
                "mocap.vrpn.ros_host",
                "VRPN rosbridge host is empty",
                "set the rosbridge hostname or IP address",
            )
        _validate_port("mocap.vrpn.ros_port", mocap.vrpn.ros_port, issue)
        if mocap.vrpn.orientation_format not in _VRPN_ORIENTATION_FORMATS:
            issue(
                "mocap.vrpn.orientation_format",
                f"unsupported orientation format {mocap.vrpn.orientation_format!r}",
                f"choose one of {_VRPN_ORIENTATION_FORMATS}",
            )

    if mocap.transport == "natnet":
        if not isinstance(mocap.natnet.server_ip, str) or not mocap.natnet.server_ip.strip():
            issue(
                "mocap.natnet.server_ip",
                "NatNet server IP is empty",
                "set the Motive computer's IPv4 address",
            )
        _validate_port("mocap.natnet.command_port", mocap.natnet.command_port, issue)
        _validate_port("mocap.natnet.data_port", mocap.natnet.data_port, issue)


def _validate_camera(cfg: LabConfig, issue: Any) -> None:
    _validate_port("camera.deck_port", cfg.camera.deck_port, issue)


def _validate_telemetry(cfg: LabConfig, issue: Any) -> list[Any]:
    telemetry = cfg.telemetry
    if telemetry.camera and not cfg.camera.enabled:
        issue(
            "telemetry.camera",
            "camera recording is enabled but the camera feed is disabled",
            "set the selected mode's SSID under camera: or disable telemetry.camera",
        )

    try:
        return build_specs(telemetry, mocap_transport=cfg.mocap.transport)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        issue(
            "telemetry.schema",
            str(exc),
            "correct the telemetry rates, custom stream names, periods, variable types, or block sizes",
        )
        return []


def _validate_port(field: str, value: Any, issue: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        issue(field, f"must be an integer from 1 through 65535, got {value!r}", "set a valid port")


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (TypeError, ValueError):
        return False


def _finite_positive(value: Any) -> bool:
    return _finite_number(value) and value > 0
