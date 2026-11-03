from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from functools import partial, reduce
from operator import getitem
from pathlib import Path
from typing import Callable

from cf_lab import firmware
from cf_lab.aideck import dory_pipeline
from cf_lab.backends.crazyflie import CrazyflieRadioBackend
from cf_lab.camera import find_deck, viewer
from cf_lab.config import LabConfig, load_config, radio_uri
from cf_lab.crazyflie.imu import (
    DEFAULT_MAX_ACCELERATION_MAGNITUDE_ERROR_G,
    DEFAULT_MAX_ACCELERATION_MOTION_G,
    DEFAULT_MAX_GYRO_MOTION_DPS,
    DEFAULT_MAX_GYRO_RATE_DPS,
)
from cf_lab.mocap.factory import build_live_pose_source
from cf_lab.mocap.stream import ExtposeStreamer
from cf_lab.preflight.camera import (
    check_camera_mocap_coexistence,
    check_camera_stream,
)
from cf_lab.preflight.configuration import check_configuration
from cf_lab.preflight.core import PreflightReport, PreflightResult, run_preflight
from cf_lab.preflight.crazyflie import (
    check_battery,
    check_expected_parameters,
    check_firmware_self_test,
    check_required_decks,
    check_static_imu,
    check_supervisor_ready,
)
from cf_lab.preflight.mocap import (
    DEFAULT_DURATION_S,
    DEFAULT_MAX_FRESHNESS_MS,
    DEFAULT_MAX_ORIENTATION_JUMP_DEG,
    DEFAULT_MAX_POSITION_JUMP_M,
    DEFAULT_MAX_POSITION_SPREAD_M,
    check_mocap_quality,
)
from cf_lab.sdk import SdkError


@dataclass(frozen=True)
class _CheckDescription:
    name: str
    resource: str
    description: str


_CHECKS = (
    _CheckDescription(
        "configuration",
        "local",
        "Validate the loaded YAML and its cross-field constraints.",
    ),
    _CheckDescription(
        "firmware_self_test",
        "radio",
        "Read the firmware's latched boot self-test result.",
    ),
    _CheckDescription(
        "supervisor_ready",
        "radio",
        "Require canArm and reject armed, tumbled, locked, or crashed states.",
    ),
    _CheckDescription(
        "battery",
        "radio",
        "Sample battery voltage, level, and power-management state.",
    ),
    _CheckDescription(
        "static_imu",
        "radio",
        "Observe a stationary, disarmed IMU window.",
    ),
    _CheckDescription(
        "expected_parameters",
        "radio",
        "Compare selected firmware parameters with expected values.",
    ),
    _CheckDescription(
        "required_decks",
        "radio",
        "Require selected deck.<name> parameters to equal one.",
    ),
    _CheckDescription(
        "mocap_quality",
        "mocap",
        "Observe live mocap rate, freshness, identity, and pose continuity.",
    ),
    _CheckDescription(
        "camera_mocap_coexistence",
        "routes",
        "Check whether the configured camera and mocap network paths can coexist.",
    ),
    _CheckDescription(
        "camera_stream",
        "camera",
        "Open the configured camera feed briefly and inspect received frames.",
    ),
)
_CHECK_BY_NAME = {check.name: check for check in _CHECKS}
_RADIO_CHECKS = frozenset(
    check.name for check in _CHECKS if check.resource == "radio"
)


class _DiscardExtposeBackend:
    """Keep the standalone mocap check read-only with respect to the drone."""

    def send_extpose(self, _pose: object) -> None:
        return


class _UnavailableCheck:
    """A selected check whose shared resource could not be opened."""

    def __init__(self, check_id: str, resource: str, error: Exception) -> None:
        self.check_id = check_id
        self._resource = resource
        self._error = error

    def __call__(self) -> PreflightResult:
        return PreflightResult(
            check_id=self.check_id,
            passed=False,
            summary=f"{self._resource.capitalize()} resource is unavailable",
            error=f"{type(self._error).__name__}: {self._error}",
            remediation=f"Restore the {self._resource} resource and rerun this check.",
        )


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _add_preflight_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/drone.yaml")
    parser.add_argument(
        "--checks",
        action="append",
        metavar="NAME[,NAME...]",
        help=(
            "Checks to run, in order. Repeat the option or use commas. "
            "Hyphens and underscores are interchangeable. Defaults to configuration only."
        ),
    )
    parser.add_argument(
        "--list-checks",
        action="store_true",
        help="List the available check names and exit without loading config or hardware.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the structured report as JSON.")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Do not execute later selected checks after the first failure.",
    )

    radio = parser.add_argument_group("Crazyflie radio checks")
    radio.add_argument("--radio-timeout-s", type=_positive_float, default=3.0)
    radio.add_argument("--supervisor-seconds", type=_positive_float, default=0.25)
    radio.add_argument("--battery-seconds", type=_positive_float, default=1.0)
    radio.add_argument("--battery-min-voltage-v", type=_nonnegative_float, default=3.7)
    radio.add_argument("--imu-seconds", type=_positive_float, default=1.5)
    radio.add_argument("--imu-period-ms", type=_positive_int, default=20)
    radio.add_argument("--imu-minimum-samples", type=_positive_int, default=10)
    radio.add_argument(
        "--imu-max-gyro-rate-dps", type=_nonnegative_float, default=DEFAULT_MAX_GYRO_RATE_DPS
    )
    radio.add_argument(
        "--imu-max-gyro-motion-dps", type=_nonnegative_float, default=DEFAULT_MAX_GYRO_MOTION_DPS
    )
    radio.add_argument(
        "--imu-max-acceleration-motion-g",
        type=_nonnegative_float,
        default=DEFAULT_MAX_ACCELERATION_MOTION_G,
    )
    radio.add_argument(
        "--imu-max-acceleration-error-g",
        type=_nonnegative_float,
        default=DEFAULT_MAX_ACCELERATION_MAGNITUDE_ERROR_G,
    )
    radio.add_argument(
        "--expect-parameter",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Expected firmware parameter; VALUE is parsed as a YAML scalar.",
    )
    radio.add_argument(
        "--numeric-tolerance",
        type=_nonnegative_float,
        default=0.0,
        help="Default absolute tolerance for numeric expected parameters.",
    )
    radio.add_argument(
        "--parameter-tolerance",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Per-parameter absolute numeric tolerance.",
    )
    radio.add_argument(
        "--require-deck",
        action="append",
        default=[],
        metavar="NAME",
        help="Required deck name, with or without the deck. prefix. Repeat as needed.",
    )

    mocap = parser.add_argument_group("mocap quality check")
    mocap.add_argument("--mocap-duration-s", type=_positive_float, default=DEFAULT_DURATION_S)
    mocap.add_argument("--mocap-minimum-rate-hz", type=_nonnegative_float)
    mocap.add_argument(
        "--mocap-max-freshness-ms", type=_nonnegative_float, default=DEFAULT_MAX_FRESHNESS_MS
    )
    mocap.add_argument("--mocap-minimum-samples", type=_positive_int, default=2)
    mocap.add_argument(
        "--mocap-max-position-spread-m",
        type=_nonnegative_float,
        default=DEFAULT_MAX_POSITION_SPREAD_M,
    )
    mocap.add_argument(
        "--mocap-max-position-jump-m",
        type=_nonnegative_float,
        default=DEFAULT_MAX_POSITION_JUMP_M,
    )
    mocap.add_argument(
        "--mocap-max-orientation-jump-deg",
        type=_nonnegative_float,
        default=DEFAULT_MAX_ORIENTATION_JUMP_DEG,
    )

    camera = parser.add_argument_group("camera stream check")
    camera.add_argument("--camera-timeout-s", type=_positive_float, default=8.0)
    camera.add_argument("--camera-sample-frames", type=_positive_int, default=8)
    camera.add_argument("--camera-minimum-fps", type=_nonnegative_float)
    camera.add_argument("--camera-width", type=_positive_int)
    camera.add_argument("--camera-height", type=_positive_int)
    camera.add_argument("--camera-format", choices=("raw", "jpeg"))
    camera.add_argument("--camera-require-extension", action="store_true")
    camera.add_argument("--camera-max-new-drops", type=_nonnegative_int)
    camera.add_argument("--camera-analyze-pixels", action="store_true")
    camera.add_argument("--camera-minimum-mean", type=_nonnegative_float)
    camera.add_argument("--camera-maximum-mean", type=_nonnegative_float)
    camera.add_argument("--camera-maximum-dark-fraction", type=_nonnegative_float)
    camera.add_argument("--camera-maximum-clipped-fraction", type=_nonnegative_float)


def _build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(prog="cf-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight",
        help="Run selected, read-only preflight checks.",
    )
    _add_preflight_arguments(preflight)

    config_value = subparsers.add_parser("config-value", help="Print one dotted config value.")
    config_value.add_argument("key", help="Dotted key, for example radio.uri.")
    config_value.add_argument("--config", default="configs/drone.yaml")

    firmware.add_commands(subparsers)

    compile_model = subparsers.add_parser(
        "compile-model",
        help="Compile a PyTorch model into a GAP8 app: NEMO quantizes it, DORY generates the C.",
    )
    dory_pipeline.add_arguments(compile_model)
    compile_model.set_defaults(func=dory_pipeline.run)

    view = subparsers.add_parser("view", help="Show the AI-deck camera stream in a window.")
    viewer.add_arguments(view)
    view.set_defaults(func=viewer.run)

    find = subparsers.add_parser("find-deck", help="Find the drone's AI-deck on the network.")
    find_deck.add_arguments(find)
    find.set_defaults(func=find_deck.run)

    reboot = subparsers.add_parser(
        "reboot", help="Power-cycle the drone over the radio and wait until it is healthy."
    )
    reboot.add_argument("--settle-s", type=float, default=6.0, help="gyro settle time after boot")
    shutdown = subparsers.add_parser(
        "shutdown", help="Power the drone fully off; only its power button brings it back."
    )
    for command, func in ((reboot, _reboot), (shutdown, _shutdown)):
        command.add_argument("--config", type=Path, default=firmware.DEFAULT_CONFIG)
        command.add_argument("--uri", help="overrides radio.uri from --config")
        command.set_defaults(func=func)
    return parser, preflight


def _reboot(args: argparse.Namespace) -> int:
    from cf_lab.backends.crazyflie import power_cycle_until_healthy

    uri = radio_uri(args.config, args.uri)
    print(f"Power-cycling the drone over {uri}. Keep it stationary.")
    if not power_cycle_until_healthy(uri, settle_s=args.settle_s, log=print):
        return 1
    print("Reboot complete. Reconnect AI-deck Wi-Fi before starting the flight runner.")
    return 0


def _shutdown(args: argparse.Namespace) -> int:
    from cf_lab.backends.crazyflie import power_off

    uri = radio_uri(args.config, args.uri)
    print(f"Shutting the drone down over {uri}.")
    if not power_off(uri, log=print):
        return 1
    print("Drone is off. Press the power button to bring it back.")
    return 0


def _requested_checks(values: list[str] | None, parser: argparse.ArgumentParser) -> tuple[str, ...]:
    if not values:
        return ("configuration",)

    names: list[str] = []
    for value in values:
        for raw_name in value.split(","):
            name = raw_name.strip().lower().replace("-", "_")
            if not name:
                parser.error("--checks contains an empty check name")
            if name not in _CHECK_BY_NAME:
                parser.error(
                    f"unknown preflight check {raw_name.strip()!r}; "
                    "use --list-checks to see the available names"
                )
            if name not in names:
                names.append(name)
    return tuple(names)


def _list_checks() -> str:
    width = max(len(check.name) for check in _CHECKS)
    return "\n".join(
        f"{check.name:<{width}}  [{check.resource}] {check.description}" for check in _CHECKS
    )


def _yaml_scalar(value: str, parser: argparse.ArgumentParser, option: str) -> object:
    # Numbers first: YAML 1.1 reads 5e-1 as a string and 010 as octal.
    for number in (int, float):
        try:
            parsed = number(value)
            break
        except ValueError:
            pass
    else:
        try:
            import yaml

            parsed = yaml.safe_load(value)
        except Exception as exc:
            parser.error(f"{option} has an invalid VALUE {value!r}: {exc}")
    if not isinstance(parsed, (str, int, float, bool)):
        parser.error(f"{option} VALUE must be a string, number, or boolean")
    if isinstance(parsed, float) and not math.isfinite(parsed):
        parser.error(f"{option} VALUE must be finite")
    return parsed


def _name_value_options(
    values: list[str],
    parser: argparse.ArgumentParser,
    option: str,
    *,
    numeric: bool = False,
) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for item in values:
        name, separator, raw_value = item.partition("=")
        name = name.strip()
        if not separator or not name or not raw_value.strip():
            parser.error(f"{option} expects NAME=VALUE, got {item!r}")
        if name in parsed:
            parser.error(f"{option} repeats {name!r}")
        if numeric:
            try:
                number = float(raw_value)
            except ValueError:
                parser.error(f"{option} VALUE for {name!r} must be numeric")
            if not math.isfinite(number) or number < 0.0:
                parser.error(
                    f"{option} VALUE for {name!r} must be finite and zero or greater"
                )
            parsed[name] = number
        else:
            parsed[name] = _yaml_scalar(raw_value.strip(), parser, option)
    return parsed


def _configuration_load_failure(path: str | Path, error: Exception) -> PreflightReport:
    return PreflightReport(
        (
            PreflightResult(
                check_id="configuration_load",
                passed=False,
                summary=f"Could not load configuration {path}",
                error=f"{type(error).__name__}: {error}",
                remediation="Correct the path or YAML contents, then rerun preflight.",
            ),
        )
    )


def _open_radio(cfg: LabConfig) -> tuple[CrazyflieRadioBackend | None, Exception | None]:
    backend: CrazyflieRadioBackend | None = None
    try:
        backend = CrazyflieRadioBackend(cfg.radio.uri)
        backend.open()
    except Exception as exc:  # resource acquisition is reported fail-closed
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        return None, exc
    return backend, None


def _start_mocap(
    cfg: LabConfig,
) -> tuple[object | None, ExtposeStreamer | None, Exception | None]:
    try:
        source = build_live_pose_source(cfg=cfg)
        # Connect first, so the check's window measures the stream, not the handshake.
        source.connect()
        streamer = ExtposeStreamer(
            source=source,
            backend=_DiscardExtposeBackend(),
            max_rate_hz=cfg.mocap.stream_rate_hz,
            pace=False,
        )
        streamer.start()
    except Exception as exc:  # resource acquisition is reported fail-closed
        close = getattr(locals().get("source"), "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        return None, None, exc
    return source, streamer, None


def _close_mocap(source: object | None, streamer: ExtposeStreamer | None) -> None:
    if streamer is not None:
        streamer.stop()
    close = getattr(source, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass
    if streamer is not None:
        streamer.join(timeout=1.0)


def _build_check(
    name: str,
    *,
    args: argparse.Namespace,
    cfg: LabConfig,
    cf: object | None,
    radio_error: Exception | None,
    streamer: ExtposeStreamer | None,
    mocap_error: Exception | None,
    expected_parameters: dict[str, object],
    parameter_tolerances: dict[str, float],
) -> Callable[[], PreflightResult]:
    if name == "configuration":
        return partial(check_configuration, cfg)
    if name in _RADIO_CHECKS and radio_error is not None:
        return _UnavailableCheck(name, "radio", radio_error)
    if name == "firmware_self_test":
        return partial(check_firmware_self_test, cf, timeout_s=args.radio_timeout_s)
    if name == "supervisor_ready":
        return partial(check_supervisor_ready, cf, seconds=args.supervisor_seconds)
    if name == "battery":
        return partial(
            check_battery,
            cf,
            minimum_voltage_v=args.battery_min_voltage_v,
            seconds=args.battery_seconds,
        )
    if name == "static_imu":
        return partial(
            check_static_imu,
            cf,
            seconds=args.imu_seconds,
            period_ms=args.imu_period_ms,
            minimum_samples=args.imu_minimum_samples,
            max_gyro_rate_dps=args.imu_max_gyro_rate_dps,
            max_gyro_motion_dps=args.imu_max_gyro_motion_dps,
            max_acceleration_motion_g=args.imu_max_acceleration_motion_g,
            max_acceleration_magnitude_error_g=args.imu_max_acceleration_error_g,
        )
    if name == "expected_parameters":
        return partial(
            check_expected_parameters,
            cf,
            expected_parameters,
            numeric_tolerance=args.numeric_tolerance,
            tolerances=parameter_tolerances,
            timeout_s=args.radio_timeout_s,
        )
    if name == "required_decks":
        return partial(
            check_required_decks,
            cf,
            tuple(args.require_deck),
            timeout_s=args.radio_timeout_s,
        )
    if name == "mocap_quality":
        if mocap_error is not None:
            return _UnavailableCheck(name, "mocap", mocap_error)
        return partial(
            check_mocap_quality,
            streamer,
            duration_s=args.mocap_duration_s,
            expected_rate_hz=cfg.mocap.stream_rate_hz,
            minimum_rate_hz=args.mocap_minimum_rate_hz,
            max_freshness_ms=args.mocap_max_freshness_ms,
            minimum_samples=args.mocap_minimum_samples,
            expected_rigid_body=cfg.mocap.rigid_body,
            max_position_spread_m=args.mocap_max_position_spread_m,
            max_position_jump_m=args.mocap_max_position_jump_m,
            max_orientation_jump_deg=args.mocap_max_orientation_jump_deg,
        )
    if name == "camera_mocap_coexistence":
        return partial(check_camera_mocap_coexistence, cfg)
    if name == "camera_stream":
        return partial(
            check_camera_stream,
            cfg,
            timeout_s=args.camera_timeout_s,
            sample_frames=args.camera_sample_frames,
            minimum_fps=args.camera_minimum_fps,
            expected_width=args.camera_width,
            expected_height=args.camera_height,
            expected_format=args.camera_format,
            require_extension=args.camera_require_extension,
            max_new_drops=args.camera_max_new_drops,
            analyze_pixels=args.camera_analyze_pixels,
            minimum_mean=args.camera_minimum_mean,
            maximum_mean=args.camera_maximum_mean,
            maximum_dark_fraction=args.camera_maximum_dark_fraction,
            maximum_clipped_fraction=args.camera_maximum_clipped_fraction,
        )
    raise AssertionError(f"unhandled preflight check {name!r}")


def _emit_report(report: PreflightReport, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(report.format())


def _run_preflight_command(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    if args.list_checks:
        print(_list_checks())
        return 0

    selected = _requested_checks(args.checks, parser)
    expected_parameters = _name_value_options(
        args.expect_parameter,
        parser,
        "--expect-parameter",
    )
    parameter_tolerances_raw = _name_value_options(
        args.parameter_tolerance,
        parser,
        "--parameter-tolerance",
        numeric=True,
    )
    parameter_tolerances = {
        name: float(value) for name, value in parameter_tolerances_raw.items()
    }
    if "expected_parameters" in selected and not expected_parameters:
        parser.error("expected_parameters requires at least one --expect-parameter NAME=VALUE")
    if "required_decks" in selected and not args.require_deck:
        parser.error("required_decks requires at least one --require-deck NAME")
    unknown_tolerances = set(parameter_tolerances).difference(expected_parameters)
    if unknown_tolerances:
        parser.error(
            "--parameter-tolerance names without --expect-parameter: "
            + ", ".join(sorted(unknown_tolerances))
        )
    nonnumeric_tolerances = [
        name
        for name in parameter_tolerances
        if not isinstance(expected_parameters[name], (int, float, bool))
    ]
    if nonnumeric_tolerances:
        parser.error(
            "--parameter-tolerance is only meaningful for numeric parameters: "
            + ", ".join(sorted(nonnumeric_tolerances))
        )

    try:
        cfg = load_config(args.config)
    except Exception as exc:  # malformed/missing configuration is a failed preflight
        report = _configuration_load_failure(args.config, exc)
        _emit_report(report, json_output=args.json)
        return 1

    radio_backend: CrazyflieRadioBackend | None = None
    radio_error: Exception | None = None
    radio_attempted = False
    source: object | None = None
    streamer: ExtposeStreamer | None = None
    mocap_error: Exception | None = None
    mocap_attempted = False

    try:
        results: list[PreflightResult] = []
        for name in selected:
            # Acquire each shared live resource immediately before its first
            # reached check. Thus --stop-on-failure can reject a bad local
            # configuration without touching the radio or mocap network.
            if name in _RADIO_CHECKS and not radio_attempted:
                radio_attempted = True
                radio_backend, radio_error = _open_radio(cfg)
            if name == "mocap_quality" and not mocap_attempted:
                mocap_attempted = True
                source, streamer, mocap_error = _start_mocap(cfg)

            cf = radio_backend.crazyflie if radio_backend is not None else None
            check = _build_check(
                name,
                args=args,
                cfg=cfg,
                cf=cf,
                radio_error=radio_error,
                streamer=streamer,
                mocap_error=mocap_error,
                expected_parameters=expected_parameters,
                parameter_tolerances=parameter_tolerances,
            )
            result = run_preflight((check,)).results[0]
            results.append(result)
            if args.stop_on_failure and not result.passed:
                break
        report = PreflightReport(tuple(results))
    finally:
        _close_mocap(source, streamer)
        if radio_backend is not None:
            radio_backend.close()

    _emit_report(report, json_output=args.json)
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    # Flash progress and cflib's own prints must stream live even when piped.
    sys.stdout.reconfigure(line_buffering=True)
    parser, preflight_parser = _build_parser()
    args = parser.parse_args(argv)

    if hasattr(args, "func"):
        try:
            return args.func(args)
        except SdkError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except FileNotFoundError as exc:
            if exc.filename != str(getattr(args, "config", None)):
                raise
            print(f"error: {exc.filename} not found; pass --uri or --config", file=sys.stderr)
            return 1

    if args.command == "preflight":
        return _run_preflight_command(args, preflight_parser)

    if args.command == "config-value":
        cfg = load_config(args.config)
        value = reduce(getitem, args.key.split("."), asdict(cfg))
        print(value)
        return 0

    parser.error(f"Unhandled command {args.command!r}.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
