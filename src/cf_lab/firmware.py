"""``cf-lab build`` and ``cf-lab flash`` for the drone's three processors.

The STM32 controller builds with the host's arm-none-eabi toolchain; the GAP8 and
ESP32 build in Docker. The GAP8 container mounts the cf-lab checkout and the app's
project at their own host paths (see sdk.docker_mounts), so an app can live anywhere.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from cf_lab.camera.find_deck import deck_hostname, deck_name
from cf_lab.config import CameraConfig, load_camera_config, load_config, radio_uri
from cf_lab.sdk import (
    DOCKER_PLATFORM,
    SdkError,
    docker_mounts,
    require_docker,
    run,
    sdk_root,
    submodule_hint,
)

GAP8_IMAGE = "cf-lab/aideck-gap8:latest"
ESP_IDF_IMAGE = "espressif/idf:v5.4.1"
DEFAULT_CONFIG = Path("configs/drone.yaml")
DEFAULT_GAP8_GOALS = ("clean", "build", "image")

# lib/gap8/cpx_wifi's enums, which an app's WIFI_MODE/WIFI_TRANSPORT macros name.
WIFI_MODES = {"station": "CPX_WIFI_MODE_STATION", "access_point": "CPX_WIFI_MODE_ACCESS_POINT"}
WIFI_TRANSPORTS = {"tcp": "CPX_WIFI_TRANSPORT_TCP", "udp": "CPX_WIFI_TRANSPORT_UDP"}


def app_dir(args: argparse.Namespace, kind: str) -> Path:
    """``--app-dir``, else ``examples/<--example>/<kind>`` in the checkout."""
    if args.app_dir is not None:
        path = args.app_dir
    elif args.example is not None:
        path = sdk_root() / "examples" / args.example / kind
    else:
        raise SdkError("pass --example NAME or --app-dir DIR")
    if not (path / "Makefile").is_file():
        raise SdkError(f"no Makefile in {path}")
    return path.resolve()


def build_controller(app: Path, jobs: int | None = None, output=None) -> None:
    """Build an STM32 out-of-tree app; ``output``, an open file, takes the build's output."""
    root = sdk_root()
    firmware = root / "external" / "crazyflie-firmware"
    if not (firmware / "Makefile").is_file():
        raise SdkError(f"crazyflie-firmware is missing; run: {submodule_hint('external/crazyflie-firmware')}")
    print(f"Building STM32 firmware in {app}", flush=True, file=output or sys.stdout)
    redirect = {} if output is None else {"stdout": output, "stderr": subprocess.STDOUT}
    # `python3` in an app's Makefile (and the firmware's own PYTHON) must be this
    # interpreter, the one with cf_lab installed, as if its venv were activated.
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    # oot.mk's `make oldconfig` may stop to ask about new Kconfig options; `yes ""`
    # takes every default, since the values that matter are already in app-config.
    yes = subprocess.Popen(["yes", ""], stdout=subprocess.PIPE)
    try:
        run(
            [
                "make", "-C", str(app), f"CF_LAB={root}", f"CRAZYFLIE_BASE={firmware}",
                f"-j{jobs or os.cpu_count() or 4}", "all",
            ],
            stdin=yes.stdout,
            env=env,
            **redirect,
        )
    finally:
        yes.kill()
        yes.wait()
        yes.stdout.close()


def c_string(text: str) -> str:
    """``text`` as a C string literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def wifi_defines(camera: CameraConfig, uri: str | None = None) -> list[str]:
    """The deck WiFi settings as -D flags for an app's WIFI_* macros.

    ``uri``, the drone's radio URI, also names the deck on the network
    (WIFI_HOSTNAME, WIFI_NAME), which is how the host finds it.
    """
    # 802.11 caps an SSID at 32 bytes and a WPA2 key at 64, and lib/gap8/cpx_wifi
    # and the ESP32 reject longer ones. An access point's 8-character minimum is
    # checked where the config is loaded, with the mode.
    if len(camera.ssid.encode()) > 32:
        raise SdkError(f"WiFi SSID {camera.ssid!r} is longer than 32 bytes")
    if len(camera.key.encode()) > 64:
        raise SdkError("a WiFi key must be at most 64 bytes")
    defines = [
        f"-DWIFI_MODE={WIFI_MODES[camera.mode]}",
        f"-DWIFI_SSID={c_string(camera.ssid)}",
        f"-DWIFI_KEY={c_string(camera.key)}",
        f"-DWIFI_TRANSPORT={WIFI_TRANSPORTS[camera.transport]}",
        f"-DWIFI_CHANNEL={camera.channel}",
    ]
    if uri is not None:
        defines += [
            f"-DWIFI_HOSTNAME={c_string(deck_hostname(uri))}",
            f"-DWIFI_NAME={c_string(deck_name(uri))}",
        ]
    return defines


def make_assignment(name: str, flags: list[str]) -> str:
    """``NAME=flags`` for a make command line that reaches the compiler intact.

    Make pastes the value into each compile command, which a shell then splits,
    so every flag is shell-quoted; ``$`` is doubled because make expands it.
    """
    return f"{name}=" + " ".join(shlex.quote(flag) for flag in flags).replace("$", "$$")


def build_gap8(
    app: Path,
    make_args: str = "",
    build_image: bool = True,
    wifi: CameraConfig | None = None,
    uri: str | None = None,
    output=None,
) -> None:
    """Build a GAP8 app in Docker.

    ``make_args`` holds make goals and ``VAR=VALUE`` settings; the goals default to
    ``clean build image``. ``wifi`` is passed as ``WIFI_DEFINES``, the -D flags an
    app's Makefile adds to use the drone config's WiFi settings, with the deck
    named after ``uri`` when given. ``output``, an open file, takes the build's
    output instead of the terminal.
    """
    root = sdk_root()
    require_docker()
    if build_image:
        run(["docker", "build", "--platform", DOCKER_PLATFORM, "-t", GAP8_IMAGE, str(root / "docker" / "aideck")])
    tokens = shlex.split(make_args)
    variables = [token for token in tokens if "=" in token]
    goals = [token for token in tokens if "=" not in token] or list(DEFAULT_GAP8_GOALS)
    if wifi is not None:
        variables.append(make_assignment("WIFI_DEFINES", wifi_defines(wifi, uri)))
    # `make clean` cannot be trusted: a stale BUILD/*.d naming a header that no longer
    # exists stops make before clean runs, and BUILD/ is root-owned (the container runs
    # as root). So "clean" becomes an rm -rf inside the container.
    clean = "rm -rf BUILD && " if "clean" in goals else ""
    goals = [goal for goal in goals if goal != "clean"]
    command = ["make", f"CF_LAB={root}", *variables, *goals]
    print(f"Building GAP8 app in {app}", flush=True, file=output or sys.stdout)
    redirect = {} if output is None else {"stdout": output, "stderr": subprocess.STDOUT}
    run(
        [
            "docker", "run", "--rm", "--platform", DOCKER_PLATFORM,
            *docker_mounts(root, app.parent), "-w", str(app), GAP8_IMAGE,
            "/bin/bash", "-lc",
            f"{clean}source /gap_sdk/configs/ai_deck.sh && {shlex.join(command)}",
        ],
        **redirect,
    )


def gap8_wifi(args: argparse.Namespace) -> tuple[CameraConfig | None, str | None]:
    """The deck WiFi for a GAP8 build, and the radio URI the deck is named after.

    The WiFi is the camera: section of --config, then the flags; the URI is --uri
    when flashing, else radio.uri of --config, else None, which leaves the deck the
    ESP32's own name. (None, None) when there is nothing to pass (no config file and
    no SSID given), which leaves the app its own defaults.
    """
    overrides = {"mode": args.wifi_mode, "ssid": args.ssid, "key": args.key,
                 "transport": args.transport, "channel": args.channel}
    if not args.config.is_file() and args.ssid is None:
        print(f"WiFi: {args.config} not found; the app keeps its own WiFi defaults", flush=True)
        return None, None
    config = args.config if args.config.is_file() else None
    # The whole file, so a misspelt key (staton_ssid) is named instead of the
    # deck quietly keeping its defaults.
    config_uri = load_config(config).radio.uri if config is not None else None
    camera = load_camera_config(config, **overrides)
    if not camera.ssid:
        print("WiFi: no SSID configured; the app keeps its own WiFi defaults", flush=True)
        return None, None
    uri = getattr(args, "uri", None) or config_uri
    mode = "access point" if camera.mode == "access_point" else "station"
    channel = f", channel {camera.channel}" if camera.mode == "access_point" and camera.channel else ""
    name = deck_hostname(uri) if uri else "aideck-XXXXXX (the ESP32's default)"
    print(f"WiFi: {mode} {camera.ssid!r}, {camera.transport.upper()}{channel}, "
          f"deck named {name}", flush=True)
    return camera, uri


def esp_firmware_dir() -> Path:
    firmware = sdk_root() / "external" / "aideck-esp-firmware"
    if not (firmware / "CMakeLists.txt").is_file():
        raise SdkError(f"aideck-esp-firmware is missing; run: {submodule_hint('external/aideck-esp-firmware')}")
    return firmware


def build_esp() -> None:
    firmware = esp_firmware_dir()
    require_docker()
    print(f"Building ESP32 firmware in {firmware}", flush=True)
    # Mounted at /project, not its host path: the IDF build caches its source path,
    # and existing build/ directories were configured under /project.
    run(
        [
            "docker", "run", "--rm", "--platform", DOCKER_PLATFORM,
            "-v", f"{firmware}:/project", "-w", "/project", ESP_IDF_IMAGE, "idf.py", "build",
        ]
    )


def controller_artifact(app: Path) -> Path:
    outputs = list(app.glob("build/*.bin"))
    if not outputs:
        raise SdkError(f"no firmware under {app}/build; build first or pass --artifact")
    return max(outputs, key=lambda path: path.stat().st_mtime)


def gap8_artifact(app: Path) -> Path:
    outputs = list(app.glob("BUILD/GAP8_V2/*/target.board.devices.flash.img"))
    if not outputs:
        raise SdkError(f"no GAP8 flash image under {app}/BUILD; build first or pass --artifact")
    return max(outputs, key=lambda path: path.stat().st_mtime)


def _build_controller(args: argparse.Namespace) -> int:
    build_controller(app_dir(args, "controller"), args.jobs)
    return 0


def _build_gap8(args: argparse.Namespace) -> int:
    wifi, uri = gap8_wifi(args)
    build_gap8(app_dir(args, "gap8"), args.make_args, build_image=not args.no_image_build,
               wifi=wifi, uri=uri)
    return 0


def _build_esp(args: argparse.Namespace) -> int:
    build_esp()
    return 0


def _flash(flash, *, target: str, artifact: Path, uri: str, done: str, **options) -> int:
    """Flash under a progress bar, print the outcome, and return the exit code:
    0 done, 1 error, 2 written but the deck is not running it, 130 interrupted."""
    from cf_lab.aideck.flash import DeckBootFailed, DeckFlashCorrupt
    from cf_lab.progress import FlashProgress

    print(f"Flashing {artifact} to {target} at {uri}")
    progress = FlashProgress()
    try:
        flash(artifact=artifact, uri=uri, progress_cb=progress, **options)
        status, message = 0, done
    except (DeckBootFailed, DeckFlashCorrupt) as exc:
        status, message = 2, f"error: {exc}"
    except KeyboardInterrupt:
        status, message = 130, "error: interrupted"
    except Exception as exc:
        status, message = 1, f"error: {exc}"
    progress.end()
    print(message, file=sys.stderr if status else sys.stdout)
    return status


def _flash_controller(args: argparse.Namespace) -> int:
    from cf_lab.stm32_flash import flash_stm32

    artifact = args.artifact
    if artifact is None:
        app = app_dir(args, "controller")
        if not args.no_build:
            build_controller(app, args.jobs)
        artifact = controller_artifact(app)
    return _flash(flash_stm32, target="STM32", artifact=artifact,
                  uri=radio_uri(args.config, args.uri),
                  done="STM32 flash complete; self-test passed.")


def _flash_gap8(args: argparse.Namespace) -> int:
    from cf_lab.aideck.flash import flash_deck

    artifact = args.artifact
    if artifact is None:
        app = app_dir(args, "gap8")
        if not args.no_build:
            wifi, uri = gap8_wifi(args)
            build_gap8(app, args.make_args, build_image=not args.no_image_build,
                       wifi=wifi, uri=uri)
        artifact = gap8_artifact(app)
    return _flash(flash_deck, target="AI-deck gap8", artifact=artifact,
                  uri=radio_uri(args.config, args.uri, silent=True),
                  done="AI-deck gap8 flash complete.", processor="gap8", cold_boot=args.cold_boot)


def _flash_esp(args: argparse.Namespace) -> int:
    from cf_lab.aideck.flash import flash_deck

    artifact = args.artifact
    if artifact is None:
        if not args.no_build:
            build_esp()
        artifact = esp_firmware_dir() / "build" / "aideck_esp.bin"
    return _flash(flash_deck, target="AI-deck esp", artifact=artifact,
                  uri=radio_uri(args.config, args.uri),
                  done="AI-deck esp flash complete.", processor="esp", cold_boot=args.cold_boot)


def add_commands(subparsers: argparse._SubParsersAction) -> None:
    app = argparse.ArgumentParser(add_help=False)
    which = app.add_mutually_exclusive_group()
    which.add_argument("--example", metavar="NAME", help="an example in the cf-lab checkout")
    which.add_argument("--app-dir", type=Path, metavar="DIR", help="the app directory (holds its Makefile)")

    jobs = argparse.ArgumentParser(add_help=False)
    jobs.add_argument("--jobs", type=int, help="parallel make jobs (default: CPU count)")

    configured = argparse.ArgumentParser(add_help=False)
    configured.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    gap8 = argparse.ArgumentParser(add_help=False)
    gap8.add_argument("--make-args", default="",
                      help='make goals and VAR=VALUE settings; goals default to "clean build image"')
    gap8.add_argument("--no-image-build", action="store_true", help="reuse the existing GAP SDK image")
    wifi = gap8.add_argument_group("deck WiFi, passed to the app as WIFI_DEFINES (default: the camera: section of --config)")
    wifi.add_argument("--wifi-mode", choices=("station", "access_point"))
    wifi.add_argument("--ssid")
    wifi.add_argument("--key")
    wifi.add_argument("--transport", choices=("tcp", "udp"))
    wifi.add_argument("--channel", type=int, metavar="1..13",
                      help="access point channel (camera.access_point_channel)")

    flashing = argparse.ArgumentParser(add_help=False, parents=[configured])
    flashing.add_argument("--artifact", type=Path, help="flash this file instead of the build output")
    flashing.add_argument("--no-build", action="store_true", help="flash the existing build output")
    flashing.add_argument("--uri", help="overrides radio.uri from --config")

    deck = argparse.ArgumentParser(add_help=False)
    deck.add_argument("--cold-boot", action="store_true",
                      help="power-cycle the drone first, to recover a wedged deck")

    build = subparsers.add_parser("build", help="Build firmware for one processor.")
    targets = build.add_subparsers(dest="target", required=True)
    targets.add_parser(
        "controller", parents=[app, jobs], help="STM32 out-of-tree controller"
    ).set_defaults(func=_build_controller)
    targets.add_parser("gap8", parents=[app, gap8, configured], help="AI-deck GAP8 app (Docker)").set_defaults(
        func=_build_gap8
    )
    targets.add_parser("esp", help="AI-deck ESP32 firmware (Docker)").set_defaults(func=_build_esp)

    flash = subparsers.add_parser("flash", help="Build (unless --no-build) and flash over the radio.")
    targets = flash.add_subparsers(dest="target", required=True)
    targets.add_parser(
        "controller", parents=[app, jobs, flashing], help="STM32 out-of-tree controller"
    ).set_defaults(func=_flash_controller)
    targets.add_parser(
        "gap8", parents=[app, gap8, flashing, deck], help="AI-deck GAP8 app"
    ).set_defaults(func=_flash_gap8)
    targets.add_parser("esp", parents=[flashing, deck], help="AI-deck ESP32 firmware").set_defaults(
        func=_flash_esp
    )
