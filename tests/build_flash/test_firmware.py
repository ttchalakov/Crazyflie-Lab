"""cf-lab build/flash: the deck WiFi baked into a GAP8 build, and the commands that build it."""
from __future__ import annotations

import argparse
import ast
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cf_lab import firmware
from cf_lab.config import CameraConfig
from cf_lab.sdk import SdkError

URI = "radio://0/90/2M/E7E7E7E701"


def defines(flags: list[str]) -> dict[str, str]:
    return dict(flag.removeprefix("-D").split("=", 1) for flag in flags)


def test_station_wifi_defines_name_the_deck_after_the_radio_uri():
    camera = CameraConfig(ssid="LabNet", key="dummy-key-123", mode="station", transport="udp")

    assert defines(firmware.wifi_defines(camera, URI)) == {
        "WIFI_MODE": "CPX_WIFI_MODE_STATION",
        "WIFI_SSID": '"LabNet"',
        "WIFI_KEY": '"dummy-key-123"',
        "WIFI_TRANSPORT": "CPX_WIFI_TRANSPORT_UDP",
        "WIFI_CHANNEL": "0",
        "WIFI_HOSTNAME": '"cf-90-2m-e7e7e7e701"',
        "WIFI_NAME": '"90/2M/E7E7E7E701"',
    }


def test_access_point_without_uri_keeps_the_apps_deck_name():
    camera = CameraConfig(ssid="deck-ap", key="", mode="access_point", channel=11)

    flags = defines(firmware.wifi_defines(camera))

    assert flags["WIFI_MODE"] == "CPX_WIFI_MODE_ACCESS_POINT"
    assert flags["WIFI_KEY"] == '""'
    assert flags["WIFI_CHANNEL"] == "11"
    assert "WIFI_HOSTNAME" not in flags and "WIFI_NAME" not in flags


@pytest.mark.parametrize("ssid, key, ok", [
    ("s" * 32, "", True),
    ("s" * 33, "", False),
    ("é" * 16, "", True),             # 32 bytes
    ("é" * 17, "", False),            # 17 characters, but 34 bytes
    ("lab", "k" * 5, True),           # a WEP key: only an access point needs 8, checked on load
    ("lab", "k" * 64, True),          # a 64-hex-digit WPA2 key
    ("lab", "k" * 65, False),
])
def test_wifi_defines_enforce_the_deck_ssid_and_key_byte_limits(ssid, key, ok):
    camera = CameraConfig(ssid=ssid, key=key, mode="station")
    if ok:
        firmware.wifi_defines(camera, URI)
    else:
        with pytest.raises(SdkError):
            firmware.wifi_defines(camera, URI)


def test_longest_radio_uri_still_fits_the_decks_name_buffer():
    # lib/gap8/cpx_wifi rejects a hostname or name over CPX_WIFI_NAME_MAX_LENGTH (23).
    flags = defines(firmware.wifi_defines(CameraConfig(ssid="lab"), "radio://0/125/250K/E7E7E7E7E7"))

    assert len(ast.literal_eval(flags["WIFI_HOSTNAME"])) <= 23
    assert len(ast.literal_eval(flags["WIFI_NAME"])) <= 23


@pytest.fixture
def docker(monkeypatch):
    """Records what build_gap8 would run in Docker."""
    commands: list[list[str]] = []
    monkeypatch.setattr(firmware, "require_docker", lambda: None)
    monkeypatch.setattr(firmware, "run", lambda command, **kwargs: commands.append(command))
    return commands


@pytest.fixture
def gap8_app(tmp_path):
    app = tmp_path / "projects" / "my_app" / "gap8"
    app.mkdir(parents=True)
    (app / "Makefile").write_text(
        "build:\n\t@printf '%s\\n' $(WIFI_DEFINES) > flags.txt\n"
        "image:\n\t@printf '%s\\n' '$(EXTRA)' > extra.txt\n"
    )
    return app


@pytest.mark.skipif(shutil.which("make") is None or shutil.which("bash") is None, reason="needs make and bash")
@pytest.mark.parametrize("ssid, key", [
    ('lab "5G" $HOME', "it's $(true) `x`;#"),
    ("a\\b  c", "pa$$word\\n%"),
])
def test_wifi_reaches_the_compiler_intact_through_docker_bash_and_make(
        cf_lab_root, docker, gap8_app, ssid, key):
    camera = CameraConfig(ssid=ssid, key=key, mode="station")

    firmware.build_gap8(gap8_app, build_image=False, wifi=camera, uri=URI)

    (command,) = docker
    script = command[-1].replace("source /gap_sdk/configs/ai_deck.sh && ", "")
    subprocess.run(["bash", "-c", script], cwd=gap8_app, check=True, capture_output=True, timeout=30)
    compiled = (gap8_app / "flags.txt").read_text().splitlines()
    assert compiled == firmware.wifi_defines(camera, URI)
    assert ast.literal_eval(defines(compiled)["WIFI_SSID"]) == ssid
    assert ast.literal_eval(defines(compiled)["WIFI_KEY"]) == key


def test_default_goals_clean_by_removing_build_inside_the_container(cf_lab_root, docker, gap8_app):
    firmware.build_gap8(gap8_app, build_image=True)

    build, run = docker
    assert build[:2] == ["docker", "build"] and build[-1] == str(cf_lab_root / "docker" / "aideck")
    script = run[-1]
    assert script.startswith("rm -rf BUILD && ")
    make = shlex.split(script.split(" && ")[-1])
    assert make == ["make", f"CF_LAB={cf_lab_root}", "build", "image"]
    assert run[run.index("-w") + 1] == str(gap8_app)


def test_make_args_split_into_variables_and_goals(cf_lab_root, docker, gap8_app):
    firmware.build_gap8(gap8_app, "IMG=qvga build 'EXTRA=a b'", build_image=False)

    (run,) = docker
    assert "rm -rf BUILD" not in run[-1]
    make = shlex.split(run[-1].split(" && ")[-1])
    assert make == ["make", f"CF_LAB={cf_lab_root}", "IMG=qvga", "EXTRA=a b", "build"]


def test_gap8_container_mounts_the_checkout_and_the_apps_project(cf_lab_root, docker, gap8_app):
    firmware.build_gap8(gap8_app, build_image=False)

    (run,) = docker
    mounts = [run[i + 1] for i, arg in enumerate(run) if arg == "-v"]
    assert sorted(mounts) == sorted(f"{d}:{d}" for d in (cf_lab_root, gap8_app.parent))


CONFIG = """\
radio:
  uri: {uri}
camera:
  mode: station
  station_ssid: LabNet
  station_key: dummy-key-123
  access_point_ssid: deck-ap
  access_point_key: ""
  access_point_channel: 6
"""


def wifi_args(config: Path, **overrides) -> argparse.Namespace:
    values = dict(config=config, wifi_mode=None, ssid=None, key=None, transport=None, channel=None)
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def drone_yaml(tmp_path):
    path = tmp_path / "drone.yaml"
    path.write_text(CONFIG.format(uri=URI))
    return path


def test_gap8_wifi_takes_the_configured_network_and_names_the_deck_after_radio_uri(drone_yaml):
    camera, uri = firmware.gap8_wifi(wifi_args(drone_yaml))

    assert (camera.mode, camera.ssid, camera.key) == ("station", "LabNet", "dummy-key-123")
    assert uri == URI


def test_gap8_wifi_mode_override_switches_to_the_other_credential_pair(drone_yaml):
    camera, _ = firmware.gap8_wifi(wifi_args(drone_yaml, wifi_mode="access_point"))

    assert (camera.mode, camera.ssid, camera.key, camera.channel) == ("access_point", "deck-ap", "", 6)


def test_gap8_wifi_names_the_deck_after_the_drone_being_flashed(drone_yaml):
    other = "radio://0/20/2M/E7E7E7E7AA"

    _, uri = firmware.gap8_wifi(wifi_args(drone_yaml, uri=other))

    assert uri == other


def test_gap8_wifi_without_config_or_ssid_leaves_the_app_defaults(tmp_path):
    assert firmware.gap8_wifi(wifi_args(tmp_path / "missing.yaml")) == (None, None)


def test_gap8_wifi_with_a_blank_ssid_leaves_the_app_defaults(tmp_path):
    # An empty WIFI_SSID would reach cpx_wifi, which refuses it: no WiFi at all.
    config = tmp_path / "drone.yaml"
    config.write_text(CONFIG.format(uri=URI).replace("station_ssid: LabNet", 'station_ssid: ""'))

    assert firmware.gap8_wifi(wifi_args(config)) == (None, None)


def test_gap8_wifi_names_a_misspelt_camera_key(tmp_path, capsys):
    # Otherwise the only sign is "no SSID configured", and the deck boots on its defaults.
    config = tmp_path / "drone.yaml"
    config.write_text(CONFIG.format(uri=URI).replace("station_ssid", "staton_ssid"))

    assert firmware.gap8_wifi(wifi_args(config)) == (None, None)

    assert "unknown keys ignored: camera.staton_ssid" in capsys.readouterr().err


def test_gap8_wifi_with_ssid_flag_and_no_config_file_keeps_the_esp32s_deck_name(tmp_path):
    camera, uri = firmware.gap8_wifi(wifi_args(tmp_path / "missing.yaml", wifi_mode="station", ssid="LabNet"))

    assert camera.ssid == "LabNet"
    assert uri is None


@pytest.mark.parametrize("finder, layout", [
    (firmware.gap8_artifact, "BUILD/GAP8_V2/{}/target.board.devices.flash.img"),
    (firmware.controller_artifact, "build/{}.bin"),
])
def test_artifact_is_the_most_recently_built_one(tmp_path, finder, layout):
    old, new = (tmp_path / layout.format(name) for name in ("GCC_RISCV_PULPOS", "GCC_RISCV_FREERTOS"))
    for age, path in ((200, old), (100, new)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        os.utime(path, (1_700_000_000 - age, 1_700_000_000 - age))

    assert finder(tmp_path) == new
    with pytest.raises(SdkError, match="build first"):
        finder(tmp_path / "never_built")


def test_controller_build_runs_make_with_this_python_first_on_path(cf_lab_root, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(firmware, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    app = tmp_path / "controller"

    firmware.build_controller(app, jobs=3)

    ((command, kwargs),) = calls
    assert command == ["make", "-C", str(app), f"CF_LAB={cf_lab_root}",
                       f"CRAZYFLIE_BASE={cf_lab_root / 'external' / 'crazyflie-firmware'}", "-j3", "all"]
    assert kwargs["env"]["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)


def test_controller_build_stops_the_yes_feeder_when_make_fails(cf_lab_root, monkeypatch, tmp_path):
    feeders = []
    real_popen = subprocess.Popen

    def popen(*args, **kwargs):
        feeders.append(real_popen(*args, **kwargs))
        return feeders[-1]

    def failing_make(command, **kwargs):
        raise SdkError("command failed (exit 2)")

    monkeypatch.setattr(firmware.subprocess, "Popen", popen)
    monkeypatch.setattr(firmware, "run", failing_make)

    with pytest.raises(SdkError):
        firmware.build_controller(tmp_path / "controller")

    assert feeders[0].poll() is not None
    assert feeders[0].stdout.closed  # a bench that builds again and again leaks no pipe


def test_flash_gap8_hands_the_image_to_the_deck_flasher_and_reports_a_deck_that_did_not_start(
        monkeypatch, tmp_path, capsys):
    from cf_lab import cli
    from cf_lab.aideck import flash

    calls = []

    def flash_deck(**kwargs):
        calls.append(kwargs)
        raise flash.DeckBootFailed("no GAP8 activity")

    monkeypatch.setattr(flash, "flash_deck", flash_deck)
    image = tmp_path / "flash.img"
    image.write_bytes(b"")

    status = cli.main(["flash", "gap8", "--artifact", str(image), "--uri", URI, "--cold-boot"])

    assert status == 2  # written, but the deck is not running it
    ((call),) = calls
    assert (call["artifact"], call["uri"], call["processor"], call["cold_boot"]) == (image, URI, "gap8", True)
    assert "error: no GAP8 activity" in capsys.readouterr().err
