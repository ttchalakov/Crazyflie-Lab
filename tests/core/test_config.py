"""cf_lab.config: YAML loading, unknown-key reporting, radio URI precedence and camera credentials."""
from operator import attrgetter
from pathlib import Path

import pytest

from cf_lab.config import (
    LabConfig, NatNetConfig, RadioConfig, VrpnConfig, load_camera_config, load_config, radio_uri,
)

REPO = Path(__file__).resolve().parents[2]


def test_load_config_reads_every_section_of_the_lab_file(lab_yaml, capsys):
    cfg = load_config(lab_yaml)

    assert cfg.radio.uri == "radio://0/80/2M/E7E7E7E7E7"
    assert cfg.mocap.transport == "natnet"
    assert cfg.mocap.natnet.server_ip == "192.0.2.10"
    assert cfg.mocap.natnet.use_multicast is False
    assert cfg.mocap.vrpn.topic == ""
    assert cfg.camera.mode == "station"
    assert (cfg.camera.ssid, cfg.camera.key) == ("lab-2g4-dummy", "station-key-dummy")
    assert cfg.camera.channel == 6
    assert cfg.telemetry.enabled and cfg.telemetry.state_hz == 100.0
    # Streams come back sorted by name, so log blocks are created in a stable order.
    assert [s.name for s in cfg.telemetry.custom] == ["sysid_imu", "sysid_motor_request"]
    imu = cfg.telemetry.custom[0]
    assert imu.period_ms == 20
    assert imu.variables[0] == "acc.x:float" and len(imu.variables) == 6
    assert capsys.readouterr().err == ""


def test_shipped_drone_yaml_loads_without_unknown_keys(capsys):
    load_config(REPO / "configs" / "drone.yaml")

    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("text", ["", "radio:\n", "camera:\n  mode: station\n"])
def test_load_config_fills_missing_values_with_defaults(write_config, text):
    cfg = load_config(write_config(text))

    assert cfg.radio.uri == LabConfig().radio.uri
    assert cfg.mocap == LabConfig().mocap
    assert cfg.mocap.transport == "natnet"
    assert cfg.telemetry == LabConfig().telemetry
    assert not cfg.camera.enabled


@pytest.mark.parametrize(("text", "expected"), [
    ("camera:\n  mode: access_point\n  access_point_ssid: deck\n  access_point_key: password1\n"
     "  access_point_channel: 11\n  deck_port: 5555\n  transport: udp\n",
     {"camera.ssid": "deck", "camera.key": "password1", "camera.channel": 11, "camera.deck_port": 5555,
      "camera.transport": "udp"}),
    ("mocap:\n  vrpn:\n    ros_host: 192.0.2.5\n    ros_port: 9191\n    topic: /pose\n"
     "    orientation_format: euler\n",
     {"mocap.vrpn": VrpnConfig("192.0.2.5", 9191, "/pose", "euler")}),
    ("mocap:\n  natnet:\n    server_ip: 192.0.2.1\n    multicast_ip: 239.0.0.1\n    command_port: 1\n"
     "    data_port: 2\n    local_ip: 192.0.2.2\n    use_multicast: false\n",
     {"mocap.natnet": NatNetConfig("192.0.2.1", "239.0.0.1", 1, 2, "192.0.2.2", False)}),
    ("telemetry:\n  controls_hz: 50\n  setpoints: false\n  mocap: false\n  extpose: false\n  camera: true\n",
     {"telemetry.controls_hz": 50.0, "telemetry.setpoints": False, "telemetry.mocap": False,
      "telemetry.extpose": False, "telemetry.camera": True}),
])
def test_every_known_key_reaches_its_setting(write_config, capsys, text, expected):
    # A key the unknown-key check accepts but from_mapping reads under another name is silently lost.
    cfg = load_config(write_config(text))

    assert {name: attrgetter(name)(cfg) for name in expected} == expected
    assert capsys.readouterr().err == ""


def test_load_config_names_unknown_keys_by_dotted_path(write_config, capsys):
    path = write_config(
        """
        radios: {}
        radio:
          url: radio://0/80/2M/E7E7E7E7E8
        mocap:
          natnet:
            server: 192.0.2.1
        camera:
          ssid: deck
          channel: 3
        telemetry:
          custom:
            imu:
              period: 10
              variables: [acc.x:float]
        """
    )

    cfg = load_config(path)

    err = capsys.readouterr().err
    assert err.startswith(f"{path}: unknown keys ignored: ")
    reported = set(err.split("unknown keys ignored: ")[1].strip().split(", "))
    # camera's file keys are per mode (station_ssid, access_point_channel), not field names.
    assert reported == {
        "radios", "radio.url", "mocap.natnet.server", "camera.ssid", "camera.channel",
        "telemetry.custom.imu.period",
    }
    assert cfg.radio.uri == LabConfig().radio.uri
    assert not cfg.camera.enabled


def test_load_config_silent_suppresses_the_unknown_key_report(write_config, capsys):
    cfg = load_config(write_config("radio:\n  url: x\n"), silent=True)

    assert capsys.readouterr().err == ""
    assert cfg.radio.uri == LabConfig().radio.uri


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("- radio\n- camera\n", ValueError),
        ("radio: radio://0/80/2M/E7E7E7E7E7\n", TypeError),
        ("camera:\n  mode: ap\n", ValueError),
        ("camera:\n  transport: quic\n", ValueError),
        ("camera:\n  access_point_channel: 14\n", ValueError),
        ("camera:\n  access_point_channel: -1\n", ValueError),
        ("telemetry:\n  state_hz: 100.5\n", ValueError),
        ("telemetry:\n  controls_hz: -1\n", ValueError),
        ("telemetry:\n  custom:\n    imu:\n      variables: acc.x:float\n", TypeError),
        ("telemetry:\n  custom:\n    imu:\n      variables: [acc.x]\n", ValueError),
        ("telemetry:\n  custom:\n    imu:\n      variables: ['acc.x:float:x']\n", ValueError),
    ],
)
def test_load_config_rejects_bad_values(write_config, text, error):
    with pytest.raises(error):
        load_config(write_config(text))


@pytest.mark.parametrize(("state_hz", "channel"), [(0, 0), (100, 13), (0.5, 1)])
def test_load_config_accepts_range_limits(write_config, state_hz, channel):
    cfg = load_config(write_config(
        f"telemetry:\n  state_hz: {state_hz}\ncamera:\n  access_point_channel: {channel}\n"
    ))

    assert cfg.telemetry.state_hz == state_hz
    assert cfg.camera.channel == channel


@pytest.mark.parametrize(("key", "ok"), [("", True), ("1234567", False), ("12345678", True)])
def test_access_point_key_must_be_open_or_at_least_8_characters(write_config, key, ok):
    path = write_config(
        f"camera:\n  mode: access_point\n  access_point_ssid: deck\n  access_point_key: '{key}'\n"
    )

    if ok:
        assert load_config(path).camera.key == key
    else:
        with pytest.raises(ValueError, match="access_point_key"):
            load_config(path)


def test_radio_uri_prefers_an_explicit_uri_without_reading_the_config(tmp_path):
    missing = tmp_path / "missing.yaml"

    assert radio_uri(missing, "radio://0/90/2M/E7E7E7E7E9") == "radio://0/90/2M/E7E7E7E7E9"


def test_radio_uri_falls_back_to_the_config(write_config, capsys):
    path = write_config("radio:\n  uri: radio://0/42/1M/E7E7E7E701\n  rate: 2M\n")

    assert radio_uri(path) == "radio://0/42/1M/E7E7E7E701"
    assert "radio.rate" in capsys.readouterr().err
    assert radio_uri(path, None, silent=True) == "radio://0/42/1M/E7E7E7E701"
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("mode", "pair"),
    [("station", ("lab-2g4-dummy", "station-key-dummy")),
     ("access_point", ("deck-ap-dummy", "ap-key-dummy"))],
)
def test_camera_mode_selects_its_credential_pair(lab_yaml, mode, pair):
    camera = load_camera_config(lab_yaml, mode=mode)

    assert camera.mode == mode
    assert (camera.ssid, camera.key) == pair


def test_load_camera_config_without_overrides_matches_load_config(lab_yaml):
    assert load_camera_config(lab_yaml) == load_config(lab_yaml).camera


def test_camera_overrides_apply_to_the_selected_mode(lab_yaml):
    camera = load_camera_config(
        lab_yaml, mode="access_point", ssid="bench-ap", key="", transport="udp", channel=11
    )

    assert (camera.mode, camera.ssid, camera.key) == ("access_point", "bench-ap", "")
    assert (camera.transport, camera.channel) == ("udp", 11)
    # The station pair in the file is untouched by an access-point override.
    assert load_camera_config(lab_yaml, mode="station").ssid == "lab-2g4-dummy"


def test_credential_overrides_follow_the_mode_in_the_file(lab_yaml):
    # The lab file is in station mode: --ssid/--key replace the station pair, not the default AP one.
    camera = load_camera_config(lab_yaml, ssid="other-lab", key="other-key")

    assert (camera.mode, camera.ssid, camera.key) == ("station", "other-lab", "other-key")


def test_load_camera_config_without_a_file_starts_from_defaults():
    camera = load_camera_config(None, ssid="deck", key="password1")

    assert (camera.mode, camera.ssid, camera.key, camera.transport) == (
        "access_point", "deck", "password1", "tcp"
    )


def test_load_camera_config_validates_overrides(lab_yaml):
    with pytest.raises(ValueError):
        load_camera_config(lab_yaml, channel=20)
    with pytest.raises(ValueError):
        load_camera_config(lab_yaml, mode="access_point", key="short")


def test_blank_ssid_disables_the_camera(write_config):
    # drone.yaml: "Leave the selected mode's SSID blank to disable the feed entirely."
    cfg = load_config(write_config("camera:\n  mode: station\n  station_ssid:\n  station_key:\n"))

    assert not cfg.camera.enabled
    assert cfg.camera.key == ""


def test_blank_radio_uri_means_the_default(write_config):
    cfg = load_config(write_config("radio:\n  uri:\n"))

    assert cfg.radio.uri == RadioConfig.uri


@pytest.mark.parametrize("value", ["'false'", "'no'", "0"])
def test_a_flag_that_is_not_true_or_false_is_rejected(write_config, value):
    # A quoted 'false' is a non-empty string: read as truthy, it would turn telemetry on.
    with pytest.raises(ValueError, match="telemetry.enabled"):
        load_config(write_config(f"telemetry:\n  enabled: {value}\n"))
