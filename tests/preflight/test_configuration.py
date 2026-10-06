"""check_configuration rejects lab settings that would misroute or silently disable a flight."""
import dataclasses
import math
from pathlib import Path

import pytest

from cf_lab.config import (
    CameraConfig,
    CustomStreamConfig,
    LabConfig,
    MocapConfig,
    NatNetConfig,
    RadioConfig,
    TelemetryConfig,
    VrpnConfig,
    load_config,
)
from cf_lab.preflight.configuration import check_configuration

LAB_YAML = Path(__file__).parent / "data" / "lab_drone.yaml"


@pytest.fixture
def lab_cfg() -> LabConfig:
    return load_config(LAB_YAML, silent=True)


def _with(cfg, **sections):
    return dataclasses.replace(cfg, **sections)


def _fields(result):
    return [issue["field"] for issue in result.details.get("issues", ())]


def test_lab_config_passes(lab_cfg):
    result = check_configuration(lab_cfg)

    assert result.passed, result.details
    assert result.summary == (
        "Configuration is internally consistent (mocap=natnet, camera=station/tcp)"
    )


@pytest.mark.parametrize("uri", ["", "   "])
def test_empty_radio_uri_fails(lab_cfg, uri):
    result = check_configuration(_with(lab_cfg, radio=RadioConfig(uri=uri)))

    assert _fields(result) == ["radio.uri"]


def test_unknown_mocap_transport_fails(lab_cfg):
    cfg = _with(lab_cfg, mocap=dataclasses.replace(lab_cfg.mocap, transport="optitrack"))

    assert _fields(check_configuration(cfg)) == ["mocap.transport"]


@pytest.mark.parametrize("rate", [0.0, -100.0, math.nan, math.inf])
def test_pose_uplink_rate_must_be_finite_and_positive(lab_cfg, rate):
    cfg = _with(lab_cfg, mocap=dataclasses.replace(lab_cfg.mocap, stream_rate_hz=rate))

    assert _fields(check_configuration(cfg)) == ["mocap.stream_rate_hz"]


def test_nan_uplink_rate_that_the_loader_accepts_is_caught(tmp_path):
    # float(".nan") parses; the loader alone would hand the streamer a NaN rate cap.
    path = tmp_path / "drone.yaml"
    path.write_text(LAB_YAML.read_text().replace("stream_rate_hz: 100.0", "stream_rate_hz: .nan"))

    result = check_configuration(load_config(path, silent=True))

    assert not result.passed
    assert _fields(result) == ["mocap.stream_rate_hz"]


def test_natnet_without_a_server_ip_fails(lab_cfg):
    # NatNetConfig's default server_ip is blank: a config that forgets it must not pass.
    cfg = _with(lab_cfg, mocap=dataclasses.replace(lab_cfg.mocap, natnet=NatNetConfig()))

    assert _fields(check_configuration(cfg)) == ["mocap.natnet.server_ip"]


@pytest.mark.parametrize(
    ("ports", "bad"),
    [
        ({"command_port": 0}, ["mocap.natnet.command_port"]),
        ({"data_port": 65536}, ["mocap.natnet.data_port"]),
        ({"command_port": True, "data_port": -1}, ["mocap.natnet.command_port", "mocap.natnet.data_port"]),
    ],
)
def test_natnet_ports_must_be_valid(lab_cfg, ports, bad):
    natnet = dataclasses.replace(lab_cfg.mocap.natnet, **ports)
    cfg = _with(lab_cfg, mocap=dataclasses.replace(lab_cfg.mocap, natnet=natnet))

    assert _fields(check_configuration(cfg)) == bad


def test_vrpn_needs_a_rosbridge_host_and_port():
    cfg = LabConfig(mocap=MocapConfig(transport="vrpn", vrpn=VrpnConfig(ros_host=" ", ros_port=0)))

    assert _fields(check_configuration(cfg)) == ["mocap.vrpn.ros_host", "mocap.vrpn.ros_port"]


def test_vrpn_does_not_require_natnet_settings():
    cfg = LabConfig(mocap=MocapConfig(transport="vrpn", natnet=NatNetConfig(server_ip="", data_port=0)))

    assert check_configuration(cfg).passed


@pytest.mark.parametrize("port", [0, 70000])
def test_camera_deck_port_must_be_valid(lab_cfg, port):
    cfg = _with(lab_cfg, camera=dataclasses.replace(lab_cfg.camera, deck_port=port))

    assert _fields(check_configuration(cfg)) == ["camera.deck_port"]


def test_camera_recording_without_a_camera_feed_fails(lab_cfg):
    cfg = _with(
        lab_cfg,
        camera=CameraConfig(ssid=""),
        telemetry=dataclasses.replace(lab_cfg.telemetry, camera=True),
    )

    assert _fields(check_configuration(cfg)) == ["telemetry.camera"]


@pytest.mark.parametrize(
    "stream",
    [
        CustomStreamConfig(name="sysid_big", variables=("acc.x:float",) * 7),
        CustomStreamConfig(name="sysid_period", period_ms=25, variables=("acc.x:float",)),
        CustomStreamConfig(name="sysid_type", variables=("acc.x:double",)),
    ],
    ids=["over_26_bytes", "off_the_10ms_grid", "unknown_type"],
)
def test_custom_telemetry_stream_that_cannot_be_logged_fails(lab_cfg, stream):
    telemetry = dataclasses.replace(lab_cfg.telemetry, custom=(stream,))

    result = check_configuration(_with(lab_cfg, telemetry=telemetry))

    assert _fields(result) == ["telemetry.schema"]


def test_every_issue_is_reported_in_one_run(lab_cfg):
    cfg = _with(
        lab_cfg,
        radio=RadioConfig(uri=""),
        mocap=dataclasses.replace(lab_cfg.mocap, stream_rate_hz=0.0),
        telemetry=TelemetryConfig(camera=True),
        camera=CameraConfig(ssid=""),
    )

    result = check_configuration(cfg)

    assert not result.passed
    assert result.summary == "Configuration has 3 issues"
    assert _fields(result) == ["radio.uri", "mocap.stream_rate_hz", "telemetry.camera"]
