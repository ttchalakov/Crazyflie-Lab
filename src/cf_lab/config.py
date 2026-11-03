from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RadioConfig:
    uri: str = "radio://0/80/2M/E7E7E7E7E7"


@dataclass(frozen=True)
class VrpnConfig:
    """Connection details for the VRPN-via-ROS (rosbridge) mocap transport."""

    ros_host: str = "localhost"
    ros_port: int = 9090
    # Empty = the vrpn_client_ros convention /vrpn_client_node/<rigid_body>/pose.
    topic: str = ""
    orientation_format: str = "quaternion"


@dataclass(frozen=True)
class NatNetConfig:
    """Connection details for the native NatNet (Motive, no ROS) mocap transport."""

    # Motive PC IP; required when mocap.transport is "natnet".
    server_ip: str = ""
    multicast_ip: str = "239.255.42.99"
    command_port: int = 1510
    data_port: int = 1511
    local_ip: str = ""
    # Must match Motive's Transmission Type (Data Streaming pane): Unicast or Multicast.
    use_multicast: bool = False


@dataclass(frozen=True)
class MocapConfig:
    rigid_body: str = "cf21bl_01"
    # Which live transport `--source optitrack` resolves to: "natnet" | "vrpn".
    transport: str = "natnet"
    # Extpose uplink rate cap (and fake-source pacing); a fixed lab/link fact.
    stream_rate_hz: float = 100.0
    # Once armed, how long without a new pose before a runner's kill check raises
    # MocapLost; shorter gaps are left to the drone's estimator.
    pose_timeout_s: float = 1.0
    vrpn: VrpnConfig = field(default_factory=VrpnConfig)
    natnet: NatNetConfig = field(default_factory=NatNetConfig)


@dataclass(frozen=True)
class CustomStreamConfig:
    """One extra STM32 log stream built from firmware TOC variables.

    Each entry in ``variables`` is ``"group.name:type"`` (e.g.
    ``"pm.vbat:float"``) using the cflib log type names. All variables of
    one stream share a single log block, so their sizes must sum to <= 26 bytes
    (checked in telemetry.schema, before flight).
    """

    name: str
    period_ms: int = 100
    variables: tuple[str, ...] = ()


@dataclass(frozen=True)
class CameraConfig:
    """How to reach the AI-deck's image stream.

    Every field here must match how the GAP8 app was *compiled* -- the deck is
    told its SSID/mode/transport by ``lib/gap8/cpx_wifi`` at boot, from values
    baked in at build time. ``cf-lab build gap8`` and ``cf-lab flash gap8`` pass
    this section to the app as ``WIFI_DEFINES``, but nothing at runtime
    reconciles the two: an app built before a change here keeps the old
    settings, and the host simply never sees a frame.

    ``mode`` is the one that matters for a mocap flight:

    ``access_point``
        The deck *is* a WiFi network (SSID ``ssid``, deck at 192.168.4.1). The
        host must join it, which on a single-NIC machine takes it off the
        network carrying mocap. Fine with a second NIC.
    ``station``
        The deck joins an existing network, so the host stays where it is and
        mocap and images coexist on one NIC.

    Either way the host finds the deck by the mDNS name its GAP8 build gave it,
    from ``radio.uri`` (see ``cf_lab.camera.find_deck``), not by an address here.
    """

    # Blank disables the feed entirely, whatever telemetry.camera says.
    ssid: str = ""
    key: str = ""
    mode: str = "access_point"  # "access_point" | "station"
    deck_port: int = 5000
    transport: str = "tcp"  # "tcp" | "udp"
    # The deck's access-point channel, 1..13; 0 keeps the ESP32 firmware's (6).
    channel: int = 0

    @property
    def enabled(self) -> bool:
        return bool(self.ssid)


@dataclass(frozen=True)
class TelemetryConfig:
    """Per-run telemetry stream toggles and rates.

    STM32 streams ride the stock CRTP log framework (10 ms period granularity,
    so rates cap at 100 Hz). ``state_hz`` drives the controller-view 12D state
    as two log blocks: EKF position+velocity, then EKF attitude+filtered gyro
    body rates. The rates are pre-EKF sensor signals used by the rate PID.
    """

    enabled: bool = False
    state_hz: float = 10.0  # 0 disables the stream
    controls_hz: float = 10.0
    setpoints: bool = True
    mocap: bool = True
    # The origin-zeroed poses actually sent up the radio (recreates the uplink).
    extpose: bool = True
    # Record every frame the deck sends, as received. Needs a `camera:` section
    # to say where the feed is; see cf_lab.telemetry.camera for the run layout.
    camera: bool = False
    custom: tuple[CustomStreamConfig, ...] = ()


@dataclass(frozen=True)
class LabConfig:
    radio: RadioConfig = field(default_factory=RadioConfig)
    mocap: MocapConfig = field(default_factory=MocapConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LabConfig":
        radio = _mapping(data.get("radio"))
        mocap = _mapping(data.get("mocap"))
        telemetry = _mapping(data.get("telemetry"))
        camera = _mapping(data.get("camera"))

        return cls(
            radio=RadioConfig(uri=str(radio.get("uri", RadioConfig.uri))),
            mocap=MocapConfig(
                rigid_body=str(mocap.get("rigid_body", MocapConfig.rigid_body)),
                transport=str(mocap.get("transport", MocapConfig.transport)),
                stream_rate_hz=float(mocap.get("stream_rate_hz", MocapConfig.stream_rate_hz)),
                pose_timeout_s=float(mocap.get("pose_timeout_s", MocapConfig.pose_timeout_s)),
                vrpn=_vrpn_config(_mapping(mocap.get("vrpn"))),
                natnet=_natnet_config(_mapping(mocap.get("natnet"))),
            ),
            telemetry=_telemetry_config(telemetry),
            camera=_camera_config(camera),
        )


def load_config(path: str | Path, *, silent: bool = False) -> LabConfig:
    """The config file at ``path``; ``silent`` skips naming its unknown keys on stderr."""
    data = _read_yaml(path)
    config = LabConfig.from_mapping(data)
    # A misspelt or removed key would otherwise do nothing, silently.
    unknown = _unknown_keys(data)
    if unknown and not silent:
        print(f"{path}: unknown keys ignored: {', '.join(unknown)}", file=sys.stderr)
    return config


def _unknown_keys(data: Mapping[str, Any]) -> list[str]:
    """Dotted paths of the keys in ``data`` that from_mapping does not read.

    A section's keys are its dataclass's fields, but for camera, whose file keys
    differ from its fields. ``data`` has been through from_mapping, so every
    section is a mapping or absent.
    """
    mocap = _mapping(data.get("mocap"))
    telemetry = _mapping(data.get("telemetry"))
    sections = [
        ("", data, _fields(LabConfig)),
        ("radio.", _mapping(data.get("radio")), _fields(RadioConfig)),
        ("mocap.", mocap, _fields(MocapConfig)),
        ("mocap.vrpn.", _mapping(mocap.get("vrpn")), _fields(VrpnConfig)),
        ("mocap.natnet.", _mapping(mocap.get("natnet")), _fields(NatNetConfig)),
        ("camera.", _mapping(data.get("camera")), CAMERA_KEYS),
        ("telemetry.", telemetry, _fields(TelemetryConfig)),
    ]
    # Stream names are the user's own: a stream's name is its key, not a field.
    sections += [
        (f"telemetry.custom.{name}.", _mapping(block), _fields(CustomStreamConfig) - {"name"})
        for name, block in _mapping(telemetry.get("custom")).items()
    ]
    return [
        f"{prefix}{key}" for prefix, keys, known in sections for key in keys if key not in known
    ]


def _fields(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}


def radio_uri(config: str | Path, uri: str | None = None, *, silent: bool = False) -> str:
    """The drone a command talks to: ``uri`` (its --uri) if given, else radio.uri of ``config``."""
    return uri or load_config(config, silent=silent).radio.uri


def load_camera_config(
    path: str | Path | None,
    *,
    mode: str | None = None,
    ssid: str | None = None,
    key: str | None = None,
    transport: str | None = None,
    channel: int | None = None,
) -> CameraConfig:
    """The ``camera:`` section of ``path`` with command-line overrides applied.

    ``mode`` selects which credential pair the file supplies, so overriding it
    alone switches to the other configured network. ``path`` None starts from
    the defaults.
    """
    data: dict[str, Any] = {}
    if path is not None:
        data = dict(_mapping(_read_yaml(path).get("camera")))
    if mode is not None:
        data["mode"] = mode
    selected = str(data.get("mode", CameraConfig.mode))
    if ssid is not None:
        data[f"{selected}_ssid"] = ssid
    if key is not None:
        data[f"{selected}_key"] = key
    if transport is not None:
        data["transport"] = transport
    if channel is not None:
        data["access_point_channel"] = channel
    return _camera_config(data)


def _vrpn_config(data: Mapping[str, Any]) -> VrpnConfig:
    return VrpnConfig(
        ros_host=str(data.get("ros_host", VrpnConfig.ros_host)),
        ros_port=int(data.get("ros_port", VrpnConfig.ros_port)),
        topic=str(data.get("topic", VrpnConfig.topic)),
        orientation_format=str(data.get("orientation_format", VrpnConfig.orientation_format)),
    )


def _natnet_config(data: Mapping[str, Any]) -> NatNetConfig:
    return NatNetConfig(
        server_ip=str(data.get("server_ip", NatNetConfig.server_ip)),
        multicast_ip=str(data.get("multicast_ip", NatNetConfig.multicast_ip)),
        command_port=int(data.get("command_port", NatNetConfig.command_port)),
        data_port=int(data.get("data_port", NatNetConfig.data_port)),
        local_ip=str(data.get("local_ip", NatNetConfig.local_ip)),
        use_multicast=_flag(data, "use_multicast", NatNetConfig.use_multicast, "mocap.natnet"),
    )


def _telemetry_config(data: Mapping[str, Any]) -> TelemetryConfig:
    return TelemetryConfig(
        enabled=_flag(data, "enabled", TelemetryConfig.enabled, "telemetry"),
        state_hz=_stream_rate(data.get("state_hz", TelemetryConfig.state_hz), "telemetry.state_hz"),
        controls_hz=_stream_rate(
            data.get("controls_hz", TelemetryConfig.controls_hz), "telemetry.controls_hz"
        ),
        setpoints=_flag(data, "setpoints", TelemetryConfig.setpoints, "telemetry"),
        mocap=_flag(data, "mocap", TelemetryConfig.mocap, "telemetry"),
        extpose=_flag(data, "extpose", TelemetryConfig.extpose, "telemetry"),
        camera=_flag(data, "camera", TelemetryConfig.camera, "telemetry"),
        custom=_custom_streams(_mapping(data.get("custom"))),
    )


CAMERA_MODES = ("access_point", "station")
CAMERA_TRANSPORTS = ("tcp", "udp")
# The keys _camera_config reads: one credential pair per mode, and the channel
# under its access-point name.
CAMERA_KEYS = {
    "mode", "transport", "deck_port", "access_point_channel",
    *(f"{mode}_{pair}" for mode in CAMERA_MODES for pair in ("ssid", "key")),
}


def _camera_config(data: Mapping[str, Any]) -> CameraConfig:
    mode = str(data.get("mode", CameraConfig.mode))
    transport = str(data.get("transport", CameraConfig.transport))
    if mode not in CAMERA_MODES:
        raise ValueError(f"camera.mode must be one of {CAMERA_MODES}, got {mode!r}.")
    if transport not in CAMERA_TRANSPORTS:
        raise ValueError(
            f"camera.transport must be one of {CAMERA_TRANSPORTS}, got {transport!r}."
        )
    # The mode selects which configured credential pair the deck uses.
    ssid = str(data.get(f"{mode}_ssid", CameraConfig.ssid))
    key = str(data.get(f"{mode}_key", CameraConfig.key))
    # The ESP32 refuses to start a secured AP with a key under 8 characters and
    # falls back to *no AP at all*, which reads downstream as "no frames".
    if mode == "access_point" and key and len(key) < 8:
        raise ValueError(
            f"camera.access_point_key must be empty (open network) or >= 8 "
            f"characters, got {len(key)}."
        )
    return CameraConfig(
        ssid=ssid,
        key=key,
        mode=mode,
        deck_port=int(data.get("deck_port", CameraConfig.deck_port)),
        transport=transport,
        channel=_channel(data.get("access_point_channel", CameraConfig.channel)),
    )


def _channel(value: Any) -> int:
    channel = int(value)
    if not 0 <= channel <= 13:
        raise ValueError(
            f"camera.access_point_channel must be 1..13 (0 for the firmware default), got {channel}."
        )
    return channel


def _stream_rate(value: Any, name: str) -> float:
    rate = float(value)
    # The firmware log task samples on a 10 ms grid, so 100 Hz is the ceiling.
    if not 0.0 <= rate <= 100.0:
        raise ValueError(f"{name} must be between 0 (off) and 100 Hz, got {rate}.")
    return rate


def _custom_streams(data: Mapping[str, Any]) -> tuple[CustomStreamConfig, ...]:
    streams = []
    for name in sorted(data):
        block = _mapping(data[name])
        variables = block.get("variables", ())
        if not isinstance(variables, (list, tuple)):
            raise TypeError(f"telemetry.custom.{name}.variables must be a list.")
        for var in variables:
            if str(var).count(":") != 1:
                raise ValueError(
                    f"telemetry.custom.{name} variable {var!r} must be 'group.name:type'."
                )
        streams.append(
            CustomStreamConfig(
                name=str(name),
                period_ms=int(block.get("period_ms", CustomStreamConfig.period_ms)),
                variables=tuple(str(var) for var in variables),
            )
        )
    return tuple(streams)


def _read_yaml(path: str | Path) -> Mapping[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file {path} must contain a YAML mapping at top level.")
    return data


def _mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping, got {type(value).__name__}.")
    # A blank value (`key:`) is unset, so the key's default applies.
    return {key: item for key, item in value.items() if item is not None}


def _flag(data: Mapping[str, Any], key: str, default: bool, section: str) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{section}.{key} must be true or false (unquoted), got {value!r}.")
    return value
