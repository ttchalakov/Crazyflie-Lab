"""Build the pose source and extpose streamer a runner needs from config.

A runner only chooses *which kind* of source to use for this run (its ``--source``
flag) and supplies its own synthetic source for dry runs; everything about the
live mocap link — transport (VRPN vs NatNet), host/IP, ports, rate — is a fixed
lab fact that lives in ``configs/drone.yaml`` (``mocap.*``). These factories read
that config so the per-demo ``run.py`` never repeats the connection wiring.
"""

from __future__ import annotations

from cf_lab.mocap.fake import DeadMocapSource
from cf_lab.mocap.natnet import NatNetPoseSource
from cf_lab.mocap.relative import OriginZeroer
from cf_lab.mocap.rosbridge import vrpn_pose_source
from cf_lab.mocap.stream import ExtposeStreamer

# ``--source`` values every mocap runner accepts.
SOURCE_KINDS = ("fake", "optitrack", "fake-dead")


def build_pose_source(source_kind, *, cfg, fake):
    """Resolve a runner's ``--source`` choice into a pose source.

    - ``"fake"``      -> ``fake`` (the demo's own synthetic source)
    - ``"fake-dead"`` -> a source that never yields (exercises the mocap gate)
    - ``"optitrack"`` -> the live source for ``cfg.mocap.transport`` (``vrpn`` or
      ``natnet``), with all connection parameters taken from ``cfg.mocap``.
    """
    if source_kind == "fake-dead":
        return DeadMocapSource()
    if source_kind == "fake":
        return fake
    if source_kind == "optitrack":
        return build_live_pose_source(cfg=cfg)
    raise ValueError(f"unknown --source {source_kind!r}; expected one of {SOURCE_KINDS}.")


def build_live_pose_source(*, cfg):
    """Build the configured live OptiTrack source (VRPN or NatNet)."""
    mocap = cfg.mocap
    transport = mocap.transport
    if transport == "vrpn":
        vrpn = mocap.vrpn
        return vrpn_pose_source(
            rigid_body=mocap.rigid_body,
            host=vrpn.ros_host,
            port=vrpn.ros_port,
            topic=vrpn.topic,
            rate_hz=mocap.stream_rate_hz,
            orientation_format=vrpn.orientation_format,
        )
    if transport == "natnet":
        natnet = mocap.natnet
        if not natnet.server_ip:
            raise ValueError(
                "mocap.transport is 'natnet' but mocap.natnet.server_ip is unset; "
                "set the Motive PC IP in the config."
            )
        return NatNetPoseSource(
            server_ip=natnet.server_ip,
            rigid_body=mocap.rigid_body,
            multicast_ip=natnet.multicast_ip,
            command_port=natnet.command_port,
            data_port=natnet.data_port,
            local_ip=natnet.local_ip,
            use_multicast=natnet.use_multicast,
        )
    raise ValueError(
        f"mocap.transport must be 'vrpn' or 'natnet', got {transport!r}."
    )


def origin_zeroed_streamer(*, source, source_kind, backend, cfg, on_origin=None):
    """Build the origin-zeroed extpose streamer for ``source``.

    Synthetic sources (``source_kind`` != ``"optitrack"``) are paced to wall
    time; the live source is decimated latest-wins so a fast mocap system cannot
    flood the radio. The uplink is capped at ``cfg.mocap.stream_rate_hz``.
    ``on_origin(position_m, yaw_rad)`` fires once when the zeroing origin locks
    (telemetry records it so the zeroed uplink can be recreated).
    """
    return ExtposeStreamer(
        source=source,
        backend=backend,
        max_rate_hz=cfg.mocap.stream_rate_hz,
        pace=source_kind != "optitrack",
        transform=OriginZeroer(on_origin=on_origin).zero,
    )
