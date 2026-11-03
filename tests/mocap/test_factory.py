"""Config-driven choice of pose source and the origin-zeroed extpose streamer; the fake sources."""

import socket

import pytest

from cf_lab.config import LabConfig
from cf_lab.mocap.factory import build_live_pose_source, build_pose_source, origin_zeroed_streamer
from cf_lab.mocap.fake import BoxSweepSource, DeadMocapSource
from cf_lab.mocap.stream import first_pose


def _cfg(**mocap):
    return LabConfig.from_mapping({"mocap": mocap})


def test_fake_kinds_never_open_a_live_link():
    fake = BoxSweepSource()
    cfg = _cfg(transport="natnet")  # no server_ip: building the live source would raise
    assert build_pose_source("fake", cfg=cfg, fake=fake) is fake
    assert isinstance(build_pose_source("fake-dead", cfg=cfg, fake=fake), DeadMocapSource)


def test_unknown_source_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown --source 'optitrak'"):
        build_pose_source("optitrak", cfg=_cfg(), fake=BoxSweepSource())


@pytest.mark.parametrize("use_multicast", [False, True])
def test_natnet_transport_connects_with_every_configured_setting(fake_motive, wire, kit, use_multicast):
    net = fake_motive(
        server=(kit.server_ip, 1610),
        server_info=wire.server_info(multicast=use_multicast),
        modeldefs=[wire.modeldef([("rb", "cf21bl_01", 3), ("rb", "cf21bl_07", 8)])],
        frames=[wire.frame(9, [(8, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0), True)])],
    )
    cfg = _cfg(
        transport="natnet",
        rigid_body="cf21bl_07",
        natnet={"server_ip": kit.server_ip, "multicast_ip": "239.1.2.3", "command_port": 1610,
                "data_port": 1611, "local_ip": "10.0.0.9", "use_multicast": use_multicast},
    )
    source = build_pose_source("optitrack", cfg=cfg, fake=None)
    try:
        pose = first_pose(source, timeout_s=3.0)
    finally:
        source.close()
    assert (pose.source_seq, pose.position_m) == (9, (1.0, 2.0, 3.0))
    data = net.data_socket
    membership = data.options.get((socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP))
    if use_multicast:
        assert data.bound == ("", 1611)
        assert membership == socket.inet_aton("239.1.2.3") + socket.inet_aton("10.0.0.9")
    else:
        assert data.bound == ("10.0.0.9", 1611) and membership is None


@pytest.mark.parametrize(
    "mocap, message",
    [
        ({"transport": "natnet"}, "server_ip is unset"),
        ({"transport": "optitrack"}, "must be 'vrpn' or 'natnet'"),
    ],
)
def test_live_source_rejects_incomplete_or_unknown_transport(mocap, message):
    with pytest.raises(ValueError, match=message):
        build_live_pose_source(cfg=_cfg(**mocap))


@pytest.mark.parametrize(
    "vrpn, topic",
    [
        ({}, "/vrpn_client_node/cf21bl_07/pose"),
        ({"topic": "/mocap/cf/pose"}, "/mocap/cf/pose"),
    ],
)
def test_vrpn_transport_subscribes_throttled_to_the_stream_rate(fake_roslibpy, vrpn, topic):
    cfg = _cfg(transport="vrpn", rigid_body="cf21bl_07", stream_rate_hz=50.0,
               vrpn={"ros_host": "10.0.0.3", "ros_port": 9095, **vrpn})
    source = build_live_pose_source(cfg=cfg)
    source.connect()
    try:
        (ros,), (subscription,) = fake_roslibpy.ros, fake_roslibpy.topics
        assert (ros.host, ros.port) == ("10.0.0.3", 9095)
        assert (subscription.name, subscription.throttle_rate) == (topic, 20)
        assert subscription.message_type == "geometry_msgs/PoseStamped"
    finally:
        source.close()


def test_dry_run_streamer_paces_and_zeroes_the_fake_source(kit):
    origins = []
    backend = kit.Backend(stop_after=3)
    cfg = _cfg(stream_rate_hz=200.0)
    fake = BoxSweepSource(start_position_m=(0.9, -0.6, 0.1), z_peak_m=0.5, period_s=6.0)
    streamer = origin_zeroed_streamer(
        source=fake, source_kind="fake", backend=backend, cfg=cfg, on_origin=lambda p, y: origins.append((p, y))
    )
    backend.streamer = streamer
    streamer.start()
    streamer.join(2.0)
    assert not streamer.is_alive() and streamer.error is None
    assert origins == [((0.9, -0.6, 0.1), 0.0)]
    assert [p.position_m for p in backend.poses] == [
        pytest.approx((0.0, 0.0, fake.height_at(k * 0.005)), abs=1e-12) for k in range(3)
    ]
    assert streamer.dropped == 0


def test_live_streamer_decimates_to_the_configured_rate(kit, fake_clock):
    arrivals = [k * 0.001 for k in range(100)]
    source = kit.Source([(t, kit.pose((5.0, 5.0, t), seq=i)) for i, t in enumerate(arrivals)], fake_clock)
    backend = kit.Backend()
    streamer = origin_zeroed_streamer(source=source, source_kind="optitrack", backend=backend,
                                      cfg=_cfg(stream_rate_hz=50.0))
    streamer.run()
    assert 4 <= streamer.sent <= 5 and streamer.sent + streamer.dropped == 100
    assert backend.poses[0].position_m == pytest.approx((0.0, 0.0, 0.0))


@pytest.mark.parametrize("t, z", [(0.0, 0.0), (1.5, 0.25), (3.0, 0.5), (4.5, 0.25), (6.0, 0.0)])
def test_box_sweep_rises_to_its_peak_and_back_once_per_period(t, z):
    source = BoxSweepSource(start_position_m=(0.9, -0.6, 0.0), z_peak_m=0.5, period_s=6.0)
    samples = source.samples(start_s=t, dt_s=0.01)
    pose = next(samples)
    assert pose.timestamp_s == t
    assert pose.position_m == pytest.approx((0.9, -0.6, z), abs=1e-12)
    assert next(samples).timestamp_s == pytest.approx(t + 0.01)
