"""NatNet wire parsing, Motive asset lookup and the receive path, against byte-exact packets."""
import math
import socket
import struct
import threading
import time
import types

import pytest

from cf_lab.mocap import natnet
from cf_lab.mocap.natnet import NatNetPoseSource, _rigid_body_descriptions
from cf_lab.mocap.stream import ExtposeStreamer, MocapBodyLost, MocapConnectionLost, first_pose
from cf_lab.telemetry.mocap import TappedPoseSource

TARGET_POS = (0.4445772469, -0.4893169701, 0.0606131069)
TARGET_Q = (0.0100801187, -0.0027049217, 0.7123106766, -0.7017867016)


def _bodies(*, target_tracked=True):
    return [
        (2, (5.0, 5.0, 5.0), (0.0, 0.0, 0.0, 1.0), True),
        (7, TARGET_POS, TARGET_Q, target_tracked),
        (9, (-1.0, -1.0, 0.0), (0.0, 0.0, 1.0, 0.0), True),
    ]


def _assert_target(pose, frame_number):
    assert pose is not None
    assert pose.position_m == pytest.approx(TARGET_POS, abs=1e-6)
    q = pose.orientation
    assert (q.qx, q.qy, q.qz, q.qw) == pytest.approx(TARGET_Q, abs=1e-6)
    assert pose.source_seq == frame_number
    assert pose.rigid_body == "cf21bl_01"


def _payload(packet):
    return packet[4:]


# -- model descriptions --------------------------------------------------------


def test_model_descriptions_list_standalone_rigid_bodies_past_every_asset_kind(wire):
    assets = [("markers", "all"), ("rb", "other", 2), ("force_plate",), ("device",),
              ("skeleton", "walker", 30, [("Hip", 31), ("cf21bl_01", 32)]), ("camera", "Prime13"),
              ("rb", "cf21bl_01", 7)]
    packet = wire.modeldef(assets)
    assert _rigid_body_descriptions(_payload(packet)) == [("other", 2), ("cf21bl_01", 7)]


def test_model_descriptions_reject_every_truncation(wire):
    payload = _payload(wire.modeldef([("markers", "all"), ("rb", "other", 2), ("rb", "cf21bl_01", 7)]))
    for cut in range(len(payload)):
        with pytest.raises(ValueError):
            _rigid_body_descriptions(payload[:cut])


def test_model_descriptions_skip_unknown_asset_records_by_size(wire):
    packet = wire.modeldef([("raw", 6, b"\x01\x02\x03" * 11), ("rb", "cf21bl_01", 7)])
    assert _rigid_body_descriptions(_payload(packet)) == [("cf21bl_01", 7)]


# -- connect: Motive query and asset resolution -------------------------------


def test_multicast_connect_resolves_name_and_yields_first_tracked_target_pose(fake_motive, wire, kit):
    frames = [
        wire.frame(100, _bodies(target_tracked=False)),
        wire.frame(101, [(2, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0), True)]),
        wire.frame(102, _bodies()),
    ]
    net = fake_motive(
        server_info=wire.server_info(multicast=True),
        modeldefs=[wire.modeldef([("rb", "other", 2), ("rb", "cf21bl_01", 7)])],
        frames=frames,
    )
    source = NatNetPoseSource(
        server_ip=kit.server_ip, rigid_body="cf21bl_01", multicast_ip="239.1.2.3", use_multicast=True
    )
    try:
        pose = first_pose(source, timeout_s=3.0)
    finally:
        source.close()
    _assert_target(pose, 102)
    assert pose.host_ns is not None
    assert pose.latency_s == pytest.approx(0.004)
    data = net.data_socket
    assert data.bound == ("", 1511)
    membership = data.options[(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP)]
    assert membership == socket.inet_aton("239.1.2.3") + socket.inet_aton("0.0.0.0")
    assert natnet.NAT_CONNECT not in net.commands_from(data)


def test_unicast_connect_registers_from_the_data_socket(fake_motive, wire, kit):
    net = fake_motive(
        server=(kit.server_ip, 1700),
        server_info=wire.server_info(),
        modeldefs=[wire.modeldef([("rb", "cf21bl_01", 7)])],
        frames=[wire.frame(5, _bodies())],
    )
    source = NatNetPoseSource(
        server_ip=kit.server_ip, command_port=1700, data_port=1600, local_ip="10.0.0.9", use_multicast=False
    )
    try:
        pose = first_pose(source, timeout_s=3.0)
    finally:
        source.close()
    _assert_target(pose, 5)
    assert pose.latency_s == pytest.approx(0.004)
    data = net.data_socket
    assert data.bound == ("10.0.0.9", 1600)
    assert (socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP) not in data.options
    assert net.commands_from(data)[:2] == [natnet.NAT_CONNECT, natnet.NAT_KEEPALIVE]
    assert all(addr == (kit.server_ip, 1700) for bound, addr, _ in net.sent if bound == data.bound)


@pytest.mark.parametrize(
    "motive_multicast, message",
    [
        (True, "Motive is streaming multicast: set mocap.natnet.use_multicast to true"),
        (False, "Motive is streaming unicast: set mocap.natnet.use_multicast to false"),
    ],
)
def test_connect_names_the_setting_when_motive_streams_the_other_mode(
    fake_motive, wire, kit, motive_multicast, message
):
    net = fake_motive(
        server_info=wire.server_info(multicast=motive_multicast),
        modeldefs=[wire.modeldef([("rb", "cf21bl_01", 7)])],
    )
    with pytest.raises(ValueError, match=message):
        NatNetPoseSource(server_ip=kit.server_ip, use_multicast=not motive_multicast).connect()
    assert net.data_socket is None


def test_connect_fails_before_opening_data_socket_when_motive_is_silent(fake_motive, wire, kit):
    net = fake_motive(server_info=None, modeldefs=[wire.modeldef([("rb", "cf21bl_01", 7)])])
    source = NatNetPoseSource(server_ip=kit.server_ip)
    with pytest.raises(RuntimeError, match="Cannot query Motive"):
        source.connect()
    assert net.data_socket is None


def test_connect_fails_when_motive_never_sends_model_descriptions(fake_motive, wire, kit):
    net = fake_motive(server_info=wire.server_info(), modeldefs=[])
    with pytest.raises(RuntimeError, match="model descriptions"):
        NatNetPoseSource(server_ip=kit.server_ip).connect()
    assert net.data_socket is None


@pytest.mark.parametrize(
    "assets, found",
    [
        ([("rb", "cf21bl_02", 3), ("rb", "wand", 4)], 0),
        ([("rb", "cf21bl_01", 3), ("rb", "cf21bl_01", 4)], 2),
        ([("skeleton", "walker", 30, [("cf21bl_01", 31)])], 0),
    ],
)
def test_asset_name_must_match_exactly_one_rigid_body(fake_motive, wire, kit, assets, found):
    net = fake_motive(server_info=wire.server_info(), modeldefs=[wire.modeldef(assets)])
    with pytest.raises(ValueError, match=rf"'cf21bl_01' must match exactly one asset; found {found}"):
        NatNetPoseSource(server_ip=kit.server_ip, rigid_body="cf21bl_01").connect()
    assert net.data_socket is None


def test_asset_lookup_error_lists_available_rigid_bodies(fake_motive, wire, kit):
    fake_motive(
        server_info=wire.server_info(),
        modeldefs=[wire.modeldef([("rb", "cf21bl_02", 3), ("rb", "wand", 4)])],
    )
    with pytest.raises(ValueError, match="Available rigid bodies: cf21bl_02, wand"):
        NatNetPoseSource(server_ip=kit.server_ip, rigid_body="cf21bl_01").connect()


def test_asset_lookup_skips_short_and_foreign_replies(fake_motive, wire, kit):
    fake_motive(
        server_info=wire.server_info(),
        modeldefs=[b"\x05", wire.server_info(), wire.modeldef([("rb", "cf21bl_01", 7)])],
        frames=[wire.frame(8, _bodies())],
    )
    source = NatNetPoseSource(server_ip=kit.server_ip)
    try:
        _assert_target(first_pose(source, timeout_s=3.0), 8)
    finally:
        source.close()


def test_asset_lookup_rejects_a_reply_shorter_than_its_header_says(fake_motive, wire, kit):
    packet = wire.modeldef([("rb", "cf21bl_01", 7)])
    fake_motive(server_info=wire.server_info(), modeldefs=[packet[:-3]])
    with pytest.raises(ValueError, match="Truncated"):
        NatNetPoseSource(server_ip=kit.server_ip).connect()


def test_receive_path_hands_the_consumer_only_the_newest_pose(fake_motive, wire, kit):
    # Motive outruns the uplink (360 Hz vs 100 Hz): a FIFO here would grow latency without bound.
    fake_motive(
        server_info=wire.server_info(),
        modeldefs=[wire.modeldef([("rb", "cf21bl_01", 7)])],
        frames=[wire.frame(n, _bodies()) for n in (40, 41, 42)],
    )
    source = NatNetPoseSource(server_ip=kit.server_ip)
    source.connect()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and [p.source_seq for p in list(source._queue.queue)] != [42]:
            time.sleep(0.005)
        _assert_target(next(source.samples()), 42)
        assert source._queue.empty()
    finally:
        source.close()


def test_close_releases_a_consumer_waiting_for_the_next_pose(fake_motive, wire, kit):
    fake_motive(
        server_info=wire.server_info(),
        modeldefs=[wire.modeldef([("rb", "cf21bl_01", 7)])],
        frames=[wire.frame(1, _bodies())],
    )
    source = NatNetPoseSource(server_ip=kit.server_ip)
    received = []

    def consume():
        for pose in source.samples():
            received.append(pose)

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    _wait_for(lambda: received)  # it now waits for a second frame that never comes
    source.close()
    consumer.join(timeout=2.0)
    assert not consumer.is_alive()
    assert [pose.source_seq for pose in received] == [1]


def _wait_for(condition, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def test_pose_watch_tells_an_untracked_body_from_a_lost_connection(fake_motive, wire, kit):
    # The flying runners' chain: NatNet, then the telemetry tap, then the extpose streamer.
    # Motive sends a frame per camera frame even while the body is untracked.
    net = fake_motive(
        server_info=wire.server_info(),
        modeldefs=[wire.modeldef([("rb", "cf21bl_01", 7)])],
        frames=[wire.frame(1, _bodies())],
    )
    source = NatNetPoseSource(server_ip=kit.server_ip)
    source.connect()
    tap = TappedPoseSource(source, types.SimpleNamespace(append_at=lambda *args, **kwargs: None))
    streamer = ExtposeStreamer(source=tap, backend=kit.Backend(), max_rate_hz=100.0, pace=False)
    streamer.start()

    def motive_streams_without_the_body(duration_s):
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            net.data_socket.inbox.put(wire.frame(2, _bodies(target_tracked=False)))
            time.sleep(0.01)

    try:
        _wait_for(lambda: streamer.sent == 1)
        motive_streams_without_the_body(0.3)
        streamer.check_poses()  # the watch starts at arming, not before
        streamer.start_pose_watch(0.2)
        with pytest.raises(MocapBodyLost):
            streamer.check_poses()

        net.data_socket.inbox.put(wire.frame(3, _bodies()))  # the body is back
        _wait_for(lambda: streamer.sent == 2)
        streamer.check_poses()

        time.sleep(0.3)  # Motive goes quiet
        with pytest.raises(MocapConnectionLost):
            streamer.check_poses()

        motive_streams_without_the_body(0.3)
        net.data_link_down = True  # the receive thread dies, its last frame still recent
        _wait_for(lambda: not source._thread.is_alive())
        with pytest.raises(MocapConnectionLost):
            streamer.check_poses()
    finally:
        streamer.stop()
        source.close()


# -- the lab's Motive, from captured packets ------------------------------------


@pytest.mark.parametrize("mode", ["multicast", "unicast"])
def test_lab_model_descriptions_list_its_rigid_bodies(motive_packets, mode):
    assert _rigid_body_descriptions(_payload(motive_packets[mode].model_def)) == [
        ("cf_ege", 62), ("cf1", 29), ("cf_teo", 63)
    ]


@pytest.mark.parametrize(
    "mode, seq, position, quat, latency_ticks",
    [
        ("multicast", 3576108, (0.609227, 0.0668315, 0.0625696),
         (0.0000154, 0.0007665, -0.0000384, 0.9999997), 39_166),
        ("unicast", 3261942, (0.6091785, 0.0667892, 0.0624674),
         (0.0000156, -0.0003797, -0.0000701, 0.9999999), 38_284),
    ],
)
def test_lab_stream_yields_the_drone_pose_and_motive_latency(
    fake_motive, motive_packets, kit, mode, seq, position, quat, latency_ticks
):
    packets = motive_packets[mode]
    fake_motive(server_info=packets.server_info, modeldefs=[packets.model_def], frames=[packets.frame])
    source = NatNetPoseSource(server_ip=kit.server_ip, rigid_body="cf_teo", use_multicast=mode == "multicast")
    try:
        pose = first_pose(source, timeout_s=3.0)
    finally:
        source.close()
    assert pose.source_seq == seq
    assert pose.position_m == pytest.approx(position, abs=1e-6)
    q = pose.orientation
    assert (q.qx, q.qy, q.qz, q.qw) == pytest.approx(quat, abs=1e-6)
    assert pose.latency_s == pytest.approx(latency_ticks / 10_000_000)  # Motive clock ticks at 10 MHz


# -- frame parsing --------------------------------------------------------------


def test_frame_yields_only_the_resolved_rigid_body(natnet_parser, wire):
    source = natnet_parser()
    _assert_target(source._pose_from_packet(wire.frame(2043627, _bodies())), 2043627)


def test_frame_with_untracked_target_yields_nothing(natnet_parser, wire):
    source = natnet_parser()
    assert source._pose_from_packet(wire.frame(1, _bodies(target_tracked=False))) is None


@pytest.mark.parametrize("message_id", [natnet.NAT_SERVERINFO, natnet.NAT_MODELDEF, natnet.NAT_KEEPALIVE])
def test_non_frame_messages_on_the_data_port_are_ignored(natnet_parser, wire, message_id):
    source = natnet_parser()
    frame = wire.frame(1, _bodies())
    assert source._pose_from_packet(struct.pack("<H", message_id) + frame[2:]) is None


def test_truncated_frames_never_raise_or_yield_a_wrong_pose(natnet_parser, wire):
    source = natnet_parser()
    packet = wire.frame(77, _bodies())
    for cut in range(len(packet)):
        pose = source._pose_from_packet(packet[:cut])
        if pose is not None:
            _assert_target(pose, 77)


@pytest.mark.parametrize(
    "stamps, clock_hz, expected",
    [
        ((1_000_000, 1_010_000, 1_040_000), 10_000_000, 0.004),
        ((5_000, 5_000, 5_000 + 360), 360_000, 0.001),
        ((1_040_000, 1_010_000, 1_000_000), 10_000_000, None),  # transmit before exposure
        ((0, 0, 20_000_000), 10_000_000, None),  # 2 s is not a frame latency
    ],
)
def test_frame_latency_comes_from_motive_clock_stamps(natnet_parser, wire, stamps, clock_hz, expected):
    source = natnet_parser(clock_hz=clock_hz)
    pose = source._pose_from_packet(wire.frame(3, _bodies(), stamps=stamps))
    assert pose.latency_s == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize(
    ("position", "quat"),
    [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
        ((math.nan, 0.1, 0.2), (0.0, 0.0, 0.0, 1.0)),
        ((0.1, 0.1, 0.2), (0.0, 0.0, math.inf, 1.0)),
    ],
    ids=["zero-quaternion", "nan-position", "inf-quaternion"],
)
def test_frame_with_a_zero_or_non_finite_pose_is_dropped_not_raised(natnet_parser, wire, position, quat):
    source = natnet_parser()
    packet = wire.frame(4, [(7, position, quat, True)])
    assert source._pose_from_packet(packet) is None
