"""NatNet wire parsing, Motive asset lookup and the receive path, against byte-exact packets."""
import math
import socket
import struct
import time
import types

import pytest

from cf_lab.mocap import natnet
from cf_lab.mocap.natnet import NatNetPoseSource, _rigid_body_descriptions
from cf_lab.mocap.stream import ExtposeStreamer, MocapBodyLost, MocapConnectionLost, first_pose
from cf_lab.telemetry.mocap import TappedPoseSource

VERSIONS = [(2, 5), (2, 6), (2, 7), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2)]
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


@pytest.mark.parametrize("version", [(2, 9), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2)])
def test_model_descriptions_list_standalone_rigid_bodies_past_every_asset_kind(wire, version):
    assets = [("markers", "all"), ("rb", "other", 2)]
    if version[0] >= 3:
        assets += [("force_plate",), ("device",)]
    assets += [("skeleton", "walker", 30, [("Hip", 31), ("cf21bl_01", 32)]), ("camera", "Prime13"),
               ("rb", "cf21bl_01", 7)]
    packet = wire.modeldef(version, assets)
    assert _rigid_body_descriptions(_payload(packet), *version) == [("other", 2), ("cf21bl_01", 7)]


def test_model_descriptions_4_1_skip_unknown_asset_records_by_size(wire):
    packet = wire.modeldef((4, 1), [("raw", 6, b"\x01\x02\x03" * 11), ("rb", "cf21bl_01", 7)])
    assert _rigid_body_descriptions(_payload(packet), 4, 1) == [("cf21bl_01", 7)]


@pytest.mark.parametrize("version", [(3, 1), (4, 0), (4, 1)])
def test_model_descriptions_reject_every_truncation(wire, version):
    payload = _payload(wire.modeldef(version, [("markers", "all"), ("rb", "other", 2), ("rb", "cf21bl_01", 7)]))
    for cut in range(len(payload)):
        with pytest.raises(ValueError):
            _rigid_body_descriptions(payload[:cut], *version)


def test_model_descriptions_reject_unknown_asset_kind_before_4_1(wire):
    packet = wire.modeldef((3, 1), [("raw", 9, b"\0" * 8), ("rb", "cf21bl_01", 7)])
    with pytest.raises(ValueError, match="Unsupported"):
        _rigid_body_descriptions(_payload(packet), 3, 1)


# -- connect: Motive query and asset resolution -------------------------------


@pytest.mark.parametrize("version", [(2, 7), (3, 1), (4, 1)])
def test_multicast_connect_resolves_name_and_yields_first_tracked_target_pose(fake_motive, wire, kit, version):
    frames = [
        wire.frame(version, 100, _bodies(target_tracked=False)),
        wire.frame(version, 101, [(2, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0), True)]),
        wire.frame(version, 102, _bodies()),
    ]
    net = fake_motive(
        server_info=wire.server_info(version),
        modeldefs=[wire.modeldef(version, [("rb", "other", 2), ("rb", "cf21bl_01", 7)])],
        frames=frames,
    )
    source = NatNetPoseSource(server_ip=kit.server_ip, rigid_body="cf21bl_01", multicast_ip="239.1.2.3")
    try:
        pose = first_pose(source, timeout_s=3.0)
    finally:
        source.close()
    _assert_target(pose, 102)
    assert pose.host_ns is not None
    assert pose.latency_s == (pytest.approx(0.004) if version >= (4, 1) else None)
    data = net.data_socket
    assert data.bound == ("", 1511)
    membership = data.options[(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP)]
    assert membership == socket.inet_aton("239.1.2.3") + socket.inet_aton("0.0.0.0")
    assert natnet.NAT_CONNECT not in net.commands_from(data)


def test_unicast_connect_registers_from_the_data_socket(fake_motive, wire, kit):
    version = (4, 1)
    net = fake_motive(
        server=(kit.server_ip, 1700),
        server_info=wire.server_info(version),
        modeldefs=[wire.modeldef(version, [("rb", "cf21bl_01", 7)])],
        frames=[wire.frame(version, 5, _bodies())],
    )
    source = NatNetPoseSource(
        server_ip=kit.server_ip, command_port=1700, data_port=1600, local_ip="10.0.0.9", use_multicast=False
    )
    try:
        pose = first_pose(source, timeout_s=3.0)
    finally:
        source.close()
    _assert_target(pose, 5)
    data = net.data_socket
    assert data.bound == ("10.0.0.9", 1600)
    assert (socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP) not in data.options
    assert net.commands_from(data)[:2] == [natnet.NAT_CONNECT, natnet.NAT_KEEPALIVE]
    assert all(addr == (kit.server_ip, 1700) for bound, addr, _ in net.sent if bound == data.bound)


def test_connect_fails_before_opening_data_socket_when_motive_is_silent(fake_motive, wire, kit):
    net = fake_motive(server_info=None, modeldefs=[wire.modeldef((4, 1), [("rb", "cf21bl_01", 7)])])
    source = NatNetPoseSource(server_ip=kit.server_ip)
    with pytest.raises(RuntimeError, match="Cannot query Motive"):
        source.connect()
    assert net.data_socket is None


def test_connect_fails_when_motive_never_sends_model_descriptions(fake_motive, wire, kit):
    net = fake_motive(server_info=wire.server_info((4, 1)), modeldefs=[])
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
    net = fake_motive(server_info=wire.server_info((3, 1)), modeldefs=[wire.modeldef((3, 1), assets)])
    with pytest.raises(ValueError, match=rf"'cf21bl_01' must match exactly one asset; found {found}"):
        NatNetPoseSource(server_ip=kit.server_ip, rigid_body="cf21bl_01").connect()
    assert net.data_socket is None


def test_asset_lookup_error_lists_available_rigid_bodies(fake_motive, wire, kit):
    fake_motive(
        server_info=wire.server_info((4, 1)),
        modeldefs=[wire.modeldef((4, 1), [("rb", "cf21bl_02", 3), ("rb", "wand", 4)])],
    )
    with pytest.raises(ValueError, match="Available rigid bodies: cf21bl_02, wand"):
        NatNetPoseSource(server_ip=kit.server_ip, rigid_body="cf21bl_01").connect()


def test_asset_lookup_skips_short_and_foreign_replies(fake_motive, wire, kit):
    version = (4, 1)
    fake_motive(
        server_info=wire.server_info(version),
        modeldefs=[b"\x05", wire.server_info(version), wire.modeldef(version, [("rb", "cf21bl_01", 7)])],
        frames=[wire.frame(version, 8, _bodies())],
    )
    source = NatNetPoseSource(server_ip=kit.server_ip)
    try:
        _assert_target(first_pose(source, timeout_s=3.0), 8)
    finally:
        source.close()


def test_asset_lookup_rejects_a_reply_shorter_than_its_header_says(fake_motive, wire, kit):
    packet = wire.modeldef((4, 1), [("rb", "cf21bl_01", 7)])
    fake_motive(server_info=wire.server_info((4, 1)), modeldefs=[packet[:-3]])
    with pytest.raises(ValueError, match="Truncated"):
        NatNetPoseSource(server_ip=kit.server_ip).connect()


def test_receive_path_hands_the_consumer_only_the_newest_pose(fake_motive, wire, kit):
    # Motive outruns the uplink (360 Hz vs 100 Hz): a FIFO here would grow latency without bound.
    version = (4, 1)
    fake_motive(
        server_info=wire.server_info(version),
        modeldefs=[wire.modeldef(version, [("rb", "cf21bl_01", 7)])],
        frames=[wire.frame(version, n, _bodies()) for n in (40, 41, 42)],
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


def _wait_for(condition, timeout_s=2.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def test_pose_watch_tells_an_untracked_body_from_a_lost_connection(fake_motive, wire, kit):
    # The flying runners' chain: NatNet, then the telemetry tap, then the extpose streamer.
    # Motive sends a frame per camera frame even while the body is untracked.
    version = (4, 1)
    net = fake_motive(
        server_info=wire.server_info(version),
        modeldefs=[wire.modeldef(version, [("rb", "cf21bl_01", 7)])],
        frames=[wire.frame(version, 1, _bodies())],
    )
    source = NatNetPoseSource(server_ip=kit.server_ip)
    source.connect()
    tap = TappedPoseSource(source, types.SimpleNamespace(append_at=lambda *args, **kwargs: None))
    streamer = ExtposeStreamer(source=tap, backend=kit.Backend(), max_rate_hz=100.0, pace=False)
    streamer.start()

    def motive_streams_without_the_body(duration_s):
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            net.data_socket.inbox.put(wire.frame(version, 2, _bodies(target_tracked=False)))
            time.sleep(0.01)

    try:
        _wait_for(lambda: streamer.sent == 1)
        motive_streams_without_the_body(0.3)
        streamer.check_poses()  # the watch starts at arming, not before
        streamer.start_pose_watch(0.2)
        with pytest.raises(MocapBodyLost):
            streamer.check_poses()

        net.data_socket.inbox.put(wire.frame(version, 3, _bodies()))  # the body is back
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
        source._publish(kit.pose())  # release the streamer thread


# -- frame parsing --------------------------------------------------------------


@pytest.mark.parametrize("version", VERSIONS)
def test_frame_yields_only_the_resolved_rigid_body(natnet_parser, wire, version):
    source = natnet_parser(version)
    _assert_target(source._pose_from_packet(wire.frame(version, 2043627, _bodies())), 2043627)


@pytest.mark.parametrize("version", [v for v in VERSIONS if v >= (2, 6)])
def test_frame_with_untracked_target_yields_nothing(natnet_parser, wire, version):
    source = natnet_parser(version)
    assert source._pose_from_packet(wire.frame(version, 1, _bodies(target_tracked=False))) is None


@pytest.mark.parametrize("message_id", [natnet.NAT_SERVERINFO, natnet.NAT_MODELDEF, natnet.NAT_KEEPALIVE])
def test_non_frame_messages_on_the_data_port_are_ignored(natnet_parser, wire, message_id):
    source = natnet_parser((4, 1))
    frame = wire.frame((4, 1), 1, _bodies())
    assert source._pose_from_packet(struct.pack("<H", message_id) + frame[2:]) is None


@pytest.mark.parametrize("version", [(2, 7), (3, 1), (4, 1)])
def test_truncated_frames_never_raise_or_yield_a_wrong_pose(natnet_parser, wire, version):
    source = natnet_parser(version)
    packet = wire.frame(version, 77, _bodies())
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
        ((1_000_000, 1_010_000, 1_040_000), 0, None),  # Motive reported no clock
    ],
)
def test_frame_latency_comes_from_motive_clock_stamps(natnet_parser, wire, stamps, clock_hz, expected):
    source = natnet_parser((4, 1), clock_hz=clock_hz)
    pose = source._pose_from_packet(wire.frame((4, 1), 3, _bodies(), stamps=stamps))
    assert pose.latency_s == (pytest.approx(expected) if expected is not None else None)


@pytest.mark.parametrize(
    ("version", "position", "quat"),
    [
        # NatNet < 2.6 has no tracking flag; Motive streams an untracked body as a zero pose.
        ((2, 5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),
        ((4, 1), (math.nan, 0.1, 0.2), (0.0, 0.0, 0.0, 1.0)),
        ((4, 1), (0.1, 0.1, 0.2), (0.0, 0.0, math.inf, 1.0)),
    ],
    ids=["zero-quaternion", "nan-position", "inf-quaternion"],
)
def test_frame_with_a_zero_or_non_finite_pose_is_dropped_not_raised(natnet_parser, wire, version, position, quat):
    source = natnet_parser(version)
    packet = wire.frame(version, 4, [(7, position, quat, True)])
    assert source._pose_from_packet(packet) is None
