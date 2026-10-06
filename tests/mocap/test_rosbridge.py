"""ROS pose message conversion and the rosbridge subscription a VRPN source opens."""
import math
import types

import pytest

from cf_lab.mocap import rosbridge
from cf_lab.mocap.rosbridge import RosbridgePoseSource, pose_sample_from_ros_message

POSE = {"position": {"x": 0.5, "y": -1.25, "z": 0.75}, "orientation": {"x": 0.0, "y": 0.0, "z": 2.0, "w": 2.0}}


@pytest.mark.parametrize(
    "stamp, expected_s",
    [
        ({"secs": 1700000000, "nsecs": 250_000_000}, 1700000000.25),  # ROS 1
        ({"sec": 1700000000, "nanosec": 5_000_000}, 1700000000.005),  # ROS 2
    ],
)
def test_pose_stamped_converts_position_quaternion_and_header_stamp(stamp, expected_s):
    message = {"header": {"stamp": stamp, "frame_id": "world"}, "pose": POSE}
    pose = pose_sample_from_ros_message(message, rigid_body="cf21bl_01")
    assert pose.timestamp_s == pytest.approx(expected_s, abs=1e-9)
    assert pose.position_m == (0.5, -1.25, 0.75)
    q = pose.orientation
    assert (q.qx, q.qy, q.qz, q.qw) == pytest.approx((0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    assert pose.euler_rad.yaw == pytest.approx(math.pi / 2)
    assert pose.rigid_body == "cf21bl_01"


def test_pose_with_covariance_reads_the_nested_pose():
    message = {"header": {"stamp": {"secs": 3, "nsecs": 0}}, "pose": {"pose": POSE, "covariance": [0.0] * 36}}
    pose = pose_sample_from_ros_message(message, rigid_body="cf21bl_01")
    assert pose.position_m == (0.5, -1.25, 0.75)
    assert pose.timestamp_s == 3.0


def test_bare_pose_without_header_is_stamped_with_host_time(monkeypatch):
    monkeypatch.setattr(rosbridge, "time", types.SimpleNamespace(time=lambda: 42.5))
    pose = pose_sample_from_ros_message(POSE, rigid_body="cf21bl_01")
    assert (pose.timestamp_s, pose.position_m) == (42.5, (0.5, -1.25, 0.75))


@pytest.mark.parametrize(
    "message",
    [
        {"pose": {"orientation": POSE["orientation"]}},
        {"pose": {"position": POSE["position"]}},
        {"pose": "not a mapping"},
    ],
)
def test_message_without_position_or_orientation_is_rejected(message):
    with pytest.raises(ValueError, match="Expected ROS"):
        pose_sample_from_ros_message(message, rigid_body="cf21bl_01")


def test_subscription_keeps_only_the_newest_pose_and_close_releases_the_link(fake_roslibpy):
    source = RosbridgePoseSource(host="10.0.0.2", port=9091, topic="/vrpn_client_node/cf21bl_01/pose")
    source.connect()
    (ros,), (topic,) = fake_roslibpy.ros, fake_roslibpy.topics
    assert (ros.host, ros.port, ros.running) == ("10.0.0.2", 9091, True)
    assert topic.queue_length == 1
    for x in (1.0, 2.0, 3.0):
        topic.callback({"header": {"stamp": {"secs": int(x), "nsecs": 0}},
                        "pose": {**POSE, "position": {"x": x, "y": 0.0, "z": 0.0}}})
    assert next(source.samples()).position_m == (3.0, 0.0, 0.0)
    source.close()
    assert topic.callback is None and not ros.running
