from __future__ import annotations

import time
from collections.abc import Iterator
from queue import Queue
from typing import Any, Mapping

from cf_lab.mocap.types import EulerAngles, PoseSample, Quaternion


class RosbridgePoseSource:
    """Read pose samples from a ROS topic through rosbridge websocket JSON."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 9090,
        topic: str,
        # Any message carrying a pose: geometry_msgs/Pose, PoseStamped,
        # PoseWithCovarianceStamped, nav_msgs/Odometry (see _extract_pose_mapping).
        message_type: str = "geometry_msgs/PoseStamped",
        orientation_format: str = "quaternion",
        rigid_body: str = "cf21bl_01",
        throttle_rate_ms: int = 0,
    ) -> None:
        self._host = host
        self._port = port
        self._topic_name = topic
        self._message_type = message_type
        self._orientation_format = orientation_format
        self._rigid_body = rigid_body
        self._queue: Queue[PoseSample] = Queue(maxsize=1)  # only the newest pose
        self._throttle_rate_ms = throttle_rate_ms
        self._ros = None
        self._topic = None

    def connect(self) -> None:
        import roslibpy

        self._ros = roslibpy.Ros(host=self._host, port=self._port)
        self._ros.run()
        # throttle_rate makes the rosbridge server drop excess messages before
        # they cross the websocket; queue_length=1 keeps only the newest one.
        self._topic = roslibpy.Topic(
            self._ros,
            self._topic_name,
            self._message_type,
            throttle_rate=self._throttle_rate_ms,
            queue_length=1,
        )
        self._topic.subscribe(self._handle_message)

    def close(self) -> None:
        if self._topic is not None:
            self._topic.unsubscribe()
            self._topic = None
        if self._ros is not None:
            self._ros.terminate()
            self._ros = None

    def __enter__(self) -> "RosbridgePoseSource":
        self.connect()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def samples(self, *, start_s: float = 0.0, dt_s: float = 0.01) -> Iterator[PoseSample]:
        del start_s, dt_s
        if self._topic is None:
            self.connect()
        while True:
            yield self._queue.get()

    def connected(self, within_s: float) -> bool:
        """Whether the rosbridge websocket is up.

        The websocket says nothing of the mocap system behind it, so a quiet
        topic on a live websocket reads as an untracked body even when the mocap
        system has stopped.
        """
        del within_s
        return self._ros is not None and self._ros.is_connected

    def _handle_message(self, message: Mapping[str, Any]) -> None:
        pose = pose_sample_from_ros_message(
            message,
            rigid_body=self._rigid_body,
            orientation_format=self._orientation_format,
        )
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(pose)


def vrpn_pose_source(
    *,
    rigid_body: str,
    host: str = "localhost",
    port: int = 9090,
    topic: str = "",
    rate_hz: float = 100.0,
    orientation_format: str = "quaternion",
) -> RosbridgePoseSource:
    """Build a rosbridge pose source for a VRPN (OptiTrack) rigid body.

    ``topic`` defaults to the ``vrpn_client_ros`` convention,
    ``/vrpn_client_node/<rigid_body>/pose``. The rosbridge server is asked to
    throttle to ``rate_hz`` so a fast mocap system cannot flood the websocket.
    """
    return RosbridgePoseSource(
        host=host,
        port=port,
        topic=topic or f"/vrpn_client_node/{rigid_body}/pose",
        rigid_body=rigid_body,
        orientation_format=orientation_format,
        throttle_rate_ms=int(1000.0 / rate_hz),
    )


def pose_sample_from_ros_message(
    message: Mapping[str, Any],
    *,
    rigid_body: str,
    orientation_format: str = "quaternion",
) -> PoseSample:
    pose = _extract_pose_mapping(message)
    position = _required_mapping(pose.get("position"), "pose.position")
    orientation = _required_mapping(pose.get("orientation"), "pose.orientation")
    quaternion, euler = _orientation_from_mapping(orientation, orientation_format=orientation_format)

    return PoseSample(
        timestamp_s=_timestamp_from_message(message),
        position_m=(
            float(position["x"]),
            float(position["y"]),
            float(position["z"]),
        ),
        orientation=quaternion,
        rigid_body=rigid_body,
        euler_override_rad=euler,
    )


def _extract_pose_mapping(message: Mapping[str, Any]) -> Mapping[str, Any]:
    if "pose" not in message:
        return message

    pose = _required_mapping(message["pose"], "pose")
    nested_pose = pose.get("pose")
    if isinstance(nested_pose, Mapping):
        return nested_pose
    return pose


def _timestamp_from_message(message: Mapping[str, Any]) -> float:
    header = message.get("header")
    if isinstance(header, Mapping):
        stamp = header.get("stamp")
        if isinstance(stamp, Mapping):
            seconds = stamp.get("sec", stamp.get("secs", 0))
            nanoseconds = stamp.get("nanosec", stamp.get("nsecs", 0))
            return float(seconds) + float(nanoseconds) / 1_000_000_000.0
    return time.time()


def _orientation_from_mapping(
    orientation: Mapping[str, Any],
    *,
    orientation_format: str,
) -> tuple[Quaternion, EulerAngles | None]:
    if orientation_format == "quaternion":
        return (
            Quaternion(
                qx=float(orientation["x"]),
                qy=float(orientation["y"]),
                qz=float(orientation["z"]),
                qw=float(orientation["w"]),
            ).normalized(),
            None,
        )

    if orientation_format in {"euler-rad", "euler-deg"}:
        euler = EulerAngles(
            roll=_float_named_axis(orientation, "roll", "x"),
            pitch=_float_named_axis(orientation, "pitch", "y"),
            yaw=_float_named_axis(orientation, "yaw", "z"),
        )
        if orientation_format == "euler-deg":
            import math

            euler = EulerAngles(
                roll=math.radians(euler.roll),
                pitch=math.radians(euler.pitch),
                yaw=math.radians(euler.yaw),
            )
        return (Quaternion.from_euler_rad(euler), euler)

    raise ValueError(
        "orientation_format must be one of: quaternion, euler-rad, euler-deg."
    )


def _float_named_axis(mapping: Mapping[str, Any], named_key: str, axis_key: str) -> float:
    if named_key in mapping:
        return float(mapping[named_key])
    return float(mapping[axis_key])


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected ROS {name} mapping.")
    return value
