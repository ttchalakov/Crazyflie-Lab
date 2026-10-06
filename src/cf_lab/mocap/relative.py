from __future__ import annotations

import dataclasses
import math

from cf_lab.mocap.types import EulerAngles, PoseSample, Quaternion, Vector3


class OriginZeroer:
    """Re-express poses in a frame anchored at the first pose's position and yaw.

    The first sample maps to position (0, 0, 0) with zero yaw; roll, pitch,
    and +z-up stay untouched. This makes the drone's starting pose on the
    ground the world origin, and it must be applied before samples reach the
    estimator so the streamed frame never changes mid-flight.
    """

    def __init__(self, *, on_origin=None) -> None:
        self._origin_position_m: Vector3 | None = None
        self._origin_yaw_rad = 0.0
        # Called once as on_origin(position_m, yaw_rad) when the origin locks,
        # so telemetry can record the transform and recreate the zeroed signal.
        self._on_origin = on_origin

    def set_origin(self, position_m: Vector3, yaw_rad: float) -> None:
        """Fix the origin explicitly (e.g. replaying a recorded transform)."""
        self._origin_position_m = position_m
        self._origin_yaw_rad = yaw_rad

    def zero(self, pose: PoseSample) -> PoseSample:
        if self._origin_position_m is None:
            self._origin_position_m = pose.position_m
            self._origin_yaw_rad = pose.euler_rad.yaw
            if self._on_origin is not None:
                self._on_origin(self._origin_position_m, self._origin_yaw_rad)

        x0, y0, z0 = self._origin_position_m
        dx = pose.position_m[0] - x0
        dy = pose.position_m[1] - y0
        dz = pose.position_m[2] - z0

        cos_yaw = math.cos(-self._origin_yaw_rad)
        sin_yaw = math.sin(-self._origin_yaw_rad)
        position = (
            cos_yaw * dx - sin_yaw * dy,
            sin_yaw * dx + cos_yaw * dy,
            dz,
        )

        unyaw = Quaternion.from_euler_rad(
            EulerAngles(roll=0.0, pitch=0.0, yaw=-self._origin_yaw_rad)
        )
        orientation = unyaw.multiply(pose.orientation).normalized()

        euler_override = None
        if pose.euler_override_rad is not None:
            euler_override = EulerAngles(
                roll=pose.euler_override_rad.roll,
                pitch=pose.euler_override_rad.pitch,
                yaw=wrap_angle_rad(pose.euler_override_rad.yaw - self._origin_yaw_rad),
            )

        return dataclasses.replace(
            pose,
            position_m=position,
            orientation=orientation,
            euler_override_rad=euler_override,
        )


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi
