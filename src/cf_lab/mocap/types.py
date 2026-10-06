from __future__ import annotations

import math
from dataclasses import dataclass


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class EulerAngles:
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class Quaternion:
    qx: float
    qy: float
    qz: float
    qw: float

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls(qx=0.0, qy=0.0, qz=0.0, qw=1.0)

    @classmethod
    def from_euler_rad(cls, euler: EulerAngles) -> "Quaternion":
        half_roll = euler.roll * 0.5
        half_pitch = euler.pitch * 0.5
        half_yaw = euler.yaw * 0.5

        cr = math.cos(half_roll)
        sr = math.sin(half_roll)
        cp = math.cos(half_pitch)
        sp = math.sin(half_pitch)
        cy = math.cos(half_yaw)
        sy = math.sin(half_yaw)

        return cls(
            qx=sr * cp * cy - cr * sp * sy,
            qy=cr * sp * cy + sr * cp * sy,
            qz=cr * cp * sy - sr * sp * cy,
            qw=cr * cp * cy + sr * sp * sy,
        ).normalized()

    def multiply(self, other: "Quaternion") -> "Quaternion":
        """Hamilton product self * other: other's rotation followed by self's."""
        return Quaternion(
            qx=self.qw * other.qx + self.qx * other.qw + self.qy * other.qz - self.qz * other.qy,
            qy=self.qw * other.qy - self.qx * other.qz + self.qy * other.qw + self.qz * other.qx,
            qz=self.qw * other.qz + self.qx * other.qy - self.qy * other.qx + self.qz * other.qw,
            qw=self.qw * other.qw - self.qx * other.qx - self.qy * other.qy - self.qz * other.qz,
        )

    def normalized(self) -> "Quaternion":
        norm = math.sqrt(self.qx**2 + self.qy**2 + self.qz**2 + self.qw**2)
        if norm == 0:
            raise ValueError("Quaternion norm must be non-zero.")
        return Quaternion(
            qx=self.qx / norm,
            qy=self.qy / norm,
            qz=self.qz / norm,
            qw=self.qw / norm,
        )

    def to_euler_rad(self) -> EulerAngles:
        q = self.normalized()

        sinr_cosp = 2.0 * (q.qw * q.qx + q.qy * q.qz)
        cosr_cosp = 1.0 - 2.0 * (q.qx * q.qx + q.qy * q.qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q.qw * q.qy - q.qz * q.qx)
        pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

        siny_cosp = 2.0 * (q.qw * q.qz + q.qx * q.qy)
        cosy_cosp = 1.0 - 2.0 * (q.qy * q.qy + q.qz * q.qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return EulerAngles(roll=roll, pitch=pitch, yaw=yaw)


@dataclass(frozen=True)
class PoseSample:
    timestamp_s: float
    position_m: Vector3
    orientation: Quaternion
    rigid_body: str
    euler_override_rad: EulerAngles | None = None
    # Source-side sample counter when the transport has one (NatNet frame
    # number); None otherwise. Telemetry uses it as the mocap clock axis.
    source_seq: int | None = None
    # time.monotonic_ns() when the packet arrived, and the source's own
    # mid-exposure-to-transmit latency, when the transport reports them.
    host_ns: int | None = None
    latency_s: float | None = None

    @property
    def euler_rad(self) -> EulerAngles:
        if self.euler_override_rad is not None:
            return self.euler_override_rad
        return self.orientation.to_euler_rad()
