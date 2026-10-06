"""IMU samples and the stationary summary behind the ``check_static_imu`` preflight check.

Before arming, the check samples the accelerometer, the corrected gyro and the
supervisor's armed flag in one firmware log block (:data:`IMU_LOG_VARIABLES`) for
a short window, and :func:`summarize_static_imu` checks that the drone was still,
read one g, and stayed disarmed. This is not telemetry: recorded runs get IMU data
from the telemetry streams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from cf_lab.backends.crazyflie import SUPERVISOR_INFO_BITS

_CORRECTED_GYRO_VARS = ("gyro.x", "gyro.y", "gyro.z")
_ACCELERATION_VARS = ("acc.x", "acc.y", "acc.z")
_SUPERVISOR_INFO_VAR = "supervisor.info"
_SUPERVISOR_ARMED_BIT = SUPERVISOR_INFO_BITS["ARMED"]

# Six floats plus supervisor.info:uint16_t use the complete 26-byte CRTP log
# payload. Keeping this as one block samples the accelerometer, gyro, and arming
# state at the same instant without spending a second radio packet.
IMU_LOG_VARIABLES = (
    *((name, "float") for name in _ACCELERATION_VARS),
    *((name, "float") for name in _CORRECTED_GYRO_VARS),
    (_SUPERVISOR_INFO_VAR, "uint16_t"),
)

DEFAULT_MAX_GYRO_RATE_DPS = 1.0  # mean corrected gyro, i.e. the bias residual
DEFAULT_MAX_GYRO_MOTION_DPS = 2.0  # gyro spread over the window
DEFAULT_MAX_ACCELERATION_MOTION_G = 0.05  # accelerometer spread over the window
DEFAULT_MAX_ACCELERATION_MAGNITUDE_ERROR_G = 0.15  # allowed distance from one g


@dataclass(frozen=True)
class ImuSample:
    """One read-only Crazyflie IMU observation.

    Accelerometer values are in g and gyro values are in deg/s, matching the
    stock firmware log variables.
    """

    acceleration_g: tuple[float, float, float]
    angular_velocity_dps: tuple[float, float, float]
    supervisor_info: int

    @classmethod
    def from_log(cls, row: Mapping[str, object]) -> ImuSample:
        """One sample from a firmware log row of :data:`IMU_LOG_VARIABLES`."""
        return cls(
            acceleration_g=tuple(float(row[name]) for name in _ACCELERATION_VARS),
            angular_velocity_dps=tuple(float(row[name]) for name in _CORRECTED_GYRO_VARS),
            supervisor_info=int(row[_SUPERVISOR_INFO_VAR]),
        )

    @property
    def armed(self) -> bool:
        """Whether the firmware supervisor reported the ARMED bit."""
        return bool(self.supervisor_info & _SUPERVISOR_ARMED_BIT)


@dataclass(frozen=True)
class ImuStaticSummary:
    """Summary of a window intended to be stationary and disarmed.

    ``stationary`` combines gyro rate/spread and accelerometer spread.
    ``reads_one_g`` is true when the mean acceleration magnitude is close to
    one g; a dead IMU reads 0 g, so it is stationary but fails this test.
    ``disarmed`` is true only when at least one sample was received and none
    carried the supervisor ARMED bit.
    The individual measurements are retained so callers can distinguish a bad
    gyro bias from motion or an accelerometer/gravity problem.
    """

    mean_acceleration_g: tuple[float, float, float]
    acceleration_magnitude_g: float
    acceleration_motion_g: float
    mean_angular_velocity_dps: tuple[float, float, float]
    angular_velocity_magnitude_dps: float
    angular_velocity_motion_dps: float
    samples: int
    armed_samples: int
    disarmed: bool
    stationary: bool
    reads_one_g: bool

    @property
    def usable(self) -> bool:
        """True when the window is non-empty, stationary, reads one g, and disarmed."""
        return self.samples > 0 and self.disarmed and self.stationary and self.reads_one_g


def summarize_static_imu(
    samples: Sequence[ImuSample],
    *,
    max_gyro_rate_dps: float = DEFAULT_MAX_GYRO_RATE_DPS,
    max_gyro_motion_dps: float = DEFAULT_MAX_GYRO_MOTION_DPS,
    max_acceleration_motion_g: float = DEFAULT_MAX_ACCELERATION_MOTION_G,
    max_acceleration_magnitude_error_g: float = DEFAULT_MAX_ACCELERATION_MAGNITUDE_ERROR_G,
) -> ImuStaticSummary:
    """Summarize whether an IMU capture was stationary and disarmed.

    This is deliberately an observation-only check. It neither disarms the
    drone nor recalibrates the IMU. If the gyro residual is poor, callers should
    ask the operator to keep the drone still and use the repository's explicit
    reboot utility before collecting another window.
    """
    thresholds = {
        "max_gyro_rate_dps": max_gyro_rate_dps,
        "max_gyro_motion_dps": max_gyro_motion_dps,
        "max_acceleration_motion_g": max_acceleration_motion_g,
        "max_acceleration_magnitude_error_g": max_acceleration_magnitude_error_g,
    }
    for name, value in thresholds.items():
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")

    rows = tuple(samples)
    if not rows:
        return ImuStaticSummary(
            mean_acceleration_g=(0.0, 0.0, 0.0),
            acceleration_magnitude_g=0.0,
            acceleration_motion_g=0.0,
            mean_angular_velocity_dps=(0.0, 0.0, 0.0),
            angular_velocity_magnitude_dps=0.0,
            angular_velocity_motion_dps=0.0,
            samples=0,
            armed_samples=0,
            disarmed=False,
            stationary=False,
            reads_one_g=False,
        )

    acceleration = tuple(sample.acceleration_g for sample in rows)
    angular_velocity = tuple(sample.angular_velocity_dps for sample in rows)
    mean_acceleration = _vector_mean(acceleration)
    mean_angular_velocity = _vector_mean(angular_velocity)
    acceleration_magnitude = _norm(mean_acceleration)
    angular_velocity_magnitude = _norm(mean_angular_velocity)
    acceleration_motion = _vector_std_norm(acceleration, mean_acceleration)
    angular_velocity_motion = _vector_std_norm(angular_velocity, mean_angular_velocity)
    armed_samples = sum(sample.armed for sample in rows)
    stationary = (
        angular_velocity_magnitude <= max_gyro_rate_dps
        and angular_velocity_motion <= max_gyro_motion_dps
        and acceleration_motion <= max_acceleration_motion_g
    )
    return ImuStaticSummary(
        mean_acceleration_g=mean_acceleration,
        acceleration_magnitude_g=acceleration_magnitude,
        acceleration_motion_g=acceleration_motion,
        mean_angular_velocity_dps=mean_angular_velocity,
        angular_velocity_magnitude_dps=angular_velocity_magnitude,
        angular_velocity_motion_dps=angular_velocity_motion,
        samples=len(rows),
        armed_samples=armed_samples,
        disarmed=armed_samples == 0,
        stationary=stationary,
        reads_one_g=abs(acceleration_magnitude - 1.0) <= max_acceleration_magnitude_error_g,
    )


def _vector_mean(rows: Sequence[tuple[float, float, float]]) -> tuple[float, float, float]:
    count = len(rows)
    return tuple(sum(row[axis] for row in rows) / count for axis in range(3))


def _vector_std_norm(
    rows: Sequence[tuple[float, float, float]],
    mean: tuple[float, float, float],
) -> float:
    count = len(rows)
    variance = tuple(
        sum((row[axis] - mean[axis]) ** 2 for row in rows) / count for axis in range(3)
    )
    return math.sqrt(sum(component for component in variance))


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in vector))
