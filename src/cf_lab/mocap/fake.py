from __future__ import annotations

import math
import time
from collections.abc import Iterator

from .types import EulerAngles, PoseSample, Quaternion, Vector3


class ScriptedTakeoffSource:
    """Fake mocap stream of a scripted takeoff and hover.

    The rigid body sits on the ground for ``ground_s``, rises linearly to
    ``hover_z_m`` over ``rise_s``, then holds there. The start position and
    yaw are expressed in the raw mocap frame so origin re-zeroing has real
    work to do during dry runs.
    """

    def __init__(
        self,
        *,
        rigid_body: str = "cf21bl_01",
        start_position_m: Vector3 = (0.0, 0.0, 0.0),
        start_yaw_rad: float = 0.0,
        ground_s: float = 2.0,
        rise_s: float = 3.0,
        hover_z_m: float = 0.5,
    ) -> None:
        if rise_s <= 0:
            raise ValueError("rise_s must be positive.")
        self._rigid_body = rigid_body
        self._start_position_m = start_position_m
        self._orientation = Quaternion.from_euler_rad(
            EulerAngles(roll=0.0, pitch=0.0, yaw=start_yaw_rad)
        )
        self._ground_s = ground_s
        self._rise_s = rise_s
        self._hover_z_m = hover_z_m

    def height_at(self, t_s: float) -> float:
        if t_s <= self._ground_s:
            return 0.0
        if t_s >= self._ground_s + self._rise_s:
            return self._hover_z_m
        return self._hover_z_m * (t_s - self._ground_s) / self._rise_s

    def samples(self, *, start_s: float = 0.0, dt_s: float = 0.01) -> Iterator[PoseSample]:
        x0, y0, z0 = self._start_position_m
        timestamp = start_s
        while True:
            yield PoseSample(
                timestamp_s=timestamp,
                position_m=(x0, y0, z0 + self.height_at(timestamp)),
                orientation=self._orientation,
                rigid_body=self._rigid_body,
            )
            timestamp += dt_s


class BoxSweepSource:
    """Fake mocap that drifts the drone up and down through a box for dry runs.

    The start pose is offset from the origin so OriginZeroer has real work to do.
    After zeroing, x and y stay at 0 (inside the box) and z follows a slow cosine
    sweep between 0 and ``z_peak_m``, so an in/out-of-box indicator toggles once
    per period.
    """

    def __init__(
        self,
        *,
        rigid_body: str = "cf21bl_01",
        start_position_m: Vector3 = (0.9, -0.6, 0.0),
        z_peak_m: float = 0.5,
        period_s: float = 6.0,
    ) -> None:
        if period_s <= 0:
            raise ValueError("period_s must be positive.")
        self._rigid_body = rigid_body
        self._start_position_m = start_position_m
        self._z_peak_m = z_peak_m
        self._period_s = period_s

    def height_at(self, t_s: float) -> float:
        return 0.5 * self._z_peak_m * (1.0 - math.cos(2.0 * math.pi * t_s / self._period_s))

    def samples(self, *, start_s: float = 0.0, dt_s: float = 0.01) -> Iterator[PoseSample]:
        x0, y0, z0 = self._start_position_m
        timestamp = start_s
        while True:
            yield PoseSample(
                timestamp_s=timestamp,
                position_m=(x0, y0, z0 + self.height_at(timestamp)),
                orientation=Quaternion.identity(),
                rigid_body=self._rigid_body,
            )
            timestamp += dt_s


class DeadMocapSource:
    """Mocap source that never yields a sample, like a mocap system left off.

    It exercises the runners' "no mocap" gate without hardware.
    """

    def samples(self, *, start_s: float = 0.0, dt_s: float = 0.01) -> Iterator[PoseSample]:
        del start_s, dt_s
        while True:
            time.sleep(3600.0)
        yield  # pragma: no cover - makes this a (never-yielding) generator
