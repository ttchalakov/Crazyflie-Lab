"""Helpers for reading Crazyflie firmware parameters through cflib."""

from __future__ import annotations

import time
from typing import Protocol


RATE_PID_PARAMETER_NAMES = tuple(
    f"pid_rate.{axis}_{term}"
    for axis in ("roll", "pitch", "yaw")
    for term in ("kp", "ki", "kd", "kff")
)


class ParameterReader(Protocol):
    """The small part of cflib's parameter API needed by these helpers."""

    def get_value(self, name: str) -> str | float: ...


def read_parameter(cf, name: str, timeout_s: float) -> str | float:
    """``name``'s value, giving cflib's parameter download at most ``timeout_s``.

    cflib's ``get_value`` ignores its own timeout and waits up to 60 s for that
    download, which on the lab's drone ends about a second after connecting.
    """
    deadline = time.monotonic() + timeout_s
    while not cf.param.is_updated:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"parameter values did not arrive within {timeout_s:g} s")
        time.sleep(0.01)
    return cf.param.get_value(name)


def read_rate_pid_parameters(param: ParameterReader) -> dict[str, float]:
    """Read the rate PID gains currently active on a connected STM32."""
    return {name: float(param.get_value(name)) for name in RATE_PID_PARAMETER_NAMES}
