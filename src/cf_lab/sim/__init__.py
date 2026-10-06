"""Differentiable batched vehicle simulation.

Imports ``torch``, which is slow to load, so this subpackage is deliberately
not imported by ``cf_lab/__init__.py``.

``ctbr``'s functions are re-exported here; ``a2rl`` uses the same names, so it
is reached as ``cf_lab.sim.a2rl``.
"""

from cf_lab.sim import a2rl
from cf_lab.sim.ctbr import (
    ACTION_DIM,
    ACTION_NAMES,
    ACTION_UNITS,
    GRAVITY_M_S2,
    OBS_DIM,
    OBS_NAMES,
    STATE_DIM,
    STATE_NAMES,
    derivative,
    hover_state,
    observe,
    quat_derivative,
    quat_multiply,
    quat_to_rotation,
    rk4_step,
    yaw_sin_cos,
)

__all__ = [
    "a2rl",
    "ACTION_DIM", "ACTION_NAMES", "ACTION_UNITS", "GRAVITY_M_S2",
    "OBS_DIM", "OBS_NAMES", "STATE_DIM", "STATE_NAMES",
    "derivative", "hover_state", "observe", "quat_derivative",
    "quat_multiply", "quat_to_rotation", "rk4_step", "yaw_sin_cos",
]
