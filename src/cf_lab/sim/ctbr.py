"""Batched, differentiable collective-thrust / body-rate quadrotor dynamics.

CTBR is the action space where the policy emits a *setpoint* -- collective
thrust plus a body-rate reference -- and the vehicle's own rate controller
tracks it. That is what makes a 100 Hz neural controller viable: the fast loop
(rate -> torque, which needs 500-1000 Hz) stays onboard next to the gyro, and
the policy sits outside it. A policy that emits torques directly *is* the rate
loop, and so inherits its bandwidth requirement.

Everything here is pure ``torch``, batched over the leading dimension, and
differentiable end to end, so it can be used for analytic-gradient training
(BPTT through a rollout) as well as ordinary RL.

Layout
------
``state``   ``[px, py, pz, vx, vy, vz, qw, qx, qy, qz]``  (10)
``action``  ``[thrust_N, wx, wy, wz]``                     (4)

World frame is z-up (FLU/ENU); thrust acts along the body z axis; the
quaternion is ``(w, x, y, z)`` mapping body -> world.

.. warning::
   **Rate tracking here is ideal: the commanded body rate is applied
   instantly.** A real vehicle's rate loop has 20-40 ms of lag, and *this
   omission is the single most common cause of sim-to-real failure for CTBR
   policies* -- the policy learns to rely on rate changes the airframe cannot
   actually deliver, then oscillates or diverges on hardware in a way that
   looks like a tuning problem but is a missing subsystem. If transfer is poor,
   model it before touching anything else: add ``w`` to the state, integrate
   ``w_dot = (w_ref - w) / tau`` with ``tau`` identified from a rate-step flight
   log, and feed the measured ``w`` into the observation so the policy can see
   that it has not yet reached its own setpoint.
"""

from __future__ import annotations

import torch

STATE_DIM = 10
ACTION_DIM = 4
OBS_DIM = 11

STATE_NAMES = ("px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz")
ACTION_NAMES = ("thrust", "wx", "wy", "wz")
ACTION_UNITS = ("N", "rad/s", "rad/s", "rad/s")
# Body-frame and yaw-decomposed, which makes the whole vector invariant to
# heading: the policy does not have to relearn the same hover at every yaw.
OBS_NAMES = (
    "ex_b", "ey_b", "ez_b",      # position error, body frame
    "vx_b", "vy_b", "vz_b",      # velocity, body frame
    "gx_b", "gy_b", "gz_b",      # world up in body frame -> roll/pitch
    "sin_yaw_err", "cos_yaw_err",
)

GRAVITY_M_S2 = 9.8066
_EPS = 1.0e-9


def quat_to_rotation(q: torch.Tensor) -> torch.Tensor:
    """``(B,4)`` quaternion ``(w,x,y,z)`` -> ``(B,3,3)`` body-to-world matrix."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1),
        ),
        dim=-2,
    )


def quat_derivative(q: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """``q_dot = 0.5 * q (x) [0, omega_body]``, the attitude kinematics."""
    w, x, y, z = q.unbind(-1)
    p, r, s = omega.unbind(-1)
    return 0.5 * torch.stack(
        (
            -x * p - y * r - z * s,
            w * p + y * s - z * r,
            w * r + z * p - x * s,
            w * s + x * r - y * p,
        ),
        dim=-1,
    )


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a (x) b``, both ``(...,4)`` in ``(w,x,y,z)``."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def yaw_sin_cos(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Heading as ``(sin, cos)``, computed without ``atan2``.

    Returning the pair rather than the angle is deliberate: a raw yaw is
    discontinuous at +/-pi, and a policy commanded to turn through that wrap
    reacts violently to a jump that is not physical.
    """
    w, x, y, z = q.unbind(-1)
    sin_raw = 2.0 * (w * z + x * y)
    cos_raw = 1.0 - 2.0 * (y * y + z * z)
    norm = torch.sqrt(sin_raw * sin_raw + cos_raw * cos_raw).clamp_min(_EPS)
    return sin_raw / norm, cos_raw / norm


def derivative(
    state: torch.Tensor,
    action: torch.Tensor,
    mass: torch.Tensor | float,
    gravity: float = GRAVITY_M_S2,
    disturbance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Continuous-time state derivative for a batch of vehicles.

    ``mass`` may be a scalar or a ``(B,)`` tensor, so a batch can carry its own
    randomized parameters. ``disturbance`` is an optional ``(B,3)`` linear
    acceleration added in the world frame.
    """
    v = state[..., 3:6]
    q = state[..., 6:10]
    thrust = action[..., 0]
    omega = action[..., 1:4]

    if not torch.is_tensor(mass):
        mass = torch.as_tensor(mass, dtype=state.dtype, device=state.device)
    body_z = quat_to_rotation(q)[..., :, 2]

    accel = body_z * (thrust / mass).unsqueeze(-1)
    accel = accel - torch.as_tensor(
        (0.0, 0.0, gravity), dtype=state.dtype, device=state.device
    )
    if disturbance is not None:
        accel = accel + disturbance
    # The commanded rate is the rate. See the module-level warning.
    return torch.cat((v, accel, quat_derivative(q, omega)), dim=-1)


def rk4_step(
    state: torch.Tensor,
    action: torch.Tensor,
    dt_s: float,
    mass: torch.Tensor | float,
    gravity: float = GRAVITY_M_S2,
    disturbance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance one zero-order-held step and renormalize the quaternion."""
    def f(s):
        return derivative(s, action, mass, gravity, disturbance)

    k1 = f(state)
    k2 = f(state + 0.5 * dt_s * k1)
    k3 = f(state + 0.5 * dt_s * k2)
    k4 = f(state + dt_s * k3)
    nxt = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    q = nxt[..., 6:10]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return torch.cat((nxt[..., 0:6], q), dim=-1)


def observe(
    state: torch.Tensor,
    position_ref: torch.Tensor | None = None,
    yaw_ref: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(B,10)`` state -> ``(B,11)`` policy observation.

    Position error and velocity are rotated into the body frame and the heading
    is carried as a ``(sin, cos)`` error, so the observation is invariant to
    yaw. Attitude enters as world-up-in-body (``g_b``), which is continuous
    everywhere and carries exactly the roll/pitch information a hover policy
    needs -- unlike Euler angles, which are discontinuous and singular.
    """
    p = state[..., 0:3]
    v = state[..., 3:6]
    q = state[..., 6:10]
    if position_ref is None:
        position_ref = torch.zeros_like(p)

    rot = quat_to_rotation(q)
    world_to_body = rot.transpose(-1, -2)
    err_b = torch.einsum("...ij,...j->...i", world_to_body, p - position_ref)
    vel_b = torch.einsum("...ij,...j->...i", world_to_body, v)
    up_b = rot[..., 2, :]  # third row = world z expressed in the body frame

    sin_yaw, cos_yaw = yaw_sin_cos(q)
    if yaw_ref is None:
        sin_ref = torch.zeros_like(sin_yaw)
        cos_ref = torch.ones_like(cos_yaw)
    else:
        sin_ref, cos_ref = torch.sin(yaw_ref), torch.cos(yaw_ref)
    sin_err = sin_ref * cos_yaw - cos_ref * sin_yaw
    cos_err = cos_ref * cos_yaw + sin_ref * sin_yaw

    return torch.cat(
        (err_b, vel_b, up_b, sin_err.unsqueeze(-1), cos_err.unsqueeze(-1)), dim=-1
    )


def hover_state(batch_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """The equilibrium: at the origin, level, at rest."""
    state = torch.zeros(batch_size, STATE_DIM, device=device, dtype=dtype)
    state[:, 6] = 1.0
    return state
