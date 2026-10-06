"""Batched, differentiable Newton-Euler quadrotor dynamics with torque inputs.

A2RL is the action space where the policy emits collective thrust and body
torques directly. Unlike CTBR (``ctbr.py``) nothing sits between the action and
the airframe, so mass, inertia and arm length all matter, and the policy *is*
the rate loop, with its 500-1000 Hz bandwidth requirement.

Everything here is pure ``torch``, batched over the leading dimension, and
differentiable end to end.

Layout
------
``state``   ``[x, y, z, vx, vy, vz, roll, pitch, yaw, p, q, r]``  (12)
``action``  ``[T, tau_x, tau_y, tau_z]``                          (4)

Units are SI: m, m/s, rad, rad/s, and ``T`` in N. ``tau_x`` and ``tau_y`` are
*differential forces* in N acting at ``arm_length``, so the roll and pitch
moments are ``arm_length * tau_x`` and ``arm_length * tau_y`` in N m; ``tau_z``
is already the yaw moment in N m.

World frame is z-up (FLU/ENU); thrust acts along the body z axis; the attitude
is ZYX Euler angles ``(roll, pitch, yaw)`` mapping body -> world, singular at
pitch = +/- pi/2; ``p, q, r`` are body-frame rates.
"""

from __future__ import annotations

import torch

from cf_lab.sim.ctbr import GRAVITY_M_S2

STATE_DIM = 12
ACTION_DIM = 4
OBS_DIM = 14

STATE_NAMES = ("x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "p", "q", "r")
ACTION_NAMES = ("T", "tau_x", "tau_y", "tau_z")
ACTION_UNITS = ("N", "N", "N", "N m")  # tau_x, tau_y act at arm_length
# ctbr's heading-invariant observation plus the body rates.
OBS_NAMES = (
    "ex_b", "ey_b", "ez_b",      # position error, body frame
    "vx_b", "vy_b", "vz_b",      # velocity, body frame
    "gx_b", "gy_b", "gz_b",      # world up in body frame -> roll/pitch
    "sin_yaw_err", "cos_yaw_err",
    "p", "q", "r",               # body rates
)

# Crazyflie 2.1 Brushless firmware defaults; pass others for a particular
# airframe and payload.
MASS_KG = 0.0393        # CF_MASS, platform_defaults_cf21bl.h (bare airframe)
IXX_KG_M2 = 16.6e-6     # diagonal of CRAZYFLIE_INERTIA, controller_brescianini.c
IYY_KG_M2 = 16.6e-6
IZZ_KG_M2 = 29.3e-6
ARM_LENGTH_M = 0.050    # ARM_LENGTH, platform_defaults_cf21bl.h (centre to motor)

_COS_PITCH_MIN = 1.0e-6


def euler_to_rotation(eta: torch.Tensor) -> torch.Tensor:
    """``(B,3)`` ZYX ``(roll, pitch, yaw)`` -> ``(B,3,3)`` body-to-world matrix."""
    s_roll, s_pitch, s_yaw = torch.sin(eta).unbind(-1)
    c_roll, c_pitch, c_yaw = torch.cos(eta).unbind(-1)
    return torch.stack(
        (
            torch.stack((
                c_yaw * c_pitch,
                c_yaw * s_pitch * s_roll - s_yaw * c_roll,
                c_yaw * s_pitch * c_roll + s_yaw * s_roll,
            ), -1),
            torch.stack((
                s_yaw * c_pitch,
                s_yaw * s_pitch * s_roll + c_yaw * c_roll,
                s_yaw * s_pitch * c_roll - c_yaw * s_roll,
            ), -1),
            torch.stack((-s_pitch, c_pitch * s_roll, c_pitch * c_roll), -1),
        ),
        dim=-2,
    )


def euler_derivative(eta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """``eta_dot = W(eta) omega_body``, the attitude kinematics.

    ``cos(pitch)`` is held at least ``1e-6`` from zero, so a rollout through
    the singularity stays finite.
    """
    roll, pitch, _ = eta.unbind(-1)
    p, q, r = omega.unbind(-1)
    s_roll, c_roll = torch.sin(roll), torch.cos(roll)
    c_pitch = torch.cos(pitch)
    c_pitch = c_pitch.abs().clamp_min(_COS_PITCH_MIN).copysign(c_pitch)
    off_axis = q * s_roll + r * c_roll
    return torch.stack(
        (p + off_axis * torch.sin(pitch) / c_pitch, q * c_roll - r * s_roll, off_axis / c_pitch),
        dim=-1,
    )


def derivative(
    state: torch.Tensor,
    action: torch.Tensor,
    mass: torch.Tensor | float = MASS_KG,
    Ixx: torch.Tensor | float = IXX_KG_M2,
    Iyy: torch.Tensor | float = IYY_KG_M2,
    Izz: torch.Tensor | float = IZZ_KG_M2,
    arm_length: torch.Tensor | float = ARM_LENGTH_M,
    gravity: float = GRAVITY_M_S2,
    disturbance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Continuous-time state derivative for a batch of vehicles.

    Each vehicle parameter may be a scalar or a ``(B,)`` tensor, so a batch can
    carry its own randomized parameters. ``disturbance`` is an optional
    ``(B,3)`` linear acceleration added in the world frame.
    """
    v = state[..., 3:6]
    eta = state[..., 6:9]
    omega = state[..., 9:12]
    p, q, r = omega.unbind(-1)
    thrust, tau_x, tau_y, tau_z = action.unbind(-1)

    body_z = euler_to_rotation(eta)[..., :, 2]
    accel = body_z * (thrust / mass).unsqueeze(-1)
    accel = accel - torch.as_tensor(
        (0.0, 0.0, gravity), dtype=state.dtype, device=state.device
    )
    if disturbance is not None:
        accel = accel + disturbance
    # J omega_dot = M - omega x J omega, with M = (L tau_x, L tau_y, tau_z).
    omega_dot = torch.stack(
        (
            ((Iyy - Izz) * q * r + arm_length * tau_x) / Ixx,
            ((Izz - Ixx) * p * r + arm_length * tau_y) / Iyy,
            ((Ixx - Iyy) * p * q + tau_z) / Izz,
        ),
        dim=-1,
    )
    return torch.cat((v, accel, euler_derivative(eta, omega), omega_dot), dim=-1)


def rk4_step(
    state: torch.Tensor,
    action: torch.Tensor,
    dt_s: float,
    mass: torch.Tensor | float = MASS_KG,
    Ixx: torch.Tensor | float = IXX_KG_M2,
    Iyy: torch.Tensor | float = IYY_KG_M2,
    Izz: torch.Tensor | float = IZZ_KG_M2,
    arm_length: torch.Tensor | float = ARM_LENGTH_M,
    gravity: float = GRAVITY_M_S2,
    disturbance: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance one zero-order-held step. Angles are not wrapped."""
    def f(s):
        return derivative(s, action, mass, Ixx, Iyy, Izz, arm_length, gravity, disturbance)

    k1 = f(state)
    k2 = f(state + 0.5 * dt_s * k1)
    k3 = f(state + 0.5 * dt_s * k2)
    k4 = f(state + dt_s * k3)
    return state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def observe(
    state: torch.Tensor,
    position_ref: torch.Tensor | None = None,
    yaw_ref: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(B,12)`` state -> ``(B,14)`` policy observation.

    ``ctbr.observe``'s yaw-invariant observation, plus the body rates: with
    torque actions the policy is the rate loop, so it must see the rates.
    """
    pos = state[..., 0:3]
    v = state[..., 3:6]
    eta = state[..., 6:9]
    omega = state[..., 9:12]
    if position_ref is None:
        position_ref = torch.zeros_like(pos)

    rot = euler_to_rotation(eta)
    world_to_body = rot.transpose(-1, -2)
    err_b = torch.einsum("...ij,...j->...i", world_to_body, pos - position_ref)
    vel_b = torch.einsum("...ij,...j->...i", world_to_body, v)
    up_b = rot[..., 2, :]  # third row = world z expressed in the body frame

    yaw_err = -eta[..., 2] if yaw_ref is None else yaw_ref - eta[..., 2]
    return torch.cat(
        (
            err_b, vel_b, up_b,
            torch.sin(yaw_err).unsqueeze(-1), torch.cos(yaw_err).unsqueeze(-1),
            omega,
        ),
        dim=-1,
    )


def hover_state(batch_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """The equilibrium: at the origin, level, at rest. Hold it with ``T = m g``."""
    return torch.zeros(batch_size, STATE_DIM, device=device, dtype=dtype)
