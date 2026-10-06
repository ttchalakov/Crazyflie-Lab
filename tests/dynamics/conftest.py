"""Shared helpers for the quadrotor-dynamics tests: rotations built from first principles."""
from types import SimpleNamespace

import numpy as np
import pytest


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_matrix(roll, pitch, yaw):
    """Body-to-world ZYX rotation, Rz(yaw) Ry(pitch) Rx(roll)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def axis_quat(axis, angle):
    """Unit quaternion (w, x, y, z) for a rotation of ``angle`` about a unit ``axis``."""
    half = 0.5 * angle
    return np.concatenate(([np.cos(half)], np.sin(half) * np.asarray(axis, dtype=float)))


def hamilton(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def euler_quat(roll, pitch, yaw):
    """Quaternion of the same ZYX rotation as ``euler_matrix``."""
    q = hamilton(axis_quat((0, 0, 1), yaw), axis_quat((0, 1, 0), pitch))
    return hamilton(q, axis_quat((1, 0, 0), roll))


@pytest.fixture
def geo():
    """The rotation helpers above, for tests (importlib mode cannot import conftest)."""
    return SimpleNamespace(
        rot_x=rot_x, rot_y=rot_y, rot_z=rot_z, euler_matrix=euler_matrix, skew=skew,
        axis_quat=axis_quat, hamilton=hamilton, euler_quat=euler_quat,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def random_states(rng):
    """(16,12) float64 A2RL states: moderate attitudes and rates, away from pitch +/- pi/2."""
    n = 16
    state = np.zeros((n, 12))
    state[:, 0:3] = rng.uniform(-2, 2, (n, 3))
    state[:, 3:6] = rng.uniform(-1.5, 1.5, (n, 3))
    state[:, 6] = rng.uniform(-1.0, 1.0, n)
    state[:, 7] = rng.uniform(-1.2, 1.2, n)
    state[:, 8] = rng.uniform(-3.0, 3.0, n)
    state[:, 9:12] = rng.uniform(-6, 6, (n, 3))
    return state


@pytest.fixture
def random_actions(rng):
    """(16,4) float64 A2RL actions [T, tau_x, tau_y, tau_z] around hover."""
    n = 16
    return np.column_stack((
        rng.uniform(0.1, 0.8, n),
        rng.uniform(-0.05, 0.05, n),
        rng.uniform(-0.05, 0.05, n),
        rng.uniform(-2e-4, 2e-4, n),
    ))
