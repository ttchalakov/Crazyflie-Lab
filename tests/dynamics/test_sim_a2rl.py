"""Physics of the batched torque-input simulator cf_lab.sim.a2rl."""
import numpy as np
import pytest
import torch

from cf_lab.sim import a2rl, ctbr

F64 = torch.float64
G = ctbr.GRAVITY_M_S2


def t64(values):
    return torch.as_tensor(np.asarray(values, dtype=float), dtype=F64)


def test_hover_is_a_fixed_point_with_thrust_mg():
    state = a2rl.hover_state(3, dtype=F64)
    action = t64([[a2rl.MASS_KG * G, 0, 0, 0]] * 3)
    assert torch.equal(a2rl.derivative(state, action), torch.zeros_like(state))
    nxt = state
    for _ in range(100):
        nxt = a2rl.rk4_step(nxt, action, 0.01)
    assert torch.allclose(nxt, state, atol=1e-12)


def test_euler_to_rotation_is_zyx_body_to_world(rng, geo):
    eta = rng.uniform(-np.pi, np.pi, (32, 3))
    eta[:, 1] = rng.uniform(-1.5, 1.5, 32)
    rot = a2rl.euler_to_rotation(t64(eta)).numpy()
    for (roll, pitch, yaw), r in zip(eta, rot):
        np.testing.assert_allclose(r, geo.euler_matrix(roll, pitch, yaw), atol=1e-12)
    np.testing.assert_allclose(rot @ rot.transpose(0, 2, 1), np.broadcast_to(np.eye(3), rot.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-12)


def test_euler_derivative_reproduces_body_rates(random_states, geo):
    # R(eta(t)) must evolve as R_dot = R [omega]_x with omega in the body frame.
    h = 1e-6
    eta = t64(random_states[:, 6:9])
    omega = t64(random_states[:, 9:12])
    eta_dot = a2rl.euler_derivative(eta, omega)
    r_dot = (a2rl.euler_to_rotation(eta + h * eta_dot) - a2rl.euler_to_rotation(eta - h * eta_dot)) / (2 * h)
    rot = a2rl.euler_to_rotation(eta).numpy()
    for r, rd, w in zip(rot, r_dot.numpy(), random_states[:, 9:12]):
        np.testing.assert_allclose(rd, r @ geo.skew(w), atol=1e-6)


def test_euler_derivative_stays_finite_at_pitch_singularity():
    eta = t64([[0.3, np.pi / 2, 0.1], [0.3, -np.pi / 2, 0.1]])
    omega = t64([[1.0, 2.0, 3.0]] * 2)
    assert torch.isfinite(a2rl.euler_derivative(eta, omega)).all()


def test_derivative_rigid_body_matches_newton_euler(random_states, random_actions, geo):
    params = dict(mass=0.05, Ixx=21e-6, Iyy=24e-6, Izz=38e-6, arm_length=0.046)
    xdot = a2rl.derivative(t64(random_states), t64(random_actions), **params).numpy()
    inertia = np.diag([params["Ixx"], params["Iyy"], params["Izz"]])
    for s, u, d in zip(random_states, random_actions, xdot):
        rot = geo.euler_matrix(*s[6:9])
        accel = rot[:, 2] * u[0] / params["mass"] - np.array([0, 0, G])
        moment = np.array([params["arm_length"] * u[1], params["arm_length"] * u[2], u[3]])
        w = s[9:12]
        omega_dot = np.linalg.solve(inertia, moment - np.cross(w, inertia @ w))
        np.testing.assert_allclose(d[0:3], s[3:6], atol=1e-12)
        np.testing.assert_allclose(d[3:6], accel, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(d[9:12], omega_dot, rtol=1e-10)


@pytest.mark.parametrize("axis, column", [(1, 9), (2, 10)])
def test_roll_pitch_torque_is_a_differential_force_at_the_arm(axis, column):
    action = torch.zeros(1, 4, dtype=F64)
    action[0, axis] = 0.01  # N
    xdot = a2rl.derivative(a2rl.hover_state(1, dtype=F64), action)
    inertia = a2rl.IXX_KG_M2 if axis == 1 else a2rl.IYY_KG_M2
    assert xdot[0, column].item() == pytest.approx(0.050 * 0.01 / inertia, rel=1e-12)


def test_yaw_torque_is_a_moment_in_newton_metres():
    action = t64([[0, 0, 0, 1e-5]])
    xdot = a2rl.derivative(a2rl.hover_state(1, dtype=F64), action)
    assert xdot[0, 11].item() == pytest.approx(1e-5 / a2rl.IZZ_KG_M2, rel=1e-12)


def test_positive_roll_tilts_thrust_towards_negative_y():
    # Constant attitude, no rates: the acceleration is constant, so RK4 is exact.
    roll, thrust, mass, dt, steps = 0.2, 0.5, 0.04, 0.01, 50
    state = torch.zeros(1, 12, dtype=F64)
    state[0, 6] = roll
    action = t64([[thrust, 0, 0, 0]])
    for _ in range(steps):
        state = a2rl.rk4_step(state, action, dt, mass=mass)
    t = dt * steps
    ay, az = -np.sin(roll) * thrust / mass, np.cos(roll) * thrust / mass - G
    np.testing.assert_allclose(state[0, 0:6].numpy(), [0, 0.5 * ay * t**2, 0.5 * az * t**2, 0, ay * t, az * t], atol=1e-10)


def test_rk4_step_is_fourth_order():
    state = t64([[0, 0, 1, 0.5, -0.3, 0.2, 0.3, -0.2, 0.4, 2.0, -1.5, 3.0]])
    action = t64([[0.5, 0.004, -0.003, 2e-5]])

    def rollout(n, total=0.4):
        s = state
        for _ in range(n):
            s = a2rl.rk4_step(s, action, total / n)
        return s

    coarse, mid, fine = rollout(8), rollout(16), rollout(32)
    order = torch.log2((coarse - mid).norm() / (mid - fine).norm()).item()
    assert 3.5 < order < 4.5


def test_per_vehicle_parameters_broadcast_over_the_batch(random_states, random_actions):
    n = len(random_states)
    params = {
        "mass": torch.linspace(0.03, 0.06, n, dtype=F64),
        "Ixx": torch.linspace(15e-6, 25e-6, n, dtype=F64),
        "Iyy": torch.linspace(16e-6, 26e-6, n, dtype=F64),
        "Izz": torch.linspace(28e-6, 40e-6, n, dtype=F64),
        "arm_length": torch.linspace(0.04, 0.06, n, dtype=F64),
    }
    batched = a2rl.rk4_step(t64(random_states), t64(random_actions), 0.002, **params)
    for i in range(n):
        one = a2rl.rk4_step(
            t64(random_states[i : i + 1]), t64(random_actions[i : i + 1]), 0.002,
            **{k: v[i].item() for k, v in params.items()},
        )
        torch.testing.assert_close(batched[i : i + 1], one, rtol=1e-12, atol=1e-12)


def test_disturbance_adds_world_frame_acceleration(random_states, random_actions, rng):
    disturbance = t64(rng.normal(0, 1, (len(random_states), 3)))
    state, action = t64(random_states), t64(random_actions)
    delta = a2rl.derivative(state, action, disturbance=disturbance) - a2rl.derivative(state, action)
    expected = torch.zeros_like(delta)
    expected[:, 3:6] = disturbance
    torch.testing.assert_close(delta, expected)


def test_rk4_step_is_differentiable_in_state_and_parameters():
    state = t64([[0.1, -0.2, 1.0, 0.3, 0.1, -0.2, 0.2, -0.1, 0.5, 1.0, -2.0, 0.5]]).requires_grad_()
    action = t64([[0.45, 0.003, -0.002, 1e-5]]).requires_grad_()
    mass = t64([0.04]).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda s, a, m: a2rl.rk4_step(s, a, 0.01, mass=m), (state, action, mass)
    )


def test_observe_layout_is_ctbr_observation_plus_body_rates(random_states, geo):
    ref = t64(np.full((len(random_states), 3), 0.3))
    yaw_ref = t64(np.full(len(random_states), 0.7))
    obs = a2rl.observe(t64(random_states), ref, yaw_ref)
    quats = np.array([geo.euler_quat(*s[6:9]) for s in random_states])
    ctbr_state = t64(np.column_stack((random_states[:, 0:6], quats)))
    assert obs.shape == (len(random_states), 14)
    torch.testing.assert_close(obs[:, :11], ctbr.observe(ctbr_state, ref, yaw_ref))
    torch.testing.assert_close(obs[:, 11:], t64(random_states[:, 9:12]))


def test_observe_is_invariant_to_heading(random_states):
    alpha = 1.1
    c, s = np.cos(alpha), np.sin(alpha)
    turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    ref = np.array([0.2, -0.4, 1.0])
    rotated = random_states.copy()
    rotated[:, 0:3] = random_states[:, 0:3] @ turn.T
    rotated[:, 3:6] = random_states[:, 3:6] @ turn.T
    rotated[:, 8] += alpha
    yaw_ref = t64(np.full(len(random_states), 0.4))
    before = a2rl.observe(t64(random_states), t64(np.tile(ref, (16, 1))), yaw_ref)
    after = a2rl.observe(t64(rotated), t64(np.tile(turn @ ref, (16, 1))), yaw_ref + alpha)
    torch.testing.assert_close(before, after)


def test_observe_expresses_position_error_in_the_body_frame():
    state = torch.zeros(1, 12, dtype=F64)
    state[0, 0] = 1.0           # 1 m along world +x ...
    state[0, 8] = np.pi / 2     # ... with the nose pointing at world +y
    obs = a2rl.observe(state)
    torch.testing.assert_close(obs[0, 0:3], t64([0.0, -1.0, 0.0]))
    torch.testing.assert_close(obs[0, 6:9], t64([0.0, 0.0, 1.0]))
    torch.testing.assert_close(obs[0, 9:11], t64([-1.0, 0.0]))  # sin/cos of yaw_ref - yaw
