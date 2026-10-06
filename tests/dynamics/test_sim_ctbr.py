"""Physics of the batched collective-thrust/body-rate simulator cf_lab.sim.ctbr."""
import numpy as np
import torch

from cf_lab.sim import a2rl, ctbr

F64 = torch.float64
G = ctbr.GRAVITY_M_S2


def t64(values):
    return torch.as_tensor(np.asarray(values, dtype=float), dtype=F64)


def random_quats(rng, n=32):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_hover_is_a_fixed_point_with_thrust_mg():
    mass = 0.0393
    state = ctbr.hover_state(4, dtype=F64)
    action = t64([[mass * G, 0, 0, 0]] * 4)
    assert torch.equal(ctbr.derivative(state, action, mass), torch.zeros_like(state))
    nxt = state
    for _ in range(100):
        nxt = ctbr.rk4_step(nxt, action, 0.01, mass)
    assert torch.allclose(nxt, state, atol=1e-12)


def test_quat_to_rotation_is_orthonormal_and_rotates_like_the_quaternion(rng, geo):
    quats = random_quats(rng)
    rot = ctbr.quat_to_rotation(t64(quats)).numpy()
    np.testing.assert_allclose(rot @ rot.transpose(0, 2, 1), np.broadcast_to(np.eye(3), rot.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-12)
    v = np.array([0.3, -1.2, 0.7])
    for q, r in zip(quats, rot):
        conj = q * np.array([1, -1, -1, -1])
        rotated = geo.hamilton(geo.hamilton(q, np.concatenate(([0.0], v))), conj)[1:]
        np.testing.assert_allclose(r @ v, rotated, atol=1e-12)


def test_quat_multiply_composes_rotations_in_hamilton_order(rng):
    a, b = t64(random_quats(rng, 16)), t64(random_quats(rng, 16))
    torch.testing.assert_close(
        ctbr.quat_to_rotation(ctbr.quat_multiply(a, b)),
        ctbr.quat_to_rotation(a) @ ctbr.quat_to_rotation(b),
    )


def test_quat_and_euler_rotations_agree(rng, geo):
    eta = rng.uniform(-1.2, 1.2, (16, 3))
    quats = t64([geo.euler_quat(*e) for e in eta])
    torch.testing.assert_close(ctbr.quat_to_rotation(quats), a2rl.euler_to_rotation(t64(eta)))


def test_quat_derivative_reproduces_body_rates(rng, geo):
    h = 1e-6
    q = t64(random_quats(rng, 16))
    omega = rng.uniform(-5, 5, (16, 3))
    q_dot = ctbr.quat_derivative(q, t64(omega))
    r_dot = (ctbr.quat_to_rotation(q + h * q_dot) - ctbr.quat_to_rotation(q - h * q_dot)) / (2 * h)
    for r, rd, w in zip(ctbr.quat_to_rotation(q).numpy(), r_dot.numpy(), omega):
        np.testing.assert_allclose(rd, r @ geo.skew(w), atol=1e-6)


def test_yaw_sin_cos_recovers_zyx_heading(rng, geo):
    eta = rng.uniform(-1.2, 1.2, (16, 3))
    eta[:, 2] = rng.uniform(-np.pi, np.pi, 16)
    sin_yaw, cos_yaw = ctbr.yaw_sin_cos(t64([geo.euler_quat(*e) for e in eta]))
    torch.testing.assert_close(sin_yaw, t64(np.sin(eta[:, 2])))
    torch.testing.assert_close(cos_yaw, t64(np.cos(eta[:, 2])))


def test_free_fall_matches_the_analytic_parabola():
    state = ctbr.hover_state(1, dtype=F64)
    state[0, 5] = 1.0
    action = torch.zeros(1, 4, dtype=F64)
    for _ in range(50):
        state = ctbr.rk4_step(state, action, 0.02, 0.04)
    t = 1.0
    np.testing.assert_allclose(state[0, 2].item(), t - 0.5 * G * t**2, atol=1e-12)
    np.testing.assert_allclose(state[0, 5].item(), 1.0 - G * t, atol=1e-12)


def test_constant_body_rate_yields_the_analytic_rotation(geo):
    w, t, steps = 2.5, 1.3, 130
    state = ctbr.hover_state(1, dtype=F64)
    action = t64([[0.0, 0.0, 0.0, w]])
    for _ in range(steps):
        state = ctbr.rk4_step(state, action, t / steps, 0.04)
    q = state[0, 6:10].numpy()
    expected = geo.axis_quat((0, 0, 1), w * t)
    np.testing.assert_allclose(q, expected, atol=1e-8)
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12


def test_rk4_step_is_fourth_order():
    q0 = np.array([0.95, 0.2, -0.1, 0.2])
    state = t64([[0, 0, 1, 0.5, -0.3, 0.2, *(q0 / np.linalg.norm(q0))]])
    action = t64([[0.5, 2.0, -1.5, 3.0]])

    def rollout(n, total=0.4):
        s = state
        for _ in range(n):
            s = ctbr.rk4_step(s, action, total / n, 0.04)
        return s

    coarse, mid, fine = rollout(8), rollout(16), rollout(32)
    order = torch.log2((coarse - mid).norm() / (mid - fine).norm()).item()
    assert 3.5 < order < 4.5


def test_per_vehicle_mass_broadcasts_over_the_batch(rng):
    n = 8
    state = t64(np.column_stack((rng.normal(size=(n, 6)), random_quats(rng, n))))
    action = t64(np.column_stack((rng.uniform(0.2, 0.6, n), rng.normal(size=(n, 3)))))
    masses = torch.linspace(0.03, 0.06, n, dtype=F64)
    batched = ctbr.rk4_step(state, action, 0.01, masses)
    for i in range(n):
        one = ctbr.rk4_step(state[i : i + 1], action[i : i + 1], 0.01, masses[i].item())
        torch.testing.assert_close(batched[i : i + 1], one)


def test_rk4_step_is_differentiable_in_state_action_and_mass():
    state = t64([[0.1, -0.2, 1.0, 0.3, 0.1, -0.2, 0.9, 0.1, -0.2, 0.3]]).requires_grad_()
    action = t64([[0.45, 0.5, -1.0, 0.3]]).requires_grad_()
    mass = t64([0.04]).requires_grad_()
    assert torch.autograd.gradcheck(lambda s, a, m: ctbr.rk4_step(s, a, 0.01, m), (state, action, mass))


def test_observe_is_invariant_to_heading(rng, geo):
    n, alpha = 16, -2.3
    p, v = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    quats = random_quats(rng, n)
    ref, yaw_ref = np.array([0.5, 0.1, 1.2]), 0.4
    turn, q_turn = geo.rot_z(alpha), geo.axis_quat((0, 0, 1), alpha)
    rotated_q = np.array([geo.hamilton(q_turn, q) for q in quats])
    before = ctbr.observe(t64(np.column_stack((p, v, quats))), t64(np.tile(ref, (n, 1))), t64(np.full(n, yaw_ref)))
    after = ctbr.observe(
        t64(np.column_stack((p @ turn.T, v @ turn.T, rotated_q))),
        t64(np.tile(turn @ ref, (n, 1))),
        t64(np.full(n, yaw_ref + alpha)),
    )
    torch.testing.assert_close(before, after)


def test_observe_layout_at_a_known_pose(geo):
    # Rolled 90 deg about body x, so world up lies along body +y.
    q = geo.axis_quat((1, 0, 0), np.pi / 2)
    state = t64([[0, 0, 0, 0, 0, 2.0, *q]])
    obs = ctbr.observe(state, t64([[0, 0, 1.0]])).numpy()[0]
    assert obs.shape == (11,)
    np.testing.assert_allclose(obs[0:3], [0, -1.0, 0], atol=1e-12)   # p - ref, the target is 1 m up
    np.testing.assert_allclose(obs[3:6], [0, 2.0, 0], atol=1e-12)    # climbing = body +y
    np.testing.assert_allclose(obs[6:9], [0, 1.0, 0], atol=1e-12)    # world up in the body frame
    np.testing.assert_allclose(obs[9:11], [0, 1.0], atol=1e-12)
