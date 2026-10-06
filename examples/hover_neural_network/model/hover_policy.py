"""Train and export the neural hover policy: 11D observation -> CTBR action.

The network input is the yaw-invariant hover observation from
``cf_lab.sim.ctbr`` -- body-frame position error and velocity, world-up in the
body frame, and the heading error as ``(sin, cos)``. The output is a
collective-thrust / body-rate setpoint ``[T, wx, wy, wz]``, which the vehicle's
own 500 Hz rate loop tracks. That is what lets the policy run at 100 Hz: the
fast loop stays onboard beside the gyro, and the policy sits outside it.

Training runs on the raw SI observation -- no input codec, no clamping, no
teacher, no pretraining, one loop. The uint8 affine that GAP8 deployment needs
comes from STATE_LIMITS and is folded into the first Linear at export time,
which is exact.

``export_nemo_onnx`` writes no C. The input limits and action affine the
firmware needs are the constants below, mirrored by hand in
``gap8/hover_neural_network.c``; the one weight-dependent value, NEMO's output
quantum, goes to ``output_epsilon.txt`` for the GAP8 Makefile. Nothing here
generates source.

This file deliberately owns the experiment constants directly. Edit the values
in the section below, train again, then rerun NEMO/DORY.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import tempfile
from pathlib import Path

from cf_lab.sim import ctbr


# ---------------------------------------------------------------------------
# Vehicle and action constants
# ---------------------------------------------------------------------------

# The assembled AI-deck vehicle's mass, estimated rather than weighed: weigh
# the final flight stack before treating it as immutable.
#
# No inertia, arm length or thrust-to-torque ratio: under CTBR the policy
# commands body rates, so the torque-to-angular-acceleration path lives entirely
# inside the onboard rate controller.
MASS_KG = 0.0518
HOVER_THRUST_N = MASS_KG * ctbr.GRAVITY_M_S2

# The network emits a dimensionless action in [-1, 1] that is ZERO at hover, so
# it learns a small correction rather than a large bias -- which also quantizes
# far better at WEIGHT_BITS.
#   T     = HOVER_THRUST_N * (1 + THRUST_DELTA_FRACTION * a0)
#   w_ref = MAX_BODY_RATE_RAD_S * a[1:]
THRUST_DELTA_FRACTION = 0.5
MAX_BODY_RATE_RAD_S = (4.0, 4.0, 2.0)

ACTION_OFFSET = [HOVER_THRUST_N, 0.0, 0.0, 0.0]
ACTION_SCALE = [HOVER_THRUST_N * THRUST_DELTA_FRACTION, *MAX_BODY_RATE_RAD_S]


# ---------------------------------------------------------------------------
# Observation, network, and quantization constants
# ---------------------------------------------------------------------------

DT_S = 0.01  # 100 Hz; one action is zero-order-held through one RK4 step.

HIDDEN_DIMS = [64, 64]
WEIGHT_BITS = 7
ACTIVATION_BITS = 8

# Per-channel input scale for deployment ONLY -- training runs on the raw SI
# observation and never sees this. DORY feeds the network a uint8 tensor, so
# something has to map SI onto [0,1]; export folds this affine into fc1 exactly
# and gap8/hover_neural_network.c holds the same eleven numbers for the encoder.
#
# Declared rather than measured, because measuring buys nothing: the last five
# channels are components of unit vectors and are bounded by 1 by construction,
# and for the rest the quantum these limits imply (2*0.6/255 = 4.7 mm of
# position, 9.4 mm/s of velocity) already sits at the observation noise the
# policy trains through. A state outside the box saturates rather than wrapping,
# which reads to the policy as "very far", so the correction stays correct.
#         ex    ey    ez    vx    vy    vz   upx  upy  upz  sin  cos
STATE_LIMITS = [0.6, 0.6, 0.5, 1.2, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = MODEL_DIR / "hover_policy.pt"
CALIBRATION_PATH = MODEL_DIR / "hover_policy_calibration.npy"


# ---------------------------------------------------------------------------
# Training, randomization, and evaluation constants
# ---------------------------------------------------------------------------

SEED = 7

# One stage: backprop through the differentiable rollout from a randomly
# initialised network. No behaviour-cloning pretrain and no teacher -- the
# horizon curriculum is what keeps the early epochs tractable, since a short
# horizon is a short backward graph.
# The learning rate and the epoch count are one setting, not two: change them
# together. The curriculum tracks them, staying at 30% of the run, so the
# horizon still finishes ramping well before the last third settles the policy.
ROLLOUT_EPOCHS = 250
ROLLOUT_BATCH_SIZE = 1024
ROLLOUT_MIN_HORIZON_S = 0.5
ROLLOUT_MAX_HORIZON_S = 3.0
ROLLOUT_CURRICULUM_EPOCHS = 75
ROLLOUT_LEARNING_RATE = 1.0e-3
GRADIENT_CLIP_NORM = 5.0

# Uniform half-ranges around hover. One quarter of each batch is sampled from a
# much tighter copy of the same box.
INITIAL_POSITION_HALF_RANGE_M = (0.20, 0.20, 0.15)
INITIAL_VELOCITY_HALF_RANGE_M_S = (0.30, 0.30, 0.25)
INITIAL_TILT_HALF_RANGE_RAD = 0.25
INITIAL_YAW_HALF_RANGE_RAD = math.pi
NEAR_EQUILIBRIUM_FRACTION = 0.25
NEAR_EQUILIBRIUM_SCALE = 0.2

MASS_RANDOMIZATION_RELATIVE = 0.08

OBSERVATION_NOISE_STD = [
    0.004, 0.004, 0.004,      # body-frame position error, m
    0.010, 0.010, 0.010,      # body-frame velocity, m/s
    0.004, 0.004, 0.004,      # world-up direction cosines
    0.004, 0.004,             # heading sin/cos
]
PROCESS_ACCELERATION_NOISE_STD = [0.08, 0.08, 0.08]
NORMALIZED_ACTION_NOISE_STD = [0.008, 0.010, 0.010, 0.008]

# Cost weights on physical error, not on observation entries.
LOSS_POSITION_WEIGHT = 1.0
LOSS_VELOCITY_WEIGHT = 0.15
LOSS_TILT_WEIGHT = 0.50
LOSS_YAW_WEIGHT = 0.20
ACTION_LOSS_WEIGHTS = [0.02, 0.01, 0.01, 0.01]
ACTION_RATE_LOSS_WEIGHT = 0.05
TERMINAL_LOSS_MULTIPLIER = 4.0

EVALUATION_ENVIRONMENTS = 512
EVALUATION_HORIZON_S = 6.0
SUCCESS_POSITION_M = 0.10
SUCCESS_VELOCITY_M_S = 0.25
SUCCESS_TILT_RAD = 0.15
SUCCESS_YAW_RAD = 0.20


def create_model():
    """Create the deployable policy solely from this file's top-level constants."""
    import torch.nn as nn

    class HoverPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            previous = ctbr.OBS_DIM
            for index, width in enumerate(HIDDEN_DIMS, start=1):
                setattr(self, "fc%d" % index, nn.Linear(previous, width))
                setattr(self, "relu%d" % index, nn.ReLU())
                previous = width
            self.fc_out = nn.Linear(previous, ctbr.ACTION_DIM)

        def forward(self, observation):
            value = observation
            for index in range(1, len(HIDDEN_DIMS) + 1):
                value = getattr(self, "relu%d" % index)(getattr(self, "fc%d" % index)(value))
            # Deliberately no tanh: DORY emits the signed final FC accumulator as
            # int32.  GAP8 multiplies by the exact NEMO epsilon, then clamps.
            return self.fc_out(value)

    return HoverPolicy()


def select_device(requested="auto"):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed=SEED):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _action_affine(dtype, device):
    import torch

    return (
        torch.as_tensor(ACTION_OFFSET, dtype=dtype, device=device),
        torch.as_tensor(ACTION_SCALE, dtype=dtype, device=device),
    )


def normalized_to_physical(normalized_action):
    """Decode the normalized network output to [T N, wx/wy/wz rad/s]."""
    import torch

    offset, scale = _action_affine(normalized_action.dtype, normalized_action.device)
    return offset + scale * torch.clamp(normalized_action, -1.0, 1.0)


def sample_initial_states(batch_size, device):
    """Draw a batch of 10D simulator states (position, velocity, quaternion)."""
    import torch

    def uniform(half_ranges):
        half = torch.tensor(half_ranges, device=device)
        return (2.0 * torch.rand(batch_size, len(half_ranges), device=device) - 1.0) * half

    position = uniform(INITIAL_POSITION_HALF_RANGE_M)
    velocity = uniform(INITIAL_VELOCITY_HALF_RANGE_M_S)
    if NEAR_EQUILIBRIUM_FRACTION > 0.0:
        near = torch.rand(batch_size, 1, device=device) < NEAR_EQUILIBRIUM_FRACTION
        position = torch.where(near, position * NEAR_EQUILIBRIUM_SCALE, position)
        velocity = torch.where(near, velocity * NEAR_EQUILIBRIUM_SCALE, velocity)

    # Tilt about a uniformly random horizontal axis, then yaw. Composing the two
    # this way keeps tilt magnitude independent of heading, so the sampled set is
    # itself yaw-symmetric -- matching the observation's own invariance.
    def half_angle_quat(angle, axis_x, axis_y, axis_z):
        half = 0.5 * angle
        sin_half = torch.sin(half)
        return torch.stack(
            (torch.cos(half), sin_half * axis_x, sin_half * axis_y, sin_half * axis_z),
            dim=-1,
        )

    tilt = (2.0 * torch.rand(batch_size, device=device) - 1.0) * INITIAL_TILT_HALF_RANGE_RAD
    direction = torch.rand(batch_size, device=device) * 2.0 * math.pi
    yaw = (2.0 * torch.rand(batch_size, device=device) - 1.0) * INITIAL_YAW_HALF_RANGE_RAD
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    quat = ctbr.quat_multiply(
        half_angle_quat(yaw, zero, zero, one),
        half_angle_quat(tilt, torch.cos(direction), torch.sin(direction), zero),
    )
    return torch.cat((position, velocity, quat), dim=-1)


def sample_vehicle_parameters(batch_size, device, randomize=True):
    """Just the mass: CTBR hides inertia and geometry behind the rate loop."""
    import torch

    mass = torch.full((batch_size,), float(MASS_KG), device=device)
    if randomize and MASS_RANDOMIZATION_RELATIVE:
        mass = mass * (
            1.0
            + MASS_RANDOMIZATION_RELATIVE * (2.0 * torch.rand(batch_size, device=device) - 1.0)
        )
    return {"mass": mass}


def _noise(batch_size, device):
    import torch

    return (
        torch.randn(batch_size, ctbr.OBS_DIM, device=device)
        * torch.tensor(OBSERVATION_NOISE_STD, device=device),
        torch.randn(batch_size, ctbr.ACTION_DIM, device=device)
        * torch.tensor(NORMALIZED_ACTION_NOISE_STD, device=device),
        torch.randn(batch_size, 3, device=device)
        * torch.tensor(PROCESS_ACCELERATION_NOISE_STD, device=device),
    )


def _clamp_state(state):
    """Bound a diverged rollout to finite magnitudes; the quaternion needs none.

    Far outside any recoverable state -- the only job is to stop one diverged
    sample turning the batch loss into inf, since a clamped trajectory still
    scores a large finite cost, which is the gradient signal we want. The
    quaternion is renormalized every step and cannot blow up.
    """
    import torch

    return torch.cat(
        (
            torch.clamp(state[:, 0:3], -5.0, 5.0),
            torch.clamp(state[:, 3:6], -20.0, 20.0),
            state[:, 6:10],
        ),
        dim=-1,
    )


def _errors(state):
    """(position error, speed, tilt, |sin yaw error|) from a batch of states."""
    import torch

    observation = ctbr.observe(state)
    position = torch.norm(state[:, 0:3], dim=-1)
    velocity = torch.norm(state[:, 3:6], dim=-1)
    # 1 - cos(tilt) via the world-up direction cosine: zero level, 2 inverted,
    # and smooth through every attitude, unlike an angle.
    tilt = 1.0 - observation[:, 8]
    yaw = torch.abs(observation[:, 9])
    return position, velocity, tilt, yaw


def _state_cost(state):
    position, velocity, tilt, yaw = _errors(state)
    return (
        LOSS_POSITION_WEIGHT * position.square()
        + LOSS_VELOCITY_WEIGHT * velocity.square()
        + LOSS_TILT_WEIGHT * tilt.square()
        + LOSS_YAW_WEIGHT * yaw.square()
    )


def rollout(policy, state, parameters, steps, noise=None, grad=True):
    """Simulate one batched episode and accumulate its cost.

    The single simulation loop behind training, evaluation and calibration, so
    an evaluation cannot quietly describe a different vehicle than the one
    training saw.

    ``policy`` maps a raw SI observation to a normalized action. ``noise`` is
    either ``None`` (noise-free) or a per-step sequence of
    ``(observation, action, process)`` tensors.
    """
    import torch

    mass = parameters["mass"]
    action_weights = torch.tensor(ACTION_LOSS_WEIGHTS, device=state.device)
    previous = None
    totals = {"state": 0.0, "action": 0.0, "rate": 0.0}
    observations = []

    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        for index in range(steps):
            if noise is None:
                observation_noise = action_noise = 0.0
                process_noise = None
            else:
                observation_noise, action_noise, process_noise = noise[index]

            observation = ctbr.observe(state) + observation_noise
            observations.append(observation)
            normalized = torch.clamp(policy(observation), -1.0, 1.0)
            applied = torch.clamp(normalized + action_noise, -1.0, 1.0)

            totals["state"] = totals["state"] + _state_cost(state).mean()
            totals["action"] = totals["action"] + (
                action_weights * normalized.square()
            ).sum(dim=-1).mean()
            if previous is not None:
                totals["rate"] = totals["rate"] + (normalized - previous).square().mean()
            previous = normalized

            state = _clamp_state(
                ctbr.rk4_step(
                    state, normalized_to_physical(applied), DT_S, mass,
                    ctbr.GRAVITY_M_S2, process_noise,
                )
            )
    return state, totals, observations


def _prerolled_noise(batch_size, device, steps):
    return [_noise(batch_size, device) for _ in range(steps)]


def _rollout_loss(model, batch_size, horizon_steps, device):
    state = sample_initial_states(batch_size, device)
    parameters = sample_vehicle_parameters(batch_size, device, randomize=True)
    noise = _prerolled_noise(batch_size, device, horizon_steps)
    state, totals, _ = rollout(model, state, parameters, horizon_steps, noise)

    running = (
        totals["state"] + totals["action"] + ACTION_RATE_LOSS_WEIGHT * totals["rate"]
    ) / float(horizon_steps)
    return running + TERMINAL_LOSS_MULTIPLIER * _state_cost(state).mean()


def _success_metrics(state):
    import torch

    position, velocity, tilt, yaw = _errors(state)
    success = (
        (position <= SUCCESS_POSITION_M)
        & (velocity <= SUCCESS_VELOCITY_M_S)
        & (tilt <= 1.0 - math.cos(SUCCESS_TILT_RAD))
        & (yaw <= math.sin(SUCCESS_YAW_RAD))
    )

    def rms(value):
        return float(torch.sqrt(torch.mean(torch.square(value))).cpu())

    return {
        "success_fraction": float(success.float().mean().cpu()),
        "final_position_rms_m": rms(position),
        "final_velocity_rms_m_s": rms(velocity),
        "final_tilt_rms": rms(tilt),
        "final_yaw_rms": rms(yaw),
    }


def _evaluate_model(model, device, environments, noisy=True):
    steps = max(1, round(EVALUATION_HORIZON_S / DT_S))
    state = sample_initial_states(environments, device)
    parameters = sample_vehicle_parameters(environments, device, randomize=noisy)
    noise = _prerolled_noise(environments, device, steps) if noisy else None
    model.eval()
    state, _, _ = rollout(model, state, parameters, steps, noise, grad=False)
    return _success_metrics(state)


def _collect_calibration(model, device, environments=256, horizon_s=6.0, max_samples=8192):
    """Raw SI observations the trained policy actually visits.

    NEMO calibrates activation ranges from these, so they must come from the
    policy in flight rather than from the sampling box.
    """
    import numpy as np

    steps = max(1, round(float(horizon_s) / DT_S))
    state = sample_initial_states(environments, device)
    parameters = sample_vehicle_parameters(environments, device, randomize=True)
    model.eval()
    _, _, observations = rollout(
        model, state, parameters, steps, _prerolled_noise(environments, device, steps),
        grad=False,
    )
    stride = max(1, math.ceil(environments * steps / max_samples))
    return np.concatenate(
        [batch.cpu().numpy() for batch in observations[::stride]], axis=0
    )[:max_samples].astype("float32")


def train_policy(
    output_dir=MODEL_DIR,
    device_name="auto",
    rollout_epochs=ROLLOUT_EPOCHS,
    batch_size=ROLLOUT_BATCH_SIZE,
    evaluation_environments=EVALUATION_ENVIRONMENTS,
    calibration_environments=256,
):
    """Train by backprop through the rollout; save weights and calibration."""
    import numpy as np
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(device_name)
    seed_everything()
    model = create_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ROLLOUT_LEARNING_RATE)
    model.train()

    print("device %s | %d epochs x %d envs" % (device, rollout_epochs, batch_size))
    for epoch in range(1, int(rollout_epochs) + 1):
        fraction = min(1.0, (epoch - 1) / max(1, ROLLOUT_CURRICULUM_EPOCHS - 1))
        horizon_s = ROLLOUT_MIN_HORIZON_S + fraction * (
            ROLLOUT_MAX_HORIZON_S - ROLLOUT_MIN_HORIZON_S
        )
        loss = _rollout_loss(model, int(batch_size), max(1, round(horizon_s / DT_S)), device)
        # One unlucky draw is not evidence of anything: the initial states and
        # noise differ every epoch, so a non-finite batch is simply skipped.
        if not torch.isfinite(loss):
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        if epoch % 50 == 0 or epoch == rollout_epochs:
            print("  %4d  horizon %.1fs  loss %.4f" % (epoch, horizon_s, loss.item()))

    evaluation = {
        "clean": _evaluate_model(model, device, int(evaluation_environments), noisy=False),
        "randomized": _evaluate_model(model, device, int(evaluation_environments), noisy=True),
    }
    calibration = _collect_calibration(model, device, int(calibration_environments))
    torch.save(model.state_dict(), str(output_dir / CHECKPOINT_PATH.name))
    np.save(str(output_dir / CALIBRATION_PATH.name), calibration)
    _print_evaluation(evaluation)
    return evaluation


def _print_evaluation(evaluation):
    for name, metrics in evaluation.items():
        print(
            "%-11s success %.3f | rms  pos %.3f m  vel %.3f m/s  yaw %.3f"
            % (
                name,
                metrics["success_fraction"],
                metrics["final_position_rms_m"],
                metrics["final_velocity_rms_m_s"],
                metrics["final_yaw_rms"],
            )
        )


def load_trained_model(checkpoint_path=CHECKPOINT_PATH, device_name="auto"):
    import torch

    device = select_device(device_name)
    model = create_model().to(device)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.eval()
    return model, device


def evaluate_checkpoint(checkpoint_path=CHECKPOINT_PATH, device_name="auto"):
    model, device = load_trained_model(checkpoint_path, device_name)
    evaluation = {
        "clean": _evaluate_model(model, device, EVALUATION_ENVIRONMENTS, noisy=False),
        "randomized": _evaluate_model(model, device, EVALUATION_ENVIRONMENTS, noisy=True),
    }
    _print_evaluation(evaluation)
    return evaluation


def _encode_state(observation):
    """Raw SI observation -> the [0,1] domain the deployed integer graph takes."""
    import torch

    limits = torch.as_tensor(STATE_LIMITS, dtype=observation.dtype, device=observation.device)
    return 0.5 * (torch.clamp(observation / limits, -1.0, 1.0) + 1.0)


def _fold_input_codec(model):
    """Absorb ``raw = STATE_LIMITS * (2u - 1)``, u in [0,1], into the first Linear.

    Exact, not an approximation -- an affine input map composed with a Linear is
    another Linear -- so the network trains on raw SI and still deploys as a
    genuine uint8-input integer network with no accuracy cost from the fold.
    """
    import torch

    with torch.no_grad():
        first = model.fc1 if HIDDEN_DIMS else model.fc_out
        limits = torch.as_tensor(STATE_LIMITS, dtype=first.weight.dtype, device=first.weight.device)
        first.bias.data.sub_(first.weight.data @ limits)
        first.weight.data.mul_(2.0 * limits)


def _output_epsilon(value):
    """Unwrap NEMO's per-edge epsilon, which may be a list, tensor, or float."""
    if isinstance(value, (list, tuple)):
        value = value[0]
    if hasattr(value, "numel"):
        value = value.detach().cpu().item()
    return float(value)


def _quantize_linear_biases(model, layer_names, input_epsilon):
    """Snap every Linear bias to its exactly deployable accumulator lattice.

    The pinned NEMO integerizes ``PACT_Linear`` weights but leaves its bias in
    real units.  First snapping the QD graph to representable real biases lets
    its output remain the honest quantized reference for a bias-bearing MLP.
    """
    import torch

    accumulator_epsilons = {}
    for name in layer_names:
        layer = getattr(model, name)
        layer_input_epsilon = model.get_eps_at(name, input_epsilon)
        if isinstance(layer_input_epsilon, (list, tuple)):
            # DeployGraph has no predecessor edge from which to recover the
            # graph input epsilon for the first Linear.
            layer_input_epsilon = (
                layer_input_epsilon[0] if layer_input_epsilon else input_epsilon
            )
        epsilon = _output_epsilon(layer.get_output_eps(layer_input_epsilon))
        accumulator_epsilons[name] = epsilon
        if layer.bias is not None:
            layer.bias.data.copy_(torch.round(layer.bias.detach() / epsilon) * epsilon)
    return accumulator_epsilons


def _integerize_linear_biases(model, accumulator_epsilons):
    """Convert already-snapped real biases to DORY int32 accumulator values."""
    import torch

    for name, epsilon in accumulator_epsilons.items():
        layer = getattr(model, name)
        if layer.bias is not None:
            layer.bias.data.copy_(torch.round(layer.bias.detach() / epsilon))


def export_nemo_onnx(
    *, model, checkpoint_path, calibration, input_shape, output_path, work_dir,
    requantization_factor=None,
):
    """Post-training quantize to int8 and emit the DORY ONNX plus its goldens.

    NEMO owns all quantization: the network trained here is plain float32.
    ``requantization_factor`` trades integer-requantizer precision against int32
    accumulator headroom, and is the one knob a caller may sweep without
    retraining.
    """
    import nemo
    import numpy as np
    import torch

    requant_factor = int(requantization_factor or 512)
    # DORY feeds the graph a uint8 tensor, so the input quantum is fixed at
    # 1/255. NEMO calibrates *activation* ranges from statistics, but eps_in is
    # the contract with whatever feeds the network: hardware, not a knob.
    input_epsilon = 1.0 / 255.0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # The calibration holds raw SI observations. Fold STATE_LIMITS into fc1; from
    # here on the graph is the uint8-input network NEMO and DORY expect.
    calibration_batch = torch.from_numpy(np.load(str(calibration))).float()
    _fold_input_codec(model)
    calibration_batch = _encode_state(calibration_batch)

    parity_input = torch.floor(calibration_batch[:512] * 255.0 + 0.5) / 255.0
    model.eval()
    with torch.no_grad():
        fp_output = model(parity_input).detach()

    model = nemo.transform.quantize_pact(
        model, dummy_input=torch.zeros(1, ctbr.OBS_DIM),
        requantization_factor=requant_factor,
    )
    precision = {"fc_out": {"W_bits": WEIGHT_BITS}}
    for index in range(1, len(HIDDEN_DIMS) + 1):
        precision["fc%d" % index] = {"W_bits": WEIGHT_BITS}
        precision["relu%d" % index] = {"x_bits": ACTIVATION_BITS}
    model.change_precision(bits=1, min_prec_dict=precision)

    model.set_statistics_act()
    with torch.no_grad():
        model(calibration_batch)
    model.unset_statistics_act()
    model.reset_alpha_act()
    model.qd_stage(eps_in=input_epsilon)
    # get_eps_at returns the epsilon at a module's *input*; the raw final int32
    # accumulator needs the Linear's output epsilon too.
    output_epsilon = _output_epsilon(
        model.fc_out.get_output_eps(model.get_eps_at("fc_out", input_epsilon))
    )

    linear_names = ["fc%d" % index for index in range(1, len(HIDDEN_DIMS) + 1)]
    accumulator_epsilons = _quantize_linear_biases(
        model, linear_names + ["fc_out"], input_epsilon
    )
    with torch.no_grad():
        qd_output = model(parity_input).detach()

    _integerize_linear_biases(model, accumulator_epsilons)
    model = nemo.transform.integerize_pact(
        model, eps_in=input_epsilon, requantization_factor=requant_factor
    )
    with torch.no_grad():
        integer_output = model(torch.floor(parity_input * 255.0 + 0.5)).detach()
    decoded_output = integer_output * output_epsilon

    # The integer graph must reproduce its own dequantized reference. This is a
    # correctness check on the codec, not a quality measurement: the policy is
    # only ever quantized one way, so there is nothing to compare it against.
    parity = float(torch.max(torch.abs(decoded_output - qd_output)))
    print(
        "[nemo] parity id_vs_qd %.3g  id_vs_fp %.3g"
        % (parity, float(torch.max(torch.abs(decoded_output - fp_output)))),
        flush=True,
    )
    if parity > 0.02:
        raise RuntimeError(
            "NEMO output codec parity %.6g exceeds 0.02; raise requantization_factor"
            % parity
        )

    with torch.no_grad():
        nemo.utils.export_onnx(
            str(output_path), model, model, (ctbr.OBS_DIM,), batch_size=1
        )

    # DORY compares against these per-layer goldens when checking its codegen.
    node_names = ["relu%d" % index for index in range(1, len(HIDDEN_DIMS) + 1)] + ["fc_out"]
    captured = {}
    handles = [
        getattr(model, name).register_forward_hook(
            lambda _m, _i, output, name=name: captured.__setitem__(
                name, output.detach().round().to(torch.int64).flatten().tolist()
            )
        )
        for name in node_names
    ]
    integer_input = torch.floor(calibration_batch[0:1] * 255.0 + 0.5)
    with torch.no_grad():
        model(integer_input)
    for handle in handles:
        handle.remove()

    def write_values(name, values):
        (output_path.parent / name).write_text(
            "".join("%d\n" % value for value in values), encoding="utf-8"
        )

    write_values("input.txt", integer_input.to(torch.int64).flatten().tolist())
    for index, name in enumerate(node_names):
        write_values("out_layer%d.txt" % index, captured[name])
    write_values("output.txt", captured["fc_out"])

    # The one number the firmware needs that is not a design constant: NEMO's
    # output quantum, which depends on the trained weights. Everything else in
    # the deck's codec is fixed at the top of this file. Written as data, not as
    # C -- the GAP8 Makefile reads this file and passes the value in as a -D, so
    # it cannot be pasted wrong and cannot go stale.
    (output_path.parent / "output_epsilon.txt").write_text(
        "%.9g\n" % output_epsilon, encoding="utf-8"
    )
    return output_path


def smoke_test(device_name="auto"):
    """Two epochs end to end, so a broken pipeline fails in seconds not hours."""
    with tempfile.TemporaryDirectory(prefix="hover-policy-smoke-") as directory:
        train_policy(
            output_dir=Path(directory),
            device_name=device_name,
            rollout_epochs=2,
            batch_size=32,
            evaluation_environments=32,
            calibration_environments=16,
        )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="train and write the checkpoint")
    commands.add_parser("evaluate", help="evaluate the saved checkpoint")
    commands.add_parser("smoke", help="run a tiny end-to-end training check")
    command = parser.parse_args(argv).command

    if command == "train":
        train_policy()
    elif command == "evaluate":
        evaluate_checkpoint()
    else:
        return smoke_test()
    return 0


if __name__ == "__main__":
    sys.exit(main())
