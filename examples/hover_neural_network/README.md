# Neural Hover

A small PyTorch policy trained in a differentiable simulator, quantized with
NEMO, compiled by DORY, and flown at 100 Hz on the AI-deck of a Crazyflie 2.1
Brushless — while the same GAP8 cluster also encodes QVGA JPEG video.
[lib/gap8/sched](../../lib/gap8/sched/README.md) schedules the two on the deck.

The policy is **CTBR**: it outputs a collective thrust and a body-rate setpoint,
which the vehicle's own 500 Hz rate loop tracks. That is what makes a 100 Hz
neural controller over a link with milliseconds of dead time reasonable — the
fast loop stays onboard beside the gyro, and the policy sits outside it.

## The contract

The STM32 sends the 11-float yaw-invariant observation at 100 Hz; the GAP8
replies once per request with four floats.

| observation (SI, FLU body frame) | |
|---|---|
| `ex, ey, ez` | position error, rotated into the body frame |
| `vx, vy, vz` | velocity, rotated into the body frame |
| `upx, upy, upz` | world "up" expressed in the body frame |
| `sin, cos` | heading error as a pair, never an angle |

| action | |
|---|---|
| `thrust` | newtons |
| `wx, wy, wz` | body rates, rad/s |

Attitude enters as world-up-in-body rather than Euler angles because it is
continuous everywhere. Heading is a `(sin, cos)` pair because a raw yaw jumps at
±π and a policy commanded through that wrap reacts to a discontinuity that is not
physical. Both together make the observation invariant to heading, so the policy
only has to learn hovering once instead of once per direction.

The flight is fixed and easy to read: PID takeoff to `HOVER_HEIGHT_M`, settle,
immediate handover to the policy for `EXPERIMENT_S`, immediate return to the
still-running PID, then a slow landing over the recorded start.

## Configuration

There is no example YAML and no wrapper scripts. Edit the constants at the top of
the file that owns each behavior:

- `model/hover_policy.py` — vehicle, network shape, dynamics, training,
  randomization, and the input/output scales the deck decodes against;
- `controller/src/hover_nn_controller.c` — request rate, freshness limit, and the
  final command clamps;
- `gap8/Makefile` — camera and video: QVGA at 60 fps, JPEG, 60 Hz anti-flicker
  for the mocap strobes;
- `gap8/hover_neural_network.c` — the state/action codec;
- `host/run.py` — hover height, phase durations, and the mocap settle window.

`configs/drone.yaml` stays what it is: radio, mocap, telemetry, and the deck's
WiFi. It is hardware configuration, not policy configuration. `cf-lab flash gap8`
builds its `camera:` section into the GAP8 app, and `host/run.py` connects with
the same section, so the two always agree. For flight use `mode: station`, so
video and mocap share one network interface; `--wifi-mode access_point` makes a
bench build that needs no router.

## Train

```bash
.venv/bin/python examples/hover_neural_network/model/hover_policy.py train
.venv/bin/python examples/hover_neural_network/model/hover_policy.py evaluate
```

Roughly nine minutes on a mid-range CUDA card. Training is backpropagation
through a differentiable RK4 rollout with a horizon curriculum — no teacher, no
pretraining, no behaviour cloning. It converges from a random initialization
because a zero action already decodes to hover thrust, so an untrained network
starts near equilibrium rather than falling out of the sky.

`train` writes exactly two files, `hover_policy.pt` and
`hover_policy_calibration.npy`, and prints its evaluation instead of serializing
it. `smoke` runs the same path for two epochs so a broken pipeline fails in
seconds; its checkpoint cannot fly.

Expect `success 1.000` on both the clean and randomized evaluations. The
randomized one varies mass ±8%, which under CTBR is the only vehicle parameter
that still matters — inertia, arm length and the thrust-to-torque ratio all live
inside the rate loop, which the policy does not command.

## Deploy

```bash
cf-lab compile-model \
  --models-root examples/hover_neural_network/build \
  --integrate-into examples/hover_neural_network/gap8/policy \
  --model-module examples/hover_neural_network/model/hover_policy.py \
  --checkpoint examples/hover_neural_network/model/hover_policy.pt \
  --calibration examples/hover_neural_network/model/hover_policy_calibration.npy \
  --input-shape 1,11 --name policy --dory-target PULP.GAP8_L2 \
  --dory-l2-bytes 228000 --dory-verbose-level None
cf-lab build controller --example hover_neural_network
cf-lab build gap8 --example hover_neural_network
```

NEMO owns all quantization; the network trained here is plain float32. The export
prints one `[nemo]` line reporting the integer graph's error against its own
dequantized reference — a correctness check on the codec, not a quality
measurement, since the policy is only ever quantized one way.

The deck reproduces the integer network bit for bit, encoding and decoding
included.

Nothing generates C. The single deployment value that depends on the trained
weights is NEMO's output quantum, which the export drops beside its ONNX as
`build/dory/policy/nemo/output_epsilon.txt`; the GAP8 Makefile reads that file and
passes it in as `-DHNN_POLICY_OUTPUT_EPSILON`. It cannot be pasted wrong and
cannot go stale. Everything else the deck decodes against — input limits, the
action affine — is a constant at the top of `hover_policy.py`, mirrored once in
`gap8/hover_neural_network.c`.

`--dory-target PULP.GAP8_L2` is not optional here. At ~70 kB the weights fit L2
outright, so they link as resident arrays and inference touches HyperRAM zero
times. The app is built for that memory model only.

## Flash and fly

```bash
cf-lab flash controller --example hover_neural_network \
  --config configs/drone.yaml
cf-lab flash gap8 --example hover_neural_network \
  --config configs/drone.yaml
cf-lab reboot --config configs/drone.yaml
.venv/bin/python examples/hover_neural_network/host/run.py --config configs/drone.yaml
```

Flash the STM32 before the deck. The controller stops streaming after
`HNN_LINK_IDLE_MS` of silence for a reason: a GAP8 radio flash travels over the
same STM32→deck link, and a steady 100 Hz of state packets starves it — the write
either stalls at 0% or completes and produces an image that never boots. If a
deck flash misbehaves, check that the STM32 carries a controller that backs off.

Keep the vehicle still through the reboot so the firmware can acquire a clean
gyro bias. The runner never reboots either processor; once the radio-loss deadman
is armed, reboot again before another flight.

The policy is disabled at boot and the PID keeps running underneath it the whole
time. Control returns to the PID immediately if the deck goes quiet, a reply is
malformed, or the newest command is older than `HNN_MAX_AGE_MS`. Before arming,
the host requires a sustained 95 replies/s. SPACE cuts the motors; Ctrl-C disables
the policy and lands under the PID.

The run log carries the two rates that matter: the camera preflight line gives
the video FPS, and the readiness line the policy's replies per second. The
`hnn_link` telemetry stream records the reply count through the flight.

Runs are recorded under `artifacts/`. Visualize the newest one with:

```bash
.venv/bin/python examples/hover_neural_network/host/visualize.py
```

## Known limits

The first two are unverified in flight; both are noted at the top of
`host/run.py`.

**Body-frame pitch sign.** The controller sends the rate setpoint as
`.pitch = -wy`, because the firmware's pitch convention is inverted relative to
the FLU frame the policy trains in. If that is wrong, pitch diverges while roll
behaves.

**Standing yaw bias.** At the exact hover equilibrium the trained policy can
still command a small yaw rate, so expect a slow rotation in hover. It is a
training artifact, not a firmware bug — the lever is `LOSS_YAW_WEIGHT` followed by
a retrain.

**Video beside control.** The deck answers every request with or without a video
client, but with video the replies sometimes pause and then catch up. The
STM32's 30 ms freshness limit hands those moments to the PID. To fly
control-only, connect no video client (the deck encodes only for one), or build
the GAP8 app without video: `cf-lab flash gap8 --example hover_neural_network
--make-args "STREAM=0"`.
