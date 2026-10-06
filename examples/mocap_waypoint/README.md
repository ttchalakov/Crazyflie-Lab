# Mocap Waypoint

A **flying** mocap demo and position-control test: the drone takes off straight
up and streams a timed sequence of absolute position setpoints, then lands
straight down from its latest measured position. Because motion capture gives a
full 6D pose, the estimator fuses external position and orientation while the
built-in PID position controller tracks the setpoints streamed from the host.

```text
mocap ──natnet──> host:  pose (origin-zeroed to the start)
host ──radio──> STM32:   extpose (into EKF) + position setpoints, 50 Hz
STM32 onboard:           built-in PID position controller
```

## Flight profile

All frames +z up, origin at the drone's starting pose.

1. **Ground:** the first mocap sample fixes the world origin at `(0, 0, 0)`, the
   estimator settles on that zeroed frame.
2. **Timed setpoints:** each point in `TIMED_SETPOINTS` is streamed as an absolute setpoint
   for its own duration, then the runner advances to the next point. The **first
   waypoint doubles as the takeoff**, so give it enough time and keep it straight
   up over the origin (e.g. `(3.0, 0, 0, height, 0)`).
3. **Land:** after the final timed setpoint, the runner reads the latest zeroed
   mocap position and descends straight down there, then stops the motors and
   disarms.

The duration includes both travel time and any remaining hold time. The schedule
advances when that time expires regardless of how close the drone is to the
target. SPACE / Ctrl-C / radio loss remain active throughout, and losing mocap
for `mocap.pose_timeout_s` (1 s by default) before the landing lands early.

## Timed setpoints

`host/run.py` contains four `TIMED_SETPOINTS` profiles in the start-pose frame:
a hover with small +x adjustments, an independent axis/yaw sweep, an aggressive
stress profile, and the generated three-dimensional figure eight. Leave exactly
one assignment active to select the flight.

For example, `(3.0, 0.2, 0.0, 0.4, 0.0)` streams the position `(0.2, 0.0,
0.4)` with zero-degree yaw for three seconds.

## PID values

The runner always applies the exposed x/y and vertical-position values at the
top of `host/run.py`:

```python
POS_XY_KP = 3.0
POS_XY_KI = 0.25
VEL_XY_KP = 20.0
VEL_XY_KI = 2.0
XY_VEL_MAX = 2.0
POS_Z_KP = 2.0
POS_Z_KI = 0.0
Z_VEL_MAX = 1.0
```

The stock cf21bl vertical-position values are `zKp=2.0`, `zKi=0.5`, and
`zVelMax=1.0`. This demo keeps the stock vertical P and velocity limit but sets
`zKi=0.0` as the conservative first test for reducing integral windup after
large altitude steps. The inner vertical-velocity gains remain untouched.

The x and y axes receive the same horizontal values while the drone is
disarmed. All exposed values are written to RAM, recorded with the run
telemetry, and reset by the next reboot.

## Pieces

- `controller/` — the **default firmware** build for the Crazyflie 2.1
  Brushless, the built-in PID controller plus the cf21bl platform + brushless
  motor options.
- `host/run.py` — the demo runner (built on `cf_lab.runner`): streams
  origin-zeroed extpose, gates arming on a live mocap stream, arms with the
  radio-loss deadman, then flies the waypoint sequence and handles the kill controls.

## Flash

Flash the default firmware for this example:

```bash
cf-lab build controller --example mocap_waypoint
cf-lab flash controller --example mocap_waypoint --config configs/drone.yaml
```

## Run

This is a live-hardware-only example: it always opens the OptiTrack source and
Crazyflie radio configured in `configs/drone.yaml`. Props ON, clear space, stand
clear. Motion capture comes from Motive's NatNet stream, set up as in
[mocap_positioning](../mocap_positioning/README.md#real-run-with-optitrack).

```bash
cf-lab reboot --config configs/drone.yaml
# Keep the drone still until the script completes, then reconnect AI-deck Wi-Fi.
.venv/bin/python examples/mocap_waypoint/host/run.py --config configs/drone.yaml
```

The reboot is explicit: `run.py` never reboots the STM32 or AI deck. Run
`cf-lab reboot` before a flight to clear the radio-loss deadman and let the stock
firmware acquire a fresh gyro bias.

Kill controls, live during every phase:

- **SPACE** — emergency stop: the supervisor cuts the motors immediately. Exit code 2.
- **Ctrl-C** — graceful shutdown: land straight down in place, then disarm.

## Preflight checks

The runner performs three read-only preflight stages and keeps the drone
disarmed if any required result fails:

1. Before opening hardware, it validates `drone.yaml` and checks that the
   configured AI-deck camera network can coexist with the mocap route.
2. After opening the radio, `standard_flight_checks()` reads the firmware's
   latched boot self-test, requires a ready supervisor, samples the battery, and
   measures a stationary, disarmed IMU window. An IMU that failed to start
   fails both the self-test and the IMU window, since a dead accelerometer
   reads 0 g instead of one g.
3. After the existing mocap settle gate, it observes a one-second live window
   for rate, freshness, rigid-body identity, valid quaternions, stationary
   spread, and discontinuous position/orientation jumps.

Stock firmware locks the gyro bias once at boot. `cf-lab reboot` performs that
reboot and holds for calibration; the runner does not. The static-IMU
stage checks the resulting corrected `gyro.x/y/z`, motion, acceleration near one
g, and disarmed state immediately before flight, so moving the vehicle during
boot calibration is detected before arming.

Every executed result is recorded as a `preflight_check` event when telemetry
is enabled. These checks never write a parameter, reset the estimator, arm,
command a motor, or reboot the vehicle. [Safety](../../docs/safety.md) shows
the same checks from the command line, `cf-lab preflight`.

## Telemetry

With `telemetry.enabled: true` in the config (the default in `configs/drone.yaml`)
every run records to `artifacts/run_<timestamp>/`: one Parquet file per stream
plus `manifest.json`, the human `debug_log.txt`, and `crazyflie_toc.json`. The
TOC sidecar records the connected firmware's log/parameter interface metadata,
not the current parameter values, and is captured before hardware preflight so
it remains available even when a failed check refuses the flight. The
controller-view 12D state is recorded at `telemetry.state_hz` as two firmware
log blocks:

- `state_pos`: EKF `x`, `y`, `z`, `vx`, `vy`, `vz` (translational 6D);
- `state_att`: EKF `roll`, `pitch`, `yaw`, and filtered gyro rates `p`, `q`, `r`.

This split is only due to the firmware's log-block size limit; both blocks use
the same configured rate. The gyro rates are pre-EKF measurements used directly
by the rate PID, not estimator outputs. `ctrl` contains the four controller
outputs, while `setpoints`, `mocap`, and `extpose` are recorded independently.

The two pose records deliberately have different roles:

- `mocap` is the full raw 6D OptiTrack pose before origin-zeroing or rate capping;
- `extpose` is the origin-zeroed 6D pose after the uplink rate cap, as forwarded
  toward the EKF.

The full origin-zeroed `extpose` position and orientation are fused by the
estimator. Both raw mocap and forwarded extpose telemetry record their complete
poses, and the `extpose_config` event marks orientation as fused and trusted as
an analysis baseline. The `origin_zeroed` event records the coordinate transform
used by the run.

The normal streams are the first telemetry level and should remain enabled for
ordinary experiments. `configs/drone.yaml` also contains a commented sys-ID
profile for the filtered six-axis IMU, uncapped motor requests and battery
voltage, capped motor outputs, and bidirectional-DShot RPM feedback. Uncomment
those `telemetry.custom` blocks only for lower-level identification runs and
check `manifest.json` for drops afterward.

Visualize the newest recorded run with Plotly:

```bash
.venv/bin/python examples/mocap_waypoint/host/visualize.py
```

Pass a run folder explicitly when needed:

```bash
.venv/bin/python examples/mocap_waypoint/host/visualize.py \
  examples/mocap_waypoint/artifacts/run_<timestamp>
```

> First flights: edit `TIMED_SETPOINTS` to a short, low sequence (e.g.
> `(3.0, 0, 0, 0.5, 0)`, `(4.0, 0.5, 0, 0.5, 0)`, `(4.0, 0, 0, 0.5, 0)`),
> confirm takeoff, the moves, and the straight-down land, and have a finger on
> SPACE. Verify the OptiTrack axes/Z-up and the origin-zeroed frame on the first
> slow legs before trusting the full sequence.
