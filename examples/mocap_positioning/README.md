# Mocap Positioning Demo

A propellers-off demo of motion-capture positioning. The drone never spins its
motors; it is a live position indicator. Place it on the floor in the centre of
the mocap room, start the runner, and the first mocap sample fixes the world
origin (0,0,0) at the drone's starting pose. A safety box, configured on the
host, sits in that frame, and the drone's **front LEDs** turn **green inside the
box** and **red outside** as you carry it around by hand. This is not a
standalone system.

```text
OptiTrack (Motive) ──NatNet──> host (cf_lab)

host ──(origin-zeroed extpose, radio)──> Crazyflie EKF ──> out-of-tree controller
  │                                                           │
  │                                       box.inside log variable (onboard check
  │                                       on the estimator state; nothing acts on it)
  │
  └──(host's own in/out check on the mocap pose)──> front LEDs via led.bitmask
                                                     (green in / red out)
```

Extpose carries position and orientation only, so the Crazyflie EKF fuses it
with the IMU's accelerometer and gyroscope. The out-of-tree controller is there
for the demonstration only.

The host decides inside/outside: the runner checks each mocap pose against the
box and lights the front LEDs through the firmware's `led.bitmask` override. The
controller runs the same check onboard on the estimator state, but only reports
it as the `box.inside` log variable; nothing on the drone acts on it. It is there
to show where an onboard check would go. In the future if you wanted to include
velocity or angular velocity, then you would need to run an estimator on the host
side unless you want to send the Crazyflie EKF estimates back to the host runner
over radio.

## Conventions

- **+z is up**.
- **The world origin is the drone's starting pose.** The first few mocap samples
  fixes position (0, 0, 0) and zero yaw (`OriginZeroer`); that one frame is
  streamed for the whole run, so the estimator never sees a frame change.
- **The safety box is axis-aligned**, given by a center and full side widths in
  that frame. Default center (0, 0, 1.5) m, widths (2.0, 1.0, 1.0) m — so the
  box spans x in ±1.0 m, y in ±0.5 m and z in [1.0, 2.0] m.

## Configuring the box

Edit `BOX_CENTER_M` and `BOX_SIZE_M` at the top of `host/run.py`.

`BOX_CENTER_M` is the centre (x, y, z) and `BOX_SIZE_M` the full side widths
(x, y, z), both in metres. The runner sets the controller's `box.*` parameters on connect.

## Safety / disarm policy

This demo intentionally does the opposite of a flight safety monitor:

- **Propellers OFF, always.** The drone is never armed, and the firmware
  force-stops the motors of an un-armed drone, so the props physically cannot
  spin. The controller also commands zero thrust and the
  app-config sets `CONFIG_MOTORS_DEFAULT_IDLE_THRUST=0`.
- **Never armed, so nothing ever auto-stops.** The out-of-tree controller runs
  on every stabilizer tick regardless of arm state, so the LED indicator works
  while disarmed. Because the drone is never armed, the firmware's pre-flight
  timeout (~30 s) and tilt/tumble auto-disarm never apply: tilting or carrying
  the drone out of the box only changes the LED. (Arming an idle drone would hit
  those auto-disarms on the pinned firmware, which is why the demo skips it.)
- **Stops only on operator command:** SPACE = emergency stop (firmware cuts and
  locks the motors), Ctrl-C = stop streaming and exit. With `--duration > 0` the
  runner also stops when the timer elapses; the default `--duration 0` runs
  until you stop it.
- **Mocap gate:** the runner refuses to start unless the mocap samples were
  actually forwarded during the settle phase, so a dead mocap system cannot
  leave the LED stuck on a stale estimate.
- **Radio loss:** if the link drops there is nothing to disarm (the drone was
  never armed); the runner just stops and exits.

## Debug log

On a radio run the firmware console (STM32 `DEBUG_PRINT`, plus the GAP8/ESP32
lines forwarded through it) is captured to `artifacts/debug_log/debug_log.txt`.
Every line is prefixed with its source and a shared `[t=…]` time tag:

```
[t=  1.23s] [HOST] Settling estimator on the start frame for 3.0s.
[t=  1.44s] [STM32] BOXVIZ: Mocap positioning controller loaded: motors stay off ...
[t=  1.90s] [GAP8] -- camera_stream GAP8 app --
```

`[HOST]` lines (run.py's own flow) also print to the terminal; the firmware
tiers (`[GAP8]`/`[STM32]`/`[ESP32]`) go to the log file only. Pass
`--no-debug-log` to skip the capture.

## Dry run (no hardware)

```bash
.venv/bin/python examples/mocap_positioning/host/run.py --source fake --backend fake --duration 8
```

The fake source starts from an offset pose (so the origin re-zeroing does real
work) and sweeps the drone up and down through the box, so the INSIDE/OUTSIDE
log lines toggle.

## Bench run (drone on the floor, propellers OFF)

Build and flash the controller:

```bash
cf-lab flash controller --example mocap_positioning --config configs/drone.yaml
```

Hardware-in-the-loop with the fake sweep (real drone, synthetic pose):

```bash
.venv/bin/python examples/mocap_positioning/host/run.py --source fake --backend radio --config configs/drone.yaml
```

## Real run with OptiTrack

The runner reads Motive's **NatNet** stream directly, with a pure-Python UDP
client (`src/cf_lab/mocap/natnet.py`, standard library only).
`mocap.transport: natnet` in `configs/drone.yaml` selects it for
`--source optitrack`.

1. **Motive (OptiTrack):** create a rigid body for the drone and name it to
   match `mocap.rigid_body` in `configs/drone.yaml` (default `cf21bl_01`). In the
   **Data Streaming** pane set the **Up Axis to Z**, so the streamed frame is +z
   up to match the estimator (Motive defaults to Y-up), and tick **Broadcast
   Frame Data** under **NatNet**. Note its **Transmission Type** (Unicast or
   Multicast): `mocap.natnet.use_multicast` must match it.
2. **Host:** put the Motive PC's IP in the config and run:

```yaml
mocap:
  transport: natnet
  natnet:
    server_ip: 192.168.1.10
```

```bash
.venv/bin/python examples/mocap_positioning/host/run.py \
  --source optitrack --backend radio --config configs/drone.yaml
```

Your laptop must be on the **same network/subnet** as the Motive PC for
Motive's frames to reach it. Useful `mocap.natnet` keys:

- `local_ip: <laptop IP>` — pins the NIC that receives the frames (use it if
  the laptop has several interfaces, e.g. WiFi + Ethernet).
- `use_multicast` — `false` (the default) for Motive's Unicast, where the client
  registers with Motive's command port so it streams to you directly; `true`
  for Multicast. If it does not match Motive, the connection stops with a
  message saying which value to set.

On every connection the client asks Motive for the NatNet 4.2 bitstream, the
only one it parses. That setting is Motive-wide, for unicast and multicast alike:
a client that asked for an older bitstream (some ROS drivers ask for 4.0) gets
4.2 from then on.
- `multicast_ip` / `data_port` / `command_port` — only if you changed Motive's
  defaults (`239.255.42.99`, `1511`, `1510`).

Set `mocap.rigid_body` to the exact Motive asset name. At connection time, the
client queries Motive for its Streaming ID; no numeric ID is configured. Missing
or ambiguous names stop the connection. Reconnect after changing assets in
Motive.
