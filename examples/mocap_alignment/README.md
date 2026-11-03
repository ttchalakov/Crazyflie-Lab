# Mocap Alignment

Aligns a Motive rigid body with the drone's IMU, in about two minutes, by rotating the
drone by hand. A rigid body made in Motive takes the drone's pose at creation as
identity, and its pivot at the centroid of its markers. The estimator flies in the
IMU's axes and treats extpose as the pose of the IMU, so the difference matters: the
estimator trusts mocap orientation to about 0.26 deg, and one asset measured here was
4.4 deg off.

Propellers stay off. The drone is never armed and no parameter is written; the radio
only logs. No level floor is needed and Motive's ground plane is not touched.

## What comes from where

| Quantity | Comparison | Notes |
|---|---|---|
| Asset to IMU rotation | gyro rates vs mocap rotation rates | gravity and the floor do not enter |
| IMU position, accelerometer bias, tilt of Motive's vertical | accelerometer vs mocap motion | turning the drone separates bias (fixed to the drone) from the tilt (fixed to the room) |
| Mechanical front and centre | deck beads vs their holes | a cross-check on the rotation; the prior for the IMU position |

**The origin** comes from the motion fit only if the fit repeats across the recording,
its two estimates of the IMU's height agree, and it lands near the geometry. Otherwise
it comes from the geometry, and the report shows both.

## Setting up the geometry

`configs/marker_geometry/cf21bl_mocap_deck.yaml` holds what the script cannot measure:

- `deck_holes_mm`: the holes holding deck beads, in body axes from the deck centre. The
  [deck datasheet](https://www.bitcraze.io/documentation/hardware/passive_marker_deck/passive_marker_deck-datasheet.pdf)
  shows the layout, on a 5 mm pitch. Three or more beads, in an asymmetric pattern.
- `bead_diameter_mm`: beads sit flush on the deck.
- `imu_xy_mm`, `imu_z_from_deck_top_mm`: where the IMU is, as a prior. The height shifts
  with the battery between the deck and the main PCB.

The deck beads must be part of the Motive asset.

## Run

`mocap.natnet.use_multicast` must match Motive's Transmission Type, and
`mocap.rigid_body` must name the asset. From the repository root:

```bash
.venv/bin/python examples/mocap_alignment/host/run.py calibrate
```

1. It checks the deck beads against the geometry file, then reboots the drone for a
   fresh gyro calibration. Keep the drone still until it says to pick it up.
2. Pick the drone up, holding it by the battery or underside with the beads in view.
   Turn it through every orientation until the progress line reaches 100%, about 30 s.
3. Put it down anywhere and let go. The analysis runs by itself.
4. Enter the printed pivot edit in Motive's Builder pane, then:

```bash
.venv/bin/python examples/mocap_alignment/host/run.py verify examples/mocap_alignment/artifacts/run_<timestamp>
```

`verify` re-reads the asset from Motive and shows what the edit actually did. The script
prints the edit in the streamed Z-up axes and in Motive's native Y-up axes. On Motive
3.3.4 the Builder pane takes the Y-up values; `verify` catches it if that ever changes.
Running `calibrate` again afterwards should report almost no rotation.

After a battery swap or a crash, run `calibrate` again. Recordings stay in `artifacts/`
(gitignored); `run.py analyze RUN_DIR` analyzes one again.

## Reading the report

| Check | Failing means |
|---|---|
| time sync | the IMU and mocap do not line up: too little motion, or a bad recording |
| rotation repeatable | parts of the motion disagree, often occluded beads |
| deck beads fit their holes | a wrong hole in the geometry file, or an unseated bead |
| deck yaw matches the IMU | the deck is rotated on the drone, or a hole is wrong |
| bias and Motive tilt separated | not enough orientations: turn the drone further |
| closure at rest | the results do not predict the accelerometer at rest |

`Motive's up` is the tilt of Motive's vertical from true gravity. It is measured and
reported, and nothing compensates for it yet.

## Known limits

- Motive's orientation varies by about 0.3 deg across the capture volume, which sets
  the floor on the rotation.
- The IMU log is limited to 100 Hz by the firmware's log framework.
- The accelerometer fit leaves a residual of about 0.3 m/s^2 (0.02 m/s^2 in simulation).
  It is fast hand jitter inside the 10 Hz comparison band, not a model error: with a
  3 Hz band it falls to 0.08 m/s^2 and the IMU position stays within a few mm.
- `imu_z_from_deck_top_mm` was first set to -23 mm by eye, which put the geometry 17 mm
  from the motion fit. Measured from the side it is about -10 mm. Measure it rather than
  guess, because the origin comes from the fit only when the two agree to 8 mm.
