# Box-Visualiser Out-of-Tree Controller

Position indicator for the demo in `examples/mocap_positioning/`. It **never
spins the propellers** — it commands zero thrust and zero attitude every tick
(same output as the `zero_controller` template) and the app-config floors the
armed idle thrust at 0, so an armed drone sits completely still.

Its useful work is the onboard box check. Each tick it reads the onboard state
estimate (which the host feeds from motion capture over extpose, re-zeroed onto
the drone's start pose) and reports whether the drone is inside the configured
safety box via the `box.inside` log variable.

The **LED indicator is driven host-side**: the runner lights the front LEDs
through the firmware's `led.bitmask` parameter (both greens inside the box, both
reds outside). `led.bitmask` is the firmware's sanctioned LED override and drives
the GPIOs directly; a raw `ledSet()` from a controller is gated by the LED
sequencer, so the host path is the reliable one.

The box is an axis-aligned rectangle defined by a centre and per-axis
half-extents in the host-zeroed mocap frame (origin at the start pose, +z up).
The host pushes the configured geometry into the `box.*` parameters on connect;
the firmware defaults (centre (0, 0, 1.5), widths 2.0 x 1.0 x 1.0) are a
fallback.

There is deliberately **no tilt or out-of-box disarm**: tilting or carrying the
drone out of the box only changes the LED colour. The runner never arms the
drone — the stabilizer calls this controller every tick regardless of arm state,
and the firmware force-stops the motors of an un-armed drone — so the props
physically cannot spin and the firmware's pre-flight/tumble auto-disarms never
apply. The demo ends only when the host asks (Ctrl-C or SPACE in the runner).

Parameters: `box.centerX/Y/Z`, `box.halfX/Y/Z` (metres).
Log group `box`: `inside` (uint8), `posX`, `posY`, `posZ`.

Build and flash:

```bash
cf-lab build controller --example mocap_positioning
cf-lab flash controller --example mocap_positioning --config configs/drone.yaml
```
