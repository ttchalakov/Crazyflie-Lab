# Safety

## What stops the drone

1. **The firmware, onboard.** The Crazyflie supervisor handles arming, cuts the
   motors when the drone tumbles, disarms a drone left armed without taking
   off, and runs the emergency-stop watchdog. The host cannot override it. An
   emergency stop locks the motors off until the drone is power-cycled
   (`cf-lab reboot` does that).
2. **The flying examples' host runners** (`mocap_waypoint`, `hover_neural_network`):
    - **SPACE** sends an emergency stop.
    - **Ctrl-C** lands and disarms.
    - The runner does not arm unless mocap poses reached the drone while it
      settled.
    - Once armed, if no new pose reaches the drone for `mocap.pose_timeout_s`
      (1 s by default), the drone lands.
    - If sending poses to the drone fails, the runner sends an emergency stop.
    - The runner feeds the firmware's watchdog four times a second. After one
      second without it, for example when the radio link drops, the drone cuts
      its own motors.

The host keeps no geofence or attitude monitor of its own: limits like these
belong onboard, where they survive a host or radio failure.

## Preflight checks

`cf-lab preflight` runs read-only checks. Without `--checks` it only validates
the config and does not touch the drone:

```bash
cf-lab preflight --list-checks
cf-lab preflight --checks configuration,firmware-self-test,supervisor-ready,battery,static-imu \
  --stop-on-failure
```

Run the drone checks with the drone still and disarmed.
