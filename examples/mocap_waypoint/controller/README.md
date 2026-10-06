# Default Firmware (mocap waypoint)

The **stock** Crazyflie firmware for the mocap waypoint demo in
`examples/mocap_waypoint/` — no out-of-tree controller.

`app-config` sets:

- The Crazyflie 2.1 Brushless platform (`CONFIG_PLATFORM_CF21BL`) and its DShot
  motor options, keeping the firmware's default idle thrust once armed.
- The built-in **PID controller** (`CONFIG_CONTROLLER_PID`), with no app and no
  out-of-tree controller (`CONFIG_APP_ENABLE` / `CONFIG_CONTROLLER_OOT` unset).
  `src/mocap_waypoint.c` is an empty placeholder so the out-of-tree build still
  emits the `built-in.o` the firmware links; it adds no behavior.
- Larger radio downlink buffers for telemetry (`CONFIG_CRTP_TX_QUEUE_SIZE`,
  `CONFIG_LOG_MAX_BLOCKS`, `CONFIG_LOG_MAX_OPS`).
- `# CONFIG_RADIO_ASSERT_ON_COM_PROBLEM is not set`: a full radio queue or a bad
  checksum drops the packet instead of asserting, which would reboot the drone
  and cut the motors in flight.

Build and flash:

```bash
cf-lab build controller --example mocap_waypoint
cf-lab flash controller --example mocap_waypoint --config configs/drone.yaml
```
