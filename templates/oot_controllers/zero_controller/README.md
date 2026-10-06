# Zero Out-Of-Tree Controller

This is the smallest useful example of a C controller that runs inside the Crazyflie stabilizer loop.

It demonstrates how to:

- enable Bitcraze's out-of-tree controller hook with `CONFIG_CONTROLLER_OOT=y`
- implement `controllerOutOfTreeInit`, `controllerOutOfTreeTest`, and `controllerOutOfTree`
- write a `control_t` output

The output is intentionally zero:

```text
roll = 0
pitch = 0
yaw = 0
thrust = 0
```

`CONFIG_CONTROLLER_OOT=y` makes the out-of-tree controller the firmware's
default: `stabilizer.controller` starts at 0 (auto-select), which picks it. So
once flashed this controller is the one running, and it produces no lift. Set
`stabilizer.controller` to 1 to switch to PID without reflashing, or set it back
to 6 (the out-of-tree controller's index) or 0.

Build:

```bash
cf-lab build controller --app-dir templates/oot_controllers/zero_controller
```

Build, then flash:

```bash
cf-lab flash controller --app-dir templates/oot_controllers/zero_controller --config configs/drone.yaml
```
