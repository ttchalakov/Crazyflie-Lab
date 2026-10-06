# Idle

- `gap8/`: parks the GAP8 with the camera and cluster unused.
- `controller/`: minimal STM32 out-of-tree app for the Crazyflie 2.1 Brushless.
  Uses the standard firmware controllers; the app itself does nothing.

Build and flash either target from the repository root:

```bash
cf-lab build controller --example idle
cf-lab flash controller --example idle --config configs/drone.yaml

cf-lab build gap8 --example idle
cf-lab flash gap8 --example idle --config configs/drone.yaml
```
