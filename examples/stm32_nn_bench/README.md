# STM32 network bench

Times small MLPs in every `cf_lab.nn` format on the Crazyflie's STM32 and checks that the drone computes the same
output bits as the host build of the same C. A second mode runs the networks as a periodic task beside the
stabilizer and times the stabilizer loop around the stock INDI, to see what a network task does to the inner loop.
The drone is never armed: the bench only writes parameters, and the stabilizer keeps the motors stopped.

- `controller/`: the bench app (`src/nn_bench.c`), and an out-of-tree controller (`src/nnb_controller.c`) that runs
  the stock INDI (or PID, `nnbench.inner = 0`) unchanged and times every stabilizer tick.
- `host/bench.py`: quantizes the networks, writes `controller/src/nnb_generated.{c,h}`, builds, flashes, runs the
  bench and compares the drone's hashes with the host's.

```bash
# networks from a PyTorch checkpoint (state dict under KEY, Linear layers under PREFIX), or random ones
.venv/bin/python examples/stm32_nn_bench/host/bench.py \
  --net policy=ckpt:model.pt:policy:net:silu --calibration policy=features.npy
.venv/bin/python examples/stm32_nn_bench/host/bench.py --net wide=shape:32,256,256,4:silu

# the load test: networks at 20-200 Hz and task priorities 1-4 beside the stabilizer
.venv/bin/python examples/stm32_nn_bench/host/bench.py --net ... --load [--traffic] [--no-flash]
```

The integer formats are calibrated on `--calibration` rows (inputs the deployed network sees); without them, on
standard normal inputs, which is enough for timing but not for accuracy. `--traffic` adds a 100 Hz external-position
uplink and a 100 Hz log stream, about what a mocap flight puts on the radio. Results land in
`examples/stm32_nn_bench/artifacts/<time>/` (or `--out`).

## What the report means

- `hash ... match`: the drone's outputs on the test inputs hash to the host's (identical bits).
- `alone us`: one network with the scheduler suspended, the compute alone; `task med/p99/max`: at the app's
  priority, where the stabilizer and the other tasks preempt it.
- `place`: one layer with its weights copied to SRAM or the core-coupled RAM.
- `load`: per network set, priority and rate, the load's execution time and start jitter, and the stabilizer's
  period, late ticks, controller time and IMU-to-output latency over 4 s.

## Findings (Crazyflie 2.1 Brushless with the AI-deck, two 34-128-128 SiLU networks)

- Each network takes 0.47 ms (`s16s8`) to 1.1 ms (`f32x4`) of compute; the bench app's own int8 kernel is 12 %
  faster than the vendored CMSIS-NN `arm_fully_connected_s8` on the same layer.
- At task priorities 1-3 the stabilizer is unchanged by the load at any rate up to 200 Hz, with or without radio
  traffic: the same period spread (0.72-1.29 ms, 5 % of ticks more than 10 % late, as without load), the same
  IMU-to-output latency (mean 0.52 ms, max 0.75 ms), INDI at 11 µs mean, 29 µs max.
- At priority 4, shared with the sensors task the stabilizer waits on, ticks run up to 3.1 ms apart and the
  IMU-to-output latency reaches 2.6 ms: keep network tasks at priority 3 or below. Scheduler-suspended runs block the
  stabilizer for their whole length (`stabilizer loop rate is off` on the console).
