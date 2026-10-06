# Telemetry Bench (stress test the recording pipeline)

A **benchtop** stress test for `cf_lab.telemetry` — the drone is never armed and
never flies. It loads every stream type at once, calls the shared
`cf_lab.telemetry.validate_run()` checker, and adds one bench-specific waveform
check:

```text
STM32 --radio--> host: 7 base blocks at --log-hz (12D state, PID commands, and
                       the tba/tbb STM32 and tga/tgb GAP8 waveforms), plus
                       any telemetry.custom blocks enabled in the YAML config
host fake mocap:       self-paced orbit at --mocap-hz, tapped raw, origin-zeroed,
                       decimated to the config cap and sent as extpose uplink
host:                  synthetic setpoints at 50 Hz, event marks, firmware console
```

## Why the firmware app

`controller/` is stock firmware plus an app (`CONFIG_APP_ENABLE=y`, no OOT
controller) that updates synthetic waveforms at 1 kHz and publishes them as log
groups `tba`/`tbb`. Every value has a closed form in the onboard tick, so the
host can verify each logged sample against ground truth — corruption, ordering
bugs, or block mixups show up as hard failures rather than plausible-looking
data. `tbb.blk` echoes `tba.tick` so cross-block timeline consistency is
checkable too.

`gap8/` generates matching synthetic signals at 1 kHz and sends them over CPX
at 100 Hz. STM32 exposes them as `tga`/`tgb`; the host records both groups and
checks GAP8 tick progression and waveform values. These carry STM32 logging
timestamps; no camera or WiFi connection is required. Flash both targets.
GAP8 log fields use one snapshot per log timestamp, so incoming CPX updates
cannot mix the tick from one sample with waveforms from another.

## Integrity checks (exit code 0 = all applicable checks pass)

- waveforms match ground truth computed from `tick` (tolerating the one-tick
  log-task sampling race; residual error is float32 `sinf` noise, ~1e-4)
- `tick` strictly increasing; delta distribution reported (log-task jitter)
- **extpose recreation**: the recorded `extpose` stream is rebuilt exactly
  (< 1e-9) from the raw `mocap` stream + the `origin_zeroed` event — proving a
  recorded run contains everything needed to recreate the signal the estimator
  received
- mocap tap keeps up at `--mocap-hz` with sequence numbers gapless
- clock fits converge (synthetic frame period recovered; STM32 rms < 5 ms)
- zero writer-queue drops (drops are reported per stream from the manifest)

Required host and radio streams must deliver at least 90% of their expected
rows; lower delivery, data corruption, or an integrity invariant failure makes
the bench exit nonzero.

With `--backend fake`, STM32 streams are explicitly reported as `SKIP`: fake
mode validates only the host mocap/extpose/setpoint path. Use the default radio
backend to require non-empty STM32 state, control, and custom streams.

The bench appends `telemetry.custom` from the selected config to its seven base
STM32 log streams. This makes it a safe, props-off way to validate that an opt-in
sys-ID profile exists in the connected firmware's log TOC and delivers data.
Generic manifest, delivery, sequence, clock, drop, and extpose-replay checks
live in `cf_lab.telemetry`; only the closed-form waveform checks (`tba/tbb` from
the STM32, `tga` from the GAP8) remain in this example.

## Run

```bash
cf-lab build controller --example telemetry_bench
cf-lab build gap8 --example telemetry_bench
cf-lab flash controller --example telemetry_bench --config configs/drone.yaml
cf-lab flash gap8 --example telemetry_bench --config configs/drone.yaml

# drone on, props off:
.venv/bin/python examples/telemetry_bench/host/run.py --duration-s 30

# heavier: 2 kHz mocap tap for a minute
.venv/bin/python examples/telemetry_bench/host/run.py --duration-s 60 --mocap-hz 2000

# hardware-free (mocap/extpose/setpoints paths only):
.venv/bin/python examples/telemetry_bench/host/run.py --backend fake --duration-s 5
```

Results land in `artifacts/run_<timestamp>/` like every telemetry run.
