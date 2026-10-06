# Telemetry demo

Two counters increment at 10 Hz: STM32 publishes `demoStm.tick`; GAP8 sends its
counter over CPX, which STM32 publishes as `demoGap.tick`. `demoGap.received`
counts received packets: if GAP8 stops, both GAP8 values stop changing while
STM32 telemetry continues. Samples carry STM32 log timestamps, not GAP8 timestamps.

From the repository root:

```bash
cf-lab build controller --example telemetry_demo
cf-lab build gap8 --example telemetry_demo
cf-lab flash controller --example telemetry_demo --config examples/telemetry_demo/config.yaml
cf-lab flash gap8 --example telemetry_demo --config examples/telemetry_demo/config.yaml
.venv/bin/python examples/telemetry_demo/host/run.py --config examples/telemetry_demo/config.yaml --duration-s 30
```

`run.py` uses the supplied YAML telemetry settings unchanged. The included config
records only the two custom streams; edit `custom` and `period_ms` to choose
variables/rates. Other configs must include these custom entries to record the
counters. Debug text and telemetry share a run folder under `artifacts/`.
This passive demo never arms or sends setpoints/extpose, so those streams have
no samples even if enabled. Configured state, controls, mocap, and camera can
also be recorded through the existing recorder.

No camera or WiFi is needed. Keep `telemetry.camera: false` for this firmware,
which sends counters only. If camera recording is enabled, the recorder tries
the configured feed and continues if it cannot connect; there is currently no
YAML setting to require a camera. `telemetry.enabled: false` skips recording.
