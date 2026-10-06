# Flash bench

Profiling and verification benchmark for **production GAP8 flashing over Crazyradio**. Builds its own
fixtures, invokes `cf_lab.aideck.flash.flash_gap8`, checks the written image's MD5,
and checks the running fixture's identity and complete readfs payload. No flight
commands, camera, WiFi association or other examples' GAP8 binaries are needed.

From the repository root, with the drone powered and props off:

```bash
# List the exhaustive matrix without building or touching hardware.
.venv/bin/python examples/flash_bench/host/bench.py --set exhaustive --list
# Five common regression cases.
.venv/bin/python examples/flash_bench/host/bench.py
# All 17 lifecycle cases.
.venv/bin/python examples/flash_bench/host/bench.py --set exhaustive
# Repeat a specific transition; source setup is automatic and recorded.
.venv/bin/python examples/flash_bench/host/bench.py --case warm-a-b --repeat 6
# Select just the two power-cycle persistence cases.
.venv/bin/python examples/flash_bench/host/bench.py --case persist-small --case persist-large
```

The existing `cf-lab/aideck-gap8:latest` Docker image is required. The bench first
builds all fixtures and normally builds/flashes the **idle STM32 controller**
from [examples/idle](../idle), ensuring the current firmware supplies MD5 evidence. `--config`
selects the radio URI; `--uri` overrides it. `--keep-controller` retains the current
STM32 firmware, but missing MD5 evidence fails the bench. ESP32 firmware is left
in place. STM32 preparation is not itself an exhaustive STM32 flashing test;
ESP32/USB flashing are outside this matrix.

`host/cases.py` defines the fixtures, named cases, and selections. Exhaustive
covers A/B transitions and same-image reflashes, single-attempt writes, transitions
to/from silent and busy apps, small/1 MiB readfs image transitions, trailing-empty-
area repair, explicit cold-start flashes, and small/large image persistence after
power cycling. Setup flashes only occur when the last verified fixture does not
match the case's source. `--repeat` repeats the complete selection serially.
Persistence checks wait the production cold-start settling interval after power-up
before reconnecting to validate the stored image.

Allow roughly **20–25 minutes** for exhaustive on a 2 Mbit/s radio link,
including builds, controller preparation, observation and cleanup; the 1 MiB
readfs fixture takes the longest to write, and retries and radio conditions can
extend the run.

Normal cases retain production's warm-flash, image-repair, timeout, MD5, and
stall-only retry behavior. Every stall/retry is recorded; a recovered stall is a
passing functional result, not a clean first attempt. No failure is classified
as packet loss solely from its percentage. The benchmark run never intentionally
corrupts written firmware.

A reporting fixture must emit three advancing records with its expected ID,
byte count, full-payload hash, and no read error. Silent images require MD5 and
absence of GAP8 console output; absence alone is **not** a boot claim. Missing
MD5, wrong/stale identity, firmware faults, incomplete output, timeout, or console
overflow fail. Cases stop on failure; unrun coverage remains explicit. A worker
process bounds each device operation. The flash deadline scales with image size;
`--flash-timeout` overrides it.

Every benchmark run writes a new `artifacts/bench_<timestamp>/` (`--output` overrides,
inside the repository). It contains:

- `results.json`, `summary.txt`: incremental results, elapsed times, coverage,
  retries, controller/cleanup outcomes, and source/toolchain provenance.
- `images/`: build logs, immutable image snapshots and payloads, source/written
  hashes, and repair counts.
- Per-case `setup/`, `flash/`, `persistence/`: request/result JSON, timestamped
  progress events, worker log, raw post-flash console, and live STM32 revision.

After success, failure, or Ctrl-C the bench attempts to flash and verify its
silent GAP8 fixture. Cleanup failure fails the benchmark run and is preserved. It does
not restore the previous GAP8 application. STM32 remains on idle unless
`--keep-controller` was used. A second interruption may prevent cleanup.

Implementation is split into `bench.py` (lifecycle/CLI), `cases.py` (matrix),
`firmware.py` (fixture builds), `device.py` (production flash/observation), and
`results.py` (output). `gap8/` contains the small test firmware.

## Why the flasher works this way

- **Warm reflashing works.** Alternating two images with no power cycle, the
  GAP8 bootloader's MD5 of the region it wrote matches every time, so
  `flash_gap8` does not cold-boot by default.
- **A stall mid-transfer is a dropped radio packet.** cflib does not retransmit
  with safelink on, so the transfer stops. The flasher bounds the wait, names the
  stall and retries once.
- **The STM32 stays in its firmware.** The image goes through the running
  firmware's deck-memory subsystem over Crazyradio, then one reboot starts it;
  the STM32 never enters its own bootloader.

Recovering a deck that will not flash is in
[Getting started](../../docs/getting-started.md#if-a-flash-fails).

## Open items

| Item | Where | Why it matters |
| --- | --- | --- |
| The GAP8 "start app" command (`0x06`) is sent once at ESP boot and never re-sent. | `external/aideck-esp-firmware/main/system.c` | If it is lost, the GAP8 stays in its bootloader and looks like a failed flash until a power cycle. |
| "GAP8 in bootloader" is set after a fixed 100 ms delay, not on the ESP's reply. | `external/crazyflie-firmware/src/deck/drivers/src/aideck.c` (`resetGap8ToBootloader`) | A reset crowded out by a busy app is still reported as done, and the flash writes into a running app. |
| The ESP32 asserts on a CPX CRC mismatch from the STM32, which reboots it. | `external/aideck-esp-firmware/main/uart_transport.c` | Line noise can reboot the ESP32. A fix drops the frame but always returns the flow-control credit, as the STM32 side does, or the link wedges. |
| `flash_gap8` still keeps its boot-liveness fallback (`verify_boot`, `gap8_is_alive`, `DeckBootFailed`). | `src/cf_lab/aideck/flash.py` | Needed only for firmware that reports no MD5; delete once every drone runs our fork. |
| An STM32 assert can leave the drone unreachable until a physical power cycle (mechanism untested, probably the nRF51 cutting power when syslink stops). | STM32 / nRF51 | No host-side retry can recover from it. |
