# debug_demo — tagged debug/console logging on real hardware

A minimal end-to-end showcase of cf-lab's source-tagged logging. A paired STM32
controller and GAP8 app each print **one line per second**; the host connects over
the radio, captures the merged firmware console, and tags every line with its tier
and a shared `[t=..]` time tag.

```
GAP8 app  --cpxPrintToConsole(LOG_TO_CRTP)-->  STM32  --CRTP console--> radio --> host
controller --DEBUG_PRINT-------------------->  STM32  --CRTP console--> radio --> host
```

- **[gap8/debug_demo.c](gap8/debug_demo.c)** — `cpxInit()` then a 1 Hz
  `cpxPrintToConsole(LOG_TO_CRTP, "gap8 tick N")`. The STM32 forwards it marked
  `CPX: GAP8:`, so the host tags it `[GAP8]`.
- **[controller/src/debug_demo_controller.c](controller/src/debug_demo_controller.c)** —
  an `appMain()` with a 1 Hz `DEBUG_PRINT("controller tick N")` (module `DBGDEMO`),
  tagged `[STM32]`.
- **[host/run.py](host/run.py)** — connects over the radio, captures for
  `--duration` seconds (10 by default), saves every tagged line to
  `artifacts/debug_log/debug_log.txt` and prints that file at the end
  (`make_run_logger(..., echo_console=True)` also echoes them live).

The tagging lives in [make_run_logger](../../src/cf_lab/runner/control.py) and is
shared with the flight runners (see `mocap_positioning`).

## Run it

Flash both firmwares, then capture over the radio:

```bash
# STM32 controller (1 Hz DEBUG_PRINT)
cf-lab build controller --example debug_demo
cf-lab flash controller --example debug_demo --config configs/drone.yaml

# GAP8 app (1 Hz cpxPrintToConsole)
cf-lab build gap8 --example debug_demo
cf-lab flash gap8 --example debug_demo --config configs/drone.yaml

# capture the tagged console
.venv/bin/python examples/debug_demo/host/run.py --config configs/drone.yaml --duration 10
```

Expected (captured file, printed at the end):

```
[t=  0.01s] [HOST] Connecting to radio://0/80/2M/E7E7E7E7E7; capturing firmware console for 10s.
[t=  1.01s] [HOST] host heartbeat
[t=  1.12s] [STM32] DBGDEMO: controller tick 4
[t=  1.20s] [GAP8] gap8 tick 4
[t=  2.01s] [HOST] host heartbeat
[t=  2.12s] [STM32] DBGDEMO: controller tick 5
[t=  2.20s] [GAP8] gap8 tick 5
```
