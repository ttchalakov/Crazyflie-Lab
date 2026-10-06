# Where to look

## The repository

| Path | What it holds |
|---|---|
| `examples/` | Complete apps, each laid out as `controller/` (STM32), `gap8/` (GAP8) and `host/` (your computer), with a README |
| `lib/` | The C libraries for the GAP8 and STM32; `lib/README.md` lists every component |
| `src/cf_lab/` | The Python package and the `cf-lab` command |
| `templates/` | Starting points: the smallest out-of-tree controller and the smallest neural network |
| `configs/drone.yaml` | The drone's radio address, mocap, camera WiFi and telemetry settings |
| `external/` | Pinned submodules: crazyflie-firmware, the AI-deck ESP32 firmware, NEMO and DORY. `external/README.md` says which are forks; each fork's `CHANGES.md` lists what it changes |
| `docker/` | The images for GAP8 builds and for NEMO/DORY |
| `tests/` | The Python tests: `pytest -n auto` |
| `docs/` | This site |

## Examples

| Example | What it shows | Parts |
|---|---|---|
| `idle` | Minimal STM32 app and a parked GAP8: the starting template | controller, gap8 |
| `telemetry_demo` | GAP8 and STM32 counters over CPX, recorded on the host | controller, gap8, host |
| `debug_demo` | Source-tagged console logging from the STM32 and GAP8 | controller, gap8, host |
| `telemetry_bench` | Benchtop stress test of the telemetry recorder | controller, gap8, host |
| `camera_stream` | The AI-deck camera as a WiFi video server, viewed and recorded with mocap | gap8, host |
| `camera_bench` | Regression bench for camera_stream: builds, flashes and measures every camera mode | gap8, host |
| `wifi_bench` | Regression bench for the AI-deck WiFi link, GAP8 to ESP32 to host, over TCP and UDP | gap8, host |
| `mocap_positioning` | Propellers-off live mocap position indicator | controller, host |
| `mocap_waypoint` | Flying mocap waypoint demo on the stock PID controller | controller, host |
| `hover_neural_network` | PyTorch hover policy run at 100 Hz on the GAP8 alongside JPEG video | controller, gap8, host, model |
| `nn_bench` | Whole DORY networks in GVSOC and on the GAP8, every layer checked against the host | gap8, host |
| `scheduler_bench` | DORY networks sharing the GAP8 with the camera and the WiFi video stream | gap8, host |
| `kernel_bench` | GAP8 quantized-kernel development and profiling against PULP-NN | gap8, host |
| `jpeg_cluster_bench` | Verification bench for the GAP8 JPEG encoder, FC and cluster | gap8, host |
| `hyperbus_bench` | AI-deck HyperFlash, flash filesystem and HyperRAM checks | gap8, host |
| `flash_bench` | Regression bench for GAP8 flashing over the radio | gap8, host |

Each example's README has its build, flash and run steps. Run the examples
from the repository root.

## The Python package

| Module | What it does |
|---|---|
| `cf_lab.config` | Loads `configs/drone.yaml` |
| `cf_lab.backends` | The radio link to the drone, and an in-memory fake for dry runs |
| `cf_lab.crazyflie` | Firmware helpers: parameters, log variables, the boot self-test, the IMU |
| `cf_lab.runner` | Building blocks for an example's `host/run.py`: kill switch, watchdog heartbeat, sessions |
| `cf_lab.mocap` | Motion-capture pose sources (OptiTrack over NatNet, or a fake) and the stream to the drone |
| `cf_lab.camera` | The AI-deck image stream on the host: the feed, the viewer, rendering video, finding the deck |
| `cf_lab.telemetry` | Records a run's streams to Parquet, then loads and checks them |
| `cf_lab.analysis` | Turns recorded runs into datasets; checks clock alignment |
| `cf_lab.preflight` | Read-only checks before flying |
| `cf_lab.sim` | Differentiable, batched vehicle simulation |
| `cf_lab.aideck` | Compiling a model for the GAP8 and flashing the AI-deck |

## Neural networks

`cf-lab compile-model` turns a PyTorch model into C for a GAP8 app: NEMO
quantizes it and DORY generates the code, both in Docker. Start from
`templates/neural_networks/README.md`; `examples/hover_neural_network` is a
complete example.
