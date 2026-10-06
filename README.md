# cf-lab

An SDK for research on the Crazyflie 2.1 Brushless with the AI-deck. One versioned
repository provides:

| Part | What it is |
|---|---|
| [`lib/`](lib/README.md) | Shared C for the GAP8 and STM32: CPX transport and bridge, camera streaming, JPEG, DORY integration, quantized NN kernels |
| `cf-lab` command | Builds and flashes the STM32 controller, GAP8 app, and ESP32 firmware; runs the NEMO/DORY neural-network pipeline |
| `cf_lab` Python package | Host toolkit: radio backends, mocap streaming, runners, preflight checks, telemetry recording |
| [`examples/`](#examples) | Complete apps, each laid out as `controller/`, `gap8/`, `host/` |
| [`external/`](external/README.md) | Pinned crazyflie-firmware, AI-deck ESP32 firmware, NEMO and DORY |

## Requirements

- Linux is supported; macOS is best-effort.
- Python 3.11+, Docker, and `arm-none-eabi-gcc` for STM32 builds. On Ubuntu:

  ```bash
  sudo apt-get install -y git python3-venv docker.io gcc-arm-none-eabi
  sudo usermod -aG docker "$USER"   # then log out and back in
  ```

- A Crazyradio with [USB permissions](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/usb_permissions/) set up.

## Install

### In your own project

Add cf-lab as a git submodule, pinned to a release, and install its Python package
from that checkout, so the C and Python sides always come from the same version:

```text
my_project/
  cf-lab/              git submodule, pinned to a release tag
  controller/          STM32 out-of-tree controller (Makefile, Kbuild, src/)
  gap8/                AI-deck GAP8 app (Makefile, sources)
  host/                Python that runs on your computer
  configs/drone.yaml   radio URI, mocap, camera (start from cf-lab/configs/drone.yaml)
```

```bash
cd my_project
git submodule add https://github.com/IllinoisReliableAutonomyGroup/Crazyflie-Lab cf-lab
git -C cf-lab checkout v0.1.0
git -C cf-lab submodule update --init --recursive external/crazyflie-firmware
git -C cf-lab submodule update --init external/aideck-esp-firmware
python3 -m venv .venv && source .venv/bin/activate
pip install -e ./cf-lab
mkdir -p configs && cp cf-lab/configs/drone.yaml configs/
```

For neural networks, also initialize NEMO and DORY:

```bash
git -C cf-lab submodule update --init --recursive external/nemo external/dory
```

Point your Makefiles at the checkout with `CF_LAB`; everything else is found from it.

```make
# gap8/Makefile
CF_LAB ?= $(abspath $(CURDIR)/../cf-lab)
include $(CF_LAB)/lib/gap8/cpx/cpx.mk

# controller/Makefile (exported so src/Kbuild can use it too)
export CF_LAB ?= $(abspath $(CURDIR)/../cf-lab)
CRAZYFLIE_BASE ?= $(CF_LAB)/external/crazyflie-firmware
OOT ?= $(CURDIR)
OOT_CONFIG ?= $(CURDIR)/app-config
include $(CRAZYFLIE_BASE)/tools/make/oot.mk
```

`examples/idle` is the smallest starting point. Build and flash from the project root:

```bash
cf-lab flash controller --app-dir controller
cf-lab flash gap8 --app-dir gap8
```

### Python only

For host code alone (no firmware builds), install a release directly:

```bash
pip install "cf-lab @ https://github.com/IllinoisReliableAutonomyGroup/Crazyflie-Lab/archive/refs/tags/v0.1.0.zip"
```

### Working in this repository

```bash
git clone https://github.com/IllinoisReliableAutonomyGroup/Crazyflie-Lab cf-lab && cd cf-lab
git submodule update --init --recursive external/crazyflie-firmware
git submodule update --init external/aideck-esp-firmware
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

One install includes everything: model tooling, telemetry, camera viewing, and
the test tools.

## Set up the drone

Once for a new drone, with the AI-deck attached, flash the STM32, then the GAP8,
then the ESP32:

```bash
cf-lab flash controller --example idle
cf-lab flash gap8 --example idle
cf-lab flash esp
```

The ESP32 step matters most: cf-lab's WiFi features need its firmware, and the
deck keeps it until you flash it again. [Getting started](docs/getting-started.md#set-up-the-drone)
explains the order.

## Updating

Move the submodule to a new tag and commit the change in your project:

```bash
git -C cf-lab fetch --tags && git -C cf-lab checkout v0.2.0
git -C cf-lab submodule update --recursive
git commit -am "cf-lab v0.2.0"
```

The editable install picks up the new Python automatically; rerun `pip install -e`
only when dependencies change. When a release changes the ESP32 firmware, flash it
again with `cf-lab flash esp`.

## Commands

| Command | Does |
|---|---|
| `cf-lab build controller\|gap8\|esp` | Build one processor's firmware (`--app-dir DIR` or `--example NAME`) |
| `cf-lab flash controller\|gap8\|esp` | Build, then flash over the radio (`--no-build`, `--artifact FILE`) |
| `cf-lab compile-model` | Compile a PyTorch model into a GAP8 app: NEMO quantizes it, DORY generates the C ([template](templates/neural_networks/README.md)) |
| `cf-lab reboot` / `cf-lab shutdown` | Power-cycle the drone / power it fully off |
| `cf-lab view` | Show the AI-deck camera stream in a window |
| `cf-lab find-deck` | Print the address of the drone's AI-deck, found by name over mDNS |
| `cf-lab preflight` | Read-only checks before flying ([guide](docs/safety.md)) |
| `cf-lab config-value KEY` | Print one value from the drone config, e.g. `radio.uri` |

Radio commands read `radio.uri` from `--config` (default `configs/drone.yaml` in the
current directory); `--uri` overrides it. GAP8, ESP32, NEMO, and DORY builds run in
Docker, so only the STM32 toolchain is installed on the host.

## Examples

| Example | What it shows | Parts |
|---|---|---|
| [idle](examples/idle) | Minimal STM32 app and a parked GAP8: the starting template | controller, gap8 |
| [telemetry_demo](examples/telemetry_demo) | GAP8 and STM32 counters over CPX, recorded on the host | controller, gap8, host |
| [debug_demo](examples/debug_demo) | Source-tagged console logging from the STM32 and GAP8 | controller, gap8, host |
| [telemetry_bench](examples/telemetry_bench) | Benchtop stress test of the telemetry recorder | controller, gap8, host |
| [camera_stream](examples/camera_stream) | The AI-deck camera as a WiFi video server, viewed and recorded with mocap | gap8, host |
| [camera_bench](examples/camera_bench) | Regression bench for camera_stream: builds, flashes and measures every camera mode | gap8, host |
| [wifi_bench](examples/wifi_bench) | Regression bench for the AI-deck WiFi link, GAP8 to ESP32 to host, over TCP and UDP | gap8, host |
| [mocap_positioning](examples/mocap_positioning) | Propellers-off live mocap position indicator | controller, host |
| [mocap_waypoint](examples/mocap_waypoint) | Flying mocap waypoint demo on the stock PID controller | controller, host |
| [hover_neural_network](examples/hover_neural_network) | PyTorch hover policy run at 100 Hz on the GAP8 alongside JPEG video | controller, gap8, host, model |
| [nn_bench](examples/nn_bench) | Whole DORY networks in GVSOC and on the GAP8, every layer checked against the host | gap8, host |
| [scheduler_bench](examples/scheduler_bench) | DORY networks sharing the GAP8 with the camera and the WiFi video stream | gap8, host |
| [kernel_bench](examples/kernel_bench) | GAP8 quantized-kernel development and profiling against PULP-NN | gap8, host |
| [jpeg_cluster_bench](examples/jpeg_cluster_bench) | Verification bench for the GAP8 JPEG encoder, FC and cluster | gap8, host |
| [hyperbus_bench](examples/hyperbus_bench) | AI-deck HyperFlash, flash filesystem and HyperRAM checks | gap8, host |
| [flash_bench](examples/flash_bench) | Regression bench for GAP8 flashing over the radio | gap8, host |

Each example's README has its build, flash, and run steps.

## Documentation

The documentation site is built from [`docs/`](docs/index.md) and published to
GitHub Pages with each release. Preview it locally with `zensical serve`.

- [Getting started](docs/getting-started.md): install and set up a new drone
- [Your own project](docs/your-project.md), [where to look](docs/where-to-look.md),
  [commands](docs/commands.md) and [safety](docs/safety.md)
- [The C libraries](lib/README.md) and [pinned submodules and forks](external/README.md)

## License and citation

GPL-3.0-only; see [LICENSE](LICENSE). Files derived from Bitcraze code keep their
original headers. If you use cf-lab in research, please cite it using
[CITATION.cff](CITATION.cff) (GitHub's "Cite this repository").
