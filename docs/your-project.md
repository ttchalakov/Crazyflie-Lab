# Your own project

A project adds cf-lab as a git submodule pinned to a release and installs the
`cf_lab` package from that checkout, so its C and Python always come from the
same version.

## Layout

```text
my_project/
  cf-lab/              the SDK, a git submodule pinned to a release tag
  controller/          STM32 out-of-tree controller (Makefile, app-config, Kbuild, src/)
  gap8/                AI-deck GAP8 app (Makefile, sources)
  host/                Python that runs on your computer
  configs/drone.yaml   the drone's radio address, mocap and camera settings
```

`examples/idle` in cf-lab is the smallest starting point: copy its `controller/`
and `gap8/` directories.

## Add cf-lab

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

The `cf-lab` command reads `configs/drone.yaml` from the directory you run it
in, so run it from the project root. For neural networks, also run
`git -C cf-lab submodule update --init --recursive external/nemo external/dory`.

## Makefiles

Point each Makefile at the checkout with `CF_LAB`; everything else is found from
it. In a copy of `examples/idle`, replace the `CF_LAB` line of the controller's
Makefile:

```make
# controller/Makefile (CF_LAB is exported so src/Kbuild can use it too)
export CF_LAB ?= $(abspath $(CURDIR)/../cf-lab)
CRAZYFLIE_BASE ?= $(CF_LAB)/external/crazyflie-firmware
OOT ?= $(CURDIR)
OOT_CONFIG ?= $(CURDIR)/app-config
include $(CRAZYFLIE_BASE)/tools/make/oot.mk
```

A GAP8 app pulls in cf-lab's C libraries by including their Make fragments:

```make
# gap8/Makefile
CF_LAB ?= $(abspath $(CURDIR)/../cf-lab)
include $(CF_LAB)/lib/gap8/cpx/cpx.mk
```

## Build and flash

```bash
cf-lab flash controller --app-dir controller
cf-lab flash gap8 --app-dir gap8
```

`cf-lab build` builds without flashing. A new drone still needs the setup in
[Getting started](getting-started.md#set-up-the-drone); those commands work the
same from a project.

## Python only

For host code alone, with no firmware builds, install a release directly:

```bash
pip install "cf-lab @ https://github.com/IllinoisReliableAutonomyGroup/Crazyflie-Lab/archive/refs/tags/v0.1.0.zip"
```

## Updating

Move the submodule to the new tag and commit the change in your project:

```bash
git -C cf-lab fetch --tags && git -C cf-lab checkout v0.2.0
git -C cf-lab submodule update --recursive
git commit -am "cf-lab v0.2.0"
```

`submodule update` moves every submodule you initialized to the release's
commits. The editable install picks up the new Python by itself; rerun
`pip install -e ./cf-lab` only when the dependencies change. When a release
changes the ESP32 firmware, flash it again with `cf-lab flash esp`.
