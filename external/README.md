# External submodules

The third-party code cf-lab builds against, as pinned git submodules. Where cf-lab
needs a change, the submodule points at a fork with the change committed on its
default branch, which keeps its upstream's name; a project we do not change is
pinned at its upstream. Each fork lists its changes in its own `CHANGES.md`.

| Submodule | Remote | What it is | Initialize |
|---|---|---|---|
| `crazyflie-firmware` | IllinoisReliableAutonomyGroup/crazyflie-firmware, a fork of bitcraze/crazyflie-firmware | STM32 flight firmware and the out-of-tree controller build | always |
| `aideck-esp-firmware` | IllinoisReliableAutonomyGroup/aideck-esp-firmware, a fork of bitcraze/aideck-esp-firmware | AI-deck ESP32 (NINA) WiFi firmware | to build or flash the ESP32 |
| `nemo` | pulp-platform/nemo | Quantization frontend of `cf-lab compile-model` | for neural networks |
| `dory` | IllinoisReliableAutonomyGroup/dory, a fork of pulp-platform/dory | GAP8 code generator of `cf-lab compile-model` | for neural networks |
| `dory/.../Backend_Kernels/pulp-nn` | IllinoisReliableAutonomyGroup/pulp-nn, a fork of pulp-platform/pulp-nn | The cluster kernels DORY's generated code calls | with `dory` (`--recursive`) |

```bash
git submodule update --init --recursive external/crazyflie-firmware
git submodule update --init --recursive external/nemo external/dory
git submodule update --init external/aideck-esp-firmware
```

## The forks

In each fork `origin` is the fork, and the submodule tracks its default branch
(`master`, or `main` for aideck-esp-firmware). To see what a fork changes, run
`git diff <commit>..HEAD` in it with the upstream commit below.

- **crazyflie-firmware**: bitcraze `master` at `9cf9d86c7` plus the changes in
  [CHANGES.md](crazyflie-firmware/CHANGES.md). cf-lab's STM32 builds use it, and
  `cf-lab flash gap8` checks the GAP8 MD5 it prints after a flash.
- **aideck-esp-firmware**: bitcraze tag `2025.02` (`58f15fa`) plus the changes in
  [CHANGES.md](aideck-esp-firmware/CHANGES.md), built with ESP-IDF 5.4.1.
  `cf-lab build esp` builds it in the `espressif/idf:v5.4.1` image and
  `cf-lab flash esp` flashes it over the radio.
- **dory**: upstream `add0d9c` plus the changes in [CHANGES.md](dory/CHANGES.md),
  which also covers the pulp-nn fork (one fix: `pulp_nn_linear` reads the int32
  bias). DORY copies the pulp-nn sources into each generated app, so a pulp-nn
  change reaches a network only when `cf-lab compile-model` generates it again.

## Updating a fork

Commit the change in the fork, push its default branch, then commit the new
submodule pointer in this repo. pulp-nn is nested in dory: push pulp-nn first,
then dory, then this repo, or a fresh `git submodule update --init --recursive`
cannot fetch the commit dory points at.

## Open items

- The ESP32 fork's own GNU make build, `tools/build` scripts, CI workflow and
  README still target ESP-IDF 4.3; cf-lab builds it with `idf.py` under 5.4.1.

## After switching branches or pulling

```bash
git submodule sync --recursive
git submodule update --recursive
```
