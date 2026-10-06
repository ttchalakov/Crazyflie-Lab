# Getting started

This page takes a new Crazyflie 2.1 Brushless with an AI-deck attached to a drone
that runs cf-lab firmware on all three processors.

## What you need

- A Crazyflie 2.1 Brushless with an AI-deck 1.1 attached, and a Crazyradio.
- A Linux computer (tested on Ubuntu 24.04) with Python 3.11 or newer, Docker,
  and `arm-none-eabi-gcc` for STM32 builds. On Ubuntu:

  ```bash
  sudo apt-get install -y git python3-venv docker.io gcc-arm-none-eabi
  sudo usermod -aG docker "$USER"   # then log out and back in
  ```

- [USB permissions](https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/usb_permissions/)
  for the Crazyradio.

The GAP8, ESP32, NEMO and DORY toolchains run in Docker; nothing else is
installed on your computer.

## Install

```bash
git clone https://github.com/IllinoisReliableAutonomyGroup/Crazyflie-Lab cf-lab && cd cf-lab
git submodule update --init --recursive external/crazyflie-firmware
git submodule update --init external/aideck-esp-firmware
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Run the `cf-lab` commands below from this directory: they read the drone's
radio address from `configs/drone.yaml`. A new Crazyflie answers at
`radio://0/80/2M/E7E7E7E7E7`, the address already there; if yours has another,
set `radio.uri` to it.

## Set up the drone

Turn the drone on with the AI-deck attached, plug in the Crazyradio, and flash
the three processors in this order:

```bash
cf-lab flash controller --example idle   # 1. STM32
cf-lab flash gap8 --example idle         # 2. GAP8
cf-lab flash esp                         # 3. ESP32
```

1. **STM32 first.** `cf-lab flash gap8` confirms every GAP8 flash with an MD5
   that cf-lab's STM32 firmware reports. Stock firmware does not report it, and
   its asserts can reboot the STM32 in the middle of a deck flash.
2. **GAP8 next.** The idle app parks the GAP8, so it stays quiet while the
   ESP32 is flashed.
3. **ESP32 last, and at least once for every deck.** This is the step that
   matters most. cf-lab's WiFi features need its ESP32 firmware: video that keeps
   streaming (the stock firmware stops transmitting under sustained streaming),
   UDP, the access point channel, and finding the deck by name. The ESP32 keeps
   it until you flash it again; STM32 and GAP8 flashes do not touch it.

Each command builds before it flashes. The first GAP8 and ESP32 builds download
their Docker images, which takes a while. The STM32 and GAP8 flashes take under
a minute, the ESP32 flash about four minutes.

## See it work

The camera is a quick check that the deck works end to end. Flash the
camera_stream example with the deck as its own WiFi network and JPEG video, join
that network, and open the viewer:

```bash
cf-lab flash gap8 --example camera_stream --wifi-mode access_point --make-args "FORMAT=JPEG JPEG=CLUSTER"
# join the open WiFi network "camera-stream", then:
cf-lab view
```

Joining the deck's network takes this computer off its other WiFi; a wired
connection keeps the internet. To have the deck join your own network instead,
set `camera.mode: station`, `station_ssid` and `station_key` in
`configs/drone.yaml`; the ESP32 only works on 2.4 GHz.

## If a flash fails

- A GAP8 flash that stalls is retried once on its own. If it fails again,
  power-cycle the drone and rerun the command, or add `--cold-boot`, which
  power-cycles the drone first.
- If GAP8 flashes stop at 0%, flash the ESP32 again (`cf-lab flash esp`), then
  `cf-lab flash gap8 --example idle`.

## Next

- [Your own project](your-project.md) sets up a repository that uses cf-lab.
- [Where to look](where-to-look.md) lists the examples; each one's README has its
  build, flash and run steps.
- Read [Safety](safety.md) before anything flies.
