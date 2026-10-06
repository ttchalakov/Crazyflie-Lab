# cf-lab

cf-lab is an SDK for research with the Crazyflie 2.1 Brushless and its AI-deck.
One versioned repository holds C libraries for the drone's processors, the
`cf-lab` command that builds and flashes them, the `cf_lab` Python package for
your computer, and complete examples, so the firmware and the host code always
come from the same release.

## The hardware

- **Crazyflie 2.1 Brushless.** Its STM32 runs the Crazyflie firmware and your
  controller.
- **AI-deck 1.1.** Its GAP8 runs the camera, neural networks and JPEG encoding;
  its ESP32 connects the GAP8 to WiFi.
- **Crazyradio.** Your computer reaches the drone over the radio, and the camera
  over the deck's WiFi.

The processors talk to each other over CPX, Bitcraze's packet protocol.

## Where to start

1. [Getting started](getting-started.md): install cf-lab and set up a new drone.
2. [Your own project](your-project.md): use cf-lab from your own repository.
3. [Where to look](where-to-look.md): what each part of the repository holds.
4. [Commands](commands.md): the `cf-lab` command.
5. [Safety](safety.md): read this before anything flies.
