# hyperbus_bench

Checks, from the GAP8, the path a DORY network's L3 weights take on the AI-deck:
HyperFlash → readfs → HyperRAM → cluster. The memory is the Infineon
S71KS512SC0B, a 512 Mbit HyperFlash and a 64 Mbit HyperRAM in one package on one
HyperBus.

```bash
python examples/hyperbus_bench/host/bench.py                     # a few minutes
python examples/hyperbus_bench/host/bench.py --keep-controller   # the STM32 already runs the idle controller
python examples/hyperbus_bench/host/bench.py --plan 4096:chatter:150 --iterations 20
python examples/hyperbus_bench/host/bench.py --no-load           # tests 1-3 only
```

It needs Docker (the `cf-lab/aideck-gap8` image), the drone on and a
Crazyradio; the radio address comes from `configs/drone.yaml`. Settings are in
[`host/bench.yaml`](host/bench.yaml); `--plan`, `--iterations` and `--tile-plan`
override them.

## What it checks

The bench puts the idle controller on the STM32, writes the load file, builds
the app, flashes it, power-cycles the drone and records the radio console. The
app runs the tests once, then prints its report forever (`HB|`), so a late
console still gets it.

| test | passes when |
| --- | --- |
| 1. HyperFlash reads at 0x0, 0x40000 and 0x48000 | the two inside the image (the flasher writes it at 0x40000) return the image's bytes as flashed. 0x0 is the bootloader, reported only. |
| 2. the flash filesystem | `pi_fs_mount()` succeeds and `hb_probe.txt` reads back its size and first 32 bytes |
| 3. HyperRAM | four fixed patterns and an address-dependent one read back exactly |
| 4. the L3 weight load | every load of `hb_load.bin` (608 KB of xorshift32) into HyperRAM reads back exactly, for every step of the plan |
| 5. the tiled layer | every 16 KB tile that core 0 fetches with `pi_cl_ram_read()` while 8 cores compute sums correctly |

Test 4 copies the file the way our DORY fork's `load_file_to_ram()` does:
`pi_fs_direct_read()` into an L2 heap buffer, then `pi_ram_write()`, chunk by
chunk. Each step is `chunk:spi:fc_mhz`: the chunk size, the CPX SPI traffic
around the load (`quiet`, `burst` = about 25 console lines queued just before it,
`chatter` = throughout) and the FC clock (the HyperBus clock follows it). A
watchdog task reports a load that stops making progress as `STUCK`. Test 5 runs
DORY's L3 weight tiling at a cluster clock under the same SPI modes. Everything
lands in `artifacts/bench_<stamp>/`: `summary.md`, `results.jsonl`, `run.json`,
`build.log`, the image, `console.log` and `report.json`.

## How the L3 path works

- **The flash.** The GAP8 bootloader (Bitcraze's) is at 0x0; the radio flasher
  writes the app image at 0x40000, and the image carries a readfs partition
  holding the `.hex` weight files. The flasher never touches 0x0.
- **Boot.** The ROM loads the bootloader into L2 (0x1C060000–0x1C078000). It
  waits for a CPX jump command, which the ESP32 sends when it starts (the STM32
  resets the GAP8 and the ESP32 together at its own boot), then loads the app's
  areas into L2 and jumps. It never boots the app on its own, so a GAP8 that
  reset alone looks like a silent hang. It jumps with the flash still open,
  which is harmless: the app's first `pi_flash_open()` re-initialises the
  controller.
- **mem_init().** DORY opens the HyperFlash, mounts readfs on it and opens the
  HyperRAM. Flash and RAM are two chip selects on one controller with one global
  transfer queue; the driver is reference-counted, and the last close turns the
  controller off.
- **network_initialize().** `load_file_to_ram()` copies each weight file into
  HyperRAM through a 4 KB buffer borrowed from the L2 heap for the load (halved
  down to 128 B if L2 is short). Each chunk is a flash read plus a HyperRAM
  write, and the fixed cost per transfer dominates, so bigger chunks load faster.
- **Inference.** A layer whose weights do not fit L2 is tiled: cluster core 0
  asks the FC for the next tile (`pi_cl_ram_read()`, a remote call the FC turns
  into a HyperBus transfer into L2) while the cores compute on the current one,
  moved L2 → L1 with MCHAN.
- **Bursts.** The GAP8 driver splits every transfer into 512 B bursts
  (`BURST_SIZE` in the SDK's `hyperbus_internal.c`) and starts each next burst
  from its interrupt handler, `hyper_handler`: a 264 KB weight read is 516 bursts
  and 516 FC interrupts. So HyperBus throughput is set by how fast the FC answers
  interrupts, not by the bus. It scales with the FC clock, drops when anything
  keeps interrupts masked, and moves with the build, most likely with where the
  handler's code lands.
- **Completions.** When a transfer's last burst ends, the handler pushes the
  waiting task. A callback task runs in the PMSIS event kernel, a FreeRTOS task
  at priority 2, above every app task (`pmsis_kickoff`'s is priority 1).
- **The SPI to the ESP32** (CPX, the console) is a separate uDMA channel. It
  shares only the uDMA's port into L2 with the HyperBus, not pins; loads and
  tiles are exact under any amount of it.

## What breaks the L3 path

- **A zero-length image area breaks every flash read.** A GAP8 image is a header
  and a table of areas (flash offset, L2 address, size) that the loader copies
  into L2. For some section layouts gapy ends the table with an area of size 0; a
  zero-length HyperBus transfer leaves the controller's read address
  unprogrammed, and every later `pi_flash_read()` ignores its address and
  streams on from where the loader stopped, so `pi_fs_mount()` fails.
  `repair_gap8_image()` in [`src/cf_lab/aideck/flash.py`](../../src/cf_lab/aideck/flash.py)
  drops such areas for every GAP8 app. Whether an image has one depends on where
  its sections fall, so it comes and goes with unrelated code changes.
- **readfs's `pi_fs_read()` races the event kernel.** readfs reads a file through
  a small state machine. When the caller's buffer and the file's flash address
  agree mod 8, it DMAs the flash straight into the buffer and advances its state
  (`pending_addr`, `pending_size`) only after issuing that read. The completion
  runs as a callback task in the event kernel, which preempts the loading task,
  so a read that completes first re-enters the state machine with stale state:
  the load hangs, or one chunk lands wrong in HyperRAM. When the two disagree mod
  8, readfs reads through its own cache, which is immune, so which path a buffer
  gets is a linker accident. DORY's loader in our fork and this bench use
  `pi_fs_direct_read()`, one flash read per call with no state machine; nothing
  in this repo calls `pi_fs_read()`.
- **A spinning task slows every L3 transfer.** A kickoff task ending in
  `while (1) pi_yield();` never sleeps: each yield is a FreeRTOS context switch,
  which masks interrupts and touches L2, so every 512 B burst above waits longer.
  End it with `vTaskDelete(NULL)`, as nn_bench does. This bench runs its tests in
  the kickoff task itself.

## JTAG without resetting the deck

The Olimex ARM-USB-TINY-H on the GAP8 header works, but the stock connect
**resets the deck**: the gap8 openocd target pulses SRST during `init`, and the
header's nRESET is the deck reset. Whatever you then read (PC in the
bootloader, L2 wiped) is the debugger's own reset. Point openocd's nSRST at an
unused FTDI pin and the app keeps running, so a hung app can be halted and read:

```bash
docker run --rm -it --privileged --network host -v /dev/bus/usb:/dev/bus/usb cf-lab/aideck-gap8:latest \
  gap8-openocd -s /usr/share/gap8-openocd/openocd/scripts \
    -f interface/ftdi/olimex-arm-usb-tiny-h.cfg -c 'ftdi_layout_signal nSRST -oe 0x4000' \
    -f target/gap8revb.tcl -c 'tap_select gap8_adv_debug_itf' -c 'du_select adv_dbg_unit 1' -c init
```

Do not change `ftdi_layout_init`: its bit 0x0200 at 1 asserts SRST and holds the
GAP8 and the ESP32 in reset, even after openocd exits. `init` halts the core, so
`resume` at once. Memory reads work while it runs (about 1 ms each, several per
TCL round trip on port 6666). `reg npc` reads 0; while halted, read the core
through the debug unit: GPR n at `0x1B300400 + 4n`, NPC `0x1B302000`, CSR c at
`0x1B304000 + 4c`. The HyperBus uDMA channel is at `0x1A102180` (RX address,
size, config; TX at +0x10; the external address at +0x20, flash addresses
carrying 0x01000000).

## Open items

- **Why L3 speed changes with the build.** One binary is reproducible, but
  builds differing only by a `#define` can land far apart, and only the layers
  that stage weights from HyperRAM move; compute-only layers stay flat. The GAP8
  caches instructions only, so the suspects are the FC side of the 512 B burst
  path: where the interrupt handler's code falls in the FC instruction cache, or
  where FC data (stacks, driver structures, the staging buffer) sits in L2
  relative to the DMA writes. A code-size change moves both. To separate them:
  one test reading 256 KB from HyperRAM into L2 destinations at a range of
  offsets, and one build option that pads the code by a few KB.

## Files

```text
host/bench.py            build, flash, record, judge
host/bench.yaml          the load file and the load and tile plans
gap8/hyperbus_bench.c    the five tests
gap8/Makefile            FS_PARTITIONTABLE_OFFSET=0x40000, the offset the flasher writes to
gap8/readfs/hb_probe.txt the file the mount test reads back (hb_load.bin is written by bench.py)
docs/README.md           links to the S71KS512 HyperMem datasheet, the AI-deck 1.1 datasheet and the rev C schematic
```
