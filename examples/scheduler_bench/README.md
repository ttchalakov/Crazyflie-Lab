# scheduler_bench

Measures [lib/gap8/sched](../../lib/gap8/sched)'s two schedulers on the AI-deck:
a DORY policy running beside the camera and the WiFi video stream. Each case
builds one image holding both schedulers, flashes it, and runs them back to back
on the same boot (sequential, then pipelined), so both see the same build
layout, the same network and the same scene.

```bash
python examples/scheduler_bench/host/bench.py                         # the common set (~35 min)
python examples/scheduler_bench/host/bench.py --set mlp_smoke         # one case (~4 min)
python examples/scheduler_bench/host/bench.py \
    --case convpool-c4s2mp-c8mp-c16-out4-l2-qqvga120-jpegcl \
    --case mlp-12to512to512to4-l3-arena64k-qqvga60-raw-100hz
```

It needs Docker (the `cf-lab/nemo` and `cf-lab/aideck-gap8` images), the drone
on, a Crazyradio, and this computer's WiFi free to join the deck's access point
(`aideck-sched`). The radio address comes from `configs/drone.yaml`. Settings
are in [`host/bench.yaml`](host/bench.yaml).

## Cases

A case is a network, a camera configuration and, for a state policy, a request
rate: `<network>-<sensor>-<format>[-<rate>hz]`, for example
`mlp-12to512to512to4-l3-arena64k-qqvga60-raw-100hz` or
`convpool-c4s2mp-c8mp-c16-out4-l2-qqvga120-jpegcl`.

- **Networks** have random integer weights and go through the real NEMO/DORY
  pipeline (`host/sched_net.py`). They are named as
  [nn_bench](../nn_bench/README.md) names its networks, without the input, which
  the camera supplies, and with DORY's memory model and, on `l3`, the L2 arena
  DORY may use: `<network>-<memory>[-arena<N>k]`. An `mlp` is a state policy:
  synthetic states arrive at the case's rate (100 or 200 Hz). A `convpool` is an
  image policy, units of a 3x3 convolution (`c<C>[s2]`) and a 2x2 max pool
  (`mp`): it is built at the camera's frame size and runs on the freshest frame.
  `l2` links the weights into L2; `l3` keeps them in HyperRAM. The generated DORY
  app and its `sched_net.h` land in `artifacts/networks/`, under the network's
  name with a `convpool` network's input after the family
  (`convpool-160x120c1-c4s2mp-c16mp-c32-c32-out4-l3-arena96k`).
- **Cameras** are camera_stream modes, named as camera_bench names them: the
  sensor (`qvga60`, `qqvga120`, `qqvga60`: QVGA at 60 fps, QQVGA at 120 and 60
  fps) and the video format (`jpegcl`, JPEG encoded on the cluster, or `raw`).
- **Sets**: `common`, `mlp_smoke` (one state policy) and `convpool_smoke` (one
  image policy).

## What a case does

1. Generates the network if needed, and builds the app for the case's camera
   and rate (`SCHED=BOTH SCHED_METRICS=1 SCHED_VERIFY=1`).
2. Flashes it, power-cycles the drone, joins the deck's access point and
   receives the video over UDP.
3. The deck opens everything once and checks the network against DORY's golden
   output. It reports its free L2 (`SB|mem`, `SB|l2`) and a clock check, waits
   `settle_ms` for the host, then runs each scheduler for `phase_ms`. Each
   scheduler re-checks the network every 5 s, then prints its measurements as
   `SB|` lines on the radio console (over WiFi they would compete with raw
   video).

Results land in `artifacts/bench_<stamp>/`: `summary.md`, `results.jsonl`, and
per case `build.log`, `console.log` (the radio console) and `reports.txt` (every
`SB|` line).

| column | meaning |
| --- | --- |
| inf/s | inferences per second |
| lat | state posted (or frame captured) to inference done, µs; p50, p99, max |
| wait | the part of `lat` before the inference started |
| period p99 | between consecutive inferences |
| sent/s, host fps | frames the deck queued, frames that arrived (a raw QQVGA frame spans ~20 UDP datagrams and any one lost loses it, so raw cases arrive at well under sent/s) |
| enc p50 | JPEG encode, all slices |
| send max | copying one frame into the CPX queue (FC) |
| cluster % | cluster busy |
| verify | failed / run golden checks |

## What the schedulers are built around

- **`pi_time_get_us()` steps back a whole tick** when the SysTick timer wraps
  between its two reads, because the tick interrupt is still pending.
  `sched_time_us()` counts the pending tick; every boot's `SB|clock` line counts
  both clocks' backward steps.
- **The SDK's `memcpy` and `memset` copy one byte per iteration**, and a CPX
  packet is copied three times on its way to the SPI link.
  [lib/gap8/fastmem](../../lib/gap8/fastmem) wraps both for every caller.
- **A JPEG encode runs in slices**, and a posted state is served between two of
  them, so a state waits for at most one slice of the encoder.
- **Queueing a frame is FC work**, so it runs in a background task while the
  worker waits for the cluster: the cluster's owner never waits on the link.
- **QVGA leaves a network little L2.** At QVGA the camera keeps two buffers, not
  three (three would take 230 KB), and the JPEG encoder's output budget is a
  third of the raw frame rather than its worst case.
- **PMSIS zeroes a reused cluster task's `stack_size`** (not `slave_stack_size`)
  when the task ends, so lib/gap8/sched and the JPEG encoder set both sizes
  before every send; a send that finds no L1 for its stacks is counted
  (`clerr`).
- **Where L2 goes** is reported at every boot (`SB|l2`): the camera buffers, the
  JPEG buffers, the CPX queues, DORY's arena and what is left; `static_l2` in
  `results.jsonl` is what the image itself takes.

## Open items

- **The image + state input**: one input vector holding both the frame and the
  state, assembled by the library, for policies trained on both.
- **Zero-copy sends.** CPX copies each packet three times on the FC.
  nanocockpit sends straight from the frame buffer in 4 KB SPI packets.
- **DORY per-layer yield points**, for an image policy that must also answer
  fast state requests.
- **QVGA l3 image networks.** The arena holds a copy of the frame. DORY could
  read the input from HyperRAM, or from the camera buffer in place.
- **STM32 requests** in place of the synthetic state generator.
- **Raw video under the pipelined scheduler.** It queues fewer raw frames than
  the sequential one: its default send policy skips a frame while the previous
  one is still being sent, which favours inference over video, and stays the
  default. Worth measuring: sending whenever the queue has room, and whether a
  faster link (zero-copy, TCP) changes the picture.

## Files

```text
host/bench.py, bench.yaml   the bench, its cases and sets
host/sched_net.py           the networks: model, DORY codegen, sched_net.h
gap8/scheduler_bench.c      the app: synthetic states, both phases, SB| reports
gap8/Makefile               NET_DIR + the camera and rate make variables
```
