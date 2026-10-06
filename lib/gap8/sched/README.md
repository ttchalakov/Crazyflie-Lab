# GAP8 scheduler (sched)

`sched` runs one DORY policy on the AI-deck beside the camera and the WiFi video
stream. It owns the camera, the cluster and the link, and decides what runs
when. A policy takes the camera frame (an image policy), or a state vector the
application posts (a state policy). Its outputs come back through a callback.

Two schedulers, chosen at build time:

- **PIPELINED** (the default): the camera, the cluster, the fabric controller
  (FC) and the WiFi link work at once. It serves a posted state within about
  one JPEG slice, whatever the video is doing.
- **SEQUENTIAL**: capture, infer, encode, send, wait for the send to drain, in
  turn. It is the baseline the bench compares against.

[examples/scheduler_bench](../../../examples/scheduler_bench) measures both on
the deck.

## Three levels

**1. Give it the network.** Everything else comes from build settings.

```c
#include "sched.h"
#include "sched_net.h" /* generated: describes the DORY network (SCHED_NETWORK) */

static void on_output(const sched_output_t *out, void *context) {
  /* out->data: DORY's final output, out->bytes long; valid during the call */
}

static void start(void) {
  sched_run(&SCHED_NETWORK, on_output, NULL); /* returns only if setup fails */
}
```

```make
SENSOR = QQVGA_BIN2_120   # camera_stream mode
FORMAT = JPEG             # video: JPEG (cluster) or RAW
include $(CF_LAB)/lib/gap8/dory/dory.mk
include $(CF_LAB)/lib/gap8/sched/sched.mk
```

A state policy posts its input from any task. The next inference uses the
newest state, and a state posted while an older one waits replaces it. The tag
comes back in `sched_output_t.tag`, to tell which request an output answers:

```c
sched_post_state(state, sizeof(state), tag);
```

An image policy runs on the freshest frame, which must be exactly the network's
input size (`SCHED_ERR_SIZE` otherwise).

**2. Tune it.** `sched_config_init()` fills in the build defaults. Change what
you need, then call `sched_open()` and `sched_start()`:

| field | default | what it does |
| --- | --- | --- |
| `kind` | the one built | `SCHED_PIPELINED` or `SCHED_SEQUENTIAL` |
| `stream` | `STREAM=1` | send video to the WiFi host |
| `jpeg_quality` | `QUALITY=50` | JPEG quality |
| `jpeg_slices` | 0 (auto) | cluster jobs per encode. A posted state waits at most one slice. Auto is one per 4800 pixels for a state policy (4 at QQVGA, 16 at QVGA) and 1 for an image policy |
| `cluster_hz` | 150 MHz | the cluster clock |
| `report_ms` | 1000 | the period of `SCHED_LOG`'s report lines |

**3. Take control.** `send_policy` decides which frames go to the link. It sees
the frame's packet count, the free space in the CPX queue, and whether the
previous frame has drained. The default sends only once the previous frame has
drained ("skip while sending"). `sched_camera()`, `sched_jpeg()` and
`sched_cluster()` return the objects underneath. `sched_time_us()` is the clock
of every timestamp, and `sched_l2_free()` reports what L2 is left.

Each frame's metadata carries the time the frame was complete in its buffer.
`sched_clock_anchor(stm32_ms)` records that a message stamped on the STM32's
clock has just arrived; every frame then carries the newest such pair, and
`cf_lab.telemetry` uses the pairs to put the frames on the STM32 clock.

## Build settings (sched.mk)

| make variable | default | |
| --- | --- | --- |
| `SENSOR` | `QVGA_60` | `QVGA_30 QVGA_60 QQVGA_BIN2_30 QQVGA_BIN2_60 QQVGA_BIN2_120` |
| `FORMAT` | `JPEG` | video encoding: `JPEG` (8-core cluster) or `RAW` |
| `BUFFERS` | 2 at QVGA, else 3 | camera buffers. Three keep the newest frame ready while one is held; at QVGA they would take 230 KB of L2 |
| `QUALITY`, `STREAM` | 50, 1 | |
| `SCHED` | `PIPELINED` | `PIPELINED`, `SEQUENTIAL`, or `BOTH` (a bench) |
| `SCHED_METRICS` `SCHED_VERIFY` `SCHED_LOG` | 0 | instrumentation, compiled in only when set |
| `TXQ_SIZE` | one frame's packets | CPX send-queue slots, about 1 KB of L2 each |
| `WIFI_DEFINES` | access point `aideck-sched` | as cf_lab passes them |

The instrumentation costs nothing unless you build it in, so a demo can leave
it all off:

- `SCHED_METRICS` keeps per-phase counters and 100 µs histograms: latency,
  wait, inference, encode, send and period (`sched_get_stats()`).
- `SCHED_VERIFY` runs DORY's golden input through the network at open and every
  5 s, and checks the output's byte sum.
- `SCHED_LOG` prints a report line every `report_ms`.

`sched.mk` also links [fastmem](../fastmem/src/fastmem.c), which replaces the
SDK's byte-at-a-time `memcpy` and `memset`.

## The network

A `sched_network_t` describes a DORY network: its input kind and size, its
arena (DORY's L2 buffer), its cluster stacks, and its golden check.
`examples/scheduler_bench/host/sched_net.py` generates one as `sched_net.h` next
to the DORY app; `examples/hover_neural_network` writes its own. Both of DORY's
memory models work:

- **2-level (`l2`, `PULP.GAP8_L2`):** the weights are linked into L2, and the
  network reads its input in place from the camera frame or the state buffer.
  The arena only holds two consecutive layer outputs.
- **3-level (`l3`, `PULP.GAP8`):** the weights live in HyperRAM, loaded from
  flash at open. The arena holds the input (copied in before each inference),
  the activations and the weight tiles; DORY sizes it from the L2 budget it is
  given.

One application runs one network. A policy that needs both an image and a
state takes them in one input vector, which sched does not assemble (see
Limits).

## How the pipelined scheduler works

| task | priority | does |
| --- | --- | --- |
| worker | 1 | the cluster's only user, by priority: a posted state's inference, a new frame's inference, then the frame's JPEG encode in slices. A state that arrives during an encode is served at the next slice boundary |
| frame | 1 | leases the newest frame whenever the lease is free (camera_stream allows one lease) |
| link | 0 (background) | copies a prepared frame into the CPX send queue while the worker waits for the cluster, then releases a raw frame's lease |
| CPX, camera, PMSIS events | 2 | the SDK's and our drivers' own tasks |

What it accounts for:

- **The cluster** runs one job at a time. Inferences and JPEG slices are
  ordered by urgency, and the encoder hands its L1 back to DORY between slices.
- **The FC.** Copying a frame into the CPX queue is FC work even with fastmem:
  each packet is copied three times, once inside FreeRTOS. It runs in the
  background, overlapping cluster jobs, and never delays the worker.
- **The link.** A frame goes out only when the previous one has left (the send
  policy). One being queued is never overwritten.
- **The camera lease.** A JPEG frame is released right after its encode; a raw
  frame when the link has copied it.
- **The HyperBus.** Every task blocks rather than polls. The HyperRAM driver
  restarts each 512 B burst from an FC interrupt, so a spinning task slows L3
  weight streaming.

The sequential scheduler does the same jobs in turn on one task, and serves a
posted state once per camera frame.

## Limits

- The l3 image networks fit at QQVGA, not QVGA. Their arena holds a copy of the
  77 KB frame next to a 77 KB first activation.
- Queueing a raw QVGA frame takes a large share of the FC in the background.
  Zero-copy CPX sends (as in nanocockpit) would remove most of it.
- One input vector holding both a frame and a state, for policies trained on
  both, is not assembled by the library.
- DORY inferences are not sliced. An image policy plus a fast state request
  path would need per-layer yield points.
- States come from `sched_post_state()`. examples/hover_neural_network posts
  the STM32's CPX requests and replies from the output callback.
