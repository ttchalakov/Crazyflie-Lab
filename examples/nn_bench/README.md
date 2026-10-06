# nn_bench

The bench for whole networks on the GAP8. Each network is generated, compiled by
DORY, and run in GVSOC and on the AI-deck three ways: DORY with the PULP-NN
kernels it ships (`pulp_nn`), DORY with our kernels from `lib/gap8/kernels`
(`lib_kernels`), and where one exists a fused executor that runs the whole network in
one cluster team (`fused`). Every layer of every run is checked against the
generator's golden output.

```bash
python examples/nn_bench/host/bench.py                    # the common set, GVSOC and the deck (~25 min)
python examples/nn_bench/host/bench.py --set exhaustive   # every network, memory model and runner
python examples/nn_bench/host/bench.py --target gvsoc     # the simulator only; no drone (~2 min)
python examples/nn_bench/host/bench.py --list
python examples/nn_bench/host/point.py resnet18-160x120c1-c8to64-out1000 --memory l3 --gvsoc   # one network
python examples/nn_bench/host/run.py \
    examples/nn_bench/artifacts/points/resnet18-160x120c1-c8to64-out1000-l3 --runner lib_kernels  # on the deck
```

It needs Docker (the `cf-lab/nemo` and `cf-lab/aideck-gap8` images). The deck
runs need the drone on and a Crazyradio; the radio address comes from
`configs/drone.yaml`. Kernels on their own are measured in
[kernel_bench](../kernel_bench/README.md).

## Networks

`host/networks.py` names every network by its shape, in fields joined by `-`,
the input first, so any size can be asked for by name; its groups are what the
sets use.

| name | network |
| --- | --- |
| `mlp-12to256to256to4` | fully connected layers: the input length (a 12-byte state), each hidden width, the outputs |
| `mobilenetv1-<W>x<H>c<C>-a<NNN>-c<A>to<B>-out<K>` | MobileNet v1 (13 depthwise-separable blocks) on a W x H x C input, width multiplier alpha = NNN / 100, which gives A = max(8, int(32 alpha)) channels at the stem and B = 1024 alpha at the last pointwise layer, K outputs |
| `resnet18-<W>x<H>c<C>-c<A>to<B>-out<K>` | ResNet-18 (four stages of two basic blocks), stage 0 at A channels doubling to B = 8A in stage 3 |
| `convpool-160x96c1-c8s2mp-c16mp-c32mp-c32-out4` | 3x3 convolution units; the fused `convpool` executor's network |
| `dwpw-24x24c16-c32mp-c64mp-c96mp-out4` | depthwise-separable units (3x3 depthwise, 1x1 pointwise); the fused `dwpw` executor's network |
| `resblock-16x16c3-c8-c8-c16mp-c16-c16-out4` | basic residual units after an implied 3x3 stem, a 1x1 projection where the channels change |

`<W>x<H>c<C>` is a tensor of W x H pixels and C channels. A unit `c<C>[s2][mp]`
outputs C channels; `s2`: its first 3x3 has stride 2; `mp`: then a 2x2 max pool.

The weights are random int8 (`host/graph.py`): what is measured is how a
network of that shape runs and whether it computes exactly what it specifies,
not what it recognises. Each layer is requantised to uint8 as NEMO emits it, and
the input is xorshift bytes that the GAP8 regenerates rather than stores.

The camera networks take one gray camera frame, `160x120c1` or `320x240c1` (the
camera's QQVGA and QVGA modes). bench.yaml lists them without the input field
(`resnet18-c8to64-out1000`), and a set entry's `inputs` says which it runs on.
The groups go up to the largest DORY can run:

| | 160x120c1 | 320x240c1 |
| --- | --- | --- |
| mobilenetv1 | `a100-c32to1024`, the full network (4.2 MB of weights, 225 MMAC) | `a100-c32to1024` (4.2 MB, 867 MMAC) |
| resnet18 | `c28to224` (2.4 MB, 130 MMAC) | `c20to160` (1.3 MB, 261 MMAC) |

A wider resnet18 finds no L1 tiling in DORY, and on 320x240c1 its residual add
no longer fits L2 past `c20to160` (both under [Limits](#limits)); the weights of
either would still fit the HyperRAM. The `resnet18_widths` group goes to
`c20to160`; bench.yaml adds `c28to224` on 160x120c1 by name.

| group | networks |
| --- | --- |
| `mlp_fits_l2` | the nine MLPs whose weights fit the l2 model, 64 to 384 wide, square and not |
| `mlp_needs_l3` | the four MLPs only l3 holds |
| `custom_cnns` | `convpool`, `dwpw` and `resblock` |
| `mobilenetv1_widths` | `a025-c8to256` to `a100-c32to1024` |
| `resnet18_widths` | `c8to64`, `c16to128`, `c20to160` |

The exhaustive set also names `mlp-12to2560to2560to4`: 6.6 MB of weights, most
of the 8 MB HyperRAM.

## Memory models and runners

A **memory model** is where DORY keeps the weights. `l2` (DORY's `PULP.GAP8_L2`
target) links them into L2. `l3` (`PULP.GAP8`) keeps them in flash, loads them
into the 8 MB HyperRAM at boot, and stages each layer's into an L2 arena as it
runs, tiling a layer whose weights or activations do not fit; every camera
network needs it. A **point** is one network built for one memory model, which
its runners share.

A **runner** is how the network executes:

| runner | |
| --- | --- |
| `pulp_nn` | DORY's generated app with the PULP-NN kernels it ships |
| `lib_kernels` | the same app, the same tiling and DMA, with `lib/gap8/kernels` linked in place of PULP-NN's arithmetic: a controlled comparison of the kernels |
| `fused` | a fused executor (`gap8/fused/`, `host/fused.py`): the activations stay in L1 for the whole inference, a 2x2 pool is fused into the convolution before it and a depthwise convolution into the pointwise after it |

There are three fused executors, each for one family: `mlp` (any MLP; on l3 it
streams DORY's own weight files from HyperRAM through L2 into L1), `convpool`
(3x3 convolution units, each optionally pooled) and `dwpw` (depthwise-separable
units). A layer an executor never stores is reported skipped; the network's
output is always checked. The convpool and dwpw executors read their weights
from L2, so they run on l2 only.

## What a run checks

Before anything runs, `host/reference.py` checks that DORY made one layer per
graph layer with the generator's checksums, and for an MLP that the weights DORY
emitted compute the goldens again (a square layer cannot tell a transposed matrix
from its shape).

On the GAP8 (`gap8/nn_bench.c`) the network runs six times (four in GVSOC) on one
cluster kept open at 150 MHz. After each layer the network calls back with its
output; the app takes its CRC-32, compared with the host's, and the layer's
cluster cycles. A layer whose output DORY keeps in HyperRAM is read back to be
hashed. The hashing is taken out of the timing. The report is one `NN|` line per
layer and run: printed twice in GVSOC, whose semihosting now and then drops a
line, and replayed every few seconds on the deck, whose radio console attaches
late.

A result passes when every run reported every layer exact (or skipped). The time
is the median of runs 1 onward; run 0 also warms the caches.

## Output

```text
artifacts/bench_<stamp>/
  run.json, results.jsonl, summary.md      the set; one line per point, runner and platform; the table
  logs/<job>.log                            gvsoc-<network>-<memory>, aideck-<network>-<memory>-<runner>
  points/<network>-<memory>/
    graph.json, nemo/, dory/                the generated network and DORY's input
    gap8_app/                               DORY's app
    gen/                                    nn_bench_golden.h, nn_bench_net.h, nn_bench_fused.h
    host.json                               MACs, weights, the host checks, the fused executor
    gvsoc/<runner>.log, .json               GVSOC's output and verdict
    aideck/<runner>/                        build.log, nn_bench.img, console.log, report.json
```

## Limits

The DORY in [`external/dory`](../../external/dory) is our fork, with the fixes
these networks need. What the networks and the app are built around:

- **MCHAN's length field is 15 bits, not the 16 the SDK declares**: a DMA
  transfer over 32768 bytes moves `size & 0x7FFF` bytes and reports success.
  GVSOC models 16 bits and cannot show it. DORY's `dory_dma.c` splits transfers
  at 32764 bytes, the fused executors at 32 KB.
- **GVSOC passing is not evidence for the deck**: the MCHAN length above, and the
  L2 left for the arena (the next item), differ between the two.
- **On the deck the app needs 180 KB of L2**, not DORY's default 132 KB: ~125 KB
  of code and data for a network of ~30 layers, then the readfs mount, the CPX
  queues and the task stacks at run time. `point.py` generates every point with
  180 KB reserved, which leaves DORY a 332 KB arena.
- **DORY cannot tile a 3x3 convolution with more than ~250 input channels**: its
  im2col buffer is 2 x 8 cores x 9 x C bytes, all of L1's 36.7 KB at 256
  channels, and DORY does not tile input channels. So resnet18 stops at
  `c28to224`, whose last stage runs in L1 tiles of one output channel and one
  pixel: exact, but slow.
- **A residual add runs whole in L2**: both inputs and the output, in the 332 KB
  arena, as DORY has no L3 tiling for an add; DORY refuses a larger one at
  codegen. resnet18's stage 0 on 320x240c1 is 80x60, so `c20to160` needs 288 KB
  and `c24to192` 346 KB.
- **l3 needs `initial_dir = 1`** with the input at the base of the arena before
  every run, and the flash partition table at 0x40000 on the deck but 0 in GVSOC.
- **DORY's HyperRAM reservations are rightsized from the generated app**
  (`rightsize_dory_hyperram()` in `src/cf_lab/aideck/dory_pipeline.py`), given the
  largest activation: `network.h`'s sizes are the L2 tile of a layer DORY keeps in
  HyperRAM, not its activation.
- **DORY's weight loader reads with `pi_fs_direct_read()`**: readfs's
  `pi_fs_read()` races the PMSIS event kernel; [hyperbus_bench](../hyperbus_bench/README.md)
  checks the load.
- **A zero-length area in a GAP8 flash image breaks every `pi_flash_read()`**;
  `repair_gap8_image()` in `src/cf_lab/aideck/flash.py` drops it for every app.
- **The kickoff task must not spin**: the HyperBus driver restarts every 512 B
  burst from an FC interrupt, so `nn_bench.c` ends its kickoff task with
  `vTaskDelete(NULL)`.
- **The fused MLP is slower than DORY on wide MLPs held in L2 and faster from
  HyperRAM**: with the weights in L2 its double-buffered tiles add a copy DORY's
  resident weights do not need.
- `pulp_nn_quant_u8` multiplies in int32 and narrows to int16, so a large
  accumulator can wrap without warning; the generator keeps every layer far from
  it.

## Open items

1. **Tile input channels in DORY's convolution tiler, and an add from L3**, to go
   past resnet18 `c28to224` and, on 320x240c1, `c20to160` (see [Limits](#limits)).
2. **L3 inference time moves with the build**: the HyperBus driver starts every
   512 B burst from an FC interrupt, so its throughput depends on the FC's
   response ([hyperbus_bench](../hyperbus_bench/README.md)).
3. **A fused executor for the camera networks**: the three executors keep every
   activation in L1, which a camera frame does not fit.

## Files

```text
host/bench.py, bench.yaml   the bench and its sets
host/networks.py            the networks and their groups
host/graph.py               the generator: NEMO's ONNX dialect and the golden outputs
host/point.py               one network on one memory model: generate, DORY, host checks, GVSOC
host/run.py                 one point on the deck: build, flash, record, judge
host/reference.py           the host checks and the golden CRCs
host/fused.py               which fused executor runs a network, and its header
gap8/nn_bench.c, Makefile   the app: runs a network, checks each layer, reports
gap8/fused/                 the fused executors: mlp.c, convpool.c, dwpw.c
```

The glue that lets a DORY network run inside a FreeRTOS app is shared with
hover_neural_network, in [`lib/gap8/dory/`](../../lib/gap8/dory/).
