# DORY and Deeploy on GAP8

What kernel_bench's measurements say about the compiler that deploys our networks
(DORY) and the one that might replace it (Deeploy). Every number is from
GVSOC at 150 MHz unless it says otherwise; the deck results in
[RESULTS.md from a bench run](README.md#output) agree in ranking.

## Decision

Keep DORY for deployment and as the exact-output baseline, and keep writing
DORY-compatible kernels underneath it. Deeploy is the better framework to build
a compiler on, but it has no GAP8 target, and nothing measured here says a new
schedule would pay for porting it.

- **Rebinding kernels under DORY is worth about 1.2×**, and is close to done: the
  overlay covers linear, convolution, pointwise, residual add and 3×3 depthwise.
- **Rescheduling the graph is worth 1.07–1.30× on top of that**, on the networks a
  fused executor exists for. nn_bench's executors, which keep every activation in L1
  and fuse every pool (and a depthwise into its pointwise), against DORY with our
  kernels on the same tensors, harness and clock:
  `convpool-160x96c1-c8s2mp-c16mp-c32mp-c32-out4` 1.17× in GVSOC, 1.07× on the deck;
  `dwpw-24x24c16-c32mp-c64mp-c96mp-out4` 1.28× and 1.30×. They only reach small
  networks, though: a camera frame's activations do not fit L1.
- **The largest structural cost is layers that do no arithmetic.** On MobileNet the
  four pooling layers are 15 % of the time and 0 % of the MACs, each paying a full
  layer's fixed cost and an L2 round trip. The useful question for a backend is
  whether it can emit fewer layers, not whether it can schedule them better.

## What a DORY layer costs

A graph built so that MACs and output count vary independently splits a layer
into three terms:

| term | PULP-NN kernels (`pulp_nn`) | our kernels (`lib_kernels`) | share of MobileNet |
| --- | ---: | ---: | ---: |
| per output element (requantise and store) | 2.87 cycles | 1.67 cycles | 22 % |
| per MAC | 0.1017 cycles | 0.1054 cycles | 29 % |
| fixed per layer | 3,657 cycles | 3,532 cycles | 9 % |

Our kernels' whole gain is the epilogue. About a quarter of that term is
requantisation arithmetic and three quarters is store, output DMA and pooling,
which caps epilogue work near 4 % of a graph.

The kernels themselves run at 6.5–7.3 MAC/cycle, 20–23 % of GAP8's 32. Dense and
MLP layers are 25–36 % arithmetic; convolution layers 50–79 %.

## DORY's limits on GAP8

- **It never overlaps transfers.** The shipping GAP8 configuration uses blocking
  DMA from one core, so only one transfer is ever in flight and a layer costs the
  *sum* of its transfers and compute. A transfer costs 143 cycles plus 4 bytes per
  cycle; a 16-byte copy costs 172 cycles, and DORY issues many for biases and
  batch-norm parameters. Double buffering gains nothing here, and at fine grain
  loses: its barriers cost more than the transfers they hide.
- **Upstream DORY's depthwise transfer hangs the cluster on hardware**:
  `dory_dma_memcpy_hwc_to_chw()` runs on core 0 but waits on a team barrier inside
  its per-channel loop, which cores 1–7 never reach. Our fork
  ([`external/dory`](../../external/dory)) waits on `mchan_transfer_wait()` instead.
- **MCHAN runs in the cluster clock domain**, so DORY's cycle counts barely move
  with the clock (+0.3 % from 50 to 150 MHz) while core-issued L2 traffic costs
  more cycles at a higher clock.
- **It works layer by layer**: each layer's output is written to L2 before the
  next reads it, and tiling, buffering and kernel choice are fixed heuristics.

## Deeploy

Deeploy brings modular platform and kernel bindings, per-kernel tile
constraints, whole-graph tensor lifetimes and static arenas, joint tiling and
memory placement, explicit hierarchical memory and asynchronous DMA, and a path
for binding custom kernels. With the same PULP kernels and memory limits its
published latency matches DORY's within 1 %; its larger gains need Siracusa
features GAP8 does not have.

For GAP8 it lacks a target, a runtime and a GVSOC runner; its tiled code is
still layer-wise; its optimisation is driven by memory feasibility rather than a
measured cost model; its PULPOpen L3 path wraps asynchronous DMA in a blocking
adapter; and nothing encodes GAP8's transfer-length limit or single-core MCHAN
rule.

A GAP8 target would have to model usable L1 after the cluster stacks, L2 and
HyperRAM regions; PMSIS cluster launch and the persistent-cluster contract;
MCHAN ownership, counters, chunking and barriers; asynchronous HyperRAM uDMA;
DORY-compatible quantisation, layouts and our kernels. Its first milestone is
parity: the same outputs and footprints as DORY on a pointwise network, a
residual network, an MLP and nn_bench's `resblock-16x16c3-c8-c8-c16mp-c16-c16-out4`.

## Where a narrow GAP8 backend could win

1. Choose PULP-NN's or our kernel per layer shape (`linear-256to4`, a classifier
   tail, runs 0.85× with ours while the layers above it gain 1.5×).
2. Use a cost model with kernel cycles, transfer bytes, the 143-cycle DMA floor,
   and a synchronisation term for barriers.
3. Fuse producer and consumer tiles (convolution plus pool, residual blocks),
   which also removes the pooling layers above and lets a block stay HWC,
   skipping DORY's per-channel transpose (6.6 % of MobileNet).
4. Keep reused weights in L2 when they fit, and batch small parameter copies.

## Links

- DORY: [paper](https://arxiv.org/abs/2008.07127) (Burrello et al., IEEE TC),
  [upstream](https://github.com/pulp-platform/dory),
  [our fork](https://github.com/IllinoisReliableAutonomyGroup/dory)
- Deeploy: [paper](https://arxiv.org/abs/2408.04413) (Scherer et al., CASES 2024),
  [repository](https://github.com/pulp-platform/Deeploy),
  [supported platforms](https://github.com/pulp-platform/Deeploy#supported-platforms),
  [tiling code generation](https://github.com/pulp-platform/Deeploy/blob/devel/Deeploy/TilingExtension/CodeTransformationPasses/TilingCodeGeneration.py),
  [PULPOpen L3 DMA](https://github.com/pulp-platform/Deeploy/blob/devel/Deeploy/Targets/PULPOpen/DMA/L3Dma.py)
- PULP-NN kernels: [paper](https://arxiv.org/abs/1908.11263) (Garofalo et al.),
  [repository](https://github.com/pulp-platform/pulp-nn)
- NEMO, the quantisation frontend: [repository](https://github.com/pulp-platform/nemo)
- GAP8: [GAP SDK](https://github.com/GreenWaves-Technologies/gap_sdk),
  [GVSOC](https://github.com/gvsoc/gvsoc)
- In this repository: [kernel_bench README](README.md), [nn_bench](../nn_bench/README.md),
  the DORY build glue in [`lib/gap8/dory/`](../../lib/gap8/dory/)
