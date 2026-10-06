# kernel_bench

Where GAP8 inference kernels are written and measured against the pinned PULP-NN
kernels that DORY uses. Every case is batch 1, checked exactly against an oracle
before its timing counts, and measured in GVSOC and on the AI-deck. The kernels
live in [`lib/gap8/kernels/`](../../lib/gap8/kernels/); this bench holds their
benchmark adapters. Whole networks, with these kernels under DORY and with the
fused executors, are measured in [nn_bench](../nn_bench/README.md). What the results
mean for DORY and Deeploy is in [DORY_AND_DEEPLOY.md](DORY_AND_DEEPLOY.md).

```bash
python examples/kernel_bench/host/bench.py                    # the common set, GVSOC and the deck (~5 min)
python examples/kernel_bench/host/bench.py --set exhaustive   # every suite (~1 h)
python examples/kernel_bench/host/bench.py --target gvsoc     # the simulator only; no drone
python examples/kernel_bench/host/bench.py --list
```

It needs Docker (the `cf-lab/aideck-gap8` image). The deck runs need the drone on,
a Crazyradio and the host's arm-none-eabi toolchain for the idle controller; the
radio address comes from `configs/drone.yaml`.

## What it measures

- **Kernel cycles**, the primary number: the largest `PI_PERF_CYCLES` reading across
  the cores, around the arithmetic only. Every speedup is a ratio of these. The FC
  elapsed time is reported beside it and includes cluster dispatch.
- **MAC/cycle** against GAP8's peak of 32 (eight cores, one four-lane dot product
  each per cycle).
- Each case runs one warm-up and three timed runs, or no warm-up and one run when
  its name ends in `-once`; the report gives the median and the spread. Small cases
  repeat inside one timed call to get above the timer's floor, except that a
  `-once` convolution, add or pooling case calls its kernel exactly once.
- Clocks: FC and cluster at 150 MHz on both targets. A case's name says whether its
  operands were in L1 or L2, which matters more than any kernel change.

Correctness comes first. Each case is scored against a scalar oracle computed on
the device. `configs/reference_results.json` holds golden checksums by case name
and kernel/variant, the smoke group's among them, so operands that drift show as
a diff: `host/reference.py --check <run>/gvsoc-s8bias/*/gvsoc.log` (and `--record`
after an intended change).

## Families, suites and configurations

| family | ours | compared with |
| --- | --- | --- |
| linear | `register4`: four output rows per register tile; fused uint8 epilogues | `pulp_nn_linear`, `pulp_nn_linear_out_32` |
| convolution, pointwise | `flattened_matmul4x2` (spatial partition across cores), 4x3 and 4x4 blocking, a 1x1 wrapper | `pulp_nn_conv_Ho_parallel` |
| depthwise | `direct3x3_2x1` | `pulp_nn_depthwise_generic` |
| matmul | the L1 microkernel alone, with TCDM-contention counters | `pulp_nn_matmul` |
| residual add | packed flattened traversal | PULP-NN's add |
| activation | sigmoid, tanh, LeakyReLU, hard-sigmoid, hard-swish, SiLU, GELU, softmax | the timed `scalar` variant (PULP-NN has none) |
| batchnorm | packed fixed statistics, per-frame statistics | the timed `scalar` variant |
| pooling | max, average, global and adaptive average | PULP-NN's pooling, the timed `scalar` variant |

A **case** is one problem, named from its fields as
`[<function>-]<op+shape>[-off<N>][-m<A>m<B>][-<output>]-<memory>-<N>core[-once]`,
for example `conv3x3s2-149x81c1to8-u8requant-l2_direct-8core` or
`gelu-vec1024-l1-8core`. Every field the matrix varies is in the name, so a name
stands for one case everywhere:

| field | meaning |
| --- | --- |
| `<function>` | activation cases only: `sigmoid`, `tanh`, `leaky_relu`, `hard_sigmoid`, `hard_swish`, `silu`, `gelu`, `softmax` |
| `<op+shape>` | `linear-<K>to<N>`; `matmul4x<P>-<K>to<N>` (P output pixels per call, K the im2col row); `conv<k>x<k>[s<S>]-<W>x<H>c<Cin>to<Cout>`; `dw<k>x<k>[s<S>]-<W>x<H>c<C>`; `pool<k>x<k>[s<S>]-<W>x<H>c<C>`, `global_avgpool-<W>x<H>c<C>`, `adaptive_avgpool-<W>x<H>c<C>-out<W>x<H>`; `batchnorm-<W>x<H>c<C>`; `add-<W>x<H>c<C>`; `vec<N>`. Sizes are width first; `s<S>` appears only when the stride is not 1, and `pad<P>` or `padt<T>b<B>l<L>r<R>` only when the padding differs from the op's default (k/2 rounded down for convolutions, none for pooling) |
| `off<N>` | matmul: the weights start N bytes past a 16-byte boundary in L1 |
| `m<A>m<B>` | add: the input multipliers in clip((a·A + b·B) >> 8) |
| `<output>` | linear, matmul, convolution, depthwise and pooling: `i32` (linear's raw accumulators), `u8` (pooling's own uint8 result), `u8shift` (clip(acc >> 6)), `u8requant` (one multiplier and shift per tensor), `u8bnfold` (batch norm folded in, per output channel) |
| `<memory>` | `l1`: every operand in cluster L1; `l2_direct`: the cores read the operands from L2 |
| `<N>core` | cores in the team: 1, 2, 4 or 8 |
| `once` | no warm-up and one timed run |

A **suite** in `configs/matrix.json` is a set of cases named `<op>_<what>[_l1]`.
`<op>` is the op (`linear`, or `linear_i32` and `linear_u8` for a suite of one of
the two linear kernels; `matmul4x2`, `matmul4x3`, `matmul4x4`, `conv3x3`,
`conv1x1`, `dw3x3`, `activation`, `batchnorm`, `pool`, `add`); `<what>` is what
the suite varies: `smoke` (a quick exactness check), `shapes`, `outputs`, `cores`
(1/2/4/8), `tails` (dimensions off a multiple of four), `tails_cores`,
`mlp_cores`, `kaligned`, `stems`, `offsets`, `multipliers`, `requant`,
`requant_smoke`, `tail_once`, and `dory_tiles`, the tile shapes DORY really
generates plus a control. `_l1` is the L1 counterpart of an `l2_direct` suite. Decide dispatch from the DORY-tile suites: the 8x8 and 16x16 cases can
reverse a conclusion. The group `smoke` runs every op once; `all` is every suite.

A **configuration** builds the bench a particular way: `s8bias`, where the
convolution and depthwise kernels read one int8 bias per output channel; and
`i32bias` (`KBENCH_DORY_INT32_BIAS=1`), where they read DORY's int32 bias as they do
under DORY, and the pinned PULP-NN convolution and depthwise kernels, which read
int8 bias, decline. `host/bench.yaml` gives each set its suites and
configurations, in the order RESULTS.md lists them.

## What a run does

1. Puts the **idle controller** on the STM32 (`--keep-controller` skips it).
2. **GVSOC**, several containers at once: each suite of each configuration is
   expanded into a matrix header, built in its own build folder and simulated
   until the app prints `#DONE`, then summarised (`host/sweep.py`,
   `host/report.py`).
3. **The deck**, one image at a time: each configuration's suites are built into
   one image, flashed, and the radio console recorded from power-up until `#DONE`
   (`host/run.py`). The app waits 20 ms after each record so the Crazyflie console
   does not overflow; a log with an overflow marker (`<F>`) is rejected.
4. **RESULTS.md**: every case, kernel and variant, GVSOC beside the deck
   (`host/results.py`).

## Output

Every run writes `artifacts/bench_<stamp>/`:

```text
run.json                         the set, target, jobs and commit
results.jsonl                    one line per suite or deck image
summary.md                       the same as a table
RESULTS.md                       every case, kernel and variant on both targets
logs/<job>.log                   each job's output (gvsoc-i32bias-dw3x3_outputs, aideck-s8bias, ...)
gvsoc-<configuration>/<suite>/   gvsoc.log, manifest.json (commit, image digest, clocks), report/
aideck-<configuration>/          build.log, kernel_bench.img, aideck.log, manifest.json, report/
```

A suite or image is `pass`, or `incomplete` when a kernel declined cases it does
not support and every other result was exact (`pool2x2s2-80x48c8`, a DORY tile
whose untiled input exceeds the pooling adapter's L1 budget); both count as passed.

## What the bench shows

- **Where operands live matters more than any kernel change here.** The same
  kernel on the same shape runs several times faster with its operands in L1
  instead of L2.
- **4x3 register blocking wins only at real tile sizes**, such as DORY's
  whole-layer tile `conv3x3-20x12c32to32` with L1 operands, not on small ones.
  It cannot be enabled under DORY without enlarging DORY's im2col buffer, which
  re-tiles every convolution layer. 4x4 loses to 4x3.
- **Batch-1 MLPs are bandwidth-bound**: one MAC per weight byte against the four
  GAP8 needs, which caps any weight-streaming MLP near 8 MAC/cycle. The lever is
  fewer weight bytes.
- **Local max and average pooling lose to PULP-NN** on instruction count, not
  memory, so dispatch keeps PULP-NN for them; global and adaptive average pooling
  win.
- **A stem kernel does not help where it matters**: with operands in L1, neither
  `stem_repack4` nor `stem_direct3x3` beats the flattened convolution at 1–4 input
  channels, because the stem is bound by where its operands live.
- **GVSOC and the deck rank variants alike but not identically**: choose kernels
  from the deck's numbers. One build gives the same GVSOC numbers every run, but
  changing only printed strings (a case name, say) moves some cases' cycle counts
  by several percent while every checksum stays the same, so a small GVSOC
  difference between two builds is not a kernel difference.
- What kernels and schedules are worth under DORY, and where the rest of a
  layer's time goes, are in [DORY_AND_DEEPLOY.md](DORY_AND_DEEPLOY.md).

Traps, each of which produces a clean-looking false result:

- A plain `make KBENCH_TARGET=aideck` builds whatever matrix header was generated
  last; `run.py` and `sweep.py` emit the suite's header into their output folder
  and build against it.
- GVSOC must run under `stdbuf -oL`: block-buffered output stalls part way and
  never reaches `#DONE`. GVSOC restarts the program instead of exiting, so the
  runner stops at `#DONE` and removes its named container.
- `begin()` must restore a case's inputs: a network with an even number of layers
  writes its output over its input, and a run then measures `f(f(x))`.

## Adding a kernel

Put the reusable kernel in `lib/gap8/kernels/<family>/` with a scalar oracle and a
self-locating `.mk` fragment; link upstream sources from `external/`, never copy
them. In this bench, add an adapter under `gap8/kernels/<family>/` that exports one
`kbench_kernel_t` descriptor (lifecycle `validate → plan → init → [begin →
run → check]* → deinit`), register it in `gap8/src/registry.c`,
and add its shapes and suites to `configs/matrix.json`: a shape's key is the
`<op+shape>` its fields give. Details are in
[gap8/kernels/README.md](gap8/kernels/README.md).

A kernel is ready for DORY or an application when it is exact on every case,
tail and core count, its golden is live, it passes under both bias configurations,
and the deck agrees.

## Open items

1. **Fuse pooling into the layer that produces it** (a DORY template change): a
   pooling layer does no arithmetic but costs a whole pass over its input.
2. **Enlarge DORY's im2col buffer for 4x3 blocking**, and measure the re-tiling it
   causes.
3. **Allocate DORY's L1 buffer once** instead of on every layer, part of DORY's
   fixed cost per layer.
4. **Explain the store's cost per output element**: the largest remaining term
   that needs no DORY change.
5. **Capability**: resize/upsample and concat/slice (needed for detection and
   segmentation heads), and a tiled `pool2x2s2-80x48c8`.

## Files

```text
host/bench.py, bench.yaml   the bench and its sets
host/sweep.py               expand suites into a matrix header and name the cases; build and run in GVSOC
host/run.py                 one deck image: header, build, flash, capture, report
host/report.py              KBENCH records to summary.json, results.csv, summary.md
host/results.py             RESULTS.md from a run's GVSOC and deck reports
host/reference.py           the committed golden-checksum ledger
configs/matrix.json         shapes, outputs, suites and groups
gap8/                       the bench app: main.c, registry.c, one adapter per family
```
