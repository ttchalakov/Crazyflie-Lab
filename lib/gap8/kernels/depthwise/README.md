# GAP8 DORY-compatible depthwise convolution

This directory provides a batch-1 `uint8 × int8 → uint8` depthwise convolution
experiment for GAP8. It matches the pinned DORY call to
`pulp_nn_depthwise_generic`: input is CHW, output is HWC, weights are compact
`[channel][kernel_y][kernel_x]`, and the channel multiplier is one.

The optimized path specializes 3×3 filters. It supports stride one and two,
arbitrary valid output dimensions, asymmetric padding, channel counts not
divisible by four, and the three stock uint8 epilogues. It does not allocate,
fork, issue DMA, or change DORY's L1 layout when called through the team API.

## Implementations

- `pulp_nn` is the pinned `pulp_nn_depthwise_generic` baseline. It constructs a
  per-core, per-channel column buffer and applies packed dot products to each
  output. This is the closest existing path to an im2col/implicit-GEMM
  formulation, although depthwise convolution never becomes a useful dense
  GEMM because each output channel consumes exactly one input channel.
- `direct3x3_2x1` reads CHW input planes directly. Its stride-one interior keeps
  three rows and nine weights in registers and computes two neighboring output
  pixels with three shared `v4u` loads and six `sdotusp4` instructions. Checked
  scalar loops handle the borders. Stride two uses a one-output packed direct
  micro-kernel.

Work is divided across `(channel, output row)` tasks instead of channels alone.
This preserves sequential access within a CHW row while keeping all cores useful
for short DORY height tiles.

## Algorithm choice

| method | depthwise 3×3 on GAP8 | decision |
|---|---|---|
| Implicit GEMM / im2col | There is no cross-channel or output-channel reuse. Materializing nine values per output adds stores and L1 traffic before only nine useful MACs. A dense GEMM representation would also be block-diagonal/sparse. | Keep the stock generic implementation as the measured baseline; do not add another im2col layout. |
| Direct micro-kernel | Reuses each four-byte row load for two horizontal outputs, uses GAP8's native unsigned×signed four-lane dot product, and needs no scratch. | Implemented as the default 3×3 candidate. |
| Winograd F(2×2, 3×3) | Reduces scalar multiplications from 36 to 16 per four-output tile, but the input/filter/output transforms add many operations. Transformed values need wider lanes, so GAP8 loses its efficient four-lane uint8×int8 dot product. Prepared weights expand from 9 to 16 values per channel and no longer match DORY's compact weight ABI; an on-call transform costs more than these small depthwise filters save. | Do not use for the current exact drop-in. Revisit only with an offline DORY weight-layout change and a separate int16/overflow proof. |

Winograd can be more attractive for dense 3×3 convolution because transforms
are amortized across many input/output-channel pairs. That is a separate
prepared-layout experiment; it is not a transparent replacement for the current
DORY depthwise call.

## DORY integration and ownership

Include the fragment and enable the exact-symbol source replacement:

```make
CF_QDEPTHWISE_REPLACE_PULP_NN := 1
include $(CF_LAB)/lib/gap8/kernels/depthwise/depthwise.mk
```

Set the variable before including the fragment. The build then exports
`pulp_nn_depthwise_generic` from this library instead of compiling the public
vendor definition. Calls with 3×3 weights use the direct kernel; other filter
sizes dispatch to a privately renamed copy of the pinned implementation. DORY's
templates, compact weights, arguments, and tiler do not change.

The direct kernel ignores `scratch` and `weight_buffer`, but a stock generated
network still reserves those regions. Therefore this drop-in improves compute
time without disturbing DORY's memory plan. Reclaiming that L1 space would
require a future tiler/template change and is deliberately outside kernel_bench.

The team API borrows the active cluster and uses its runtime team width. It has
one barrier at completion. There is no mid-kernel context switch: the scheduler
must treat one DORY compute tile as the run-to-completion unit and may release or
reuse scratch only after that barrier.

## Testing

kernel_bench's `dw3x3_outputs` suite runs `direct3x3_2x1` and the PULP-NN
baseline on five shapes (`dw3x3-16x16c32`, `dw3x3-17x3c16`, `dw3x3s2-17x15c24`,
`dw3x3-9x7c13` and the asymmetric `dw3x3-9x7c7-padt0b2l1r0`) with each of the three
outputs, and `dw3x3_cores` runs `dw3x3-16x16c32` and `dw3x3-17x3c16` at 1/2/4/8
cores. The direct kernel is byte-exact on every case and faster than PULP-NN on
each at eight cores. With L2-direct operands its scaling flattens after four
cores, limited by L2 traffic and team overhead.
