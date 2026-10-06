# GAP8 DORY-shaped uint8 pooling

This library adds compact-HWC, batch-one uint8 max, average, global-average,
and adaptive-average pooling with scalar correctness oracles and pinned
PULP-NN baselines.

The local fixed-window kernels support rectangular kernels and strides,
asymmetric padding, arbitrary output shapes, and channel tails. Max pooling
ignores padded samples. Average pooling supports both full-kernel
`count_include_pad` and valid-sample divisors; the kernel_bench DORY comparison
uses full-kernel division and rounds down to match the pinned uint8 PULP-NN
kernel. Average, global-average, and adaptive-average paths can also apply the
pinned DORY order `((sum * multiplier) / divisor + add) >> shift` with uint8
saturation, avoiding a separate requantization pass. Ordinary DORY uint8
ranges use a signed 32-bit product/divide on GAP8; an overflow-checked 64-bit
fallback preserves the public contract for unusually large custom windows or
multipliers.

The aligned fast path accumulates four adjacent HWC channels together and uses
one packed load/store per window position. It has no alignment penalty when
`C % 4 == 0`; other channel counts safely use the scalar team path. Global
average pooling has a separate channel-parallel implementation that visits
each input element once. Adaptive average pooling computes the conventional
floor/ceil bin boundaries and supports output sizes that do not evenly divide
the input.

The packed path is an experiment, not an unconditional dispatcher decision.
For small fixed average windows, the pinned PULP-NN traversal can remain faster,
especially when requantization division dominates. Use the matrix results to
select it per shape. The specialized global and adaptive paths are intended to
cover the missing operators and favorable channel-aligned shapes.

## Pinned baseline caveat

`cf_qpooling_pulp_nn_team` links the unchanged pooling sources from the pinned
DORY checkout. That baseline is restricted to eight cores. Its max-pool
implementation uses the input L1 tile as horizontal workspace, so the input is
not preserved; `kernel_bench` restores it outside the timed interval before each
sample. Its row partitioning also assumes that the effective core count is a
power of two; the adapter reports other shapes as unsupported instead of
misrepresenting a known bad baseline. The local kernels are out-of-place and
never modify input.

## DORY ownership contract

All `*_team` entries expect caller-owned L1 input/output tiles and an active
team. They allocate no memory, launch no DMA, fork no nested team, and own no
cluster or scheduler lock. DORY or the application remains responsible for
L2/L3 staging and may reuse the buffers only after the team returns. Pooling
has no persistent mutable state, so complete invocations are natural scheduler
handoff points.

The fixed max/average signatures map directly to the dimensions carried by a
DORY pooling layer. Global average is the same fixed contract with a full-map
window. Adaptive pooling is DORY-shaped arithmetic but stock DORY does not
currently emit that operator; generator selection remains separate and no
source under `external/dory` is edited.

## Status

Global-average and adaptive-average traversal beat the pinned baseline because
they reduce a long sample axis per channel. Local fixed-window pooling does not,
for average or max, at eight cores with operands in L1.

`cf_qpooling_average_output()` performs an integer division per output element
and RI5CY's divider is multi-cycle, while the pinned kernel avoids one on its
hot path. Hoist a reciprocal multiply for the constant interior divisor before
treating the local paths as candidates; until then the pinned max/average
kernels remain the dispatch default for fixed windows.
