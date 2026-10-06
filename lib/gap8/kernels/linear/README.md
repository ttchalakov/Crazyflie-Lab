# GAP8 batch-1 quantized linear kernel

This library is the reusable batch-1 linear arithmetic that `kernel_bench`
measures. It targets inference of one activation vector at a time:

```text
int32 output[o] = int32 bias[o] + sum_k(uint8 input[k] * int8 weight[o,k])
```

The fused production form applies the pinned DORY/PULP-NN epilogue to that
accumulator and writes uint8 directly. It supports shift-and-clip, scalar
requantization plus ReLU, and per-output batch normalization plus ReLU.

There is no training, gradient, optimizer, or backpropagation API.

Variants:

- `reference`: scalar FC-side correctness oracles for int32 and fused uint8;
- `pulp_nn`: adapters around the exact pinned `pulp_nn_linear_out_32` and
  `pulp_nn_linear` sources;
- `register4`: GAP8 packed-dot implementation which reuses an input word across
  four output accumulators and supports runtime 1/2/4/8-core teams;
- `register4_unroll2`: otherwise identical ablation with two packed K words
  unrolled per loop iteration. It shows no meaningful gain, so it is not the
  default;
- `register4_fused`: keeps four int32 accumulators in registers, applies one of
  the three DORY epilogues, packs the four uint8 results, and commits them with
  one aligned store.

The fast path accepts arbitrary logical K through a prepared layout. Set
`packed_input_dim = round_up(K, 4)`, use that multiple of four as the physical
weight-row stride, and zero the input/weight padding. The preparation helpers do
this without allocating; callers own the source and destination storage. For an
already aligned K, physical and logical dimensions are equal, so there is no
padding or extra compute. Arbitrary N is supported directly: full groups of four
use a branch-free worker and the last one to three rows use a separate tail.
The fused fast path additionally requires a four-byte-aligned output base for
its packed four-result store; N-tail outputs are still written safely one byte
at a time.

Every variant is a `*_team` function for DORY or another caller that already
activated all cluster cores. Every active core calls it once; it does not nest
`pi_cl_team_fork()` and does not add a barrier. DORY's generated post-compute
barrier supplies the synchronization; another caller must provide an equivalent
barrier. `kernel_bench` forks its own team around the same functions.

None of them allocates L1/L2, opens the cluster, or starts DMA. DORY-generated
layer code should continue to own its generated `l1_buffer` offsets, MCHAN
transfers, and double buffers, then call the `*_team` microkernel on the resident
tile. A standalone orchestrator may implement a different L1 policy around the
same compute function, but must pass its scratch arena in rather than allocating
behind DORY's back.

For DORY integration, padding is part of the generated physical layout, not a
hidden kernel allocation. Constant weight rows should be zero-padded offline;
the generated L1 size and DMA length must include the physical stride. The last
input tile needs up to three explicit zero bytes in DORY-reserved L1 storage.
Passing `packed_input_dim > input_dim` against an unchanged compact weight blob
is invalid because a packed load would read into the next row.

The fused job deliberately mirrors DORY's existing `pulp_nn_linear` parameters:
`out_mult`, `out_shift`, per-output `k`/`lambda`, and the three meaningful
ReLU/batch-normalization flag combinations. L1/DMA orchestration remains a
separate, measurable layer owned by DORY or another caller.

Include `linear.mk` from a GAP8 application. It links the PULP-NN baseline from
the pinned DORY submodule rather than copying vendor source into `lib/`.
