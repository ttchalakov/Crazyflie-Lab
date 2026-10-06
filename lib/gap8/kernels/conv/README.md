# GAP8 quantized convolution kernels

This library investigates drop-in replacements for the pinned GAP8 DORY/PULP-NN
8-bit convolution path. The public contract is batch-1 NHWC uint8 activations,
compact OHWI int8 weights, int32 accumulation, and a fused uint8 epilogue.
Supported epilogues exactly mirror the three meaningful PULP-NN flag modes:
shift/clip, scalar requantization/ReLU, and per-output-channel batch
normalization/ReLU.

The variants isolate three optimizations:

- `matmul4x2` keeps the pinned `pulp_nn_conv_Ho_parallel` im2col traversal but
  replaces its shared two-column matmul. Four output channels and two spatial
  pixels are accumulated together, the epilogue mode is selected once per call,
  and aligned four-channel results use packed stores.
- `flattened_matmul4x2` additionally partitions linearized output-pixel pairs
  across the active cluster cores. This avoids the stock kernel's output-height
  partitioning bottleneck when a DORY L1 tile has fewer rows than cores. Its
  im2col builder has a separate branch-free interior path and a bounded border
  path.
- `pointwise_flattened4x2` replaces DORY's separate 1x1 wrapper. It skips
  im2col, partitions all spatial pixel pairs rather than output height, and
  passes adjacent NHWC pixels directly to the same 4x2 register microkernel.
  Input-channel counts that are not a multiple of four use that microkernel's
  scalar lane tail with either bias layout; there is no separate fallback.

`cf_qconv_matmul4x2_krotate` is a third, experimental exact-ABI entry point. It
rotates each core's aligned K-word starting point and wraps to zero, changing
only the TCDM access phase and accumulation order. It is slower than the
sequential 4x2 path, with more load stalls, so it is kept only as a TCDM
ablation and neither exact replacement symbol uses it.

All variants accept compact DORY rows directly. `Cin * Kh * Kw` and `Cout` may
be arbitrary; packed dot products handle the main K body, and separate tails
handle the remaining one to three lanes/channels. No padded or interleaved
weight serialization is required.

## Ownership and DORY compatibility

The job mirrors `pulp_nn_conv_Ho_parallel`. Input, weight, output, quantization,
and im2col pointers are caller-owned. The im2col requirement is unchanged:

```text
2 * active_cores * Kh * Kw * Cin bytes
```

Team functions perform one inference on an already-active team. They allocate
nothing, launch no DMA, do not fork, and use only the supplied scratch.
`kernel_bench` forks its own 1/2/4/8-core team and calls the same functions.

DORY can therefore retain its generated L1 offsets, MCHAN transfers, compact
weight blobs, and post-compute barrier. For a generated application that should
replace the vendor symbols without changing its templates, exclude the
generated/vendor `pulp_nn_matmul.c`, `pulp_nn_conv_Ho_parallel.c`, and
`pulp_nn_pointwise_HoWo_parallel.c`, include `conv.mk`, and set:

```make
CF_QCONV_REPLACE_PULP_NN := 1
```

The library then exports `pulp_nn_matmul`, `pulp_nn_conv_Ho_parallel`, and
`pulp_nn_pointwise_HoWo_parallel` with the exact pinned signatures.
The exported `pulp_nn_matmul` intentionally selects the sequential 4x2 path;
the rejected K-rotation experiment is never selected implicitly.

The pinned function signature types convolution bias as `int8_t *`, but
NEMO/DORY serializes and DMAs one int32 value per output channel. The pinned
PULP-NN implementation indexes those bytes as int8. `lib/gap8/dory/kernels.mk`
sets `CF_QCONV_DORY_INT32_BIAS=1`, which makes all local fast and tail paths read
the actual DORY int32 layout without changing the public symbol. Direct
`conv.mk` users keep the standalone int8-bias job contract by default.

Include `conv.mk` from a GAP8 application. With replacement disabled (the
default), it also links the exact pinned vendor convolution, pointwise wrapper,
and matmul so kernel_bench can report baseline and optimized variants in one binary.

## Stem variant: padded K (`cf_qconv_stem.c`)

`cf_qconv_u8s8_u8_stem_repack_team` pads
K = kernel_h * kernel_w * input_channels up to a multiple of four with zeros on
both operands, so that consecutive weight rows and the second im2col column are
word aligned and the per-channel scalar tail disappears. The pad lanes multiply
by zero, so the accumulator is bit-identical to the unpadded path -- it is an
exactness-preserving transform, and `kernel_bench`'s `conv3x3_stems` suite gates
it as one.

It needs two things beyond the ordinary job:

- `job->weight_repack`, at least
  `output_channels * align_up(K, 4)` bytes, 4-byte aligned;
- `job->im2col` sized at the **padded** stride, i.e.
  `2 * cores * align_up(K, 4)`.

When `K % 4 == 0`, or when `weight_repack` is NULL, it forwards to
`cf_qconv_u8s8_u8_flattened_team` unchanged.

**Use it for Cin=3, not Cin=1.** With operands in L2 it beats the flattened
path at Cin=3 (K 27 -> 28, 4% more lanes) but loses at Cin=1 (K 9 -> 12, 33% more
lanes): what alignment gains in lane throughput does not pay for a third more
lanes. With operands in L1 it does not beat the flattened path at all.
