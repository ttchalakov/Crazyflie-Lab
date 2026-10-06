# GAP8 DORY-shaped quantized activation kernels

This library adds the batch-1 inference activations missing from the pinned
PULP-NN kernel set without changing DORY itself. It implements sigmoid, tanh,
LeakyReLU, hard-sigmoid, hard-swish, SiLU/Swish, GELU, and softmax. ReLU remains
in the linear/convolution epilogues where fusing it avoids an extra tensor pass.
LogSoftmax is not included because the deployed inference path can consume
logits or uint8 softmax probabilities; it would need a separate signed output
and numeric contract rather than pretending to be another byte LUT.

## Numeric contracts

The elementwise kernel indexes a 256-byte table with the input byte's raw bit
pattern and copies the selected raw output byte. Table preparation encodes the
input/output signedness, scales, zero points, saturation, and activation
parameters. The compute loop therefore has no activation-mode, sign, scale, or
tail branch per element. `tools/generate_lut.py` emits model-specific tables,
including an adjustable LeakyReLU slope.

`kernel_bench` ships deterministic default tables for comparison
(`examples/kernel_bench/gap8/kernels/activation/activation_luts.c`):

| activation | input | output |
|---|---|---|
| sigmoid, hard-sigmoid | signed int8 Q4.4 | uint8, scale 1/256 |
| tanh | signed int8 Q4.4 | signed int8, scale 1/128 |
| LeakyReLU (slope 1/8), hard-swish, SiLU, GELU | signed int8 Q4.4 | signed int8 Q4.4 |

These defaults are bench fixtures, not part of this library or a requirement
imposed on a network. A production exporter must build the table from that
tensor's recorded quantization. The 256-byte table is small enough to serialize
as a model constant and stage beside the activation tile.

Softmax accepts signed or unsigned 8-bit logits and a 256-entry uint16 exponent
table containing `round(exp(-d * input_scale) * 32768)`. Each output row is a
uint8 probability-mass vector that sums to exactly 255. The scalar oracle and
team implementation both assign quantization residue to the first maximum,
making ties deterministic. The current public bound is 1,024 columns. Softmax
needs no reduction scratch; core 0 normalizes complete rows while the remaining
team cores wait at the completion barrier. This deliberately favors robust
batch-1 classifier inference over barrier-heavy column reduction.

## DORY ownership contract

The `*_team` functions are the integration surface. Every active core calls the
same team function once with caller-owned pointers. They do not allocate L1 or
L2, issue DMA, fork a nested team, open/close the cluster, or take a scheduler
lock. The elementwise workers need no internal barrier because outputs are
disjoint. Softmax uses only core 0 plus a final barrier, avoiding shared
column-reduction state. `job->cores` must equal the active DORY team width.

A generated layer or future external wrapper owns the following sequence:

1. Reserve input/output tiles and the 256-byte activation LUT in its generated
   L1 plan. Softmax instead reserves its 512-byte exponent LUT; it needs no
   additional scratch.
2. Stage the input and constants using DORY's existing MCHAN transfers, then
   wait for those transfers before entering the team kernel.
3. Call the matching `*_team` function from the already-forked compute region.
4. Execute DORY's post-compute team barrier, then copy or consume the output
   using DORY's existing orchestration.
5. Only after completion may the caller reuse/release input, output, or table
   offsets. A LUT may remain resident across tiles if the generated L1
   plan explicitly treats it as persistent.

This makes the arithmetic compatible with DORY-managed compact tensors and
run-to-completion scheduling, but stock DORY does not yet emit calls to these
new symbols automatically. Adding that selection is a later generator/template
task outside this library. No source under `external/dory` is modified.

Include `activation.mk` from a GAP8 application. It links only local sources and
does not replace or compile any DORY source.
