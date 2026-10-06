# GAP8 forward-only quantized BatchNorm

This library provides two deliberately separate BatchNorm paths without
changing NEMO or DORY:

- **fixed affine inference** applies prepared per-channel `multiplier` and
  `bias` values to a compact HWC uint8 tensor. The four-channel implementation
  removes the channel-modulo operation and uses aligned packed loads/stores
  when `C % 4 == 0`; arbitrary channel tails fall back to the scalar team path.
- **batch-stat/update inference** calculates the current frame's per-channel
  mean and biased variance, normalizes the frame, and can update caller-owned
  running mean and unbiased running variance. It is forward-only: `gamma` and
  `beta` are immutable and there is no gradient, backward, optimizer, or
  training-data interface.

NEMO's folded BatchNorm remains the normal static deployment path. Folding has
no standalone activation pass and therefore cannot be beaten by a separate
kernel. The fixed kernel is for graph boundaries or shared tensors that cannot
be folded; the updating kernel is for explicit online calibration or
test-time adaptation.

## Updating numeric contract

Input and output are uint8 HWC. State and parameters use Q16.16:

| field | units |
|---|---|
| `running_mean_q16` | raw input-code units |
| `running_variance_q16` | raw input-code units squared |
| `gamma_q16` | output-code units per normalized unit |
| `beta_q16` | output-code units |
| `epsilon_q16` | input-code units squared |
| `momentum_q16` | weight assigned to the new statistics |

For each channel, the current statistics reduce over `H*W` samples. The
forward result uses population variance. `UPDATE_RUNNING` applies the usual
exponential moving average to the batch mean and unbiased variance. The
integer square root and per-channel affine preparation occur once per channel;
the output pass uses one widened multiply rather than a divide per element.

Batch-stat modes require more than one spatial sample. Batch-one convolutional
features remain valid while `H*W > 1`; a `1x1` feature map or batch-one linear
output must use fixed running statistics or a batch-independent normalization.

## DORY and scheduling boundary

The `*_team` functions are the integration surface. Every active core calls the
same function from an already-forked region. Kernels allocate nothing, issue no
DMA, do not fork a nested team, and do not acquire or release the cluster.
Input/output and immutable parameters normally live in caller-planned L1;
mutable running state remains in caller-owned L2 and may be checkpointed to L3
between complete invocations.

An updating call is a run-to-completion transaction. Each core exclusively owns
its channel range, and running state is written only after that channel's output
is complete. GAP8 cannot context-switch cluster execution in the middle of the
call. A scheduler may yield before submission or after the team returns, but
must not retry an interrupted update without accounting for any committed
channels. If stronger restart semantics are required, the caller should pass a
shadow state buffer and swap it after successful completion.

Stock DORY does not generate a mutable BatchNorm node. A future wrapper can
stage these buffers using DORY's existing memory plan and call the team entry;
no file under `external/dory` is modified here.
