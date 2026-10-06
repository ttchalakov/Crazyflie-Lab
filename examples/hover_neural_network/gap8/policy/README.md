# Generated policy

`generated/` is the NEMO/DORY artifact and is intentionally untracked. The
FreeRTOS glue DORY's runtime needs comes from `lib/gap8/dory/dory.mk`.

Train first, then integrate:

```bash
cf-lab compile-model \
  --integrate-into examples/hover_neural_network/gap8/policy \
  --models-root examples/hover_neural_network/build \
  --model-module examples/hover_neural_network/model/hover_policy.py \
  --checkpoint examples/hover_neural_network/model/hover_policy.pt \
  --calibration examples/hover_neural_network/model/hover_policy_calibration.npy \
  --input-shape 1,11 --name policy --dory-target PULP.GAP8_L2 \
  --dory-l2-bytes 228000 --dory-verbose-level None
```

`--dory-verbose-level None` removes DORY's generated per-layer golden checks and
prints from the live 100 Hz path. The offline correctness checks remain: NEMO
parity, reported on the `[nemo]` line of the compile-model log, and the DORY build
itself.

`--dory-target PULP.GAP8_L2` must not be changed. The app links resident weight
arrays and a small activations-only arena, and its Makefile refuses a 3-level
network, which would need HyperRAM and a flash filesystem for no benefit at this
size.

The export writes no C and prints none. It drops NEMO's output quantum beside its
ONNX as `nemo/output_epsilon.txt`, and the GAP8 Makefile passes that in as
`-DHNN_POLICY_OUTPUT_EPSILON`. Everything else the deck decodes against is a
constant at the top of `model/hover_policy.py`, mirrored in
`gap8/hover_neural_network.c`.

The GAP8 build fails immediately when generated artifacts are absent. Do not guess
quantization scales and do not hand-edit generated code.
