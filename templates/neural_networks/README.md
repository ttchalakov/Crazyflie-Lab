# Neural-Network Templates

Minimal, reusable starting points for getting your own model onto the AI-deck
GAP8 through the **NEMO -> DORY** pipeline (`cf-lab compile-model`).
They are templates to copy, **not** models that fly anything.

A complete, deployed model that does real work lives next to the example that
uses it (`examples/<name>/model/`, e.g. `examples/hover_neural_network/model/hover_policy.py`).
When you start a new one, copy a template here as the skeleton.

## What's here

- `bench_cnn/` — the smallest model that exercises every stage of the pipeline:
  a tiny BatchNorm-free conv/pool/linear net. Use it to verify the toolchain and
  as the blank slate for your own architecture.

## The model contract

Every model the pipeline can compile is a single Python module exposing two hooks
that `cf-lab compile-model` calls by name:

- `create_model()` — build and return the `nn.Module` (the checkpoint is loaded
  into it).
- `export_nemo_onnx(*, model, checkpoint_path, calibration, input_shape, output_path, work_dir)`
  — apply the model-specific NEMO quantization recipe and write a DORY-compatible
  integer ONNX to `output_path`. (`calibration` is a path to a single-array `.npy`;
  nets with fixed-range activations can ignore it.)

The hooks run in the NEMO container (Python 3.9, torch 1.8), so the module must
stay Python 3.9 compatible; it may import `cf_lab`.

## Build it

```bash
# 1. generate a checkpoint + calibration array next to the template
python templates/neural_networks/bench_cnn/bench_cnn.py make-artifacts

# 2. run the pipeline into the network directory of the GAP8 app that runs it
#    (intermediate files land in a gitignored build/ dir)
cf-lab compile-model \
  --model-module templates/neural_networks/bench_cnn/bench_cnn.py \
  --checkpoint templates/neural_networks/bench_cnn/bench_cnn.pt \
  --calibration templates/neural_networks/bench_cnn/bench_cnn_calibration.npy \
  --input-shape 1,1,16,16 \
  --name bench_cnn \
  --integrate-into path/to/your_app/gap8/bench_cnn
```

The generated checkpoint, calibration, and DORY output are all build artifacts
(gitignored); only the `*.py` template is tracked.
