"""Tiny CNN template for the PyTorch -> NEMO -> DORY -> GAP8 pipeline.

Copy-me example of the two hooks every deployable model must expose -
`create_model()` and `export_nemo_onnx(...)` - that cf-lab compile-model
drives. This is not a flight policy: it is a minimal model whose only job is to
push a real checkpoint through every stage of the pipeline. The
topology follows NEMO's reference MNIST example (conv-relu blocks, max
pooling, two linear layers) because that is the network family the DORY NEMO
frontend was built around. It deliberately has NO BatchNorm: NEMO's BN
deployment path (qd_stage -> bn_quantizer/calibrate_bn) depends on
deploy-graph supernode detection that is broken under the torch 1.8 pinned
in docker/nemo/Dockerfile, so BN models fail quantization. Fold or drop BN
before bringing a policy through this pipeline.

The export_nemo_onnx hook runs inside the cf-lab/nemo container
(python 3.9 + torch 1.8 — see docker/nemo/Dockerfile), so this file must
stay compatible with that interpreter.

Generate the checkpoint and calibration data with:

    python templates/neural_networks/bench_cnn/bench_cnn.py make-artifacts
"""

from __future__ import annotations

import sys
from pathlib import Path


INPUT_SHAPE = (1, 1, 16, 16)
CHECKPOINT_PATH = Path(__file__).parent / "bench_cnn.pt"
CALIBRATION_PATH = Path(__file__).parent / "bench_cnn_calibration.npy"
CALIBRATION_SAMPLES = 64

# Module names whose outputs are the DORY graph nodes, in graph order:
# Conv(+BNRelu), MaxPool, Conv(+BNRelu), MaxPool, FC(+Relu), FC.
DORY_NODE_OUTPUTS = ("relu1", "pool1", "relu2", "pool2", "fcrelu1", "fc2")


def create_model():
    import torch.nn as nn

    class BenchCnn(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(16 * 4 * 4, 32)
            self.fcrelu1 = nn.ReLU()
            self.fc2 = nn.Linear(32, 4)

        def forward(self, x):
            import torch

            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = torch.flatten(x, 1)
            return self.fc2(self.fcrelu1(self.fc1(x)))

    return BenchCnn()


def export_nemo_onnx(
    *,
    model,
    checkpoint_path,
    calibration,
    input_shape,
    output_path,
    work_dir,
):
    """Quantize with NEMO and write a DORY-ready integer ONNX plus sidecars.

    Follows NEMO's post-training quantization flow (FakeQuantized ->
    QuantizedDeployable -> IntegerDeployable) and dumps the per-layer integer
    activations DORY needs for its on-device checksums (input.txt,
    out_layer{i}.txt, output.txt — one integer per line, DORY node order).
    """
    import nemo
    import numpy as np
    import torch

    output_path = Path(output_path)
    calibration_batch = torch.from_numpy(np.load(str(calibration))).float()
    dummy_input = torch.randn(1, *input_shape[1:])

    model = nemo.transform.quantize_pact(model, dummy_input=dummy_input)
    precision = {
        "conv1": {"W_bits": 7},
        "conv2": {"W_bits": 7},
        "fc1": {"W_bits": 7},
        "fc2": {"W_bits": 7},
        "relu1": {"x_bits": 8},
        "relu2": {"x_bits": 8},
        "fcrelu1": {"x_bits": 8},
    }
    model.change_precision(bits=1, min_prec_dict=precision)

    model.set_statistics_act()
    with torch.no_grad():
        model(calibration_batch)
    model.unset_statistics_act()
    model.reset_alpha_act()

    model.qd_stage(eps_in=1.0 / 255)
    model = nemo.transform.integerize_pact(model, eps_in=1.0 / 255)

    # no_grad: export_onnx rounds parameters in place, which torch rejects
    # on grad-tracking leaves.
    with torch.no_grad():
        nemo.utils.export_onnx(str(output_path), model, model, input_shape[1:], batch_size=1)

    # IntegerDeployable forward passes carry the integer images directly,
    # so hook outputs only need rounding.
    captured = {}

    def capture(name):
        def hook(module, inputs, output):
            captured[name] = output.detach().round().to(torch.int64).flatten().tolist()

        return hook

    handles = [getattr(model, name).register_forward_hook(capture(name)) for name in DORY_NODE_OUTPUTS]
    integer_input = (calibration_batch[0:1] * 255.0).round()
    with torch.no_grad():
        model(integer_input)
    for handle in handles:
        handle.remove()

    def write_values(path, values):
        path.write_text("".join("%d\n" % v for v in values), encoding="utf-8")

    sidecar_dir = output_path.parent
    write_values(sidecar_dir / "input.txt", integer_input.to(torch.int64).flatten().tolist())
    for index, name in enumerate(DORY_NODE_OUTPUTS):
        write_values(sidecar_dir / ("out_layer%d.txt" % index), captured[name])
    write_values(sidecar_dir / "output.txt", captured[DORY_NODE_OUTPUTS[-1]])
    return output_path


def make_artifacts():
    import numpy as np
    import torch

    torch.manual_seed(7)
    np.random.seed(7)

    model = create_model()
    model.eval()
    torch.save(model.state_dict(), str(CHECKPOINT_PATH))

    calibration = np.random.rand(CALIBRATION_SAMPLES, *INPUT_SHAPE[1:]).astype(np.float32)
    np.save(str(CALIBRATION_PATH), calibration)
    print("checkpoint: %s" % CHECKPOINT_PATH)
    print("calibration: %s" % CALIBRATION_PATH)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "make-artifacts":
        make_artifacts()
    else:
        print(
            "usage: python templates/neural_networks/bench_cnn/bench_cnn.py make-artifacts",
            file=sys.stderr,
        )
        sys.exit(2)
