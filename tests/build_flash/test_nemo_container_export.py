"""The NEMO export driver that runs in the container: model hook in, one ONNX file out."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from cf_lab.aideck import nemo_container_export

MODEL = '''
from pathlib import Path

class Model:
    def load_state_dict(self, state):
        self.state = state
    def eval(self):
        pass

def create_model():
    return Model()

def export_nemo_onnx(*, model, checkpoint_path, calibration, input_shape, output_path, work_dir):
    {body}
'''


@pytest.fixture
def export(tmp_path, monkeypatch):
    # The real container has torch 1.8; the driver only loads the checkpoint with it.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda path, map_location: {"w": 1}))
    monkeypatch.syspath_prepend(str(tmp_path))  # the driver puts --nemo-dir first on sys.path

    def run(body: str):
        module = tmp_path / "policy.py"
        module.write_text(MODEL.format(body=body))
        (tmp_path / "policy.pt").write_bytes(b"")
        output = tmp_path / "out" / "nemo" / "model_int.onnx"
        nemo_container_export.main([
            "--model-module", str(module), "--checkpoint", str(tmp_path / "policy.pt"),
            "--calibration", str(tmp_path / "calibration.npy"), "--input-shape", "1x11",
            "--output", str(output), "--work-dir", str(tmp_path / "work"), "--nemo-dir", str(tmp_path),
        ])
        return output

    return run


def test_onnx_written_elsewhere_by_the_hook_is_copied_to_the_output(export, tmp_path):
    output = export(
        "assert input_shape == (1, 11)\n"
        "    side = Path(work_dir) / 'exported.onnx'\n"
        "    side.write_bytes(b'onnx-bytes')\n"
        "    return side")

    assert output.read_bytes() == b"onnx-bytes"


def test_hook_that_writes_the_output_itself_is_accepted(export):
    output = export("Path(output_path).write_bytes(b'direct')")

    assert output.read_bytes() == b"direct"


def test_hook_that_exports_nothing_is_an_error(export):
    with pytest.raises(RuntimeError, match="did not create"):
        export("return None")
