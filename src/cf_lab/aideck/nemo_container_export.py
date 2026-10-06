"""NEMO export driver that runs inside the cf-lab/nemo container.

NEMO depends on private torch.onnx APIs removed in torch 1.13, so the deploy
pipeline (cf_lab.aideck.dory_pipeline) runs this in the pinned python 3.9 +
torch 1.8 image (docker/nemo/Dockerfile). It builds the model from the module's
factory and checkpoint, then calls the module's export_nemo_onnx(...) hook.

This file must stay python 3.9 compatible and must not import cf_lab (model
modules may: the pipeline puts cf_lab on the container's PYTHONPATH).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def load_python_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load model module: %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_input_shape(value):
    parts = value.replace("x", ",").split(",")
    return tuple(int(part.strip()) for part in parts if part.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nemo-container-export")
    parser.add_argument("--model-module", type=Path, required=True)
    parser.add_argument("--model-factory", default="create_model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--input-shape", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--nemo-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.nemo_dir))
    import torch

    module = load_python_module(args.model_module)
    hook = getattr(module, "export_nemo_onnx", None)
    if hook is None:
        raise RuntimeError("%s must define export_nemo_onnx(...)" % args.model_module)
    factory = getattr(module, args.model_factory, None)
    if factory is None:
        raise RuntimeError("%s must define factory `%s`" % (args.model_module, args.model_factory))

    model = factory()
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = hook(
        model=model,
        checkpoint_path=args.checkpoint,
        calibration=args.calibration,
        input_shape=parse_input_shape(args.input_shape),
        output_path=args.output,
        work_dir=args.work_dir,
    )
    if result is not None and Path(result) != args.output:
        import shutil

        shutil.copyfile(str(result), str(args.output))
    if not args.output.is_file():
        raise RuntimeError("NEMO export did not create expected ONNX file: %s" % args.output)
    print("NEMO ONNX written: %s" % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
