# NEMO + DORY tools environment.
#
# Both tools need old dependencies and cannot run in the repo venv:
#  - NEMO builds its deployment graph with private torch.onnx APIs that were
#    removed in torch 1.13 (torch.onnx.utils._model_to_graph's
#    _retain_param_name argument).
#  - DORY's codegen converts 1-element numpy arrays with int(...), which
#    numpy >= 1.25 rejects.
# This image pins torch and numpy below those versions; the deploy pipeline
# (cf_lab.aideck.dory_pipeline) runs the model module's export_nemo_onnx hook and
# DORY's network_generate.py inside it.
FROM python:3.9-slim

RUN pip install --no-cache-dir \
    torch==1.8.1+cpu torchvision==0.9.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html \
    numpy==1.21.6 \
    scipy==1.7.3 \
    scikit-learn==1.0.2 \
    tqdm \
    packaging

# ortools 9.3 is the last protobuf-3 series, matching onnx 1.12.
RUN pip install --no-cache-dir \
    onnx==1.12.0 \
    ortools==9.3.10497 \
    mako==1.1.6

WORKDIR /work
