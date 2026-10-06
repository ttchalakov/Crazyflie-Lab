"""Integer networks in NEMO's ONNX dialect, with every layer's golden output.

DORY's NEMO frontend reads one ONNX dialect, so this writes it directly instead
of training anything. Weights are random int8. Every layer but the last is
requantised to uint8 as ``clip((sum * m) >> SHIFT, 0, 255)``, with ``m``
calibrated so the layer uses the uint8 range (the clip is the ReLU); the last
layer is a raw int32 accumulation. Each layer's output is computed here in
exact integers: that is the golden the GAP8 is checked against.

The input is xorshift bytes in the device's HWC order, so the GAP8 regenerates
it (nn_bench.c) instead of storing a frame-sized array.

    g = Graph((1, 120, 160))           # channels, height, width; or (n,) for a vector
    g.conv(8, kernel=3, stride=2)      # layers append in order
    g.global_average()
    g.fc(4, requant=False)
    write(g, out_dir)                  # nemo/ (ONNX, sidecars), dory/config.json, graph.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from numpy.lib.stride_tricks import sliding_window_view
from onnx import TensorProto, helper, numpy_helper

SHIFT = 14
TARGET_P99 = 220.0  # where each layer's 99th percentile lands in uint8
XORSHIFT_SEED = 2463534242


def xorshift_bytes(count: int, seed: int = XORSHIFT_SEED) -> np.ndarray:
    """The input bytes; nn_bench.c and lib/gap8/sched generate the same stream."""
    x, out = seed, np.empty(count, dtype=np.int64)
    for i in range(count):
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out[i] = x & 0xFF
    return out


def _conv(x: np.ndarray, w: np.ndarray, stride: int, pad: int, depthwise: bool) -> np.ndarray:
    """Exact integer convolution, CHW. float64 is exact here (every sum is far
    below 2**53) and lets numpy use BLAS."""
    k = w.shape[-1]
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad))).astype(np.float64)
    win = sliding_window_view(padded, (k, k), axis=(1, 2))[:, ::stride, ::stride]  # c, oh, ow, k, k
    if depthwise:
        sums = np.einsum("chwij,cij->chw", win, w[:, 0].astype(np.float64))
    else:
        c, oh, ow = win.shape[:3]
        patches = win.transpose(1, 2, 0, 3, 4).reshape(oh * ow, c * k * k)
        sums = (patches @ w.reshape(w.shape[0], -1).T.astype(np.float64)).T.reshape(-1, oh, ow)
    return np.rint(sums).astype(np.int64)


@dataclass
class Layer:
    """One DORY layer: its golden output (CHW or flat), its cost, and for a
    convolution or fully connected layer the tensors (fused.py hands them to the
    fused executors)."""
    label: str         # the layer's name in the graph, never holding 'weight' or 'bias': DORY
                       # tells <label>.weight from <label>.bias by those substrings
    op: str            # conv, dw, fc, pool, gap or add
    values: np.ndarray
    macs: int = 0
    weight_bytes: int = 0
    int32: bool = False
    weights: np.ndarray | None = None  # OIHW, or (out, in)
    bias: np.ndarray | None = None
    multiplier: int = 1                # the requant m; the shift is SHIFT

    def device_bytes(self) -> bytes:
        """The output as the GAP8 holds it: HWC uint8, or little-endian int32."""
        v = self.values.transpose(1, 2, 0) if self.values.ndim == 3 else self.values
        return v.astype("<i4" if self.int32 else np.uint8).tobytes()


class Graph:
    def __init__(self, shape: tuple[int, ...], seed: int = 0):
        self.shape = tuple(shape)
        self.rng = np.random.default_rng(seed)
        data = xorshift_bytes(int(np.prod(shape)))
        self.input = data.reshape(*shape[1:], shape[0]).transpose(2, 0, 1) if len(shape) == 3 else data
        self.values, self.tensor = self.input, "input.1"
        self.nodes: list = []
        self.inits: list = []
        self.layers: list[Layer] = []

    # -- ONNX plumbing ----------------------------------------------------------

    def _node(self, op: str, inputs: list[str], **attrs) -> str:
        out = str(len(self.nodes) + 1)
        self.nodes.append(helper.make_node(op, inputs, [out], name=f"{op}_{out}", **attrs))
        return out

    def _const(self, value, dtype=np.int64, scalar: bool = False) -> str:
        array = np.asarray(value if scalar else [value], dtype=dtype)
        return self._node("Constant", [], value=numpy_helper.from_array(array))

    def _param(self, values: np.ndarray, name: str) -> str:
        self.inits.append(numpy_helper.from_array(values.astype(np.float32), name))
        return name

    def _requant(self, tensor: str, sums: np.ndarray, layer: Layer) -> None:
        positive = sums[sums > 0]
        scale = float(np.percentile(positive, 99)) if positive.size else 1.0
        m = int(np.clip(round(TARGET_P99 * (1 << SHIFT) / max(scale, 1.0)), 1, 32767))
        t = self._node("Cast", [tensor], to=TensorProto.INT64)
        t = self._node("Mul", [t, self._const(m)])
        t = self._node("Div", [t, self._const(1 << SHIFT)])
        for to in (TensorProto.INT64, TensorProto.INT64, TensorProto.FLOAT):  # NEMO's Casts
            t = self._node("Cast", [t], to=to)
        self.tensor = self._node("Clip", [t], min=0.0, max=255.0)
        layer.multiplier = m
        layer.values = self.values = np.clip((sums * m) >> SHIFT, 0, 255)
        self.layers.append(layer)

    # -- layers -----------------------------------------------------------------

    def conv(self, channels: int, kernel: int = 3, stride: int = 1, label: str = "conv",
             depthwise: bool = False) -> None:
        """A convolution (or a depthwise one: one 3x3 filter per channel), padded
        to keep the size at stride 1. Zero bias, as after folding batch norm."""
        c = self.values.shape[0]
        if depthwise:
            channels = c
        w = self.rng.integers(-4, 5, size=(channels, 1 if depthwise else c, kernel, kernel))
        pad = kernel // 2
        sums = _conv(self.values, w, stride, pad, depthwise)
        t = self._node("Conv", [self.tensor, self._param(w, f"{label}.weight"),
                                self._param(np.zeros(channels), f"{label}.bias")],
                       kernel_shape=[kernel, kernel], pads=[pad] * 4, strides=[stride, stride],
                       dilations=[1, 1], group=c if depthwise else 1)
        self._requant(t, sums, Layer(label, "dw" if depthwise else "conv", sums, macs=int(w[0].size * sums.size),
                                     weight_bytes=w.size,
                                     weights=w, bias=np.zeros(channels, dtype=np.int64)))

    def depthwise(self, stride: int = 1, label: str = "dw") -> None:
        self.conv(0, 3, stride, label, depthwise=True)

    def fc(self, n: int, requant: bool = True, label: str = "fc") -> None:
        """Fully connected (Gemm). Biased, so DORY's int32 bias path is exercised.
        Without requant it is the int32 output layer."""
        x = self.values.reshape(-1)
        source = self._node("Flatten", [self.tensor], axis=1) if self.values.ndim > 1 else self.tensor
        w = self.rng.integers(-4, 5, size=(n, x.size))
        b = self.rng.integers(-64, 65, size=n)
        sums = w @ x + b
        t = self._node("Gemm", [source, self._param(w, f"{label}.weight"), self._param(b, f"{label}.bias")],
                       alpha=1.0, beta=1.0, transB=1)
        layer = Layer(label, "fc", sums, macs=w.size, weight_bytes=w.size + 4 * n, weights=w, bias=b)
        if requant:
            self._requant(t, sums, layer)
        else:
            layer.int32 = True
            self.tensor, self.values = t, sums
            self.layers.append(layer)

    def maxpool2(self, label: str = "pool") -> None:
        c, h, w = self.values.shape
        self.values = self.values[:, :h // 2 * 2, :w // 2 * 2].reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))
        self.tensor = self._node("MaxPool", [self.tensor], kernel_shape=[2, 2], strides=[2, 2])
        self.layers.append(Layer(label, "pool", self.values))

    def global_average(self, label: str = "gap") -> None:
        c, h, w = self.values.shape
        self.values = (self.values.sum(axis=(1, 2)) // (h * w)).reshape(c, 1, 1)
        self.tensor = self._node("GlobalAveragePool", [self.tensor])
        self.layers.append(Layer(label, "gap", self.values))

    def add(self, skip: tuple[str, np.ndarray], label: str = "add") -> None:
        """A residual add of `skip` (a saved (tensor, values)) and the current
        output, as NEMO emits it with equal branch scales: (a + b) >> 1."""
        a = self._node("Mul", [skip[0], self._const(1, np.float32, scalar=True)])
        b = self._node("Mul", [self.tensor, self._const(1, np.float32, scalar=True)])
        t = self._node("Div", [self._node("Add", [a, b]), self._const(2, np.float32)])
        self.tensor = self._node("Floor", [t])
        self.values = (skip[1] + self.values) >> 1
        self.layers.append(Layer(label, "add", self.values))

    @property
    def branch(self) -> tuple[str, np.ndarray]:
        """The current output, to add back later (add) or branch from."""
        return self.tensor, self.values.copy()

    def model(self, name: str) -> onnx.ModelProto:
        graph = helper.make_graph(
            self.nodes, name,
            [helper.make_tensor_value_info("input.1", TensorProto.FLOAT, [1, *self.shape])],
            [helper.make_tensor_value_info(self.tensor, TensorProto.FLOAT, [1, int(self.values.size)])],
            self.inits)
        model = helper.make_model(graph, producer_name="pytorch", opset_imports=[helper.make_opsetid("", 9)])
        model.ir_version = 6
        return model


def write(graph: Graph, out: Path, name: str = "net", code_reserved_space: int = 132000) -> None:
    """DORY's input: the ONNX and the sidecars it builds its checksums from
    (nemo/), its config (dory/), and graph.json for the host checks."""
    nemo, dory = out / "nemo", out / "dory"
    nemo.mkdir(parents=True, exist_ok=True)
    dory.mkdir(parents=True, exist_ok=True)
    onnx.save(graph.model(name), str(nemo / "model_int.onnx"))
    save = lambda path, values: np.savetxt(path, np.asarray(values).reshape(-1), fmt="%d")  # noqa: E731
    save(nemo / "input.txt", graph.input)
    for i, layer in enumerate(graph.layers):
        save(nemo / f"out_layer{i}.txt", layer.values)
    save(nemo / "output.txt", graph.layers[-1].values)
    (dory / "config.json").write_text(json.dumps({
        "BNRelu_bits": 32, "code reserved space": code_reserved_space,
        "onnx_file": "../nemo/model_int.onnx"}, indent=2) + "\n")
    (out / "graph.json").write_text(json.dumps({
        "input_shape": list(graph.shape),
        "layers": [{"label": l.label, "op": l.op, "shape": list(l.values.shape), "int32": l.int32,
                    "macs": l.macs, "weight_bytes": l.weight_bytes} for l in graph.layers]}, indent=1) + "\n")
