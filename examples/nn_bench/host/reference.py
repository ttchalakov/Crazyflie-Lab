"""The host-side checks of a generated network, and the goldens the GAP8 compares against.

Before anything runs, a point's generated graph (graph.py) is checked against
what DORY made of it:

  - DORY made one layer per graph layer, and its activations_out_checksum for
    each (the byte sum DORY computes from our sidecar files) is the byte sum of
    our golden output: DORY read the network we wrote;
  - for a network of fully connected layers, the weights DORY emitted, run
    through the specified arithmetic (run_ideal), give the goldens again: DORY
    laid the weights out the way its kernels read them. A square layer's shape
    cannot tell a transposed weight matrix, so only the arithmetic does.

The GAP8 compares every layer's output with the CRC-32 of the golden
(nn_bench_golden.h): a camera-sized network's activations are megabytes, more than
L2 can hold as reference tensors.

load() and run_ideal() also serve examples whose network came from a real NEMO
export (hover_neural_network): they read a DORY MLP back from its generated app.
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ReferenceError(RuntimeError):
    """The generated app does not look the way this file knows how to read."""


def _ints(text: str, name: str) -> list[int]:
    m = re.search(rf"{name}\[[^\]]*\](?:\[[^\]]*\])? = \{{(.*?)\}};", text, re.S)
    if not m:
        raise ReferenceError(f"no {name} in the generated network.h")
    return [int(v) for v in re.findall(r"-?\d+", m.group(1))]


def network_h(app_dir: Path, prefix: str = "") -> str:
    path = Path(app_dir) / "inc" / f"{prefix}network.h"
    if not path.is_file():
        raise ReferenceError(f"no generated header at {path}")
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# a DORY MLP, read back from its generated app
# --------------------------------------------------------------------------

@dataclass
class Layer:
    name: str
    n_in: int
    n_out: int
    out_mult: int
    out_shift: int
    weights: np.ndarray  # [n_out][n_in] int64, as the kernel reads them
    bias: np.ndarray     # [n_out] int64
    is_output_layer: bool


@dataclass
class Network:
    layers: list[Layer]
    input_vector: np.ndarray


def _c_array(text: str, name: str) -> np.ndarray | None:
    """One uint8 C array from a generated header. Weights come out as floats
    ("3., 253."), the input as hex ("0xe5")."""
    m = re.search(rf"\b{re.escape(name)}\s*\[\s*(\d+)\s*\]\s*=\s*\{{(.*?)\}}\s*;", text, re.S)
    if not m:
        return None
    tokens = re.findall(r"0[xX][0-9a-fA-F]+|[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?", m.group(2))
    values = [int(t, 16) if t[:2].lower() == "0x" else int(float(t)) for t in tokens]
    if len(values) != int(m.group(1)):
        raise ReferenceError(f"{name}: declared {m.group(1)} elements, found {len(values)}")
    return np.array(values, dtype=np.uint8)


def _bytes(app_dir: Path, prefix: str, hex_name: str, header: str, symbols: list[str]) -> np.ndarray:
    """A blob from whichever form DORY emitted: a .hex file (the 3-level model, l3)
    or a C array (the 2-level model, l2)."""
    path = app_dir / "hex" / f"{prefix}{hex_name}"
    if path.is_file():
        return np.fromfile(path, dtype=np.uint8)
    for header_path in {app_dir / "inc" / f"{prefix}{header}", app_dir / "inc" / header}:
        if header_path.is_file():
            text = header_path.read_text(encoding="utf-8")
            for symbol in symbols:
                if (array := _c_array(text, symbol)) is not None:
                    return array
    raise ReferenceError(f"neither {path} nor {'/'.join(symbols)} in {header}")


def load(app_dir: Path, prefix: str = "") -> Network:
    """A DORY-generated MLP: every layer fully connected, the last raw int32."""
    app_dir = Path(app_dir)
    prefix = prefix if not prefix or prefix.endswith("_") else prefix + "_"
    text = network_h(app_dir, prefix)
    names = re.findall(r'"([^"]+)"', re.search(r"Layers_name\[\d+\] = \{(.*?)\};", text, re.S).group(1))
    act_in, act_out = _ints(text, "activations_size"), _ints(text, "activations_out_size")
    mult, shift = _ints(text, "out_mult_vector"), _ints(text, "out_shift_vector")
    layers = []
    for i, name in enumerate(names):
        last = i == len(names) - 1
        n_in, n_out = act_in[i], act_out[i] // (4 if last else 1)
        raw = _bytes(app_dir, prefix, f"{name}_weights.hex", "weights.h", [f"Weights_{name}"])
        if raw.size != n_out * n_in + 4 * n_out:
            raise ReferenceError(f"{name}: a {raw.size} B blob is not {n_out}x{n_in} weights and "
                                 f"{n_out} int32 biases (not a fully connected layer?)")
        layers.append(Layer(name, n_in, n_out, mult[i], shift[i],
                            raw[:n_out * n_in].view(np.int8).reshape(n_out, n_in).astype(np.int64),
                            raw[n_out * n_in:].view(np.int32).astype(np.int64), last))
    x = _bytes(app_dir, prefix, "inputs.hex", "input.h", [f"{prefix}L2_input_h", "L2_input_h"])
    return Network(layers, x.astype(np.int64))


def run_ideal(net: Network) -> list[np.ndarray]:
    """Every layer's output under the specified arithmetic, in exact integers."""
    acts, x = [], net.input_vector
    for layer in net.layers:
        sums = layer.weights @ x + layer.bias
        x = sums if layer.is_output_layer else np.clip((sums * layer.out_mult) >> layer.out_shift, 0, 255)
        acts.append(x)
    return acts


# --------------------------------------------------------------------------
# a generated graph against DORY's app
# --------------------------------------------------------------------------

def check(graph, app_dir: Path) -> list[str]:
    """What can be decided without a GAP8; the failures, or [] when DORY's app
    is the network graph.py wrote."""
    text = network_h(app_dir)
    sums = _ints(text, "activations_out_checksum")
    if len(sums) != len(graph.layers):
        return [f"DORY made {len(sums)} layers of a {len(graph.layers)}-layer graph"]
    failures = [f"layer {i} ({layer.label}): DORY's checksum {want} != the golden's {got}"
                for i, (layer, want) in enumerate(zip(graph.layers, sums))
                if (got := sum(layer.device_bytes())) != want]
    if not failures and graph.input.ndim == 1:  # a vector in: every layer is fully connected
        for i, (got, layer) in enumerate(zip(run_ideal(load(app_dir)), graph.layers)):
            if not np.array_equal(got, layer.values):
                failures.append(f"layer {i}: DORY's emitted weights compute something else "
                                "(a transposed or misread weight matrix)")
    return failures


def emit_golden(graph, path: Path) -> None:
    """nn_bench_golden.h: each layer's output size and CRC-32."""
    layers = graph.layers
    path.write_text("\n".join([
        "// GENERATED by examples/nn_bench/host/reference.py -- do not edit.",
        "#ifndef NN_BENCH_GOLDEN_H",
        "#define NN_BENCH_GOLDEN_H",
        "#include <stdint.h>",
        f"#define NNBENCH_NUM_LAYERS {len(layers)}",
        f"#define NNBENCH_INPUT_LEN {graph.input.size}",
        "static const uint32_t nnbenchLayerBytes[] = {"
        + ", ".join(str(len(layer.device_bytes())) for layer in layers) + "};",
        "static const uint32_t nnbenchLayerHash[] = {"
        + ", ".join(f"0x{zlib.crc32(layer.device_bytes()):08x}u" for layer in layers) + "};",
        "#endif", ""]))
