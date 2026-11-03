"""A multilayer perceptron as float64 arrays: the reference the quantizer and the exporter start from."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

ACTIVATIONS = ("none", "relu", "silu")


def activation(x: np.ndarray, act: str) -> np.ndarray:
    if act == "relu":
        return np.maximum(x, 0.0)
    if act == "silu":
        return x / (1.0 + np.exp(-x))
    if act == "none":
        return x
    raise ValueError(f"unknown activation {act!r}; expected one of {ACTIVATIONS}")


@dataclass(frozen=True)
class Dense:
    """y = act(w x + b), with w of shape (out, in)."""

    w: np.ndarray
    b: np.ndarray
    act: str = "none"

    @property
    def n_in(self) -> int:
        return int(self.w.shape[1])

    @property
    def n_out(self) -> int:
        return int(self.w.shape[0])


@dataclass(frozen=True)
class Mlp:
    layers: tuple[Dense, ...]

    @property
    def n_in(self) -> int:
        return self.layers[0].n_in

    @property
    def n_out(self) -> int:
        return self.layers[-1].n_out

    def pre_activations(self, x: np.ndarray) -> list[np.ndarray]:
        """Each layer's w x + b on the rows of x, in float64."""
        out = []
        h = np.asarray(x, dtype=np.float64)
        for layer in self.layers:
            z = h @ layer.w.T + layer.b
            out.append(z)
            h = activation(z, layer.act)
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = self.pre_activations(x)[-1]
        return activation(z, self.layers[-1].act)


def _array(t: Any) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach().cpu().double().numpy()
    return np.asarray(t, dtype=np.float64)


def mlp_from_state_dict(state_dict: Mapping[str, Any], prefix: str, act: str = "silu") -> Mlp:
    """The Linear layers stored under ``prefix`` (``prefix.0.weight``, ``prefix.2.weight``, ...) in index order,
    with ``act`` after every layer but the last: the layout of an ``nn.Sequential`` of Linear and activation modules.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    indices = sorted(int(m.group(1)) for key in state_dict if (m := pattern.match(key)))
    if not indices:
        raise KeyError(f"no Linear weights under {prefix!r}")
    layers = []
    for n, index in enumerate(indices):
        w = _array(state_dict[f"{prefix}.{index}.weight"])
        b = _array(state_dict[f"{prefix}.{index}.bias"])
        layers.append(Dense(w=w, b=b, act=act if n < len(indices) - 1 else "none"))
    return Mlp(tuple(layers))


def mlp_from_sequential(module: Any) -> Mlp:
    """An ``nn.Sequential`` of ``nn.Linear`` layers, each optionally followed by ``nn.ReLU`` or ``nn.SiLU``."""
    from torch import nn

    layers: list[Dense] = []
    for child in module:
        if isinstance(child, nn.Linear):
            layers.append(Dense(w=_array(child.weight), b=_array(child.bias)))
        elif isinstance(child, (nn.ReLU, nn.SiLU)):
            if not layers or layers[-1].act != "none":
                raise ValueError("an activation must follow a Linear layer")
            last = layers[-1]
            layers[-1] = Dense(w=last.w, b=last.b, act="relu" if isinstance(child, nn.ReLU) else "silu")
        else:
            raise TypeError(f"unsupported module {type(child).__name__}")
    return Mlp(tuple(layers))
