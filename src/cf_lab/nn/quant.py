"""Post-training quantization of an ``Mlp`` to the formats of ``lib/shared/include/cf_lab/nn.h``.

Ranges come from calibration inputs, rows of what the deployed network will see (for a controller, the inputs along
simulated closed-loop runs), widened by ``margin`` of their span and always containing zero. Integer formats:

- the inputs are standardised per feature (mean and standard deviation over the calibration rows) before they are
  quantized, and the standardisation is folded exactly into the first layer (w' = w diag(std), b' = b + w mean), so
  features of very different scales, metres beside millimetres, each use the whole integer range;
- weights are symmetric per output channel (int8 or int16), biases are in the accumulator's units;
- ``s8`` activations are asymmetric int8 with a zero point, as in TFLite and CMSIS-NN; ``s16s8`` and ``s16``
  activations are symmetric int16;
- an activation function is a table from the requantized pre-activation to the output, computed here in float64:
  256 entries for int8, ``LUT16_SIZE`` linearly interpolated knots for int16.

The float formats keep the original weights (``f32``, ``f32x4`` and ``f32fma`` share them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cf_lab.nn.spec import Dense, Mlp, activation

FORMATS = ("f32", "f32x4", "f32fma", "s8", "s16s8", "s16")
FORMAT_CODES = {name: code for code, name in enumerate(FORMATS)}
ACT_CODES = {"none": 0, "relu": 1, "silu": 2}
LUT16_SIZE = 513
SILU_ARGMIN = -1.2784645427610738  # where x * sigmoid(x) is smallest


@dataclass
class QuantLayer:
    """One layer in the stored layout of ``nn.h``: ``w`` is (n_out, in_pad) float32, int8 (groups of four permuted
    as (w0, w2, w1, w3)) or int16; ``b`` float32, int32 or int64."""

    n_in: int
    in_pad: int
    n_out: int
    w: np.ndarray
    b: np.ndarray
    act: str = "none"
    mult: np.ndarray | None = None
    shift: np.ndarray | None = None
    in_offset: int = 0
    out_offset: int = 0
    lut: np.ndarray | None = None

    @property
    def nbytes(self) -> int:
        arrays = (self.w, self.b, self.mult, self.shift, self.lut)
        return sum(a.nbytes for a in arrays if a is not None)


@dataclass
class QuantNet:
    fmt: str
    layers: list[QuantLayer]
    n_in: int
    n_out: int
    in_scale: np.ndarray | None = None
    in_bias: np.ndarray | None = None
    out_scale: float = 1.0
    out_zero: int = 0

    @property
    def max_width(self) -> int:
        return max(max(layer.in_pad, layer.n_out) for layer in self.layers)

    @property
    def nbytes(self) -> int:
        extra = sum(a.nbytes for a in (self.in_scale, self.in_bias) if a is not None)
        return sum(layer.nbytes for layer in self.layers) + extra

    @property
    def macs(self) -> int:
        return sum(layer.in_pad * layer.n_out for layer in self.layers)


def pad4(n: int) -> int:
    return (n + 3) // 4 * 4


def _pad_columns(w: np.ndarray, in_pad: int) -> np.ndarray:
    out = np.zeros((w.shape[0], in_pad), dtype=w.dtype)
    out[:, : w.shape[1]] = w
    return out


def _pad_rows(a: np.ndarray, rows: int) -> np.ndarray:
    out = np.zeros((rows, *a.shape[1:]), dtype=a.dtype)
    out[: a.shape[0]] = a
    return out


def _blocks(w: np.ndarray, rows: int, group: int) -> np.ndarray:
    """(n_out, in_pad) -> [n_out/rows][in_pad/group][rows][group], padding n_out with zero rows (nn.h's layouts)."""
    n_out, in_pad = w.shape
    padded = _pad_rows(w, -(-n_out // rows) * rows)
    return padded.reshape(-1, rows, in_pad // group, group).transpose(0, 2, 1, 3).copy()


def _int8_layout(w: np.ndarray) -> np.ndarray:
    """int8 weights in blocks of four rows, each word of four weights stored (w0, w2, w1, w3)."""
    n_out, in_pad = w.shape
    permuted = w.reshape(n_out, in_pad // 4, 4)[:, :, [0, 2, 1, 3]].reshape(n_out, in_pad)
    return _blocks(permuted, 4, 4)


def _padded(a: np.ndarray, rows: int, fill: int = 0) -> np.ndarray:
    out = np.full(-(-a.shape[0] // rows) * rows, fill, dtype=a.dtype)
    out[: a.shape[0]] = a
    return out


def _range(values: np.ndarray, margin: float) -> tuple[float, float]:
    lo, hi = float(np.min(values)), float(np.max(values))
    span = hi - lo if hi > lo else max(abs(hi), 1e-6)
    return min(lo - margin * span, 0.0), max(hi + margin * span, 0.0)


def _asym8(lo: float, hi: float) -> tuple[float, int]:
    """Scale and zero point of an int8 tensor over [lo, hi], lo <= 0 <= hi, with zero exactly representable."""
    scale = (hi - lo) / 255.0
    return scale, int(np.clip(round(-128.0 - lo / scale), -128, 127))


def _act_range(lo: float, hi: float, act: str) -> tuple[float, float]:
    xs = [lo, hi] + ([SILU_ARGMIN] if act == "silu" and lo < SILU_ARGMIN < hi else [])
    ys = activation(np.array(xs), act)
    return min(float(ys.min()), 0.0), max(float(ys.max()), 0.0)


def _q31(r: float) -> tuple[int, int]:
    """r = mult / 2^31 * 2^shift with mult in [2^30, 2^31), CMSIS-NN's convention."""
    m, e = math.frexp(r)
    q = round(m * (1 << 31))
    if q == 1 << 31:
        q //= 2
        e += 1
    if not -31 <= e <= 0:
        raise ValueError(f"requantization scale {r:.3e} is out of the supported range")
    return q, e


def _m15(r: float) -> tuple[int, int]:
    """r = mult / 2^shift with mult in [2^14, 2^15): y = (acc * mult + 2^(shift - 1)) >> shift."""
    m, e = math.frexp(r)
    q = round(m * (1 << 15))
    if q == 1 << 15:
        q //= 2
        e += 1
    shift = 15 - e
    if not 1 <= shift <= 62:
        raise ValueError(f"requantization scale {r:.3e} is out of the supported range")
    return q, shift


def _float_layers(mlp: Mlp, interleaved: bool) -> list[QuantLayer]:
    layers = []
    for layer in mlp.layers:
        in_pad = pad4(layer.n_in)
        w = _pad_columns(layer.w, in_pad).astype(np.float32)
        b = layer.b.astype(np.float32)
        if interleaved:
            w = _blocks(w, 4, 1)
            b = _padded(b, 4)
        layers.append(QuantLayer(n_in=layer.n_in, in_pad=in_pad, n_out=layer.n_out, w=w, b=b, act=layer.act))
    return layers


def _standardise(mlp: Mlp, calib: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[Dense]]:
    mean = calib.mean(axis=0)
    std = calib.std(axis=0)
    std = np.where(std > 1e-9 * np.maximum(1.0, np.abs(mean)), std, 1.0)  # constant features keep their scale
    first = mlp.layers[0]
    folded = Dense(w=first.w * std[None, :], b=first.b + first.w @ mean, act=first.act)
    return mean, std, [folded, *mlp.layers[1:]]


def _quantize_s8(mlp: Mlp, calib: np.ndarray, margin: float) -> QuantNet:
    pre = mlp.pre_activations(calib)
    mean, std, layers = _standardise(mlp, calib)
    s_in, z_in = _asym8(*_range((calib - mean) / std, margin))
    in_scale = (1.0 / (std * s_in)).astype(np.float32)
    in_bias = (-mean / (std * s_in) + z_in).astype(np.float32)
    out = []
    for k, layer in enumerate(layers):
        in_pad = pad4(layer.n_in)
        s_w = np.abs(layer.w).max(axis=1) / 127.0
        s_w = np.where(s_w > 0, s_w, 1.0)
        wq = np.clip(np.round(layer.w / s_w[:, None]), -127, 127).astype(np.int8)
        bq = np.round(layer.b / (s_in * s_w))
        if np.abs(bq).max() >= 2**31:
            raise ValueError(f"layer {k}: bias does not fit int32")
        s_pre, z_pre = _asym8(*_range(pre[k], margin))
        mult, shift = zip(*(_q31(s_in * sw / s_pre) for sw in s_w))
        lut = None
        s_next, z_next = s_pre, z_pre
        if layer.act != "none":
            s_next, z_next = _asym8(*_act_range(s_pre * (-128 - z_pre), s_pre * (127 - z_pre), layer.act))
            post = activation(s_pre * (np.arange(-128, 128) - z_pre), layer.act)
            lut = np.clip(np.round(post / s_next) + z_next, -128, 127).astype(np.int8)
        out.append(QuantLayer(
            n_in=layer.n_in, in_pad=in_pad, n_out=layer.n_out,
            w=_int8_layout(_pad_columns(wq, in_pad)), b=_padded(bq.astype(np.int32), 4), act=layer.act,
            mult=_padded(np.array(mult, dtype=np.int32), 4), shift=_padded(np.array(shift, dtype=np.int32), 4),
            in_offset=-z_in, out_offset=z_pre, lut=lut,
        ))
        s_in, z_in = s_next, z_next
    return QuantNet("s8", out, mlp.n_in, mlp.n_out, in_scale, in_bias, out_scale=s_in, out_zero=z_in)


def _quantize_s16(mlp: Mlp, calib: np.ndarray, margin: float, wide_weights: bool) -> QuantNet:
    pre = mlp.pre_activations(calib)
    mean, std, layers = _standardise(mlp, calib)
    s_in = max(abs(v) for v in _range((calib - mean) / std, margin)) / 32767.0
    in_scale = (1.0 / (std * s_in)).astype(np.float32)
    in_bias = (-mean / (std * s_in)).astype(np.float32)
    w_max = 32767.0 if wide_weights else 127.0
    knots = -32768.0 + 128.0 * np.arange(LUT16_SIZE)
    out = []
    for k, layer in enumerate(layers):
        in_pad = pad4(layer.n_in)
        s_w = np.abs(layer.w).max(axis=1) / w_max
        s_w = np.where(s_w > 0, s_w, 1.0)
        wq = np.clip(np.round(layer.w / s_w[:, None]), -w_max, w_max)
        bq = np.round(layer.b / (s_in * s_w))
        if not wide_weights and np.max(np.abs(bq) + np.abs(wq).sum(axis=1) * 32768.0) >= 2**31:
            raise ValueError(f"layer {k}: the int32 accumulator could overflow")
        s_pre = max(abs(v) for v in _range(pre[k], margin)) / 32767.0
        mult, shift = zip(*(_m15(s_in * sw / s_pre) for sw in s_w))
        lut = None
        s_next = s_pre
        if layer.act != "none":
            lo, hi = _act_range(-32768.0 * s_pre, 32767.0 * s_pre, layer.act)
            s_next = max(-lo, hi) / 32767.0
            lut = np.clip(np.round(activation(s_pre * knots, layer.act) / s_next), -32768, 32767).astype(np.int16)
        w = _pad_columns(wq.astype(np.int16 if wide_weights else np.int8), in_pad)
        rows = 2 if wide_weights else 4
        out.append(QuantLayer(
            n_in=layer.n_in, in_pad=in_pad, n_out=layer.n_out,
            w=_blocks(w, 2, 4) if wide_weights else _int8_layout(w),
            b=_padded(bq.astype(np.int64 if wide_weights else np.int32), rows), act=layer.act,
            mult=_padded(np.array(mult, dtype=np.int32), rows),
            shift=_padded(np.array(shift, dtype=np.int32), rows, fill=1), lut=lut,
        ))
        s_in = s_next
    fmt = "s16" if wide_weights else "s16s8"
    return QuantNet(fmt, out, mlp.n_in, mlp.n_out, in_scale, in_bias, out_scale=s_in, out_zero=0)


def quantize(mlp: Mlp, calib: np.ndarray | None, fmt: str, *, margin: float = 0.1) -> QuantNet:
    """``mlp`` in format ``fmt`` (one of ``FORMATS``), calibrated on the rows of ``calib`` (integer formats only)."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    if fmt.startswith("f32"):
        return QuantNet(fmt, _float_layers(mlp, interleaved=fmt != "f32"), mlp.n_in, mlp.n_out)
    if calib is None:
        raise ValueError(f"format {fmt} needs calibration inputs")
    calib = np.asarray(calib, dtype=np.float64).reshape(-1, mlp.n_in)
    if fmt == "s8":
        return _quantize_s8(mlp, calib, margin)
    return _quantize_s16(mlp, calib, margin, wide_weights=fmt == "s16")
