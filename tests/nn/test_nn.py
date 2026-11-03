"""cf_lab.nn: quantization, the C export, and the host build of lib/shared's MLP code (needs a C compiler)."""

import ctypes
import itertools
import math

import numpy as np
import pytest

from cf_lab.nn import Dense, HostLibrary, Mlp, build_host_library, quantize, write_c
from cf_lab.nn.spec import activation


def random_mlp(rng, sizes, act="silu"):
    layers = []
    for k, (n_in, n_out) in enumerate(itertools.pairwise(sizes)):
        layers.append(Dense(w=rng.normal(scale=1.0 / np.sqrt(n_in), size=(n_out, n_in)),
                            b=rng.normal(scale=0.1, size=n_out), act=act if k < len(sizes) - 2 else "none"))
    return Mlp(tuple(layers))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A 34-64-64-15 network (inputs and outputs not multiples of four) in every format, with features of very
    different scales, exported and built for the host."""
    rng = np.random.default_rng(1)
    mlp = random_mlp(rng, [34, 64, 64, 15])
    scales = np.geomspace(1e-3, 3.0, 34)
    calib = rng.normal(size=(4000, 34)) * scales + rng.normal(size=34)
    nets = {f"t_{fmt}": quantize(mlp, calib, fmt) for fmt in ("f32", "f32x4", "f32fma", "s8", "s16s8", "s16")}
    out = tmp_path_factory.mktemp("nn")
    write_c(nets, out / "nets.c", out / "nets.h")
    lib = HostLibrary(build_host_library([out / "nets.c"], include_dirs=[out], build_dir=out / "build"))
    test = calib[:500]
    return mlp, lib, test, out


def test_the_four_row_float_kernel_gives_the_reference_kernels_bits(built):
    _, lib, test, _ = built
    a, b = lib.net("t_f32")(test), lib.net("t_f32x4")(test)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


@pytest.mark.parametrize("fmt,tolerance", [("f32", 1e-5), ("f32fma", 1e-5), ("s16", 2e-3), ("s16s8", 2e-2),
                                           ("s8", 1e-1)])
def test_every_format_tracks_float64(built, fmt, tolerance):
    mlp, lib, test, _ = built
    ref = mlp.forward(test)
    err = np.abs(lib.net(f"t_{fmt}")(test) - ref).max() / np.abs(ref).max()
    assert err < tolerance


def test_shared_float_weights_are_written_once(built):
    _, _, _, out = built
    source = (out / "nets.c").read_text()
    assert source.count("t_f32x4_w0[") == 1  # f32x4 writes the interleaved weights
    assert "t_f32fma_w0[" not in source       # f32fma reuses them


def cmsis_requantize(acc, mult, shift):
    """CMSIS-NN arm_nn_requantize as the Cortex-M4 computes it (shift <= 0)."""
    right = -shift
    prod = (1 - (1 << 30) if (acc < 0) != (mult < 0) else 1 << 30) + acc * mult
    hi = int(prod / (1 << 31)) if prod >= 0 else -int(-prod / (1 << 31))  # truncation toward zero
    hi = max(min(hi, 2**31 - 1), -(2**31))
    mask = (1 << right) - 1
    result = hi >> right
    threshold = (mask >> 1) + (1 if result < 0 else 0)
    return result + (1 if (hi & mask) > threshold else 0)


def test_requantization_is_cmsis_arithmetic(built):
    _, lib, _, _ = built
    fn = lib.lib.cf_nn_requant_q31
    fn.argtypes = [ctypes.c_int32] * 3
    fn.restype = ctypes.c_int32
    rng = np.random.default_rng(2)
    for _ in range(2000):
        acc = int(rng.integers(-(2**24), 2**24))
        mult = int(rng.integers(2**30, 2**31))
        shift = -int(rng.integers(0, 20))
        assert fn(acc, mult, shift) == cmsis_requantize(acc, mult, shift)


def test_portable_math_is_accurate(built):
    _, lib, _, _ = built
    for name in ("cf_expf", "cf_siluf", "cf_tanhf"):
        getattr(lib.lib, name).argtypes = [ctypes.c_float]
        getattr(lib.lib, name).restype = ctypes.c_float
    xs = np.linspace(-20, 20, 4001, dtype=np.float32)
    exp = np.array([lib.lib.cf_expf(float(x)) for x in xs])
    assert np.max(np.abs(exp / np.exp(xs.astype(np.float64)) - 1)) < 3e-7
    silu = np.array([lib.lib.cf_siluf(float(x)) for x in xs])
    assert np.max(np.abs(silu - activation(xs.astype(np.float64), "silu"))) < 1e-5
    tanh = np.array([lib.lib.cf_tanhf(float(x)) for x in xs])
    assert np.max(np.abs(tanh - np.tanh(xs.astype(np.float64)))) < 1e-6
    sincos = lib.lib.cf_sincosf
    sincos.argtypes = [ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    s, c = ctypes.c_float(), ctypes.c_float()
    for x in np.linspace(-10, 10, 2001):
        sincos(float(x), ctypes.byref(s), ctypes.byref(c))
        assert abs(s.value - math.sin(x)) < 1e-6 and abs(c.value - math.cos(x)) < 1e-6
