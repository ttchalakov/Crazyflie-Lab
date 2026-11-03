"""Build ``lib/shared``'s network code for this computer and run networks exactly as the STM32 does.

The library is compiled from the same sources as the firmware, with the flags that make float32 arithmetic round the
same way on both (no contraction into fused multiply-adds, no fast-math), so a network exported by
``cf_lab.nn.export`` gives the drone's bits here. Libraries are cached under ``build_dir`` by a hash of their sources
and flags.
"""

from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from cf_lab.sdk import SdkError, sdk_root

CFLAGS = ("-std=c11", "-O2", "-fPIC", "-shared", "-ffp-contract=off", "-fno-fast-math",
          "-Wall", "-Wextra", "-Werror")


def shared_dir() -> Path:
    return sdk_root() / "lib" / "shared"


def build_host_library(
    extra_sources: Iterable[Path] = (),
    *,
    include_dirs: Iterable[Path] = (),
    build_dir: Path,
    name: str = "cfnn",
    defines: Iterable[str] = (),
) -> Path:
    """Compile lib/shared's nn.c and fpmath.c with ``extra_sources`` into a shared library; returns its path."""
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        raise SdkError("no C compiler (cc or gcc) on PATH")
    shared = shared_dir()
    sources = [shared / "src" / "nn.c", shared / "src" / "fpmath.c", *map(Path, extra_sources)]
    includes = [shared / "include", *map(Path, include_dirs)]
    flags = [*CFLAGS, *(f"-D{d}" for d in defines)]
    digest = hashlib.sha256(" ".join(flags).encode())
    for path in sorted(sources) + sorted(h for d in includes for h in Path(d).rglob("*.h")):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    out = Path(build_dir) / f"lib{name}-{digest.hexdigest()[:12]}.so"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        command = [compiler, *flags, *(f"-I{d}" for d in includes), *map(str, sources), "-o", str(out), "-lm"]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SdkError(f"host build failed:\n{result.stderr}")
    return out


class _Net(ctypes.Structure):
    """cf_nn_net_t, field for field."""

    _fields_ = [
        ("format", ctypes.c_uint8),
        ("n_layers", ctypes.c_uint8),
        ("n_in", ctypes.c_uint16),
        ("n_out", ctypes.c_uint16),
        ("max_width", ctypes.c_uint16),
        ("layers", ctypes.c_void_p),
        ("in_scale", ctypes.c_void_p),
        ("in_bias", ctypes.c_void_p),
        ("out_scale", ctypes.c_float),
        ("out_zero", ctypes.c_int32),
    ]


class HostLibrary:
    def __init__(self, path: Path) -> None:
        self.lib = ctypes.CDLL(str(path))
        self.lib.cf_nn_work_bytes.argtypes = [ctypes.c_void_p]
        self.lib.cf_nn_work_bytes.restype = ctypes.c_uint32
        self.lib.cf_nn_run_batch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_void_p]
        self.lib.cf_nn_run_batch.restype = None

    def net(self, symbol: str) -> HostNet:
        return HostNet(self, symbol)


class HostNet:
    """A ``cf_nn_net_t`` exported into the host library, callable on rows of float32 inputs."""

    def __init__(self, library: HostLibrary, symbol: str) -> None:
        self._lib = library.lib
        self._net = _Net.in_dll(self._lib, symbol)
        self.n_in = int(self._net.n_in)
        self.n_out = int(self._net.n_out)
        self._work = ctypes.create_string_buffer(self._lib.cf_nn_work_bytes(ctypes.byref(self._net)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, self.n_in)
        y = np.empty((x.shape[0], self.n_out), dtype=np.float32)
        self._lib.cf_nn_run_batch(ctypes.byref(self._net), x.ctypes.data, y.ctypes.data, x.shape[0], self._work)
        return y


def fnv1a(data: bytes, h: int = 0x811C9DC5) -> int:
    """32-bit FNV-1a, as the bench firmware hashes its outputs."""
    for byte in data:
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return h
