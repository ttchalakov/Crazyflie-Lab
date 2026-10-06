from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from cf_lab.sdk import DOCKER_PLATFORM, SdkError, docker_mounts, sdk_root, submodule_hint


DEFAULT_MODEL_NAME = "policy"
# NEMO and DORY run only in this image: NEMO needs torch < 1.13 (private torch.onnx
# APIs), which has no wheels for the Python cf_lab needs, and DORY needs numpy < 1.25.
NEMO_IMAGE = "cf-lab/nemo:latest"
PULP_NN_KERNELS = Path("dory/Hardware_targets/PULP/Backend_Kernels/pulp-nn")
# Export-only knobs read by a model module inside the NEMO container. Without
# explicit forwarding, `VAR=... cf-lab compile-model ...` sets the
# variable on the host only and the container silently uses the default.
NEMO_EXPORT_FORWARDED_ENV = ("HOVER_NEMO_REQUANTIZATION_FACTOR",)
DEFAULT_DORY_TARGET = "PULP.GAP8"
DEFAULT_DORY_OPTIONAL = "auto"
# DORY's per-layer golden checks and performance counters are opt-in
# (--dory-verbose-level Check_all+Perf_final); by default the network has neither.
DEFAULT_DORY_VERBOSE = "None"
DEFAULT_CODE_RESERVED_SPACE = 132000
DEFAULT_BNRELU_BITS = 32
# DORY emits a standalone pulp-os test app; the example provides its own main(),
# so these test-harness files are dropped when integrating into an example.
DORY_TEST_HARNESS = ("main.c", "network_main.c")
DORY_ACCEPTED_ONNX_OPS = frozenset(
    {
        "Add",
        "AveragePool",
        "Cast",
        "Clip",
        "Concat",
        "Constant",
        "Conv",
        "Div",
        "Flatten",
        "Floor",
        "Gather",
        "Gemm",
        "GlobalAveragePool",
        "LogSoftmax",
        "MatMul",
        "MaxPool",
        "Mul",
        "Pad",
        "Relu",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Unsqueeze",
    }
)


class PipelineError(SdkError):
    """Raised when the NEMO/DORY deployment pipeline cannot continue."""


@dataclass(frozen=True)
class ArtifactLayout:
    work_dir: Path
    nemo_dir: Path
    nemo_onnx_path: Path
    dory_dir: Path
    dory_config_path: Path
    dory_app_dir: Path
    metadata_path: Path


@dataclass(frozen=True)
class DoryCodegenConfig:
    target: str = DEFAULT_DORY_TARGET
    optional: str = DEFAULT_DORY_OPTIONAL
    verbose_level: str = DEFAULT_DORY_VERBOSE
    code_reserved_space: int = DEFAULT_CODE_RESERVED_SPACE
    bnrelu_bits: int = DEFAULT_BNRELU_BITS
    n_inputs: int = 1
    # DORY prepends this to network-specific symbols/files (network.c -> <p>network.c,
    # main.c -> <p>main.c) so a network can coexist with the rest of a GAP8 app.
    prefix: str = ""


@dataclass(frozen=True)
class DoryHyperramPlan:
    weights_bytes: int
    input_bytes: int
    output_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class IntegrationResult:
    generated_dir: Path
    synced_src: int
    synced_inc: int
    synced_hex: int


@dataclass(frozen=True)
class CompileResult:
    layout: ArtifactLayout
    dory_config: Path
    dory_app_dir: Path
    integration: IntegrationResult


def default_artifact_layout(models_root: Path, model_name: str = DEFAULT_MODEL_NAME) -> ArtifactLayout:
    return artifact_layout_for_work_dir(models_root / "dory" / model_name)


def artifact_layout_for_work_dir(work_dir: Path) -> ArtifactLayout:
    nemo_dir = work_dir / "nemo"
    dory_dir = work_dir / "dory"
    return ArtifactLayout(
        work_dir=work_dir,
        nemo_dir=nemo_dir,
        nemo_onnx_path=nemo_dir / "model_int.onnx",
        dory_dir=dory_dir,
        dory_config_path=dory_dir / "config.json",
        dory_app_dir=work_dir / "gap8_app",
        metadata_path=work_dir / "metadata.json",
    )


def parse_input_shape(value: str) -> tuple[int, ...]:
    parts = value.replace("x", ",").split(",")
    try:
        shape = tuple(int(part.strip()) for part in parts if part.strip())
    except ValueError as exc:
        raise ValueError(f"Input shape must contain integers: {value}") from exc
    if not shape:
        raise ValueError("Input shape must contain at least one dimension.")
    if any(dim <= 0 for dim in shape):
        raise ValueError("Input shape dimensions must be positive.")
    return shape


def write_dory_config(
    *,
    config_path: Path,
    onnx_path: Path,
    settings: DoryCodegenConfig,
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_relative_path = os.path.relpath(onnx_path, start=config_path.parent)
    payload = {
        "BNRelu_bits": settings.bnrelu_bits,
        "onnx_file": onnx_relative_path,
        "code reserved space": settings.code_reserved_space,
    }
    if settings.n_inputs != 1:
        payload["n_inputs"] = settings.n_inputs
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dory_hw_description_path(dory_dir: Path, target: str) -> Path:
    """Path to the HW_description.json for a DORY target (e.g. PULP.GAP8)."""
    return dory_dir / "dory" / "Hardware_targets" / Path(*target.split(".")) / "HW_description.json"


@contextlib.contextmanager
def dory_l2_override(*, dory_dir: Path, target: str, l2_bytes: int | None):
    """Temporarily set DORY's L2 tiling budget for codegen, then restore the file.

    DORY exposes no CLI flag for the L2 working-set size; it reads it from the
    target's HW_description.json. A network destined for a larger GAP8 app must be
    tiled to fit beside it, so we patch memory.L2.dimension for the duration of
    codegen and restore the original file byte-for-byte afterward. No-op when
    ``l2_bytes`` is None.
    """
    if l2_bytes is None:
        yield
        return
    hw_path = dory_hw_description_path(dory_dir, target)
    if not hw_path.is_file():
        raise PipelineError(f"Cannot set L2 budget: {hw_path} not found (target {target}).")
    # Read/restore exact bytes so we never rewrite the upstream file's line endings
    # (it ships CRLF; text-mode read would silently normalize to LF).
    original_bytes = hw_path.read_bytes()
    data = json.loads(original_bytes.decode("utf-8"))
    try:
        data["memory"]["L2"]["dimension"] = l2_bytes
    except (KeyError, TypeError) as exc:
        raise PipelineError(f"{hw_path} has no memory.L2.dimension to override.") from exc
    hw_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    try:
        yield
    finally:
        hw_path.write_bytes(original_bytes)


def build_dory_codegen_command(
    *,
    dory_dir: Path,
    config_path: Path,
    onnx_path: Path,
    app_dir: Path,
    settings: DoryCodegenConfig,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        DOCKER_PLATFORM,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        *docker_mounts(dory_dir, config_path, onnx_path, app_dir),
        "-w",
        str(dory_dir.resolve()),
        NEMO_IMAGE,
        "python",
        "network_generate.py",
        "NEMO",
        settings.target,
        str(config_path.resolve()),
        "--app_dir",
        str(app_dir.resolve()),
        "--verbose_level",
        settings.verbose_level,
        "--optional",
        settings.optional,
    ]
    if settings.prefix:
        command += ["--prefix", settings.prefix]
    return command


def unsupported_dory_onnx_ops(op_types: list[str]) -> list[str]:
    unsupported: list[str] = []
    for op_type in op_types:
        if op_type not in DORY_ACCEPTED_ONNX_OPS and op_type not in unsupported:
            unsupported.append(op_type)
    return unsupported


def validate_dory_onnx(onnx_path: Path) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    unsupported = unsupported_dory_onnx_ops([node.op_type for node in model.graph.node])
    if unsupported:
        joined = ", ".join(unsupported)
        raise PipelineError(f"ONNX contains operators not accepted by the DORY frontend: {joined}")


def build_nemo_docker_command(
    *,
    model_module: Path,
    model_factory: str,
    checkpoint: Path,
    calibration: Path,
    input_shape: str,
    output_path: Path,
    work_dir: Path,
    nemo_dir: Path,
) -> list[str]:
    src = sdk_root() / "src"
    driver = src / "cf_lab" / "aideck" / "nemo_container_export.py"
    # The export knobs a model module reads from the environment are only useful
    # if they survive into the container that actually runs it.
    forwarded_env: list[str] = []
    for name in NEMO_EXPORT_FORWARDED_ENV:
        value = os.environ.get(name)
        if value is not None:
            forwarded_env += ["-e", f"{name}={value}"]
    paths = [model_module, checkpoint, calibration, output_path, work_dir, nemo_dir]
    return [
        "docker",
        "run",
        "--rm",
        "--platform",
        DOCKER_PLATFORM,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        # Model modules run here too, and may `import cf_lab` like any host code.
        "-e",
        f"PYTHONPATH={src}",
        *forwarded_env,
        *docker_mounts(src, *paths),
        "-w",
        str(work_dir.resolve()),
        NEMO_IMAGE,
        "python",
        str(driver),
        "--model-module",
        str(model_module.resolve()),
        "--model-factory",
        model_factory,
        "--checkpoint",
        str(checkpoint.resolve()),
        "--calibration",
        str(calibration.resolve()),
        "--input-shape",
        input_shape,
        "--output",
        str(output_path.resolve()),
        "--work-dir",
        str(work_dir.resolve()),
        "--nemo-dir",
        str(nemo_dir.resolve()),
    ]


def copy_existing_nemo_onnx(*, source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise PipelineError(f"NEMO ONNX file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    for sidecar in _dory_sidecar_files(source.parent):
        normalize_dory_sidecar(source=sidecar, destination=destination.parent / sidecar.name)
    return destination


def _dory_sidecar_files(directory: Path) -> list[Path]:
    names = ["input.txt", "output.txt"]
    files = [directory / name for name in names if (directory / name).is_file()]
    files.extend(sorted(directory.glob("out_layer*.txt")))
    return files


def normalize_dory_sidecar(*, source: Path, destination: Path) -> None:
    values: list[str] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        token = raw_line.strip()
        if not token or token.startswith("#"):
            continue
        token = token.rstrip("\\").strip()
        if token.endswith(","):
            token = token[:-1].strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise PipelineError(f"{source} contains non-numeric sidecar value on line {line_number}: {token}") from exc
        if not value.is_integer():
            raise PipelineError(f"{source} contains non-integer sidecar value on line {line_number}: {token}")
        values.append(str(int(value)))
    destination.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    if cwd is not None and not cwd.is_dir():
        raise PipelineError(f"Working directory not found: {cwd}")
    try:
        subprocess.run(command, check=True, cwd=str(cwd) if cwd is not None else None)
    except FileNotFoundError as exc:
        raise PipelineError(f"Required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise PipelineError(f"Command failed with exit code {exc.returncode}: {' '.join(command)}") from exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sync_dir(*, src_dir: Path, dst_dir: Path, pattern: str) -> int:
    """Replace dst_dir's contents with the pattern-matching files from src_dir."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob(pattern))
    for source in files:
        shutil.copyfile(source, dst_dir / source.name)
    return len(files)


def _dory_int_array(header: str, name: str) -> list[int]:
    match = re.search(
        rf"static\s+int\s+{re.escape(name)}\s*\[\s*\d+\s*\]\s*=\s*\{{(.*?)\}}\s*;",
        header,
        flags=re.DOTALL,
    )
    if match is None:
        raise PipelineError(f"DORY network header has no parseable {name} array.")
    values = [int(value) for value in re.findall(r"-?\d+", match.group(1))]
    if not values or any(value < 0 for value in values):
        raise PipelineError(f"DORY network header has invalid {name}: {values!r}.")
    return values


def largest_activation(int_onnx: Path) -> int:
    """The largest activation in NEMO's integer ONNX, in bytes: DORY runs them at 8 bits.

    rightsize_dory_hyperram needs it for a layer DORY keeps in L3, whose header
    sizes are only its L2 tile. Every tensor but the weights counts, so one that
    DORY fuses away (a convolution's output before its pooling) can only make the
    reservation larger.
    """
    import onnx

    graph = onnx.shape_inference.infer_shapes(onnx.load(str(int_onnx))).graph
    weights = {tensor.name for tensor in graph.initializer}
    return max(
        (math.prod(dim.dim_value for dim in value.type.tensor_type.shape.dim)
         for value in (*graph.input, *graph.value_info, *graph.output) if value.name not in weights),
        default=0,
    )


def rightsize_dory_hyperram(*, app_dir: Path, prefix: str = "", largest_activation: int = 0) -> DoryHyperramPlan:
    """Replace DORY's fixed 7 MB L3 reservations with graph-derived sizes.

    The GAP8 template reserves 4 MB for weights and two 1.5 MB activation
    buffers for every graph.  Those values are unrelated to the generated
    network and leave almost no room in an AI-deck's 8 MB HyperRAM.  Derive the
    weight total and a conservative full-activation ping-pong size from DORY's
    own generated arrays, validate the weight blobs, and patch the pristine app
    before it is built or integrated.  Exact source patterns make template
    drift fail closed instead of silently editing unfamiliar generated code.

    For a layer whose activations DORY keeps in L3, the header's sizes are the
    L2 tile, not the activation, so a caller that knows the graph passes its
    largest activation (largest_activation, bytes); without it such a layer
    writes past its HyperRAM buffer.
    """
    norm = prefix if not prefix or prefix.endswith("_") else prefix + "_"
    source_path = app_dir / "src" / f"{norm}network.c"
    header_path = app_dir / "inc" / f"{norm}network.h"
    if not source_path.is_file() or not header_path.is_file():
        raise PipelineError(
            f"DORY network runtime missing: expected {source_path} and {header_path}."
        )

    # The L2-only target (memory.levels == 2) keeps every weight resident in L2
    # as a C array and emits no L3 reservations at all, so there is nothing to
    # rightsize. Detect it by the absence of the macro rather than by target
    # name, so it stays correct if another 2-level target is added.
    source_probe = source_path.read_text(encoding="utf-8")
    if "#define L3_WEIGHTS_SIZE" not in source_probe:
        return DoryHyperramPlan(
            weights_bytes=0, input_bytes=0, output_bytes=0, total_bytes=0
        )

    header = header_path.read_text(encoding="utf-8")
    weight_sizes = _dory_int_array(header, "weights_size")
    layers_with_weights = _dory_int_array(header, "layer_with_weights")
    activation_sizes = _dory_int_array(header, "activations_size")
    output_sizes = _dory_int_array(header, "activations_out_size")
    if len(layers_with_weights) != len(weight_sizes) or any(
        flag not in (0, 1) for flag in layers_with_weights
    ):
        raise PipelineError(
            "DORY weights_size and layer_with_weights arrays are inconsistent."
        )
    # DORY's per-layer `weights_size[i]` is the size of the L2 *working buffer*
    # the layer allocates (network.c: `L2_weights = dmalloc(weights_size[i])`),
    # not the size of the weight blob in L3.  For a layer small enough to load in
    # one shot the two coincide, but once a layer is big enough that DORY tiles
    # its weights out of L3 (e.g. a 512x512 FC) `weights_size[i]` collapses to the
    # double-buffered *tile* footprint (2 weight tiles + 2 bias tiles) while the
    # real blob stays full size -- a 512x512 layer emits a 264192 B .hex but
    # reports weights_size == 33024 (== 2*16384 + 2*128).  The authoritative L3
    # size is the generated .hex file itself: `<prefix>network_initialize()`
    # streams each file into L3 with `load_file_to_ram()` and advances the L3
    # cursor by the returned file length.  So size the L3 weight reservation from
    # the .hex files, and use `weights_size[i]` only as a lower bound per blob.
    l2_weight_buffers = [
        size for size, present in zip(weight_sizes, layers_with_weights, strict=True) if present
    ]
    weight_names_match = re.search(
        r"static\s+const\s+char\s*\*\s*L3_weights_files\s*\[\s*\]\s*=\s*\{(.*?)\}\s*;",
        header,
        flags=re.DOTALL,
    )
    if weight_names_match is None:
        raise PipelineError("DORY network header has no parseable L3_weights_files array.")
    weight_names = re.findall(r'"([^"]+)"', weight_names_match.group(1))
    if len(weight_names) != len(l2_weight_buffers):
        raise PipelineError(
            "DORY weight file/size counts differ: "
            f"{len(weight_names)} files versus {len(l2_weight_buffers)} sizes."
        )
    blob_sizes: list[int] = []
    overread = 0
    for name, l2_buffer in zip(weight_names, l2_weight_buffers, strict=True):
        weight_path = app_dir / "hex" / name
        if not weight_path.is_file():
            raise PipelineError(f"DORY weight blob {weight_path} is missing.")
        blob = weight_path.stat().st_size
        # A layer's L2 buffer can be larger than its blob, and cl_ram_read() fills
        # the whole buffer, reading on past the blob. Pad the reservation so the
        # last one stays in it.
        overread = max(overread, l2_buffer - blob)
        blob_sizes.append(blob)

    weights_bytes = max(4, sum(blob_sizes) + overread)
    activation_bytes = max(4, largest_activation, *activation_sizes, *output_sizes)
    plan = DoryHyperramPlan(
        weights_bytes=weights_bytes,
        input_bytes=activation_bytes,
        output_bytes=activation_bytes,
        total_bytes=weights_bytes + 2 * activation_bytes,
    )

    source = source_path.read_text(encoding="utf-8")
    replacements = {
        "L3_WEIGHTS_SIZE": plan.weights_bytes,
        "L3_INPUT_SIZE": plan.input_bytes,
        "L3_OUTPUT_SIZE": plan.output_bytes,
    }
    for macro, value in replacements.items():
        pattern = rf"^#define\s+{macro}\s+\d+\s*$"
        source, count = re.subn(
            pattern,
            f"#define {macro} {value}",
            source,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise PipelineError(
                f"DORY template drift: expected one numeric {macro} definition in {source_path}."
            )

    # The same template repeats its fixed input-reservation literal in residual
    # allocation paths.  Tie those paths to the derived macro as well.
    source = source.replace("cl_ram_malloc(1500000)", "cl_ram_malloc(L3_INPUT_SIZE)")
    source = source.replace(
        "1500000 - activations_out_size[i]",
        "L3_INPUT_SIZE - activations_out_size[i]",
    )
    source_path.write_text(source, encoding="utf-8")
    return plan


def verify_fc_weight_orientation(*, generated_dir: Path, int_onnx: Path, prefix: str = "") -> list[str]:
    """Assert every fully-connected weight blob is serialized ``[out][in]``.

    That is the order both the generated DMA descriptor (``length_1d_copy = in``,
    ``number_of_2d_copies = out``) and ``pulp_nn_linear`` (``pB = pWeights +
    i*dim_vec``) read back, so a blob in the other order is a silently incorrect
    network on the deck. Nothing downstream catches it on its own: the shape stops
    disambiguating the axes once a layer is square, and a byte-sum
    ``weights_checksum`` cannot see a transpose at all, because permuting bytes
    does not change their multiset.

    Our DORY fork's frontends read the layout from the ONNX ``transB`` attribute
    (``DORY_node.fc_weights_layout``), so this function only confirms the result
    -- it deliberately does **not** rewrite anything. A failure here means the
    frontend regressed, and rewriting the blob would hide that.

    Returns the names of the blobs it checked.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(int_onnx))
    initializers = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    gemms = [n for n in model.graph.node if n.op_type == "Gemm"]

    norm = prefix if (not prefix or prefix.endswith("_")) else prefix + "_"
    blobs = _locate_weight_blobs(generated_dir=generated_dir, norm=norm)
    if len(blobs) != len(gemms):
        # Convolutional or mixed graphs are out of scope for this check; it cannot
        # pair a blob with an ONNX node without guessing.
        return []

    checked: list[str] = []
    for node, blob in zip(gemms, blobs):
        weight_name = next((i for i in node.input if i in initializers), None)
        if weight_name is None:
            continue
        ref = np.rint(initializers[weight_name]).astype(np.int8)
        if ref.ndim != 2:
            continue
        # ONNX Gemm stores B as [out][in] when transB=1 and [in][out] otherwise;
        # the kernel always wants [out][in].
        trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
        if not trans_b:
            ref = ref.T
        out_f, in_f = ref.shape
        raw = np.frombuffer(blob.read(), dtype=np.uint8)
        n = out_f * in_f
        if raw.size < n:
            raise PipelineError(f"{blob.name} is smaller than {out_f}x{in_f} weights.")
        got = raw[:n].view(np.int8).reshape(out_f, in_f)
        if np.array_equal(got, ref):
            checked.append(blob.name)
            continue
        transposed = np.array_equal(raw[:n].view(np.int8).reshape(in_f, out_f).T, ref)
        raise PipelineError(
            f"{blob.name} is {'the transpose of' if transposed else 'not'} "
            f"{weight_name} ({out_f}x{in_f}, transB={trans_b}); pulp_nn_linear reads "
            "[out][in], so this network would compute the wrong answer. "
            + (
                "The DORY frontend's weight-layout fix (DORY_node.fc_weights_layout) "
                "has regressed -- check external/dory."
                if transposed
                else "DORY's weight serialization changed in a way this check does not model."
            )
        )
    return checked


class _WeightBlob:
    """One layer's weight bytes, wherever DORY chose to put them.

    The L3 target writes each layer to its own ``.hex`` file in the readfs
    partition; the L2-only target emits them as C arrays in ``<prefix>weights.h``
    instead. The orientation check is identical either way, so it reads through
    this accessor rather than caring which target produced the network.
    """

    def __init__(self, name: str, read, write) -> None:
        self.name = name
        self.read = read
        self.write = write


def _locate_weight_blobs(*, generated_dir: Path, norm: str) -> list[_WeightBlob]:
    header = generated_dir / "inc" / f"{norm}network.h"
    if not header.is_file():
        return []

    # L3 target: layer order comes from the generated header, not from sorting
    # filenames -- alphabetically "FullyConnected2" precedes "ReluFullyConnected0".
    match = re.search(
        r"L3_weights_files\[\]\s*=\s*\{(.*?)\}", header.read_text(encoding="utf-8"), re.S
    )
    if match is not None:
        hex_dir = generated_dir / "hex"
        paths = [hex_dir / name for name in re.findall(r'"([^"]+)"', match.group(1))]
        if not all(p.is_file() for p in paths):
            return []
        return [
            _WeightBlob(p.name, p.read_bytes, p.write_bytes) for p in paths
        ]

    # L2-only target: arrays declared in <prefix>weights_definition.h, defined in
    # <prefix>weights.h. The declaration order is the graph order.
    definition = generated_dir / "inc" / f"{norm}weights_definition.h"
    source = generated_dir / "inc" / f"{norm}weights.h"
    if not definition.is_file() or not source.is_file():
        raise PipelineError(
            f"DORY network header has neither an L3_weights_files array nor "
            f"{definition.name}/{source.name}; cannot locate the weight blobs."
        )
    names = re.findall(r"extern\s+uint8_t\s+(\w+)\s*\[\s*\d+\s*\]", definition.read_text("utf-8"))
    return [_WeightBlob(n, *_c_array_accessors(source, n)) for n in names]


def _c_array_accessors(path: Path, symbol: str):
    """Read/write the byte values of one ``uint8_t <symbol>[N] = {...};`` array."""
    pattern = re.compile(
        r"(uint8_t\s+" + re.escape(symbol) + r"\s*\[\s*\d+\s*\]\s*=\s*\{)(.*?)(\};)", re.S
    )

    def _match(text: str):
        m = pattern.search(text)
        if m is None:
            raise PipelineError(f"Array {symbol} not found in {path}.")
        return m

    def read() -> bytes:
        m = _match(path.read_text(encoding="utf-8"))
        # DORY writes the initializers as "3., 235., ..." -- float syntax for
        # uint8_t values, so take the integer part of each token.
        return bytes(int(float(tok)) & 0xFF for tok in m.group(2).replace("\n", " ").split(",") if tok.strip())

    def write(data: bytes) -> None:
        text = path.read_text(encoding="utf-8")
        m = _match(text)
        body = "\n  " + ",\n  ".join(
            ", ".join(f"{b:3d}" for b in data[i : i + 16]) for i in range(0, len(data), 16)
        ) + "\n"
        path.write_text(text[: m.start(2)] + body + text[m.end(2) :], encoding="utf-8")

    return read, write


def integrate_generated_app(
    *,
    app_dir: Path,
    network_dir: Path,
    prefix: str = "",
    test_harness: tuple[str, ...] = DORY_TEST_HARNESS,
    int_onnx: Path | None = None,
) -> IntegrationResult:
    """Wire DORY's freshly generated app into an example's GAP8 build, with no
    manual editing of generated sources.

    Given the DORY output ``app_dir`` and an example network dir (e.g.
    ``examples/hover_neural_network/gap8/policy``) it: (1) syncs ``src/*.c``,
    ``inc/*.h``, ``hex/*.hex`` into ``<network_dir>/generated``; (2) drops DORY's
    standalone test harness (the example provides its own ``main()``); and (3)
    verifies the result rather than patching it. Reusable for any model.

    Nothing here patches generated sources: the kernel and runtime fixes are
    committed to our DORY fork, and ``lib/gap8/dory/dory.mk`` force-includes the
    freeRTOS compat shim, so there is nothing to re-derive when DORY is updated.
    """
    app_dir = app_dir.resolve()
    generated_dir = network_dir / "generated"
    if not (app_dir / "src").is_dir():
        raise PipelineError(f"DORY app dir has no src/: {app_dir} (did codegen run?)")

    synced_src = _sync_dir(src_dir=app_dir / "src", dst_dir=generated_dir / "src", pattern="*.c")
    synced_inc = _sync_dir(src_dir=app_dir / "inc", dst_dir=generated_dir / "inc", pattern="*.h")
    synced_hex = _sync_dir(src_dir=app_dir / "hex", dst_dir=generated_dir / "hex", pattern="*.hex")

    # DORY prefixes the test harness too (main.c -> <prefix>_main.c), so drop the
    # prefixed names; the example provides its own main().
    norm = prefix if (not prefix or prefix.endswith("_")) else prefix + "_"
    for stem in test_harness:
        (generated_dir / "src" / f"{norm}{stem}").unlink(missing_ok=True)

    # A cluster task argument array declared smaller than the words written into
    # it corrupts the wrapper's stack frame. Our fork's templates size it
    # correctly, so this only checks it and fails loudly -- silently rewriting
    # generated code would hide a regression.
    network_source = generated_dir / "src" / f"{norm}network.c"
    if network_source.is_file():
        source = network_source.read_text(encoding="utf-8")
        assigned = [int(m) for m in re.findall(r"\bargs\[(\d+)\]\s*=", source)]
        declared = re.search(r"\bunsigned int args\[(\d+)\];", source)
        if assigned and declared is None:
            raise PipelineError(
                f"DORY template drift: {network_source} writes args[{max(assigned)}] "
                "but no 'unsigned int args[N];' declaration was found to check."
            )
        if assigned and int(declared.group(1)) < max(assigned) + 1:
            raise PipelineError(
                f"{network_source}: cluster task args[] is declared "
                f"{declared.group(1)} words but args[{max(assigned)}] is written. "
                "The overrun corrupts network_run_async()'s stack frame. "
                "Our DORY fork fixes this in network.c.t -- is the submodule at the "
                "right commit?"
            )

    if int_onnx is not None and int_onnx.is_file():
        checked = verify_fc_weight_orientation(
            generated_dir=generated_dir, int_onnx=int_onnx, prefix=prefix
        )
        if checked:
            print(f"note: weight orientation verified against {int_onnx.name} ({len(checked)} blobs).")

    return IntegrationResult(
        generated_dir=generated_dir,
        synced_src=synced_src,
        synced_inc=synced_inc,
        synced_hex=synced_hex,
    )


def compile_model(args: argparse.Namespace) -> CompileResult:
    input_shape = parse_input_shape(args.input_shape)
    model_name = args.name or args.checkpoint.stem
    layout = (
        default_artifact_layout(args.models_root, model_name)
        if args.work_dir is None
        else artifact_layout_for_work_dir(args.work_dir)
    )
    layout.work_dir.mkdir(parents=True, exist_ok=True)

    # The DORY network prefix keeps the network's files and symbols from colliding
    # with the rest of the app, which #includes "<name>_network.h".
    dory_prefix = args.dory_prefix if args.dory_prefix is not None else model_name

    settings = DoryCodegenConfig(
        target=args.dory_target,
        optional=args.dory_optional,
        verbose_level=args.dory_verbose_level,
        code_reserved_space=args.code_reserved_space,
        bnrelu_bits=args.bnrelu_bits,
        n_inputs=args.n_inputs,
        prefix=dory_prefix,
    )

    root = sdk_root()
    nemo_dir = args.nemo_dir or root / "external" / "nemo"
    dory_dir = args.dory_dir or root / "external" / "dory"
    needs_nemo = args.nemo_onnx is None
    needs_dory = not args.skip_dory_codegen
    if (needs_nemo and not (nemo_dir / "nemo" / "__init__.py").is_file()) or (
        needs_dory and not (dory_dir / PULP_NN_KERNELS / "__init__.py").is_file()
    ):
        raise PipelineError(
            f"NEMO/DORY sources are missing; run: {submodule_hint('external/nemo', 'external/dory')}"
        )
    if (needs_nemo or needs_dory) and not args.no_docker_image_build:
        run_command(
            ["docker", "build", "--platform", DOCKER_PLATFORM, "-t", NEMO_IMAGE, str(root / "docker" / "nemo")]
        )

    if args.nemo_onnx is not None:
        copy_existing_nemo_onnx(source=args.nemo_onnx, destination=layout.nemo_onnx_path)
    else:
        # Docker would create a missing -w directory as root, unwritable by --user.
        layout.nemo_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            build_nemo_docker_command(
                model_module=args.model_module,
                model_factory=args.model_factory,
                checkpoint=args.checkpoint,
                calibration=args.calibration,
                input_shape=args.input_shape,
                output_path=layout.nemo_onnx_path,
                work_dir=layout.nemo_dir,
                nemo_dir=nemo_dir,
            )
        )
        if not layout.nemo_onnx_path.is_file():
            raise PipelineError(f"NEMO container export did not create {layout.nemo_onnx_path}")

    if not args.skip_onnx_validation:
        validate_dory_onnx(layout.nemo_onnx_path)

    write_dory_config(
        config_path=layout.dory_config_path,
        onnx_path=layout.nemo_onnx_path,
        settings=settings,
    )

    metadata = {
        "model_name": model_name,
        "input_shape": list(input_shape),
        "checkpoint": str(args.checkpoint),
        "calibration": str(args.calibration),
        "dory": asdict(settings),
        "dory_l2_bytes": args.dory_l2_bytes,
        "dory_l2_arena_bytes": (
            args.dory_l2_bytes - settings.code_reserved_space
            if args.dory_l2_bytes is not None
            else None
        ),
    }

    if needs_dory:
        with dory_l2_override(dory_dir=dory_dir, target=settings.target, l2_bytes=args.dory_l2_bytes):
            run_command(
                build_dory_codegen_command(
                    dory_dir=dory_dir,
                    config_path=layout.dory_config_path,
                    onnx_path=layout.nemo_onnx_path,
                    app_dir=layout.dory_app_dir,
                    settings=settings,
                )
            )
        hyperram_plan = rightsize_dory_hyperram(
            app_dir=layout.dory_app_dir,
            prefix=settings.prefix,
            largest_activation=largest_activation(layout.nemo_onnx_path),
        )
        metadata["dory_hyperram"] = asdict(hyperram_plan)

    layout.metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # The generated network is compiled into the app that uses it, which is built
    # and flashed with `cf-lab flash gap8`; DORY's own test harness is dropped.
    integration = integrate_generated_app(
        app_dir=layout.dory_app_dir,
        network_dir=args.integrate_into,
        prefix=settings.prefix,
        int_onnx=layout.nemo_onnx_path,
    )

    return CompileResult(
        layout=layout,
        dory_config=layout.dory_config_path,
        dory_app_dir=layout.dory_app_dir,
        integration=integration,
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-module", type=Path, required=True)
    parser.add_argument("--model-factory", default="create_model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-shape", required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Path to a .npy file holding the single calibration input tensor. "
        "Models that need no activation calibration (e.g. fixed-range ReLU6 nets) "
        "still require the argument; pass a placeholder array.",
    )
    parser.add_argument("--name")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("build"),
        help="Root for pipeline output; results land in <models-root>/dory/<name>/ "
        "(default: build/ in the current directory).",
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--nemo-dir", type=Path)
    parser.add_argument("--nemo-onnx", type=Path)
    parser.add_argument("--skip-onnx-validation", action="store_true")
    parser.add_argument("--dory-dir", type=Path)
    parser.add_argument("--dory-target", default=DEFAULT_DORY_TARGET)
    parser.add_argument("--dory-optional", default=DEFAULT_DORY_OPTIONAL)
    parser.add_argument("--dory-verbose-level", default=DEFAULT_DORY_VERBOSE)
    parser.add_argument(
        "--dory-prefix",
        default=None,
        help="Prefix DORY prepends to the network's generated files/symbols "
        "(network.c -> <prefix>_network.c), so the app can include "
        "<prefix>_network.h. Defaults to --name.",
    )
    parser.add_argument("--code-reserved-space", type=int, default=DEFAULT_CODE_RESERVED_SPACE)
    parser.add_argument(
        "--dory-l2-bytes",
        type=int,
        default=None,
        help="Temporarily set DORY's L2 tiling budget (memory.L2.dimension in the "
        "target HW_description.json) for codegen, restoring the file afterward. "
        "DORY has no CLI flag for this; the network must be tiled to fit beside "
        "the rest of the GAP8 app (hover_neural_network uses 228000).",
    )
    parser.add_argument("--bnrelu-bits", type=int, default=DEFAULT_BNRELU_BITS)
    parser.add_argument("--n-inputs", type=int, default=1)
    parser.add_argument(
        "--integrate-into",
        type=Path,
        required=True,
        help="The app's network directory: the generated network is synced into "
        "<DIR>/generated and verified, never patched. Build and flash the app "
        "with `cf-lab flash gap8` afterward. "
        "E.g. examples/hover_neural_network/gap8/policy.",
    )
    parser.add_argument("--skip-dory-codegen", action="store_true")
    parser.add_argument("--no-docker-image-build", action="store_true")


def run(args: argparse.Namespace) -> int:
    try:
        result = compile_model(args)
    except (SdkError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"NEMO ONNX: {result.layout.nemo_onnx_path}")
    print(f"DORY config: {result.dory_config}")
    print(f"DORY output: {result.dory_app_dir}")
    integ = result.integration
    print(
        f"Integrated into: {integ.generated_dir} "
        f"({integ.synced_src} src, {integ.synced_inc} inc, {integ.synced_hex} hex)"
    )
    print(f"Next: cf-lab flash gap8 --app-dir {integ.generated_dir.parent.parent}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cf-lab compile-model")
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
