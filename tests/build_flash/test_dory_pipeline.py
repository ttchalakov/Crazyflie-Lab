"""NEMO/DORY deployment: HyperRAM rightsizing, the L2 budget override, FC weight layout, integration."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from cf_lab.aideck import dory_pipeline
from cf_lab.aideck.dory_pipeline import (
    DoryCodegenConfig,
    DoryHyperramPlan,
    PipelineError,
    dory_l2_override,
    integrate_generated_app,
    rightsize_dory_hyperram,
    verify_fc_weight_orientation,
)

DATA = Path(__file__).parent / "data"
# Real DORY output for a 24x24 MobileNet on the L3 target (depthwise layers included),
# exactly as codegen left it, before rightsizing.
MOBILENET = DATA / "dory_mobilenet_l3"
# Real DORY output for the hover policy MLP on the L2-only target, prefix "policy",
# with the NEMO integer ONNX it was generated from.
POLICY = DATA / "dory_hover_policy_l2"
MOBILENET_PLAN = DoryHyperramPlan(weights_bytes=15168, input_bytes=18432, output_bytes=18432, total_bytes=52032)


@pytest.fixture
def mobilenet(tmp_path):
    return Path(shutil.copytree(MOBILENET, tmp_path / "gap8_app"))


@pytest.fixture
def policy(tmp_path):
    return Path(shutil.copytree(POLICY, tmp_path / "policy_app"))


def macros(source: str) -> dict[str, int]:
    return {name: int(value) for name, value in re.findall(r"^#define (L3_\w+_SIZE) (\d+)$", source, re.M)}


def test_rightsize_reserves_hex_blobs_plus_depthwise_overread(mobilenet):
    plan = rightsize_dory_hyperram(app_dir=mobilenet)

    # 11328 B of blobs, plus 3840 B: the largest depthwise L2 buffer past its blob
    # (4672 - 832); activations: the largest one DORY reports, 18432 B.
    assert plan == MOBILENET_PLAN
    source = (mobilenet / "src" / "network.c").read_text()
    assert macros(source) == {"L3_WEIGHTS_SIZE": 15168, "L3_INPUT_SIZE": 18432, "L3_OUTPUT_SIZE": 18432}
    assert "1500000" not in source
    assert "cl_ram_malloc(L3_INPUT_SIZE)" in source
    assert "L3_INPUT_SIZE - activations_out_size[i]" in source


def test_rightsize_uses_the_blob_size_for_a_layer_dory_tiles_out_of_l3(mobilenet):
    header = mobilenet / "inc" / "network.h"
    # A tiled layer reports its double-buffered L2 tile, smaller than its 6528 B blob.
    header.write_text(header.read_text().replace(
        "weights_size[11] = {1168, 640, 0, 2336, 2304, 0, 4672, 6528,",
        "weights_size[11] = {1168, 640, 0, 2336, 2304, 0, 4672, 1632,"))

    assert rightsize_dory_hyperram(app_dir=mobilenet).weights_bytes == MOBILENET_PLAN.weights_bytes


def test_rightsize_takes_a_larger_activation_from_the_graph(mobilenet):
    plan = rightsize_dory_hyperram(app_dir=mobilenet, largest_activation=100_000)

    assert (plan.input_bytes, plan.output_bytes) == (100_000, 100_000)
    assert plan.total_bytes == MOBILENET_PLAN.weights_bytes + 200_000


@pytest.mark.parametrize("prefix", ["net", "net_"])
def test_rightsize_finds_prefixed_network_files(mobilenet, prefix):
    for sub, name in (("src", "network.c"), ("inc", "network.h")):
        (mobilenet / sub / name).rename(mobilenet / sub / f"net_{name}")

    assert rightsize_dory_hyperram(app_dir=mobilenet, prefix=prefix) == MOBILENET_PLAN


def test_rightsize_leaves_an_l2_only_network_alone(policy):
    before = (policy / "src" / "policy_network.c").read_bytes()

    plan = rightsize_dory_hyperram(app_dir=policy, prefix="policy")

    assert plan.total_bytes == 0
    assert (policy / "src" / "policy_network.c").read_bytes() == before


@pytest.mark.parametrize("break_app, message", [
    (lambda app: (app / "hex" / "ReluConvolution6_weights.hex").unlink(), "is missing"),
    (lambda app: (app / "inc" / "network.h").write_text(
        (app / "inc" / "network.h").read_text().replace(', "FullyConnected10_weights.hex"', "")),
     "counts differ"),
    (lambda app: (app / "src" / "network.c").write_text(
        (app / "src" / "network.c").read_text().replace(
            "#define L3_OUTPUT_SIZE 1500000", "#define L3_OUTPUT_SIZE (1500000)")),
     "template drift"),
], ids=["missing-blob", "file-count", "template-drift"])
def test_rightsize_fails_closed_on_unexpected_output(mobilenet, break_app, message):
    break_app(mobilenet)
    before = (mobilenet / "src" / "network.c").read_bytes()

    with pytest.raises(PipelineError, match=message):
        rightsize_dory_hyperram(app_dir=mobilenet)

    assert (mobilenet / "src" / "network.c").read_bytes() == before


HW_DESCRIPTION = '{\r\n\t"name": "PULP/GAP8",\r\n\t"memory": {\r\n\t\t"L2": {"dimension": 512000}\r\n\t}\r\n}\r\n'


def test_l2_override_applies_during_codegen_and_restores_the_file_byte_for_byte(tmp_path):
    hw = dory_pipeline.dory_hw_description_path(tmp_path, "PULP.GAP8")
    hw.parent.mkdir(parents=True)
    hw.write_bytes(HW_DESCRIPTION.encode())

    with pytest.raises(RuntimeError, match="codegen failed"), \
            dory_l2_override(dory_dir=tmp_path, target="PULP.GAP8", l2_bytes=228000):
        assert json.loads(hw.read_text())["memory"]["L2"]["dimension"] == 228000
        raise RuntimeError("codegen failed")

    assert hw.read_bytes() == HW_DESCRIPTION.encode()


def test_l2_override_for_a_target_without_description_fails(tmp_path):
    with pytest.raises(PipelineError, match="not found"), \
            dory_l2_override(dory_dir=tmp_path, target="PULP.GAP9", l2_bytes=228000):
        pass


# -- FC weight orientation -------------------------------------------------------


def l2_array(path: Path, symbol: str) -> tuple[re.Match, list[int]]:
    text = path.read_text()
    match = re.search(rf"{symbol}\[\d+\] = \{{(.*?)\}};", text, re.S)
    return match, [int(float(tok)) for tok in match.group(1).split(",") if tok.strip()]


def test_real_policy_weights_are_verified_out_by_in(policy):
    checked = verify_fc_weight_orientation(
        generated_dir=policy, int_onnx=policy / "model_int.onnx", prefix="policy")

    assert checked == ["Weights_policy_ReluFullyConnected0", "Weights_policy_ReluFullyConnected1",
                       "Weights_policy_FullyConnected2"]


def test_transposed_square_layer_is_caught(policy):
    # The 64x64 layer: a shape cannot tell its axes apart, and a byte checksum cannot either.
    weights = policy / "inc" / "policy_weights.h"
    match, values = l2_array(weights, "Weights_policy_ReluFullyConnected1")
    block = np.array(values[:64 * 64]).reshape(64, 64).T.ravel().tolist()
    body = ", ".join(f"{v}." for v in block + values[64 * 64:])
    weights.write_text(weights.read_text()[:match.start(1)] + body + weights.read_text()[match.end(1):])

    with pytest.raises(PipelineError, match="transpose of"):
        verify_fc_weight_orientation(generated_dir=policy, int_onnx=policy / "model_int.onnx", prefix="policy")


def mlp_onnx(path: Path, layers: list[tuple[np.ndarray, int]]) -> None:
    """An integer-valued Gemm chain as NEMO exports it: weights stored as floats."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, inits, name = [], [], "x"
    for i, (weight, trans_b) in enumerate(layers):
        inits.append(numpy_helper.from_array(weight.astype(np.float32), f"w{i}"))
        nodes.append(helper.make_node("Gemm", [name, f"w{i}"], [f"y{i}"], transB=trans_b))
        name = f"y{i}"
    in_features = layers[0][0].shape[1 if layers[0][1] else 0]
    graph = helper.make_graph(nodes, "mlp", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, in_features])],
                              [helper.make_tensor_value_info(name, TensorProto.FLOAT, None)], inits)
    onnx.save(helper.make_model(graph), str(path))


@pytest.fixture
def l3_mlp(tmp_path):
    """A 12-16-16-4 MLP on the L3 target: one [in][out] Gemm, one square, one transB=1."""
    rng = np.random.default_rng(7)
    w = [rng.integers(-128, 128, size) for size in ((12, 16), (16, 16), (4, 16))]
    trans_b = [0, 1, 1]
    out_by_in = [w[0].T, w[1], w[2]]
    names = ["ReluFullyConnected0", "ReluFullyConnected1", "FullyConnected2"]
    (tmp_path / "inc").mkdir()
    (tmp_path / "hex").mkdir()
    (tmp_path / "inc" / "network.h").write_text(
        "static const char * L3_weights_files[] = {\n  "
        + ", ".join(f'"{n}_weights.hex"' for n in names) + "\n};\n")
    for name, weight in zip(names, out_by_in):
        bias = np.zeros(weight.shape[0] * 4, dtype=np.uint8)
        (tmp_path / "hex" / f"{name}_weights.hex").write_bytes(weight.astype(np.int8).tobytes() + bias.tobytes())
    mlp_onnx(tmp_path / "model_int.onnx", list(zip(w, trans_b)))
    return tmp_path


def test_l3_blobs_are_checked_in_header_order_for_both_gemm_layouts(l3_mlp):
    checked = verify_fc_weight_orientation(generated_dir=l3_mlp, int_onnx=l3_mlp / "model_int.onnx")

    assert checked == ["ReluFullyConnected0_weights.hex", "ReluFullyConnected1_weights.hex",
                       "FullyConnected2_weights.hex"]


def test_l3_blob_left_in_onnx_in_by_out_order_is_caught(l3_mlp):
    blob = l3_mlp / "hex" / "ReluFullyConnected0_weights.hex"
    data = np.frombuffer(blob.read_bytes(), dtype=np.int8)
    blob.write_bytes(data[:16 * 12].reshape(16, 12).T.tobytes() + data[16 * 12:].tobytes())

    with pytest.raises(PipelineError, match="ReluFullyConnected0_weights.hex is the transpose of w0"):
        verify_fc_weight_orientation(generated_dir=l3_mlp, int_onnx=l3_mlp / "model_int.onnx")


# -- integrating a generated network into an app ---------------------------------


def test_integration_syncs_the_network_and_drops_dorys_test_harness(policy, tmp_path):
    network_dir = tmp_path / "example" / "gap8" / "policy"
    stale = network_dir / "generated" / "src" / "policy_OldLayer.c"
    stale.parent.mkdir(parents=True)
    stale.write_text("// from the previous network\n")

    result = integrate_generated_app(app_dir=policy, network_dir=network_dir, prefix="policy",
                                     int_onnx=policy / "model_int.onnx")

    generated = network_dir / "generated"
    assert sorted(p.name for p in (generated / "src").iterdir()) == ["policy_network.c"]
    assert sorted(p.name for p in (generated / "inc").iterdir()) == sorted(
        p.name for p in (policy / "inc").iterdir())
    assert (generated / "src" / "policy_network.c").read_bytes() == (policy / "src" / "policy_network.c").read_bytes()
    assert (result.synced_src, result.synced_hex) == (2, 0)


def test_integration_rejects_a_cluster_args_array_too_small_for_its_writes(policy, tmp_path):
    source = policy / "src" / "policy_network.c"
    source.write_text(source.read_text().replace("unsigned int args[6];", "unsigned int args[5];"))

    with pytest.raises(PipelineError, match=r"args\[5\] is written"):
        integrate_generated_app(app_dir=policy, network_dir=tmp_path / "net", prefix="policy")


# -- compile-model end to end, with Docker faked ---------------------------------


@pytest.fixture
def checkout(cf_lab_root):
    (cf_lab_root / "external" / "nemo" / "nemo").mkdir(parents=True)
    (cf_lab_root / "external" / "nemo" / "nemo" / "__init__.py").write_text("")
    dory = cf_lab_root / "external" / "dory"
    (dory / dory_pipeline.PULP_NN_KERNELS).mkdir(parents=True)
    (dory / dory_pipeline.PULP_NN_KERNELS / "__init__.py").write_text("")
    hw = dory_pipeline.dory_hw_description_path(dory, "PULP.GAP8")
    hw.parent.mkdir(parents=True)
    hw.write_bytes(HW_DESCRIPTION.encode())
    return cf_lab_root


def assert_host_paths_mounted(command: list[str]) -> None:
    mounts = [Path(command[i + 1].split(":")[0]) for i, arg in enumerate(command) if arg == "-v"]
    paths = [Path(arg) for arg in command if arg.startswith("/") and ":" not in arg]
    for path in paths:
        assert any(path.is_relative_to(m) for m in mounts), f"{path} is not visible in the container"


@pytest.fixture
def docker(checkout, monkeypatch):
    """Runs the pipeline's containers as fakes that produce real NEMO and DORY output."""
    seen: dict = {"commands": [], "l2_during_codegen": None}
    hw = dory_pipeline.dory_hw_description_path(checkout / "external" / "dory", "PULP.GAP8")

    def run_command(command, *, cwd=None):
        seen["commands"].append(command)
        if "--model-module" in command:
            assert_host_paths_mounted(command)
            shutil.copyfile(POLICY / "model_int.onnx", command[command.index("--output") + 1])
        elif "network_generate.py" in command:
            assert_host_paths_mounted(command)
            seen["l2_during_codegen"] = json.loads(hw.read_text())["memory"]["L2"]["dimension"]
            app = Path(command[command.index("--app_dir") + 1])
            prefix = command[command.index("--prefix") + 1] + "_" if "--prefix" in command else ""
            shutil.copytree(MOBILENET, app)
            for sub, name in (("src", "network.c"), ("inc", "network.h")):
                (app / sub / name).rename(app / sub / f"{prefix}{name}")

    monkeypatch.setattr(dory_pipeline, "run_command", run_command)
    return seen


def compile_args(tmp_path, *extra: str) -> argparse.Namespace:
    for name in ("policy.py", "policy.pt", "calibration.npy"):
        (tmp_path / name).write_bytes(b"")
    parser = argparse.ArgumentParser()
    dory_pipeline.add_arguments(parser)
    return parser.parse_args([
        "--model-module", str(tmp_path / "policy.py"), "--checkpoint", str(tmp_path / "policy.pt"),
        "--calibration", str(tmp_path / "calibration.npy"), "--input-shape", "1x11",
        "--name", "policy", "--models-root", str(tmp_path / "build"),
        "--integrate-into", str(tmp_path / "app" / "gap8" / "policy"), *extra,
    ])


def test_compile_generates_the_network_into_the_app(docker, tmp_path, checkout):
    result = dory_pipeline.compile_model(compile_args(tmp_path, "--dory-l2-bytes", "228000"))

    assert result.layout.work_dir == tmp_path / "build" / "dory" / "policy"
    assert (tmp_path / "app" / "gap8" / "policy" / "generated" / "src" / "policy_network.c").is_file()
    assert docker["l2_during_codegen"] == 228000
    hw = dory_pipeline.dory_hw_description_path(checkout / "external" / "dory", "PULP.GAP8")
    assert hw.read_bytes() == HW_DESCRIPTION.encode()
    metadata = json.loads(result.layout.metadata_path.read_text())
    assert metadata["dory_hyperram"]["total_bytes"] == MOBILENET_PLAN.total_bytes
    assert metadata["dory_l2_arena_bytes"] == 228000 - DoryCodegenConfig().code_reserved_space
    assert macros((result.layout.dory_app_dir / "src" / "policy_network.c").read_text())["L3_WEIGHTS_SIZE"] == 15168


def test_compile_rejects_an_onnx_dory_cannot_take(docker, tmp_path, monkeypatch):
    def nemo_exports_softmax(command, *, cwd=None):
        if "--model-module" in command:
            import onnx
            from onnx import TensorProto, helper

            x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
            y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
            graph = helper.make_graph([helper.make_node("Softmax", ["x"], ["s"]),
                                       helper.make_node("Relu", ["s"], ["y"])], "g", [x], [y])
            onnx.save(helper.make_model(graph), command[command.index("--output") + 1])

    monkeypatch.setattr(dory_pipeline, "run_command", nemo_exports_softmax)

    with pytest.raises(PipelineError, match="Softmax"):
        dory_pipeline.compile_model(compile_args(tmp_path))


@pytest.mark.parametrize("text, shape", [("1x11", (1, 11)), ("1,3, 24,24", (1, 3, 24, 24))])
def test_parse_input_shape(text, shape):
    assert dory_pipeline.parse_input_shape(text) == shape


@pytest.mark.parametrize("text", ["", "1x0", "1x-3", "1xa"])
def test_parse_input_shape_rejects_bad_shapes(text):
    with pytest.raises(ValueError):
        dory_pipeline.parse_input_shape(text)


def test_dory_config_points_at_the_onnx_relative_to_itself(tmp_path):
    config = tmp_path / "work" / "dory" / "config.json"
    onnx_path = tmp_path / "work" / "nemo" / "model_int.onnx"

    dory_pipeline.write_dory_config(config_path=config, onnx_path=onnx_path, settings=DoryCodegenConfig())

    payload = json.loads(config.read_text())
    assert (config.parent / payload["onnx_file"]).resolve() == onnx_path.resolve()
    assert payload["code reserved space"] == DoryCodegenConfig().code_reserved_space


def test_compile_reserves_the_largest_activation_for_a_layer_kept_in_l3(docker, tmp_path, monkeypatch):
    # The graph's widest activation is 8x96x96 = 73728 B. DORY keeps layer 3's output
    # in HyperRAM, so network.h reports only its L2 tile, and the plan must come from
    # the graph.
    def onnx_with_a_wide_activation(path: str) -> None:
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 96, 96])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 96, 96])
        w = numpy_helper.from_array(np.ones((8, 1, 1, 1), dtype=np.float32), "w")
        graph = helper.make_graph([helper.make_node("Conv", ["x", "w"], ["c"]),
                                   helper.make_node("Relu", ["c"], ["y"])], "g", [x], [y], [w])
        onnx.save(helper.make_model(graph), path)

    codegen = dory_pipeline.run_command

    def run_command(command, *, cwd=None):
        if "--model-module" in command:
            onnx_with_a_wide_activation(command[command.index("--output") + 1])
            return
        codegen(command, cwd=cwd)
        if "network_generate.py" not in command:
            return
        header = Path(command[command.index("--app_dir") + 1]) / "inc" / "policy_network.h"
        text = header.read_text()
        if "L3_output_layers[11] = {0, 0, 0, 0," not in text:
            raise LookupError("the MobileNet sample no longer has the L3_output_layers row this test edits")
        header.write_text(text.replace("L3_output_layers[11] = {0, 0, 0, 0,", "L3_output_layers[11] = {0, 0, 0, 1,"))

    monkeypatch.setattr(dory_pipeline, "run_command", run_command)

    result = dory_pipeline.compile_model(compile_args(tmp_path))
    plan = json.loads(result.layout.metadata_path.read_text())["dory_hyperram"]
    assert plan["input_bytes"] >= 8 * 96 * 96
