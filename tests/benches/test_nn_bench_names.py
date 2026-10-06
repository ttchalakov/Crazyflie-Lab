"""nn_bench's network names say what each network is, and build from their own numbers
(examples/nn_bench/host)."""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
HOST = REPO / "examples" / "nn_bench" / "host"
KERNELS_MK = REPO / "lib" / "gap8" / "dory" / "kernels.mk"
# nn_bench's host modules import each other by bare, common names (graph, point,
# reference, bench, ...), which other benches use too.
SIBLINGS = ("bench", "fused", "graph", "networks", "point", "reference", "run")


def load(*names: str) -> list:
    """Import nn_bench host modules, leaving sys.path and sys.modules as they were."""
    path, saved = list(sys.path), {n: sys.modules.pop(n) for n in SIBLINGS if n in sys.modules}
    sys.path.insert(0, str(HOST))
    try:
        return [importlib.import_module(name) for name in names]
    finally:
        sys.path[:] = path
        for name in SIBLINGS:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


networks, points, bench = load("networks", "point", "bench")
SETS = yaml.safe_load((HOST / "bench.yaml").read_text())["sets"]
NETWORKS = sorted({row["network"] for spec in SETS.values() for row in bench.expand(spec, "both")})

INPUT = r"(\d+)x(\d+)c(\d+)"
GRAMMAR = {
    "mlp": re.compile(r"mlp-(\d+(?:to\d+)+)"),
    "mobilenetv1": re.compile(rf"mobilenetv1-{INPUT}-a(\d{{3}})-c(\d+)to(\d+)-out(\d+)"),
    "resnet18": re.compile(rf"resnet18-{INPUT}-c(\d+)to(\d+)-out(\d+)"),
    "custom": re.compile(rf"(convpool|dwpw|resblock)-{INPUT}((?:-c\d+(?:s2)?(?:mp)?)+)-out(\d+)"),
}


def name_of(family: str, graph) -> str:
    """The name a graph's structure gives, family by family."""
    layers = graph.layers
    if family == "mlp":
        return "mlp-" + "to".join(str(n) for n in [graph.shape[0]] + [l.values.size for l in layers])
    c, h, w = graph.shape
    head, outputs = f"{family}-{w}x{h}c{c}", f"out{layers[-1].values.size}"
    if family == "mobilenetv1":
        first, last = layers[0].values.shape[0], layers[-3].values.shape[0]
        return f"{head}-a{round(100 * last / 1024):03d}-c{first}to{last}-{outputs}"
    if family == "resnet18":
        return f"{head}-c{layers[0].values.shape[0]}to{layers[-3].values.shape[0]}-{outputs}"
    # A unit opens at a convolution (convpool), a depthwise (dwpw) or the first
    # 3x3 after an add or the stem (resblock), and takes the channels its last layer
    # gives; s2 when its spatial size halves before a pool, mp when it pools.
    units, size, opens = [], h, {"convpool": ("conv",), "dwpw": ("dw",), "resblock": ("conv",)}[family]
    body = layers[1:-2] if family == "resblock" else layers[:-2]  # resblock's stem is implied
    for prev, layer in zip([None] + body[:-1], body):
        starts = layer.op in opens and (family != "resblock" or prev is None or prev.op in ("add", "pool"))
        if family == "resblock" and layer.op == "conv" and layer.weights.shape[-1] == 1:
            starts = True  # a projection opens its unit
        if starts:
            units.append({"c": 0, "s2": False, "mp": False})
        unit, height = units[-1], layer.values.shape[1]
        if layer.op == "pool":
            unit["mp"] = True
        else:
            unit["c"] = layer.values.shape[0]
            unit["s2"] |= height < size and not unit["mp"]
        size = height
    if family == "resblock":
        assert layers[0].values.shape[0] == units[0]["c"], "the implied stem gives the first unit's channels"
    text = "-".join(f"c{u['c']}{'s2' if u['s2'] else ''}{'mp' if u['mp'] else ''}" for u in units)
    return f"{head}-{text}-{outputs}"


def family_of(name: str) -> str:
    for family, pattern in GRAMMAR.items():
        if m := pattern.fullmatch(name):
            return m.group(1) if family == "custom" else family
    raise AssertionError(f"{name} does not follow the grammar")


def test_sets_name_networks_in_the_grammar():
    assert NETWORKS
    for name in NETWORKS:
        family_of(name)
    assert set(networks.FIXED) <= set(NETWORKS)


@pytest.mark.parametrize("name", NETWORKS)
def test_a_network_builds_what_its_name_says(name):
    graph = networks.build(name)
    assert name_of(family_of(name), graph) == name
    labels = [layer.label for layer in graph.layers]
    assert len(set(labels)) == len(labels), "labels name the ONNX initializers, so they are unique"
    assert not [label for label in labels if "weight" in label or "bias" in label]


def test_mobilenetv1_alpha_gives_the_channel_range():
    for name in NETWORKS:
        if m := GRAMMAR["mobilenetv1"].fullmatch(name):
            alpha = int(m.group(4)) / 100
            assert (int(m.group(5)), int(m.group(6))) == (max(8, int(32 * alpha)), int(1024 * alpha)), name


def test_a_camera_network_takes_each_input_after_its_family():
    assert networks.names("resnet18-c8to64-out1000", ["160x120c1"]) == ["resnet18-160x120c1-c8to64-out1000"]
    assert networks.names(["mobilenetv1-a050-c16to512-out1000"]) == [
        f"mobilenetv1-{size}-a050-c16to512-out1000" for size in networks.INPUTS]
    assert networks.names("mlp_needs_l3", ["160x120c1"]) == networks.GROUPS["mlp_needs_l3"]


def test_group_names_never_look_like_network_names():
    for group, members in networks.GROUPS.items():
        assert re.fullmatch(r"[a-z0-9_]+", group), group
        assert len(set(members)) == len(members), group
        assert all("-" in member for member in members), group


def test_jobs_points_and_containers_are_named_by_network_memory_and_runner():
    out = REPO / "examples" / "nn_bench" / "artifacts" / "bench_plan"  # never made
    rows = bench.expand(SETS["exhaustive"], "both")
    jobs, settles = bench.plan(rows, out, keep_controller=True)
    for job in jobs:
        for row in settles[job.name]:
            point = f"{row['network']}-{row['memory']}"
            assert bench.row_dir(out, row).name == point
            expected = f"gvsoc-{point}" if row["platform"] == "gvsoc" else f"aideck-{point}-{row['runner']}"
            assert job.name == expected
            container = f"nnbench_1234567_{point}_{row['runner']}"  # as point.simulate() names it
            assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", container), container
    assert {row["memory"] for row in rows} == set(points.MEMORY)
    assert {row["runner"] for row in rows} == set(points.RUNNERS)


def test_dory_runners_are_the_kernel_variants_kernels_mk_takes():
    """gap8/Makefile passes NNBENCH_RUNNER straight through as DORY_KERNEL_VARIANT."""
    text = KERNELS_MK.read_text()
    variants = set(re.findall(r"\(DORY_KERNEL_VARIANT\),(\w+)\)", text))
    assert re.search(r"^DORY_KERNEL_VARIANT \?= pulp_nn$", text, re.M)
    assert variants == {runner for runner in points.RUNNERS if runner != "fused"}
