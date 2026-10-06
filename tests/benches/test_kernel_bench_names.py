"""kernel_bench's case names say what each case runs, and its matrix, records, ledger and sets agree with them (examples/kernel_bench/host)."""

from __future__ import annotations

import importlib
import itertools
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "examples" / "kernel_bench"
HOST = BENCH / "host"
# kernel_bench's host modules import each other by bare names, some of which
# (bench, reference, results, run) other benches use too.
SIBLINGS = ("bench", "reference", "report", "results", "run", "sweep")


def load(*names: str) -> list:
    """Import kernel_bench host modules, leaving sys.path and sys.modules as they were."""
    path, saved = list(sys.path), {n: sys.modules.pop(n) for n in SIBLINGS if n in sys.modules}
    sys.path.insert(0, str(HOST))
    try:
        return [importlib.import_module(name) for name in names]
    finally:
        sys.path[:] = path
        for name in SIBLINGS:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


sweep, bench, report, results = load("sweep", "bench", "report", "results")
CONFIG = sweep.load_config()
BENCH_YAML = yaml.safe_load((HOST / "bench.yaml").read_text())

PAD = r"(?:-pad\d+|-padt\d+b\d+l\d+r\d+)?"
CASE = re.compile(
    r"(?:(?P<function>sigmoid|tanh|leaky_relu|hard_sigmoid|hard_swish|silu|gelu|softmax)-)?"
    r"(?P<shape>vec\d+|linear-\d+to\d+|matmul4x[234]-\d+to\d+"
    rf"|conv\d+x\d+(?:s\d+)?-\d+x\d+c\d+to\d+{PAD}|dw\d+x\d+(?:s\d+)?-\d+x\d+c\d+{PAD}"
    rf"|pool\d+x\d+(?:s\d+)?-\d+x\d+c\d+{PAD}|global_avgpool-\d+x\d+c\d+"
    r"|adaptive_avgpool-\d+x\d+c\d+-out\d+x\d+|batchnorm-\d+x\d+c\d+|add-\d+x\d+c\d+)"
    r"(?:-off(?P<offset>\d+))?(?:-m(?P<m1>\d+)m(?P<m2>\d+))?"
    r"(?:-(?P<output>i32|u8|u8shift|u8requant|u8bnfold))?"
    r"-(?P<memory>l1|l2_direct)-(?P<cores>\d+)core(?P<once>-once)?")


def _pad(shape: dict, default: int) -> str:
    sides = [shape.get(f"padding_{side}", 0) for side in ("top", "bottom", "left", "right")]
    if sides == [default] * 4:
        return ""
    return f"-pad{sides[0]}" if len(set(sides)) == 1 else "-padt{}b{}l{}r{}".format(*sides)


def _window(op: str, shape: dict) -> str:
    assert shape.get("stride_h", 1) == shape.get("stride_w", 1)
    stride = shape.get("stride_h", 1)
    return f"{op}{shape['kernel_h']}x{shape['kernel_w']}" + (f"s{stride}" if stride != 1 else "")


def shape_name(shape: dict) -> str:
    """The <op+shape> a matrix shape's fields give; its key must be this."""
    family = shape["family"]
    if family == "linear":
        return f"linear-{shape['k']}to{shape['n']}"
    if family == "matmul":
        return f"matmul4x{shape['m']}-{shape['k']}to{shape['n']}"
    if family == "activation":
        return f"vec{shape['n']}"
    tensor = f"{shape['width']}x{shape['height']}c{shape['input_channels']}"
    if family == "conv":
        return f"{_window('conv', shape)}-{tensor}to{shape['output_channels']}{_pad(shape, shape['kernel_h'] // 2)}"
    if family == "dw":
        return f"{_window('dw', shape)}-{tensor}{_pad(shape, shape['kernel_h'] // 2)}"
    if family == "pool":
        if shape["kernel_h"] == 0:
            return f"adaptive_avgpool-{tensor}-out{shape['n']}x{shape['m']}"
        if (shape["kernel_h"], shape["kernel_w"]) == (shape["height"], shape["width"]):
            return f"global_avgpool-{tensor}"
        return f"{_window('pool', shape)}-{tensor}{_pad(shape, 0)}"
    return f"{family}-{tensor}"  # batchnorm, add


def expansions() -> dict[str, list[dict]]:
    """Every matrix the bench builds: each suite alone (GVSOC), each group, all."""
    found = {suite: sweep.expand_suite(CONFIG, suite) for suite in CONFIG["suites"]}
    for name in [*CONFIG["groups"], "all"]:
        found[name] = sweep.combine_suites(CONFIG, sweep.resolve_suites(CONFIG, [name]))
    return found


EXPANSIONS = expansions()


def fingerprint(case: dict) -> str:
    return json.dumps({k: v for k, v in case.items() if k not in {"suite", "name", "id"}}, sort_keys=True)


def test_shape_keys_are_the_names_their_fields_give():
    assert {name: shape_name(shape) for name, shape in CONFIG["shapes"].items()} == \
        {name: name for name in CONFIG["shapes"]}


@pytest.mark.parametrize("matrix", sorted(EXPANSIONS))
def test_case_names_follow_the_grammar_and_are_unique(matrix):
    names = [case["name"] for case in EXPANSIONS[matrix]]
    assert len(set(names)) == len(names)
    for name in names:
        assert CASE.fullmatch(name), name
        assert len(name) <= sweep.MAX_CASE_NAME, name
        assert isinstance(report._value(name), str), name  # stays a string in reports
        sweep._c_string(name)


def test_a_name_means_one_case_in_every_matrix():
    seen: dict[str, str] = {}
    for case in itertools.chain.from_iterable(EXPANSIONS.values()):
        assert seen.setdefault(case["name"], fingerprint(case)) == fingerprint(case), case["name"]


def test_names_carry_every_field_the_matrix_varies():
    for case in EXPANSIONS["all"]:
        fields = CASE.fullmatch(case["name"]).groupdict()
        assert fields["shape"] == case["shape"]
        assert fields["function"] == (None if case["activation"] == "none" else case["activation"])
        assert fields["offset"] == (str(case["misalignment_offset_bytes"]) if case["family"] == "matmul" else None)
        multipliers = (str(case["input1_multiplier"]), str(case["input2_multiplier"]))
        assert (fields["m1"], fields["m2"]) == (multipliers if case["family"] == "add" else (None, None))
        several = len(CONFIG["outputs"][case["family"]]) > 1
        assert fields["output"] == (case["output"] if several else None)
        assert (fields["memory"], int(fields["cores"])) == (case["memory"], case["cores"])
        assert bool(fields["once"]) == ((case["warmups"], case["runs"]) == (0, 1))
        assert (case["warmups"], case["runs"]) in {(0, 1), (1, 3)}, case["name"]


def test_every_family_has_outputs_with_a_c_epilogue():
    assert set(CONFIG["outputs"]) == set(sweep.FAMILY_ENUM) == sweep.SUPPORTED_FAMILIES
    for family, outputs in CONFIG["outputs"].items():
        for token, output in outputs.items():
            assert output["epilogue"] in sweep.EPILOGUE_ENUM, (family, token)
    assert {shape["family"] for shape in CONFIG["shapes"].values()} == set(sweep.FAMILY_ENUM)


def _returned_strings(source: str, function: str) -> set[str]:
    body = re.search(rf"{function}\([^)]*\) \{{(.*?)\n\}}", source, re.S).group(1)
    return set(re.findall(r'return "([^"]+)";', body)) - {"invalid"}


def test_records_print_the_names_the_matrix_uses():
    main_c = (BENCH / "gap8" / "src" / "main.c").read_text()
    assert _returned_strings(main_c, "epilogue_name") == set(sweep.EPILOGUE_ENUM)
    assert _returned_strings(main_c, "memory_path_name") == set(sweep.MEMORY_ENUM)


def test_smoke_group_runs_every_family():
    assert {case["family"] for case in EXPANSIONS["smoke"]} == set(sweep.FAMILY_ENUM)


def test_tracked_header_is_the_linear_smoke_suite(tmp_path):
    sweep.emit_header(sweep.expand_suite(CONFIG, "linear_smoke"), tmp_path / "kbench_matrix.h")
    assert (tmp_path / "kbench_matrix.h").read_text() == \
        (BENCH / "gap8" / "generated" / "kbench_matrix.h").read_text()


def test_ledger_keys_name_cases_and_registered_descriptors():
    ledger = json.loads((BENCH / "configs" / "reference_results.json").read_text())["cases"]
    names = {case["name"] for case in EXPANSIONS["all"]}
    literals = set(itertools.chain.from_iterable(
        re.findall(r'"([a-z0-9_]+)"', path.read_text()) for path in (BENCH / "gap8" / "kernels").rglob("*.c")))
    assert set(ledger) <= names
    for case, entries in ledger.items():
        for key in entries:
            kernel, variant = key.split("/")
            assert {kernel, variant} <= literals, (case, key)


def test_sets_name_configurations_and_suites():
    for set_name, spec in BENCH_YAML["sets"].items():
        assert set(spec) <= set(BENCH_YAML["configurations"]), set_name
        for platforms in spec.values():
            assert set(platforms) <= {"gvsoc", "aideck"}
            for names in platforms.values():
                assert set(sweep.resolve_suites(CONFIG, names)) <= set(CONFIG["suites"])


def test_plan_names_jobs_and_folders_by_platform_configuration_and_suite():
    spec = BENCH_YAML["sets"]["exhaustive"]
    out = BENCH / "artifacts" / "bench_plan"  # never made
    jobs, rows = bench.plan(BENCH_YAML, spec, out, "both", keep_controller=True)
    expected = []
    for configuration, platforms in spec.items():
        expected += [f"gvsoc-{configuration}-{suite}"
                     for suite in sweep.resolve_suites(CONFIG, platforms["gvsoc"])]
        expected.append(f"aideck-{configuration}")
    assert [job.name for job in jobs] == expected
    for job in jobs:
        folder = out / "-".join(job.name.split("-")[:2])
        assert str(folder) in job.commands[0], job.name
        assert rows[job.name]["configuration"] == job.name.split("-")[1]


def test_results_lists_the_configurations_bench_yaml_gives():
    assert [(c.hardware_dir, c.gvsoc_dir) for c in results.configurations()] == \
        [(bench.folder("aideck", name), bench.folder("gvsoc", name)) for name in BENCH_YAML["configurations"]]
    labels = [c.label for c in results.configurations()]
    assert labels == ["s8bias", "i32bias (KBENCH_DORY_INT32_BIAS=1)"]


def test_results_labels_rows_by_case_name(tmp_path):
    cases = sweep.expand_suite(CONFIG, "linear_smoke")
    provenance = {"recorded_utc": "t", "git": {"commit": "c"}, "matrix": {"sha256_16": "m"},
                  "image": {"digest": "d"}, "build": {}}
    rows = [{"case": case["id"], "kernel": "linear_u8s8_i32", "variant": variant, "status": "pass",
             "exact": 1, "samples": 3, "kernel_cycle_samples": 3, "median_kernel_cycles": cycles}
            for case in cases for variant, cycles in (("pulp_nn", 200.0), ("register4", 100.0))]
    for folder in ("aideck-s8bias", "gvsoc-s8bias/linear_smoke"):
        (tmp_path / folder / "report").mkdir(parents=True)
        (tmp_path / folder / "manifest.json").write_text(json.dumps({"cases": cases, "provenance": provenance}))
        (tmp_path / folder / "report" / "summary.json").write_text(json.dumps({"results": rows}))

    text = results.render(tmp_path)

    assert "**s8bias — Linear**" in text
    assert "| linear-16to16-i32-l2_direct-8core · `linear_u8s8_i32/register4` | PULP-NN |" in text


def test_mac_kernels_count_every_matmul_block():
    for kernel in ("linear_u8s8_i32", "conv2d_u8s8_u8", "depthwise_conv2d_u8s8_u8",
                   "matmul4x2_u8s8_u8", "matmul4x3_u8s8_u8", "matmul4x4_u8s8_u8"):
        assert report._is_mac_kernel({"kernel": kernel}), kernel
    for kernel in ("maxpool2d_u8", "activation_lut8", "batchnorm_fixed_u8", "add_u8_u8_u8"):
        assert not report._is_mac_kernel({"kernel": kernel}), kernel
