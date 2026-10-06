"""jpeg_cluster_bench's case names say what each case builds and runs (examples/jpeg_cluster_bench/host)."""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
HOST = REPO / "examples" / "jpeg_cluster_bench" / "host"
ENCODER = REPO / "lib" / "gap8" / "jpeg" / "src" / "jpeg_cl.c"
# <W>x<H>[-no_<switch>], or soak: 320x240 for longer on the deck.
CASE = re.compile(r"(\d+)x(\d+)(?:-(no_fast_gather|no_early_eob|no_stage_on_cluster))?|soak")
SWITCH_OFF = re.compile(r"-DJPEG_CL_([A-Z_]+)=0")


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((HOST / "bench.yaml").read_text())


def name_from_fields(case: dict) -> str:
    if "seconds" in case:
        return "soak"
    switches = SWITCH_OFF.findall(case.get("defines", ""))
    return "-".join([case["window"], *(f"no_{switch.lower()}" for switch in switches)])


def test_case_keys_are_the_names_their_fields_give(config):
    assert {name: name_from_fields(case) for name, case in config["cases"].items()} == \
        {name: name for name in config["cases"]}


def test_case_names_follow_the_grammar(config):
    for name in config["cases"]:
        assert CASE.fullmatch(name), name
    assert "corpus" not in config["cases"]  # the host's frame-corpus row


def test_defines_only_turn_encoder_switches_off(config):
    """Any other define would build something the case name does not say."""
    encoder = ENCODER.read_text()
    for name, case in config["cases"].items():
        defines = case.get("defines", "")
        assert re.fullmatch(r"(?:-DJPEG_CL_[A-Z_]+=0(?: |$))*", defines), name
        for switch in SWITCH_OFF.findall(defines):
            assert f"#ifndef JPEG_CL_{switch}\n" in encoder, switch


def test_soak_is_built_like_320x240(config):
    cases = config["cases"]
    assert {**cases["soak"], "seconds": 60} == {**cases["320x240"], "seconds": 60}


def test_sets_name_cases(config):
    for name, platforms in config["sets"].items():
        assert set(platforms) <= {"gvsoc", "aideck"}, name
        for platform, cases in platforms.items():
            assert set(cases) <= set(config["cases"]), (name, platform)


def test_plan_names_jobs_rows_and_containers_by_case(config):
    spec = importlib.util.spec_from_file_location("jpeg_cluster_bench_host_bench", HOST / "bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    exhaustive = config["sets"]["exhaustive"]
    out = bench.BENCH_DIR / "artifacts" / "bench_plan"  # inside the repo, as the containers mount it; never made

    jobs, rows = bench.plan(exhaustive, config["cases"], out, "both", keep_controller=True)

    names = [f"gvsoc-{case}" for case in exhaustive["gvsoc"]] + [f"aideck-{case}" for case in exhaustive["aideck"]]
    assert [job.name for job in jobs] == names
    assert rows == {**{f"gvsoc-{case}": [{"platform": "gvsoc", "case": case}] for case in exhaustive["gvsoc"]},
                    **{f"aideck-{case}": [{"platform": "aideck", "case": case}] for case in exhaustive["aideck"]}}
    for job, case in zip(jobs, exhaustive["gvsoc"]):
        assert f"--name jpeg_cluster_bench_gvsoc_{os.getpid()}_{case} " in job.commands[0][-1]
    soak = jobs[names.index("aideck-soak")].commands[0]
    assert soak[soak.index("--window") + 1:] == ["320x240", "--defines=", "--seconds", "900"]
