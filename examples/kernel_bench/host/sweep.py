#!/usr/bin/env python3
"""Expand kernel_bench's benchmark matrix and optionally run it in GVSOC.

The matrix is compiled into one application so a sweep does not pay one GAP8
build per point. Registered implementations validate each case and emit an
explicit skip for unsupported layouts or memory/lifecycle modes.

A case is named from its fields,
[<function>-]<op+shape>[-off<N>][-m<A>m<B>][-<output>]-<memory>-<N>core[-once],
for example conv3x3s2-149x81c1to8-u8requant-l2_direct-8core: every field the
matrix varies is in the name, so two different cases never share one.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
REPO = BENCH.parent.parent
REPO_ROOT = REPO
DEFAULT_CONFIG = BENCH / "configs" / "matrix.json"
DEFAULT_OUT = BENCH / "artifacts" / "sweep"
# Compile jobs per container. The build parallelises and the simulation that
# follows does not, so this may exceed the core count across concurrent
# containers; the number of concurrent simulations is what must not.
DEFAULT_BUILD_JOBS = 4
# Mirrors gap8/Makefile's defaults. Recorded in the manifest so a
# microsecond figure can be tied to the clock that produced it -- DORY
# generates at 100 MHz and this bench defaults to 150.
DEFAULT_FC_FREQ_HZ = "150000000"
DEFAULT_CL_FREQ_HZ = "150000000"
# Case names go into the app's console records, which must stay below the GAP
# SDK's printf buffer. A longer name is an error rather than being shortened, so
# a name always spells every field of its case.
MAX_CASE_NAME = 72

FAMILY_ENUM = {
    "linear": "KBENCH_PROBLEM_LINEAR",
    "matmul": "KBENCH_PROBLEM_MATMUL",
    "conv": "KBENCH_PROBLEM_CONV2D",
    "dw": "KBENCH_PROBLEM_DEPTHWISE_CONV2D",
    "activation": "KBENCH_PROBLEM_ACTIVATION",
    "batchnorm": "KBENCH_PROBLEM_NORMALIZATION",
    "pool": "KBENCH_PROBLEM_POOLING",
    "add": "KBENCH_PROBLEM_ADDITION",
}
MEMORY_ENUM = {
    "l2_direct": "KBENCH_MEMORY_L2_DIRECT",
    "l1": "KBENCH_MEMORY_L2_L1",
}
# The C epilogue each output applies, spelled as gap8/src/main.c prints it.
EPILOGUE_ENUM = {
    "none": "KBENCH_EPILOGUE_INT32",
    "u8shift": "KBENCH_EPILOGUE_SHIFT_CLIP_U8",
    "u8requant": "KBENCH_EPILOGUE_REQUANT_RELU_U8",
    "u8bnfold": "KBENCH_EPILOGUE_BATCHNORM_RELU_U8",
    "add": "KBENCH_EPILOGUE_ADD_U8",
}
ACTIVATION_ENUM = {
    "none": "KBENCH_ACTIVATION_NONE",
    "sigmoid": "KBENCH_ACTIVATION_SIGMOID",
    "tanh": "KBENCH_ACTIVATION_TANH",
    "leaky_relu": "KBENCH_ACTIVATION_LEAKY_RELU",
    "hard_sigmoid": "KBENCH_ACTIVATION_HARD_SIGMOID",
    "hard_swish": "KBENCH_ACTIVATION_HARD_SWISH",
    "silu": "KBENCH_ACTIVATION_SILU",
    "gelu": "KBENCH_ACTIVATION_GELU",
    "softmax": "KBENCH_ACTIVATION_SOFTMAX",
}

# Families with at least one descriptor in gap8/src/registry.c.
SUPPORTED_FAMILIES = {
    "linear",
    "matmul",
    "conv",
    "dw",
    "activation",
    "batchnorm",
    "pool",
    "add",
}

PROBLEM_FIELDS = (
    "m",
    "n",
    "k",
    "height",
    "width",
    "input_channels",
    "output_channels",
    "kernel_h",
    "kernel_w",
    "stride_h",
    "stride_w",
    "padding_top",
    "padding_bottom",
    "padding_left",
    "padding_right",
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 2:
        raise ValueError(f"unsupported matrix schema in {path}")
    return data


def case_name(case: dict[str, Any], outputs: dict[str, Any]) -> str:
    """The case's name, from its fields; `outputs` is its family's outputs table.

    The output is named only for a family with more than one, the weight offset
    on every matmul case, the input multipliers on every add case, and `once`
    on a case timed with no warm-up and one run.
    """
    fields = [case["activation"]] if case["activation"] != "none" else []
    fields.append(case["shape"])
    if case["family"] == "matmul":
        fields.append(f"off{case['misalignment_offset_bytes']}")
    if case["family"] == "add":
        fields.append(f"m{case['input1_multiplier']}m{case['input2_multiplier']}")
    if len(outputs) > 1:
        fields.append(case["output"])
    fields += [case["memory"], f"{case['cores']}core"]
    if (case["warmups"], case["runs"]) == (0, 1):
        fields.append("once")
    return "-".join(str(field) for field in fields)


def expand_suite(config: dict[str, Any], suite_name: str) -> list[dict[str, Any]]:
    suites = config["suites"]
    selected = list(suites) if suite_name == "all" else [suite_name]
    missing = [name for name in selected if name not in suites]
    if missing:
        # Name the available suites so the caller need not open matrix.json.
        raise SystemExit(
            f"unknown suite(s): {', '.join(missing)}\n"
            f"available: {', '.join(sorted(suites))}"
        )

    expanded: list[dict[str, Any]] = []
    seen: set[str] = set()
    used_names: set[str] = set()
    for selected_name in selected:
        suite = suites[selected_name]
        axes = suite["axes"]
        keys = list(axes)
        for values in itertools.product(*(axes[key] for key in keys)):
            case = dict(config["defaults"])
            case.update(suite.get("overrides", {}))
            case.update(dict(zip(keys, values, strict=True)))

            shape_name = case["shape"]
            shape = config["shapes"].get(shape_name)
            if shape is None:
                raise KeyError(f"suite {selected_name}: unknown shape {shape_name}")
            case.update(shape)

            if case["memory"] not in MEMORY_ENUM:
                raise KeyError(f"suite {selected_name}: unknown memory {case['memory']}")
            # A suite that names no output gets its family's first.
            outputs = config["outputs"][case["family"]]
            case.setdefault("output", next(iter(outputs)))
            output = outputs.get(case["output"])
            if output is None:
                raise KeyError(
                    f"suite {selected_name}: {case['family']} has no output {case['output']}"
                )
            case.update(output)
            case["suite"] = selected_name

            # "all" combines suites and should not rerun an identical point.
            fingerprint = json.dumps(
                {
                    key: value
                    for key, value in case.items()
                    if key not in {"suite", "name"}
                },
                sort_keys=True,
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            name = case_name(case, outputs)
            if len(name) > MAX_CASE_NAME:
                raise ValueError(
                    f"suite {selected_name}: case name over {MAX_CASE_NAME} "
                    f"characters: {name}"
                )
            if name in used_names:
                raise ValueError(
                    f"suite {selected_name}: two different cases are named {name}"
                )
            used_names.add(name)
            case["name"] = name
            case["id"] = len(expanded)
            expanded.append(case)
    return expanded


def _c_string(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9_-]+", value):
        raise ValueError(f"case name is not C-string safe: {value!r}")
    return json.dumps(value)


def emit_header(cases: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "/* Generated by examples/kernel_bench/host/sweep.py. */",
        "#ifndef CF_KERNEL_BENCH_GENERATED_MATRIX_H",
        "#define CF_KERNEL_BENCH_GENERATED_MATRIX_H",
        "",
        '#include "kernel_bench/kernel.h"',
        "",
        "static const kbench_case_t kbench_cases[] = {",
    ]
    for case in cases:
        lines += [
            "    {",
            f"        .id = {case['id']}u,",
            f"        .name = {_c_string(case['name'])},",
            "        .problem = {",
            f"            .family = {FAMILY_ENUM[case['family']]},",
            f"            .batch = {int(case['batch'])}u,",
        ]
        for field in PROBLEM_FIELDS:
            value = int(case.get(field, 1 if field in {"kernel_h", "kernel_w", "stride_h", "stride_w"} else 0))
            lines.append(f"            .{field} = {value}u,")
        lines += [
            "        },",
            f"        .seed = {int(case['seed'])}u,",
            f"        .warmups = {int(case['warmups'])}u,",
            f"        .runs = {int(case['runs'])}u,",
            f"        .cores = {int(case['cores'])}u,",
            f"        .element_bits = {int(case['element_bits'])}u,",
            f"        .base_alignment_bytes = {int(case['base_alignment_bytes'])}u,",
            "        .misalignment_offset_bytes = "
            f"{int(case['misalignment_offset_bytes'])}u,",
            f"        .memory_path = {MEMORY_ENUM[case['memory']]},",
            f"        .activation = {ACTIVATION_ENUM[case['activation']]},",
            f"        .epilogue = {EPILOGUE_ENUM[case['epilogue']]},",
            f"        .input1_multiplier = {int(case['input1_multiplier'])}u,",
            f"        .input2_multiplier = {int(case['input2_multiplier'])}u,",
            f"        .output_multiplier = {int(case['output_multiplier'])}u,",
            f"        .output_shift = {int(case['output_shift'])},",
            "    },",
        ]
    lines += [
        "};",
        "",
        "#define KBENCH_CASE_COUNT (sizeof(kbench_cases) / sizeof(kbench_cases[0]))",
        "",
        "#endif /* CF_KERNEL_BENCH_GENERATED_MATRIX_H */",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _git(*args: str) -> str:
    """A git query that never raises: provenance must not break a run."""
    try:
        out = subprocess.run(("git", *args), cwd=REPO_ROOT, text=True,
                             capture_output=True, timeout=30, check=False)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _image_digest(image: str) -> str:
    """The repo digest of the image actually used.

    `latest` is mutable, so the tag alone does not identify what ran; the digest
    does.
    """
    try:
        out = subprocess.run(
            ("docker", "image", "inspect", image, "--format",
             "{{if .RepoDigests}}{{index .RepoDigests 0}}"
             "{{else}}{{.Id}}{{end}}"),
            text=True, capture_output=True, timeout=30, check=False)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def collect_provenance(image: str, make_vars: tuple[str, ...],
                       config_path: Path) -> dict[str, Any]:
    """Everything needed to reproduce a run, recorded beside its cases: the
    commit, the image digest, the make variables, the clocks and the matrix hash."""
    try:
        config_hash = hashlib.sha256(
            config_path.read_bytes()).hexdigest()[:16]
    except OSError:
        config_hash = ""
    dirty = _git("status", "--porcelain")
    return {
        "recorded_utc": datetime.datetime.now(
            datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(dirty),
            "dirty_paths": sorted(
                line.split(maxsplit=1)[-1]
                for line in dirty.splitlines() if line.strip())[:64],
            "submodules": _git("submodule", "status", "--recursive"),
        },
        "image": {"tag": image, "digest": _image_digest(image)},
        "build": {
            "make_vars": list(make_vars or ()),
            "fc_freq_hz": _make_var_value(make_vars, "KBENCH_FC_FREQ_HZ",
                                          DEFAULT_FC_FREQ_HZ),
            "cl_freq_hz": _make_var_value(make_vars, "KBENCH_CL_FREQ_HZ",
                                          DEFAULT_CL_FREQ_HZ),
            "dory_int32_bias": _make_var_value(make_vars, "KBENCH_DORY_INT32_BIAS", "0"),
        },
        "matrix": {"path": str(config_path), "sha256_16": config_hash},
    }


def _make_var_value(make_vars: tuple[str, ...], key: str, default: str) -> str:
    """Last assignment wins, matching make's own precedence."""
    value = default
    for var in make_vars or ():
        if var.startswith(f"{key}="):
            value = var.split("=", 1)[1]
    return value


def write_manifest(cases: list[dict[str, Any]], path: Path,
                   provenance: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"schema_version": 3, "cases": cases}
    if provenance is not None:
        payload["provenance"] = provenance
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_cases(cases: list[dict[str, Any]]) -> None:
    print(f"{'id':>3} {'suite':22} name")
    print("-" * 80)
    for case in cases:
        print(f"{case['id']:>3} {case['suite']:22} {case['name']}")
    print(f"\n{len(cases)} case(s)")


def assert_guest_passed(log_text: str, log_path: Path) -> None:
    """Refuse a log whose guest did not finish `status=pass`.

    `#DONE` is printed unconditionally on every exit path, including
    `cluster_open_failed`, and the container's exit code is deliberately
    ignored once we have killed it ourselves. So the sentinel alone says only
    that the guest reached its exit, not that anything ran or passed. The
    guest's own verdict is the one that distinguishes them.
    """
    match = re.search(r"^KBENCH\|DONE\|status=([a-z_]+)(?:\|results=(\d+))?",
                      log_text, re.MULTILINE)
    if match is None:
        raise RuntimeError(
            f"{log_path} has no KBENCH|DONE record: the guest exited without "
            f"reporting a verdict."
        )
    status, results = match.group(1), int(match.group(2) or 0)
    if status != "pass":
        raise RuntimeError(f"{log_path} finished status={status}")
    if results == 0:
        raise RuntimeError(
            f"{log_path} finished status=pass with zero results: every "
            f"descriptor declined the matrix."
        )


def assert_ran_expected_matrix(log_text: str, cases: list[dict[str, Any]],
                               log_path: Path) -> None:
    """Refuse a log whose first case is not the matrix we asked for.

    A run that compiles someone else's matrix still reports `status=pass`, so
    nothing downstream can tell the difference. Comparing the first emitted
    case name against the expanded suite catches it at the source.
    """
    if not cases:
        return
    match = re.search(r"^KBENCH\|CASE\|id=0\|name=([^|]*)\|", log_text, re.MULTILINE)
    if match is None:
        return
    observed, expected = match.group(1), cases[0]["name"]
    if observed != expected:
        raise RuntimeError(
            f"{log_path} benchmarked the wrong matrix: expected first case "
            f"{expected!r}, got {observed!r}. This usually means two concurrent "
            f"runs shared a generated header path."
        )


def run_gvsoc(image: str, timeout: int, out_dir: Path, matrix_header: Path,
              make_vars: tuple[str, ...] = (),
              cases: list[dict[str, Any]] | None = None,
              build_jobs: int = DEFAULT_BUILD_JOBS) -> Path:
    # The digest must cover the output directory, not just the bench: concurrent
    # suites run as threads of one process and would otherwise all try to claim
    # the same container name.
    digest = hashlib.sha1(str(out_dir.resolve()).encode()).hexdigest()[:10]
    container = f"kernel_bench_gvsoc_{os.getpid()}_{digest}"
    log_path = out_dir / "gvsoc.log"
    try:
        matrix_dir = matrix_header.resolve().parent.relative_to(REPO.resolve())
    except ValueError as error:
        raise ValueError("the GVSOC matrix header must be generated inside the repository") from error
    container_matrix_dir = f"/module/{matrix_dir}"
    try:
        build_dir = out_dir.resolve().relative_to(REPO.resolve()) / "sdk_build"
    except ValueError as error:
        raise ValueError("the GVSOC output directory must live inside the repository") from error
    # Each suite gets its own GAP SDK BUILDDIR so concurrent containers cannot
    # `make clean` each other's objects while sharing one bind mount.
    container_build_dir = f"/module/{build_dir}"
    # Build-time switches (KBENCH_DORY_INT32_BIAS) reach the Makefile from here,
    # so a suite runs with DORY's int32 bias without editing the Makefile.
    extra = "".join(f" {shlex.quote(var)}" for var in (make_vars or ()))
    make_vars_text = (
        f"platform=gvsoc io=host "
        f"KBENCH_MATRIX_DIR={shlex.quote(container_matrix_dir)} "
        f"BUILDDIR={shlex.quote(container_build_dir)}{extra}"
    )
    # Three make invocations, not `make clean all run`: goals on one command
    # line may run concurrently under -j, so `clean` could race `all` and `all`
    # race `run`. Only the compile is parallel. The toolchain and SDK versions
    # exist only inside the container, so they go into the log as KBENCH|META
    # records beside the numbers they produced.
    probe = (
        "printf 'KBENCH|META|toolchain=%s\\n' "
        "\"$(riscv32-unknown-elf-gcc -dumpversion 2>/dev/null || echo unknown)\"; "
        "printf 'KBENCH|META|gap_sdk=%s\\n' "
        "\"$(git -C /gap_sdk rev-parse --short HEAD 2>/dev/null || echo unknown)\""
    )
    inner = (
        f"{probe}; "
        f"make clean {make_vars_text} && "
        f"make -j{int(build_jobs)} all {make_vars_text} && "
        f"make run {make_vars_text}"
    )
    # stdbuf line-buffers GVSOC's stdout: block-buffered, a run killed by the
    # timeout would lose up to 4 KB from the end of its log.
    command = (
        "source /gap_sdk/configs/ai_deck.sh >/dev/null && "
        f"timeout {timeout} stdbuf -oL -eL "
        f"/bin/bash -c {shlex.quote(inner)} 2>&1 "
        "| sed -u '/^#DONE/q'"
    )
    docker = [
        "docker",
        "run",
        "--rm",
        "--name",
        container,
        "--platform",
        "linux/amd64",
        "-v",
        f"{REPO}:/module",
        "-w",
        "/module/examples/kernel_bench/gap8",
        image,
        "/bin/bash",
        "-lc",
        command,
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    finished_cleanly = False
    try:
        # Stream the output and kill the container at `#DONE`: GVSOC keeps
        # simulating after pmsis_exit and, printing nothing more, never takes
        # the SIGPIPE that would stop it, so it would run until `timeout`.
        proc = subprocess.Popen(
            docker, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            captured.append(line)
            sys.stdout.write(line)
            if line.startswith("#DONE"):
                finished_cleanly = True
                subprocess.run(["docker", "kill", container],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=False)
                break
        proc.stdout.close()
        returncode = proc.wait()
        text = "".join(captured)
        log_path.write_text(text, encoding="utf-8")
        assert_ran_expected_matrix(text, cases or [], log_path)
        assert_guest_passed(text, log_path)
        if not finished_cleanly and "#DONE" not in text:
            raise RuntimeError(f"GVSOC never reached #DONE; see {log_path}")
        # A non-zero status after our own `docker kill` is expected and means
        # nothing; only judge the status of a run we did not interrupt.
        if not finished_cleanly and returncode != 0:
            raise subprocess.CalledProcessError(returncode, docker)
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    return log_path


def resolve_suites(config: dict[str, Any], names: list[str]) -> list[str]:
    """Expand any group name into the suites it stands for.

    A suite's axes fit one family, so a group, a named list of suites one level
    deep, is how several families run together; `all` is every suite.
    """
    groups = config.get("groups", {})
    resolved: list[str] = []
    for name in names:
        if name == "all":
            for member in config.get("suites", {}):
                if member not in resolved:
                    resolved.append(member)
            continue
        members = groups.get(name)
        if members is None:
            resolved.append(name)
            continue
        for member in members:
            if member in groups:
                raise SystemExit(
                    f"group {name!r} contains group {member!r}: groups are "
                    f"one level deep"
                )
            if member not in config.get("suites", {}):
                raise SystemExit(
                    f"group {name!r} names unknown suite {member!r}"
                )
            if member not in resolved:
                resolved.append(member)
    return resolved


def combine_suites(
    config: dict[str, Any], names: list[str]
) -> list[dict[str, Any]]:
    """Expand resolved suites into one stable, deduplicated board matrix."""
    missing = [name for name in names if name not in config.get("suites", {})]
    if missing:
        raise SystemExit(f"unknown suite(s): {', '.join(missing)}")
    selected = dict(config)
    selected["suites"] = {name: config["suites"][name] for name in names}
    return expand_suite(selected, "all")


def run_one(config: dict[str, Any], name: str, args: argparse.Namespace) -> int:
    """Build and simulate one suite into its own output and BUILDDIR."""
    cases = expand_suite(config, name)
    # A suite that expands to nothing, or to a family with no registered
    # descriptor, would produce an all-skip log that reads as a pass.
    if not cases:
        raise RuntimeError(f"suite {name!r} expanded to zero cases")
    unsupported = {case["family"] for case in cases} - SUPPORTED_FAMILIES
    if unsupported:
        raise RuntimeError(
            f"suite {name!r}: no registered kernel implements: "
            + ", ".join(sorted(unsupported))
        )
    out_dir = args.out / name
    header = out_dir / "generated" / "kbench_matrix.h"
    emit_header(cases, header)
    manifest = out_dir / "manifest.json"
    write_manifest(cases, manifest,
                   collect_provenance(args.image, tuple(args.make_var),
                                      args.config))
    log = run_gvsoc(args.image, args.timeout, out_dir, header,
                    tuple(args.make_var), cases, args.build_jobs)
    print(f"[{name}] GVSOC log: {log}")
    return 0


def run_many(config: dict[str, Any], names: list[str],
             args: argparse.Namespace) -> int:
    """Suites are independent GAP SDK builds, so run them concurrently.

    Each gets its own --out subdirectory, generated header, and BUILDDIR;
    container names already carry the host PID. One gvsoc_launcher pins one
    core, so serial runs waste most of the machine.
    """
    if not args.gvsoc:
        for name in names:
            print_cases(expand_suite(config, name))
        return 0
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(args.jobs, len(names)))) as pool:
        futures = {pool.submit(run_one, config, name, args): name
                   for name in names}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - reported per suite
                failures.append(f"{name}: {error}")
    for failure in failures:
        print(f"FAILED {failure}", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite", default="linear_smoke",
                        help="suite or group name, comma-separated list, or all")
    parser.add_argument("--jobs", type=int, default=1,
                        help="suites to build and simulate concurrently")
    parser.add_argument("--list", action="store_true", help="print expanded cases")
    # Defaults to <out>/generated/kbench_matrix.h, so --out isolates a run: a
    # shared path would let concurrent runs benchmark each other's matrix.
    parser.add_argument("--emit-header", type=Path, default=None)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--gvsoc", action="store_true")
    parser.add_argument("--image", default=os.environ.get("AIDECK_IMAGE", "cf-lab/aideck-gap8:latest"))
    parser.add_argument("--timeout", type=int, default=2700)
    parser.add_argument(
        "--make-var",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra make variable, repeatable (e.g. KBENCH_DORY_INT32_BIAS=1)",
    )
    parser.add_argument("--build-jobs", type=int, default=DEFAULT_BUILD_JOBS,
                        help="compile jobs per container (build only; the "
                             "simulation after it is single-threaded)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    header = args.emit_header or (args.out / "generated" / "kbench_matrix.h")

    config = load_config(args.config)
    names = resolve_suites(config,
                           [n.strip() for n in args.suite.split(",") if n.strip()])
    # Every simulated run goes through run_one, which gives each suite its own
    # output directory, header and BUILDDIR under --out, so concurrent runs
    # never share them.
    if args.gvsoc and not args.emit_header:
        return run_many(config, names, args)
    cases = combine_suites(config, names)
    if not cases:
        raise SystemExit("matrix expanded to zero cases")
    families = {case["family"] for case in cases}
    unsupported = families - SUPPORTED_FAMILIES
    if unsupported:
        # A suite whose family has no registered descriptor produces an
        # all-skip log that reads like a pass. Refuse it instead.
        raise SystemExit(
            "no registered kernel implements: " + ", ".join(sorted(unsupported))
        )
    emit_header(cases, header)
    if args.manifest:
        # An AI-deck build's manifest carries the same provenance as a GVSOC run's.
        write_manifest(
            cases,
            args.manifest,
            collect_provenance(args.image, tuple(args.make_var), args.config),
        )
    if args.list or not args.gvsoc:
        print_cases(cases)
    if args.gvsoc:
        manifest = args.manifest or (args.out / "manifest.json")
        write_manifest(cases, manifest,
                       collect_provenance(args.image, tuple(args.make_var),
                                          args.config))
        log = run_gvsoc(args.image, args.timeout, args.out, header,
                        tuple(args.make_var), cases, args.build_jobs)
        print(f"GVSOC log: {log}")
        print(f"Matrix manifest: {manifest}")
        print(f"Report: .venv/bin/python {BENCH / 'host' / 'report.py'} {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
