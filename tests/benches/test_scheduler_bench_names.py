"""scheduler_bench's network and case names say what each case builds and runs
(examples/scheduler_bench/host)."""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
HOST = REPO / "examples" / "scheduler_bench" / "host"
APP = REPO / "examples" / "scheduler_bench" / "gap8" / "scheduler_bench.c"
# scheduler_bench's host and the nn_bench host modules sched_net imports have
# bare, common names (bench, graph, point, reference, ...) that other benches use too.
SIBLINGS = ("bench", "sched_net", "fused", "graph", "networks", "point", "reference", "run")
# The STM32 relays a GAP8 console line in one CPX packet: CPX_MAX_PAYLOAD_SIZE
# (100) less its 2-byte header, the terminating NUL included. A longer line
# arrives split, and bench.py reads the SB|open fields from one line.
CONSOLE_LINE_BYTES = 98
# camera_bench's sensor tokens and scheduler_bench's camera_stream SENSOR values.
SENSORS = {"qvga60": "QVGA_60", "qqvga120": "QQVGA_BIN2_120", "qqvga60": "QQVGA_BIN2_60"}
FORMATS = {"jpegcl": "JPEG", "raw": "RAW"}


def load(*names: str) -> list:
    """Import scheduler_bench host modules, leaving sys.path and sys.modules as they were."""
    path, saved = list(sys.path), {n: sys.modules.pop(n) for n in SIBLINGS if n in sys.modules}
    sys.path.insert(0, str(HOST))
    try:
        return [importlib.import_module(name) for name in names]
    finally:
        sys.path[:] = path
        for name in SIBLINGS:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


bench, sched_net = load("bench", "sched_net")
CONFIG = yaml.safe_load((HOST / "bench.yaml").read_text())
NETWORKS = CONFIG["networks"]
SIZES = sorted({bench.frame_size(camera) for camera in CONFIG["cameras"].values()})


def name_from_fields(name: str, entry: dict) -> str:
    """The network name its bench.yaml fields give; the output count from the built graph."""
    spec = sched_net.parse_spec(name, entry, SIZES[0])
    if spec.family == "mlp":
        layers = "to".join(str(d) for d in spec.dims)
    else:
        layers = "-".join(spec.units) + f"-out{sched_net.build(spec).layers[-1].values.size}"
    arena = f"-arena{spec.arena_kb}k" if spec.memory == "l3" else ""
    return f"{spec.family}-{layers}-{spec.memory}{arena}"


def test_network_keys_are_the_names_their_fields_give():
    assert {name: name_from_fields(name, entry) for name, entry in NETWORKS.items()} == \
        {name: name for name in NETWORKS}
    for name, entry in NETWORKS.items():
        assert ("arena_kb" in entry) == (entry["memory"] == "l3"), name  # the name shows every arena


@pytest.mark.parametrize("name", [name for name, entry in NETWORKS.items() if entry["family"] == "mlp"])
def test_an_mlp_builds_the_widths_its_name_lists(name):
    graph = sched_net.build(sched_net.parse_spec(name, NETWORKS[name]))
    widths = [graph.shape[0]] + [layer.values.size for layer in graph.layers]
    assert name.startswith(f"mlp-{'to'.join(str(w) for w in widths)}-")


@pytest.mark.parametrize("name", [name for name, entry in NETWORKS.items() if entry["family"] == "convpool"])
@pytest.mark.parametrize("size", SIZES)
def test_a_convpool_builds_its_units_on_the_camera_frame(name, size):
    spec = sched_net.parse_spec(name, NETWORKS[name], size)
    graph = sched_net.build(spec)
    assert graph.shape == (1, size[1], size[0])
    labels = [layer.label for layer in graph.layers]
    assert labels[-2:] == ["gap", "out"]
    height = size[1]
    for index, unit in enumerate(spec.units):
        channels, stride, pooled = re.fullmatch(r"c(\d+)(s2)?(mp)?", unit).groups()
        conv = graph.layers[labels.index(f"unit{index}_conv")]
        assert conv.weights.shape[-1] == 3 and conv.values.shape[0] == int(channels)
        assert (conv.values.shape[1] < height) == bool(stride)
        assert (f"unit{index}_pool" in labels) == bool(pooled)
        height = graph.layers[labels.index(f"unit{index}_pool" if pooled else f"unit{index}_conv")].values.shape[1]


def test_each_build_has_its_own_cache_directory():
    builds: dict[str, set] = {}
    for name, entry in NETWORKS.items():
        for size in SIZES:
            spec = sched_net.parse_spec(name, entry, size)
            builds.setdefault(spec.dir_name, set()).add((name, spec.image))
    assert all(len(found) == 1 for found in builds.values()), builds
    name = "convpool-c4s2mp-c8mp-c16-out4-l2"
    assert sched_net.parse_spec(name, NETWORKS[name], (160, 120)).dir_name == \
        "convpool-160x120c1-c4s2mp-c8mp-c16-out4-l2"


def test_camera_keys_name_the_sensor_and_format():
    for name, camera in CONFIG["cameras"].items():
        sensor, fmt = name.split("-")
        assert (SENSORS[sensor], FORMATS[fmt]) == (camera["SENSOR"], camera["FORMAT"]), name


def test_cases_compose_network_camera_and_rate():
    cases = bench.cases_of(CONFIG)
    for name, case in cases.items():
        rate = f"-{case.rate_hz}hz" if case.rate_hz else ""
        assert name == f"{case.network}-{case.camera}{rate}"
        assert bool(case.rate_hz) == (NETWORKS[case.network]["family"] == "mlp")
    assert len(cases) == len(CONFIG["cameras"]) * sum(
        len(CONFIG["rates"]) if entry["family"] == "mlp" else 1 for entry in NETWORKS.values())
    for set_name, names in CONFIG["sets"].items():
        assert set(names) <= set(cases), set_name


def test_every_open_line_fits_one_console_packet():
    """SB|open carries the network's name; bench.py parses its fields from one line."""
    fmt = re.search(r'"(SB\|open [^"]*)"', APP.read_text()).group(1).replace("\\n", "\n")
    assert fmt == "SB|open net=%s input=%d arena=%lu rate=%u\n"
    rate = max(len(str(rate)) for rate in CONFIG["rates"])
    for name in NETWORKS:
        # input is SCHED_INPUT_STATE or _IMAGE (one digit); an arena is under 512 KiB (6 digits)
        line = fmt.replace("%s", name).replace("%d", "0").replace("%lu", "9" * 6).replace("%u", "9" * rate)
        assert len(line.encode()) + 1 <= CONSOLE_LINE_BYTES, name
