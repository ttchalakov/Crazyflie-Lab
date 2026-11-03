"""Benchmark small MLPs on the STM32 in every cf_lab.nn format, and check the drone computes the host's bits.

    python examples/stm32_nn_bench/host/bench.py \
        --net policy=ckpt:path/to/model.pt:policy:net:silu --calibration policy=features.npy \
        --net wide=shape:32,256,256,4:silu

Each ``--net`` is a PyTorch checkpoint (``ckpt:PATH:KEY:PREFIX:ACT``: the state dict under ``KEY``, empty when the file
is the state dict, and the Linear layers under ``PREFIX``) or random weights (``shape:IN,H1,...,OUT:ACT``). Integer
formats are calibrated on ``--calibration`` rows (a .npy of shape (n, in)); without them, standard normal inputs.

The script writes controller/src/nnb_generated.{c,h}, builds the firmware, flashes it (the drone is never armed),
sets ``nnbench.run``, reads the console report and writes ``results.json`` into ``--out``. Every output hash the
drone reports is compared with the same C built for this computer (cf_lab.nn.host): ``match`` means identical bits.
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

from cf_lab.nn import (
    FORMATS,
    HostLibrary,
    Mlp,
    build_host_library,
    fnv1a,
    mlp_from_state_dict,
    quantize,
    write_c,
)
from cf_lab.nn.export import c_array_body
from cf_lab.nn.spec import Dense

APP = Path(__file__).resolve().parents[1] / "controller"
CPU_HZ = 168e6
RAM_BYTES = 16384  # SRAM copy buffer of the placement test
CCM_BYTES = 18432  # core-coupled RAM copy buffer
WORK_BYTES = 2048


def load_net(spec: str, seed: int) -> Mlp:
    kind, _, rest = spec.partition(":")
    if kind == "shape":
        dims, _, act = rest.partition(":")
        sizes = [int(n) for n in dims.split(",")]
        rng = np.random.default_rng(seed)
        layers = []
        for k, (n_in, n_out) in enumerate(itertools.pairwise(sizes)):
            w = rng.normal(scale=1.0 / np.sqrt(n_in), size=(n_out, n_in))
            b = rng.normal(scale=0.1, size=n_out)
            layers.append(Dense(w=w, b=b, act=(act or "silu") if k < len(sizes) - 2 else "none"))
        return Mlp(tuple(layers))
    if kind == "ckpt":
        path, key, prefix, act = (rest.split(":") + ["", "", ""])[:4]
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False)
        if key:
            state = state[key]
        return mlp_from_state_dict(state, prefix, act=act or "silu")
    raise SystemExit(f"--net {spec!r}: expected ckpt:... or shape:...")


def generate(nets: dict[str, Mlp], calib: dict[str, np.ndarray], formats: list[str], n_test: int):
    """Quantize, write nnb_generated.{c,h}, and return the model table the firmware will report on."""
    quantized, models, tests = {}, [], {}
    for name, mlp in nets.items():
        tests[name] = calib[name][:n_test].astype(np.float32)
        for fmt in formats:
            symbol = f"nnb_{name}_{fmt}"
            quantized[symbol] = quantize(mlp, calib[name], fmt)
            models.append({"net": name, "fmt": fmt, "symbol": symbol})
    places = []
    first = next(iter(nets))
    for index, model in enumerate(models):
        if model["net"] != first:
            continue
        q = quantized[model["symbol"]]
        fitting = [k for k, layer in enumerate(q.layers) if layer.w.nbytes <= max(RAM_BYTES, CCM_BYTES)]
        if fitting:
            places.append((index, max(fitting, key=lambda k: q.layers[k].w.nbytes)))
    load_sets = [(fmt, [i for i, m in enumerate(models) if m["fmt"] == fmt]) for fmt in ("s16s8", "f32x4")]
    load_sets = [(fmt, idx) for fmt, idx in load_sets if idx] or [(models[0]["fmt"], [0])]
    cmsis = next((i for i, m in enumerate(models) if m["fmt"] == "s8"), -1)
    lut16 = next((i for i, m in enumerate(models) if m["fmt"] == "s16s8"), -1)
    extra_h = (
        "typedef struct {\n  const char *net;\n  const char *fmt;\n  const cf_nn_net_t *model;\n"
        "  const float *test_x;\n  uint16_t n_test;\n} nnb_model_t;\n\n"
        "typedef struct {\n  uint8_t model;\n  uint8_t layer;\n} nnb_place_t;\n\n"
        "typedef struct {\n  const char *name;\n  uint8_t n;\n  uint8_t models[8];\n} nnb_load_set_t;\n\n"
        f"#define NNB_N_LOAD_SETS {len(load_sets)}\n"
        "extern const nnb_load_set_t nnb_load_sets[NNB_N_LOAD_SETS];\n"
        f"#define NNB_N_MODELS {len(models)}\n#define NNB_N_PLACES {len(places)}\n"
        f"#define NNB_MAX_OUT {max(max(m.n_out for m in nets.values()), 32)}\n"
        f"#define NNB_WORK_BYTES {WORK_BYTES}\n#define NNB_RAM_BYTES {RAM_BYTES}\n#define NNB_CCM_BYTES {CCM_BYTES}\n"
        f"#define NNB_CMSIS_MODEL {cmsis}\n#define NNB_CMSIS_LAYER {1 if cmsis >= 0 else 0}\n"
        f"#define NNB_LUT16_MODEL {lut16}\n\n"
        "extern const nnb_model_t nnb_models[NNB_N_MODELS];\nextern const nnb_place_t nnb_places[NNB_N_PLACES];\n"
    )
    extra_c = "".join(
        f"static const float nnb_x_{name}[{x.size}] = {{\n{c_array_body(x)}\n}};\n\n" for name, x in tests.items()
    )
    extra_c += "const nnb_model_t nnb_models[NNB_N_MODELS] = {\n" + "".join(
        f'  {{"{m["net"]}", "{m["fmt"]}", &{m["symbol"]}, nnb_x_{m["net"]}, {len(tests[m["net"]])}}},\n'
        for m in models
    ) + "};\n\n"
    extra_c += "const nnb_load_set_t nnb_load_sets[NNB_N_LOAD_SETS] = {\n" + "".join(
        f'  {{"{fmt}", {len(idx)}, {{{", ".join(map(str, idx))}}}}},\n' for fmt, idx in load_sets
    ) + "};\n\n"
    extra_c += "const nnb_place_t nnb_places[NNB_N_PLACES] = {\n" + "".join(
        f"  {{{i}, {layer}}},\n" for i, layer in places
    ) + "};\n"
    write_c(quantized, APP / "src" / "nnb_generated.c", APP / "src" / "nnb_generated.h",
            extra_c=extra_c, extra_h=extra_h)
    return quantized, models, tests


def host_reference(quantized, models, tests, nets, build_dir: Path) -> dict:
    """Expected output hashes (the drone's arithmetic on this computer) and errors against float64."""
    lib = HostLibrary(build_host_library([APP / "src" / "nnb_generated.c"], include_dirs=[APP / "src"],
                                         build_dir=build_dir))
    expected = {}
    for m in models:
        net = lib.net(m["symbol"])
        y = net(tests[m["net"]])
        h = 0x811C9DC5
        for row in y:
            h = fnv1a(row.astype("<f4").tobytes(), h)
        ref = nets[m["net"]].forward(tests[m["net"]].astype(np.float64))
        err = np.abs(y - ref)
        q = quantized[m["symbol"]]
        expected[(m["net"], m["fmt"])] = {
            "hash": f"{h:08x}", "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
            "weight_bytes": q.nbytes, "macs": q.macs,
        }
    return expected


def build_and_flash(uri: str, flash: bool, log_path: Path) -> dict:
    from cf_lab.firmware import build_controller

    with open(log_path, "w") as log:
        build_controller(APP, output=log)
    memory = {}
    for line in log_path.read_text().splitlines():
        if m := re.match(r"^(Flash|RAM|CCM)\s*\|\s*(\d+)/(\d+)", line):
            memory[m.group(1).lower()] = {"used": int(m.group(2)), "total": int(m.group(3))}
    if flash:
        from cf_lab.firmware import controller_artifact
        from cf_lab.stm32_flash import flash_stm32

        flash_stm32(artifact=controller_artifact(APP), uri=uri)
    return memory


def radio_traffic(cf, stop: threading.Event) -> None:
    """Flight-like radio load: a 100 Hz external position uplink and a 100 Hz log block of six floats."""
    from cflib.crazyflie.log import LogConfig

    block = LogConfig(name="nnb_traffic", period_in_ms=10)
    for name in ("stateEstimate.x", "stateEstimate.y", "stateEstimate.z", "gyro.x", "gyro.y", "gyro.z"):
        block.add_variable(name, "float")
    cf.log.add_config(block)
    block.start()
    t = 0.0
    while not stop.wait(0.01):
        t += 0.01
        cf.extpos.send_extpos(0.1 * np.cos(t), 0.1 * np.sin(t), 1.0)
    block.stop()


def run_on_drone(uri: str, timeout_s: float, log, trigger: str = "nnbench.run", last: str = "done",
                 traffic: bool = False) -> list[str]:
    from cf_lab.backends.crazyflie import CrazyflieRadioBackend

    lines: list[str] = []
    done = threading.Event()

    def sink(line: str) -> None:
        log(line)
        # The report lines are tagged NNB; a firmware built to prefix the module name adds "NNB: " too.
        if m := re.match(r"^(?:NNB: )?NNB (.*)$", line.strip()):
            lines.append(m.group(1))
            if lines[-1] == last:
                done.set()

    # Only a parameter write: nothing here sends an arming request or a setpoint.
    with CrazyflieRadioBackend(uri, console_sink=sink) as backend:
        time.sleep(1.0)
        stop = threading.Event()
        pump = threading.Thread(target=radio_traffic, args=(backend.crazyflie, stop), daemon=True)
        if traffic:
            pump.start()
        try:
            backend.crazyflie.param.set_value(trigger, "1")
            if not done.wait(timeout_s):
                raise TimeoutError(f"the drone did not finish the benchmark within {timeout_s:.0f}s")
        finally:
            stop.set()
            if traffic:
                pump.join(timeout=2.0)
    return lines


def parse(lines: list[str]) -> list[dict]:
    records = []
    for line in lines:
        kind, *fields = line.split()
        record = {"kind": kind}
        for field in fields:
            key, _, value = field.partition("=")
            record[key] = int(value) if key != "h" and re.fullmatch(r"-?\d+", value) else value
        records.append(record)
    return records


def summarize(records: list[dict], expected: dict) -> list[str]:
    def us(cycles: float) -> float:
        return cycles / CPU_HZ * 1e6

    rows = []
    hashes = {(r["net"], r["fmt"]): r["h"] for r in records if r["kind"] == "hash"}
    runs = {}
    for r in records:
        if r["kind"] == "run":
            runs.setdefault((r["net"], r["fmt"]), {})[r["mode"]] = r
    rows.append(f"{'net':10s} {'format':8s} {'bits':6s} {'KB':>6s} {'max err':>9s} "
                f"{'alone us':>9s} {'task med':>9s} {'task p99':>9s} {'task max':>9s}")
    for key, exp in expected.items():
        drone = hashes.get(key)
        match = "match" if drone and drone.lower() == exp["hash"] else ("MISS" if drone else "-")
        run = runs.get(key, {})
        alone = run.get("suspended", {}).get("min")
        task = run.get("task", {})
        cells = [f"{us(alone):9.1f}" if alone else f"{'-':>9s}"]
        cells += [f"{us(task[k]):9.1f}" if k in task else f"{'-':>9s}" for k in ("med", "p99", "max")]
        rows.append(f"{key[0]:10s} {key[1]:8s} {match:6s} {exp['weight_bytes'] / 1024:6.1f} "
                    f"{exp['max_abs_err']:9.2e} " + " ".join(cells))
    for r in records:
        if r["kind"] == "place":
            rows.append(f"place {r['net']} {r['fmt']} layer {r['l']} ({r['bytes']} B) in {r['mem']}: "
                        f"{us(r['cyc']):.1f} us")
        elif r["kind"] == "cmsis":
            rows.append(f"CMSIS-NN arm_fully_connected_s8 {r['in']}x{r['out']}: {us(r['cyc']):.1f} us")
        elif r["kind"] == "act":
            rows.append(f"activation {r['type']} x{r['n']}: {us(r['cyc']):.1f} us")
        elif r["kind"] == "load":
            rows.append(f"load {r['set']:6s} prio {r['prio']} {r['rate']:3d} Hz: exec med {r['exec_med']} "
                        f"p99 {r['exec_p99']} max {r['exec_max']} us, start jitter {r['late']} us, missed {r['miss']} | "
                        f"stabilizer period {r['tick_min']}-{r['tick_max']} us, late>10% {r['late10']}, "
                        f"controller mean {r['ctrl_mean']} max {r['ctrl_max']} us, IMU-to-output mean "
                        f"{r['lat_mean']} p99 {r['lat_p99']} max {r['lat_max']} us")
        elif r["kind"] == "clock":
            rows.append(f"cycle counter: {r['cycles_per_100ms'] / 0.1 / 1e6:.2f} MHz")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--net", action="append", required=True, metavar="NAME=SPEC")
    parser.add_argument("--calibration", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--formats", default=",".join(FORMATS))
    parser.add_argument("--n-test", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/drone.yaml"))
    parser.add_argument("--uri", default=None)
    parser.add_argument("--no-flash", action="store_true", help="build, but run on the firmware already flashed")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--load", action="store_true", help="run the load test (nnbench.load) instead of the bench")
    parser.add_argument("--traffic", action="store_true", help="flight-like radio traffic during the run")
    args = parser.parse_args()

    from cf_lab.config import radio_uri

    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = args.out or Path(__file__).resolve().parents[1] / "artifacts" / stamp
    out.mkdir(parents=True, exist_ok=True)
    nets = dict(item.split("=", 1) for item in args.net)
    nets = {name: load_net(spec, args.seed) for name, spec in nets.items()}
    calib = {}
    for item in args.calibration:
        name, path = item.split("=", 1)
        calib[name] = np.load(path)
    rng = np.random.default_rng(args.seed)
    for name, mlp in nets.items():
        if name not in calib:
            print(f"{name}: no calibration rows, using standard normal inputs", file=sys.stderr)
            calib[name] = rng.normal(size=(4096, mlp.n_in))
    formats = args.formats.split(",")

    quantized, models, tests = generate(nets, calib, formats, args.n_test)
    expected = host_reference(quantized, models, tests, nets, out / "host_build")
    print(f"generated {len(models)} models; host reference built")
    if args.generate_only:
        return 0
    uri = radio_uri(args.config, args.uri)
    memory = build_and_flash(uri, flash=not args.no_flash, log_path=out / "build.log")
    print(f"firmware memory: {memory}")
    trigger, last = ("nnbench.load", "load-done") if args.load else ("nnbench.run", "done")
    with open(out / "console.log", "w", buffering=1) as console:
        lines = run_on_drone(uri, args.timeout, lambda line: console.write(line + "\n"), trigger, last,
                             args.traffic)
    records = parse(lines)
    summary = summarize(records, expected)
    print("\n".join(summary))
    (out / "results.json").write_text(json.dumps({
        "memory": memory,
        "expected": {f"{k[0]}/{k[1]}": v for k, v in expected.items()},
        "records": records,
        "summary": summary,
    }, indent=1))
    print(f"results in {out}")
    misses = [k for k, v in expected.items()
              if any(r["kind"] == "hash" and (r["net"], r["fmt"]) == k and r["h"].lower() != v["hash"]
                     for r in records)]
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())
