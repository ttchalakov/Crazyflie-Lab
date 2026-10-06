#!/usr/bin/env python3
"""kernel_bench on the AI-deck: build a suite's matrix, flash it, capture its report.

  python examples/kernel_bench/host/run.py --suite linear_smoke
  python examples/kernel_bench/host/run.py --suite all --out examples/kernel_bench/artifacts/aideck
  python examples/kernel_bench/host/run.py --suite conv3x3_outputs,dw3x3_outputs,conv1x1_outputs \\
      --make-var KBENCH_DORY_INT32_BIAS=1 --out examples/kernel_bench/artifacts/aideck-i32bias
  python examples/kernel_bench/host/run.py --out <dir> --listen      # capture only

The suites (names, groups, or `all`, as for sweep.py) are expanded into one
matrix header and manifest in --out, the bench is built for the deck against that
header, flashed over the radio, and the drone power-cycled; the radio console is
recorded from power-up until the guest prints #DONE. Emitting the header here is
what keeps a deck build from silently running whatever matrix was generated
last.

--out gets generated/kbench_matrix.h, manifest.json, build.log, the flashed image,
aideck.log (the console) and report/ (report.py's summary). The exit status is 0
only for a complete passing run: KBENCH|DONE|status=pass with results, #DONE, the
matrix that was asked for, and no console overflow (<F>).

The radio URI comes from configs/drone.yaml.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

HOST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HOST_DIR))

import report  # noqa: E402
import sweep  # noqa: E402
from cf_lab.bench import RadioConsole, flash_and_boot  # noqa: E402
from cf_lab.config import load_config  # noqa: E402

APP_DIR = HOST_DIR.parent / "gap8"
ROOT = HOST_DIR.parents[2]
DEFAULT_OUT = HOST_DIR.parent / "artifacts" / "aideck"
GAP8_IMAGE = os.environ.get("AIDECK_IMAGE", "cf-lab/aideck-gap8:latest")


def note(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build(header_dir: Path, make_vars: list[str], build_log: Path) -> Path:
    """Build the bench for the deck against this matrix header; return the image."""
    from cf_lab.firmware import build_gap8, gap8_artifact

    make = (f"clean build image KBENCH_TARGET=aideck KBENCH_MATRIX_DIR={header_dir} "
            + " ".join(make_vars))
    with build_log.open("w") as out:
        build_gap8(APP_DIR, make, build_image=False, output=out)
    return gap8_artifact(APP_DIR)


def capture(uri: str, path: Path, seconds: float) -> str:
    """Record the radio console until the guest prints #DONE."""
    console = RadioConsole(uri, path)
    started = time.monotonic()
    next_note = started + 30.0
    try:
        while time.monotonic() - started < seconds:
            time.sleep(1.0)
            text = path.read_text(encoding="utf-8", errors="replace")
            if "#DONE" in text:
                time.sleep(2.0)  # KBENCH|STACK follows #DONE
                break
            if time.monotonic() >= next_note:
                next_note += 30.0
                note(f"capturing: {time.monotonic() - started:.0f} s, {len(text)} characters")
    finally:
        console.close()
    return path.read_text(encoding="utf-8", errors="replace")


def records(raw: str) -> str:
    """The guest's own lines, as GVSOC would print them."""
    # The STM32 prefixes every CPX console packet, so a long record can carry
    # the prefix mid-line; the console log adds a timestamp in front.
    text = raw.replace("CPX: GAP8: ", "").replace("CPX: GAP8:", "")
    lines = []
    for line in text.splitlines():
        if "KBENCH|" in line:
            lines.append(line[line.find("KBENCH|"):])
        elif "#DONE" in line:
            lines.append("#DONE")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="linear_smoke",
                        help="suite name, group, comma-separated list, or all")
    parser.add_argument("--make-var", action="append", default=[], metavar="KEY=VALUE",
                        help="extra make variable, repeatable (e.g. KBENCH_DORY_INT32_BIAS=1)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seconds", type=float, default=900.0,
                        help="longest to wait for #DONE after power-up")
    parser.add_argument("--listen", action="store_true", help="capture only: no build, no flash")
    parser.add_argument("--image", type=Path,
                        help="flash this image instead of building (its matrix must match --suite)")
    parser.add_argument("--matrix", type=Path, default=sweep.DEFAULT_CONFIG)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "drone.yaml",
                        help="the drone config; only radio.uri is used")
    args = parser.parse_args(argv)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    uri = load_config(args.config).radio.uri

    config = sweep.load_config(args.matrix)
    names = sweep.resolve_suites(config, [n.strip() for n in args.suite.split(",") if n.strip()])
    cases = sweep.combine_suites(config, names)
    if not cases:
        parser.error("matrix expanded to zero cases")

    if not args.listen:
        header_dir = out / "generated"
        sweep.emit_header(cases, header_dir / "kbench_matrix.h")
        sweep.write_manifest(cases, out / "manifest.json",
                             sweep.collect_provenance(GAP8_IMAGE, tuple(args.make_var),
                                                      args.matrix))
        image = args.image
        if image is None:
            note(f"build {len(cases)} cases from {', '.join(names)} (log: {out / 'build.log'})")
            try:
                image = build(header_dir, args.make_var, out / "build.log")
            except Exception as exc:
                note(f"build failed: {exc}")
                return 2
        pinned = out / "kernel_bench.img"
        if image.resolve() != pinned:
            shutil.copy2(image, pinned)
        note(f"flash {pinned.stat().st_size} B and power-cycle")
        try:
            flash_and_boot(pinned, uri, log=lambda m: note(f"  {m}"))
        except Exception as exc:  # a failed flash must not end an unattended queue
            note(f"flash failed: {exc}")
            return 2

    log_path = out / "aideck.log"
    log_path.unlink(missing_ok=True)
    note(f"capture the radio console (up to {args.seconds:.0f} s)")
    raw = capture(uri, log_path, args.seconds)

    if "<F>" in raw:
        note("the Crazyflie console overflowed (<F>): bytes were dropped, so this run "
             "cannot be used as a profile")
        return 1
    guest = records(raw)
    if "#DONE" not in guest:
        note("the guest never printed #DONE")
        return 1
    try:
        sweep.assert_ran_expected_matrix(guest, cases, log_path)
    except RuntimeError as exc:
        note(str(exc))
        return 1
    # A complete log is reported whatever its verdict: `incomplete` (an expected
    # skip, such as a case too large for its adapter's L1 budget) still carries
    # every other result.
    report.write_report(report.summarize(report.parse_records(guest.splitlines())),
                        out / "report")
    note(f"report: {out / 'report' / 'summary.md'}")
    try:
        sweep.assert_guest_passed(guest, log_path)
    except RuntimeError as exc:
        note(str(exc))
        return 1
    note("complete passing run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
