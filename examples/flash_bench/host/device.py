"""Bounded worker for production flashing and independent fixture observation.

The parent runs this in a separate process so a cflib regression cannot hang the
benchmark run indefinitely. No flasher methods are patched on the hardware path.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from threading import Event

from cf_lab.aideck import flash

RECORD = re.compile(
    r"FBENCH\|id=(\d+)\|seq=(\d+)\|bytes=(\d+)\|hash=([0-9a-f]{8})\|error=(\d+)")
VERIFIED = re.compile(r"GAP8 md5 verified \(([0-9a-f]{32})\)")


def assess_console(text: str, fixture: dict) -> dict:
    """Validate identity, complete readfs content, and sustained execution."""
    records = [dict(zip(("id", "seq", "bytes", "hash", "error"), match.groups()))
               for match in RECORD.finditer(text)]
    errors = []
    if "<F>" in text:
        errors.append("console overflow")
    if "Assert failed" in text or "Hardfault" in text or "FBENCH|fatal=" in text:
        errors.append("firmware fault")
    if fixture["silent"]:
        if "GAP8:" in text or records:
            errors.append("silent fixture emitted GAP8 console output")
    else:
        if len(records) < 3:
            errors.append("fewer than three fixture heartbeats")
        for row in records:
            if (int(row["id"]) != fixture["identity"] or int(row["error"]) != 0
                    or int(row["bytes"]) != fixture["payload_bytes"]
                    or row["hash"] != fixture["payload_hash"]):
                errors.append("fixture identity or readfs contents disagree")
                break
        sequences = [int(row["seq"]) for row in records]
        if any(b <= a for a, b in zip(sequences, sequences[1:])):
            errors.append("fixture restarted or sequence did not advance")
    return {"passed": not errors, "errors": errors, "records": records,
            "boot_verified": not fixture["silent"] and not errors}


def observe(uri: str, fixture: dict, directory: Path, timeout_s: float = 30) -> dict:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cf_lab.backends.crazyflie import connect, toc_cache

    cflib.crtp.init_drivers()
    cf = Crazyflie(rw_cache=toc_cache())
    chunks: list[str] = []
    changed = Event()
    with (directory / "console.log").open("w", buffering=1) as stream:
        def receive(text):
            stream.write(text)
            chunks.append(text)
            changed.set()

        cf.console.receivedChar.add_callback(receive)
        try:
            connect(cf, uri, timeout_s=20, retry=True)
            identity = {}
            for key in ("firmware.revision0", "firmware.revision1", "firmware.modified", "deck.bcAI"):
                try:
                    identity[key] = cf.param.get_value(key, timeout=2)
                except (KeyError, RuntimeError, TimeoutError) as exc:
                    identity[key] = str(exc)
            start = time.monotonic()
            while time.monotonic() - start < timeout_s:
                elapsed = time.monotonic() - start
                if fixture["silent"]:
                    if elapsed >= 5:
                        break
                elif elapsed >= (5 if fixture["interval_ms"] < 1000 else 1) and len(
                        RECORD.findall("".join(chunks))) >= 3:
                    break
                changed.wait(0.2)
                changed.clear()
        finally:
            cf.close_link()
    return {**assess_console("".join(chunks), fixture), "stm32": identity}


def execute(request: dict, directory: Path) -> dict:
    """Invoke the same public GAP8 flasher as cf-lab; retain all progress events."""
    start = time.monotonic()
    result = {"passed": False, "status": "failed", "action": request["action"]}
    events = []
    with (directory / "progress.jsonl").open("w", buffering=1) as stream:
        def progress(message, percent):
            event = {"elapsed_s": round(time.monotonic() - start, 3),
                     "percent": int(percent), "message": message}
            events.append(event)
            stream.write(json.dumps(event) + "\n")
            if percent in (0, 100) or "verified" in message:
                print(f"[{event['elapsed_s']:.1f}s] {percent}% {message}", flush=True)

        try:
            if request["action"] == "controller":
                from cf_lab.firmware import build_controller, controller_artifact
                from cf_lab.stm32_flash import flash_stm32
                app = Path(request["app"])
                with (directory / "build.log").open("w") as log:
                    build_controller(app, output=log)
                image = directory / "controller.bin"
                image.write_bytes(controller_artifact(app).read_bytes())
                flash_stm32(artifact=image, uri=request["uri"], progress_cb=progress)
                result["passed"] = True
            elif request["action"] == "flash":
                fixture = request["fixture"]
                kwargs = {"artifact": Path(fixture["path"]), "uri": request["uri"],
                          "cold_boot": request.get("cold_boot", False), "progress_cb": progress}
                if request.get("attempts") is not None:
                    kwargs["attempts"] = request["attempts"]
                # Keep production boot verification enabled, including for silent
                # images: a valid MD5 must short-circuit the liveness fallback.
                flash.flash_gap8(**kwargs)
                result["flash_s"] = round(time.monotonic() - start, 3)
                if any("Power-cycle failed" in e["message"] or
                       "could not reboot the drone" in e["message"] or
                       "the drone did not answer" in e["message"] for e in events):
                    raise RuntimeError("flasher reported a power/reboot failure; see progress.jsonl")
                digests = [m.group(1) for event in events
                           if (m := VERIFIED.fullmatch(event["message"]))]
                result["md5"] = digests[-1] if digests else None
                result["stalled_attempts"] = sum(
                    e["message"].startswith("stalled on attempt ") for e in events)
                result["fallback_used"] = any("verifying the GAP8 started" in e["message"]
                                              for e in events)
                if result["md5"] != fixture["written_md5"]:
                    raise RuntimeError("production flasher did not verify the expected image MD5")
                if fixture["repaired_areas"] and not any(
                        e["message"].startswith(f"dropping {fixture['repaired_areas']} zero-length")
                        for e in events):
                    raise RuntimeError("production flasher did not report the expected image repair")
                result["observation"] = observe(request["uri"], fixture, directory)
                result["passed"] = result["observation"]["passed"]
            elif request["action"] == "power-cycle":
                from cf_lab.backends.crazyflie import power_cycle
                if not power_cycle(request["uri"], log=lambda msg: progress(msg, 0)):
                    raise RuntimeError("power cycle failed")
                # Match the production cold-start path: power_cycle returns
                # immediately after power-up, before the radio is ready.
                progress(f"waiting {flash.COLD_BOOT_SETTLE_S:g}s for cold-start readiness", 0)
                time.sleep(flash.COLD_BOOT_SETTLE_S)
                result["observation"] = observe(request["uri"], request["fixture"], directory)
                result["passed"] = result["observation"]["passed"]
            else:
                raise ValueError(f"unknown action {request['action']}")
            result["status"] = "pass" if result["passed"] else "failed"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_s"] = round(time.monotonic() - start, 3)
    result["events"] = events
    return result


def main() -> int:
    request_path = Path(sys.argv[1])
    result = execute(json.loads(request_path.read_text()), request_path.parent)
    (request_path.parent / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
