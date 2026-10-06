"""Build and snapshot bench-owned GAP8 fixtures with independently known payloads."""
from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import asdict
from pathlib import Path

from cf_lab.aideck.flash import repair_gap8_image
from cf_lab.firmware import build_gap8, gap8_artifact

from .cases import Fixture

BENCH = Path(__file__).resolve().parents[1]


def payload(fixture: Fixture) -> bytes:
    pattern = bytes((i * 7 + fixture.identity) & 255 for i in range(256))
    return (pattern * ((fixture.payload_bytes + 255) // 256))[:fixture.payload_bytes]


def fnv1a(data: bytes) -> str:
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return f"{value:08x}"


def describe_image(path: Path) -> dict:
    source = path.read_bytes()
    written, dropped = repair_gap8_image(source)
    return {"path": str(path), "bytes": len(source),
            "sha256": hashlib.sha256(source).hexdigest(),
            "written_md5": hashlib.md5(written).hexdigest(),
            "written_sha256": hashlib.sha256(written).hexdigest(),
            "repaired_areas": dropped}


def append_empty_area(data: bytes) -> bytes:
    """Reproduce gapy's trailing-empty-area defect without shifting any payload.

    Only use unused header padding before the first load area. Refuse images
    whose padding cannot accommodate a descriptor. Production must repair it.
    """
    data, _ = repair_gap8_image(data)
    if len(data) < 32:
        raise ValueError("image header is truncated")
    count = int.from_bytes(data[4:8], "little")
    end = 16 + count * 16
    if not 1 <= count < 64 or end + 16 > len(data):
        raise ValueError("invalid area count or missing header padding")
    first = min(int.from_bytes(data[16 + i * 16:20 + i * 16], "little")
                for i in range(count))
    partition = int.from_bytes(data[:4], "little")
    if first < end + 16 or (partition and partition < end + 16):
        raise ValueError("no unused header padding for a zero-length area")
    if any(byte not in (0, 255) for byte in data[end:end + 16]):
        raise ValueError("header padding contains data")
    changed = bytearray(data)
    changed[4:8] = (count + 1).to_bytes(4, "little")
    changed[end:end + 16] = bytes(16)
    repaired, dropped = repair_gap8_image(bytes(changed))
    if dropped != 1 or repaired[16: end] != data[16:end] or repaired[first:] != data[first:]:
        raise ValueError("empty-area fixture changed a load area")
    return bytes(changed)


def build(fixture: Fixture, directory: Path) -> dict:
    directory.mkdir(parents=True)
    blob = payload(fixture)
    payload_path = directory / "payload.bin"
    payload_path.write_bytes(blob)
    variables = ["clean", "build", "image", f"FB_ID={fixture.identity}",
                 f"FB_INTERVAL_MS={fixture.interval_ms}", f"FB_SILENT={int(fixture.silent)}",
                 f"FB_PAYLOAD_BYTES={fixture.payload_bytes}",
                 f"FB_PAYLOAD={payload_path if blob else ''}"]
    with (directory / "build.log").open("w") as log:
        build_gap8(BENCH / "gap8", shlex.join(variables), build_image=False, output=log)
    image = directory / "flash.img"
    image.write_bytes(gap8_artifact(BENCH / "gap8").read_bytes())
    metadata = {**asdict(fixture), "payload_hash": fnv1a(blob), **describe_image(image)}
    (directory / "image.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
