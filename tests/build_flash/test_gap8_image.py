"""GAP8 flash-image repair: dropping trailing zero-length load areas, and nothing else."""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from cf_lab.aideck.flash import repair_gap8_image

# Real gapy output: examples/idle (4 areas) and examples/debug_demo (5, the last empty).
DATA = Path(__file__).parent / "data"


def areas(image: bytes) -> list[tuple[int, int, int, int]]:
    count = struct.unpack_from("<I", image, 4)[0]
    return [struct.unpack_from("<4I", image, 16 + 16 * i) for i in range(count)]


def make_image(sizes: list[int], payload: bytes = b"\xab" * 64) -> bytes:
    header = struct.pack("<4I", 0x400, len(sizes), 0x1C000094, 0x1C000000)
    table = b"".join(struct.pack("<4I", 0x400 + 0x400 * i, 0x1C000000 + i, size, size and 1)
                     for i, size in enumerate(sizes))
    return header + table + payload


def test_real_image_with_trailing_empty_area_loses_only_that_area():
    original = (DATA / "gap8" / "debug_demo.flash.img").read_bytes()
    assert areas(original)[-1][2] == 0

    repaired, dropped = repair_gap8_image(original)

    assert dropped == 1
    assert areas(repaired) == areas(original)[:-1]
    assert len(repaired) == len(original)
    # Only the area count changes: every byte the ROM loads is untouched.
    assert repaired[:4] == original[:4] and repaired[8:] == original[8:]


def test_real_image_without_empty_area_is_returned_unchanged():
    original = (DATA / "gap8" / "idle.flash.img").read_bytes()

    assert repair_gap8_image(original) == (original, 0)


def test_several_trailing_empty_areas_are_all_dropped():
    repaired, dropped = repair_gap8_image(make_image([12, 0x5BC, 0x6000, 0, 0]))

    assert dropped == 2
    assert [area[2] for area in areas(repaired)] == [12, 0x5BC, 0x6000]


def test_empty_area_before_a_loaded_one_is_kept():
    image = make_image([12, 0, 0x6000])

    assert repair_gap8_image(image) == (image, 0)


def test_image_of_only_empty_areas_keeps_one_area():
    repaired, dropped = repair_gap8_image(make_image([0, 0, 0]))

    assert dropped == 2
    assert len(areas(repaired)) == 1


@pytest.mark.parametrize("image", [
    b"",
    b"\x00" * 15,
    make_image([])[:16] + b"\x00" * 32,                  # zero areas
    struct.pack("<4I", 0, 65, 0, 0) + b"\x00" * 65 * 16,  # more areas than the ROM allows
    make_image([12, 0])[:16 + 16],                        # area table runs past the end
    b"\x7fELF" + b"\x02" * 60,                            # not a flash image at all
], ids=["empty", "short", "no-areas", "too-many-areas", "truncated-table", "elf"])
def test_unparseable_image_is_left_alone(image):
    assert repair_gap8_image(image) == (image, 0)
