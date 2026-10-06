"""AI-deck flashing against a fake deck memory: MD5 verification, stall retries, bounded cflib waits."""
from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

import pytest
from cflib.crazyflie.mem import _WriteRequest
from cflib.crazyflie.mem.deck_memory import DeckMemory, SyncDeckMemoryManager

from cf_lab.aideck import flash
from cf_lab.aideck.flash import (
    DeckBootFailed,
    DeckFlashCorrupt,
    FlashStalled,
    flash_deck,
    gap8_reported_md5,
    repair_gap8_image,
)

DATA = Path(__file__).parent / "data"
URI = "radio://0/80/2M/E7E7E7E7E7"


@pytest.fixture
def gap8_image(tmp_path):
    """A real GAP8 image whose last load area is empty, so the flasher repairs it."""
    path = tmp_path / "target.board.devices.flash.img"
    shutil.copyfile(DATA / "gap8" / "debug_demo.flash.img", path)
    return path


def run_flash(artifact, processor="gap8", **kwargs):
    notes: list[tuple[str, int]] = []
    kwargs.setdefault("stall_timeout_s", 0.05)
    flash_deck(artifact=artifact, uri=URI, processor=processor,
               progress_cb=lambda message, percent: notes.append((message, percent)), **kwargs)
    return notes


def test_gap8_flash_writes_repaired_image_from_offset_zero_and_verifies_md5(fake_deck, gap8_image):
    notes = run_flash(gap8_image)

    repaired, dropped = repair_gap8_image(gap8_image.read_bytes())
    assert dropped == 1
    assert fake_deck.completed == [repaired]
    # The deck is put in its bootloader and told the size before one write at offset 0
    # (the firmware maps that to 0x40000; it never addresses the GAP8 bootloader).
    assert [e for e in fake_deck.events if e != "query" and e != "connect" and e != "params_settled"] == [
        "reset_to_bootloader", ("size", len(repaired)), ("write", 0), "close_link", "reboot_to_fw",
    ]
    assert (f"GAP8 md5 verified ({hashlib.md5(repaired).hexdigest()})", 100) in notes
    assert fake_deck.liveness_checks == 0
    assert list(fake_deck.snapshots.iterdir()) == []


def test_gap8_md5_mismatch_raises_corrupt_without_retrying(fake_deck, gap8_image):
    fake_deck.corrupt_offset = 1000

    with pytest.raises(DeckFlashCorrupt):
        run_flash(gap8_image)

    assert fake_deck.events.count(("write", 0)) == 1


def test_stalled_write_is_retried_from_offset_zero_and_completes(fake_deck, gap8_image):
    fake_deck.stall_attempts = {1}

    run_flash(gap8_image)

    assert fake_deck.events.count(("write", 0)) == 2
    assert fake_deck.completed == [repair_gap8_image(gap8_image.read_bytes())[0]]
    # Each attempt ends with a reboot, which leaves the deck clean for the retry.
    assert fake_deck.events.count("reboot_to_fw") == 2


def test_retry_flashes_the_snapshot_not_an_artifact_rebuilt_meanwhile(fake_deck, gap8_image):
    original = repair_gap8_image(gap8_image.read_bytes())[0]
    fake_deck.stall_attempts = {1}
    fake_deck.on_attempt = lambda n: n == 2 and gap8_image.write_bytes(b"a different app" * 100)

    run_flash(gap8_image)

    assert fake_deck.completed == [original]


def test_stall_on_every_attempt_raises_flash_stalled_after_all_attempts(fake_deck, gap8_image, monkeypatch):
    calls = []

    def stalled_write(**kwargs):
        calls.append(kwargs["data"])
        raise FlashStalled("no ack")

    monkeypatch.setattr(flash, "_write_deck", stalled_write)

    with pytest.raises(FlashStalled, match="all 3 attempts"):
        run_flash(gap8_image, attempts=3)

    assert len(calls) == 3 and len(set(calls)) == 1


def test_failed_write_is_reported_at_once_and_still_reboots(fake_deck, gap8_image):
    fake_deck.fail_write = True

    with pytest.raises(RuntimeError, match="failed write"):
        run_flash(gap8_image)

    assert fake_deck.events.count(("write", 0)) == 1
    assert fake_deck.events[-2:] == ["close_link", "reboot_to_fw"]


@pytest.mark.parametrize("setup, message", [
    (lambda deck: setattr(deck, "name", "bcLighthouse4"), "no bcAI:gap8 deck"),
    (lambda deck: setattr(deck, "supports_upgrade", False), "cannot update"),
    (lambda deck: setattr(deck, "enter_bootloader", False), "did not enter its bootloader"),
], ids=["no-ai-deck", "no-upgrade-support", "stays-out-of-bootloader"])
def test_deck_that_cannot_take_the_image_fails_without_writing(fake_deck, gap8_image, setup, message):
    setup(fake_deck)

    with pytest.raises(RuntimeError, match=message):
        run_flash(gap8_image)

    assert not any(isinstance(e, tuple) for e in fake_deck.events)
    assert "reboot_to_fw" in fake_deck.events


def test_ctrl_c_mid_flash_reboots_the_drone_and_is_not_retried(fake_deck, gap8_image):
    fake_deck.interrupt = True

    with pytest.raises(KeyboardInterrupt):
        run_flash(gap8_image)

    assert fake_deck.events.count(("write", 0)) == 1
    assert fake_deck.events[-2:] == ["close_link", "reboot_to_fw"]
    assert list(fake_deck.snapshots.iterdir()) == []


def test_connect_timeout_counts_as_a_stall_and_is_retried(fake_deck, gap8_image):
    fake_deck.connect_timeouts = 1

    run_flash(gap8_image)

    assert fake_deck.events.count("connect") == 2
    assert len(fake_deck.completed) == 1


def test_deck_is_not_touched_until_the_parameter_download_finishes(fake_deck, gap8_image):
    fake_deck.params_delay_s = 0.1

    run_flash(gap8_image)

    assert fake_deck.events.index("params_settled") < fake_deck.events.index("query")


def test_stock_firmware_without_md5_falls_back_to_boot_check(fake_deck, gap8_image):
    fake_deck.report_md5 = False

    notes = run_flash(gap8_image)

    assert fake_deck.liveness_checks == 1
    assert "cold_boot" not in fake_deck.events
    assert notes[-1][1] == 100


def test_gap8_that_never_starts_after_a_cold_boot_raises_boot_failed(fake_deck, gap8_image):
    fake_deck.report_md5 = False
    fake_deck.alive = [False]

    with pytest.raises(DeckBootFailed):
        run_flash(gap8_image)

    assert fake_deck.events.count("cold_boot") == 1
    assert fake_deck.liveness_checks == 2


def test_gap8_that_starts_after_the_cold_boot_passes(fake_deck, gap8_image):
    fake_deck.report_md5 = False
    fake_deck.alive = [False, True]

    run_flash(gap8_image)

    assert fake_deck.events.count("cold_boot") == 1


def test_verify_boot_off_skips_liveness_for_a_quiet_app(fake_deck, gap8_image):
    fake_deck.report_md5 = False

    run_flash(gap8_image, verify_boot=False)

    assert fake_deck.liveness_checks == 0


def test_esp_image_is_flashed_byte_exact_and_not_boot_checked(fake_deck, tmp_path):
    fake_deck.name = "bcAI:esp"
    fake_deck.report_md5 = False
    # Bytes that happen to parse as a GAP8 header with a trailing empty area.
    image = tmp_path / "aideck_esp.bin"
    image.write_bytes((DATA / "gap8" / "debug_demo.flash.img").read_bytes()[:4096])

    run_flash(image, processor="esp")

    assert fake_deck.completed == [image.read_bytes()]
    assert fake_deck.liveness_checks == 0


@pytest.mark.parametrize("processor, exists, error", [
    ("stm32", True, ValueError),
    ("gap8", False, FileNotFoundError),
])
def test_bad_request_fails_before_the_radio(fake_deck, tmp_path, processor, exists, error):
    artifact = tmp_path / "app.img"
    if exists:
        artifact.write_bytes(b"\0" * 64)

    with pytest.raises(error):
        run_flash(artifact, processor=processor)

    assert fake_deck.events == []


@pytest.mark.parametrize("console, expected", [
    ("boot\nGAP8 md5: 0123456789abcdef0123456789ABCDEF len=17\n", "0123456789abcdef0123456789abcdef"),
    ("GAP8 md5: 0123456789abcdef0123456789abcdef len=1\n", None),
    ("GAP8 md5: no reply within 5165 ms\n", None),
    ("", None),
], ids=["upper-case", "short-reply", "timeout", "stock-firmware"])
def test_gap8_reported_md5(console, expected):
    assert gap8_reported_md5(console) == expected


# -- the bounded cflib waits in _hardened_cflib ---------------------------------


class SilentManager:
    """A deck-memory manager whose answers are lost on the radio."""

    def __init__(self):
        self.cleared = False

    def query_decks(self, on_success, on_failure=None):
        pass

    def _clear_query_cb(self):
        self.cleared = True

    def _write(self, *args):
        pass


def test_unanswered_deck_query_raises_stall_and_clears_the_pending_query(monkeypatch):
    monkeypatch.setattr(flash, "DECK_QUERY_TIMEOUT_S", 0.05)
    manager = SilentManager()

    with flash._hardened_cflib(1.0), pytest.raises(FlashStalled):
        SyncDeckMemoryManager(manager).query_decks()

    assert manager.cleared


def test_unacked_deck_command_raises_stall():
    deck = DeckMemory(SilentManager(), 0x2000)
    deck._bit_field1 = DeckMemory.MASK_IS_VALID | DeckMemory.MASK_IS_STARTED

    start = time.monotonic()
    with flash._hardened_cflib(0.05), pytest.raises(FlashStalled):
        deck.set_fw_new_flash_size(1234)

    assert time.monotonic() - start < 0.5


def test_cflib_is_restored_after_a_stall():
    def patched():
        return (DeckMemory.write_sync, DeckMemory._write_command_data,
                SyncDeckMemoryManager.query_decks, _WriteRequest.write_done)

    originals = patched()

    with pytest.raises(FlashStalled), flash._hardened_cflib(0.05):
        raise FlashStalled("lost packet")

    assert patched() == originals


def deck_memory(fake_deck):
    found = []
    fake_deck.query_decks(lambda decks: found.extend(decks.values()))
    return found[0]


def test_last_chunk_ack_may_take_the_gap8_verify_time(fake_deck):
    # 200 kB gives the GAP8 1 s to hash before its last ack; it takes 0.7 s.
    fake_deck.final_ack_delay_s = 0.7
    data = bytes(range(256)) * 800

    with flash._hardened_cflib(0.05):
        assert DeckMemory.write_sync(deck_memory(fake_deck), 0, data) is True

    assert fake_deck.completed == [data]


def test_small_image_gets_no_verify_allowance(fake_deck):
    fake_deck.final_ack_delay_s = 1.2

    with flash._hardened_cflib(0.05), pytest.raises(FlashStalled):
        DeckMemory.write_sync(deck_memory(fake_deck), 0, bytes(2000))


def test_write_that_stops_mid_image_raises_stall_without_the_verify_allowance(fake_deck):
    # 200 kB earns 1 s of verify time, but only for the last chunk's ack.
    fake_deck.stall_attempts = {0}

    start = time.monotonic()
    with flash._hardened_cflib(0.05), pytest.raises(FlashStalled):
        DeckMemory.write_sync(deck_memory(fake_deck), 0, bytes(200_000))

    assert time.monotonic() - start < 1.2
