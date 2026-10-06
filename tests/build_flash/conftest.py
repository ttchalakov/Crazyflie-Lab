"""Fakes for the build_flash tests: the drone's deck memory, driven through cflib's own code."""
from __future__ import annotations

import hashlib
import struct
import threading
from types import SimpleNamespace

import pytest

DEVICE = "Crazyflie 2.1 Brushless"


class FakeDeck:
    """The STM32's deck-memory subsystem as cflib sees it over the radio.

    cflib's real DeckMemory and _WriteRequest objects talk to it, so the patches
    in cf_lab.aideck.flash._hardened_cflib run against cflib's own chunking and
    ack path. Like the drone, one instance lives across flash attempts; each
    Crazyflie the flasher builds is one attempt. It mirrors gap8DeckFlasherWrite:
    a write at offset 0 restarts the image, and once the last byte lands the
    firmware prints the MD5 the GAP8 bootloader computed over what it holds.
    """

    id = 7
    DATA_BASE = 0x1000_0000
    COMMAND_BASE = 0x2000_0000

    def __init__(self, name: str = "bcAI:gap8"):
        self.name = name
        self.device_type = DEVICE
        self.supports_upgrade = True
        self.supports_reset = True
        self.bootloader = False
        self.enter_bootloader = True
        self.interrupt = False
        self.answer_queries = True
        self.ack_commands = True
        self.report_md5 = True
        self.md5_reply_len = 17
        self.fail_write = False
        self.stall_attempts: set[int] = set()
        self.stall_after_chunks = 5
        self.corrupt_offset: int | None = None
        self.final_ack_delay_s = 0.0
        self.connect_timeouts = 0
        self.params_delay_s = 0.0
        self.on_attempt = None
        self.attempt = 0
        self.flash = bytearray()
        self.completed: list[bytes] = []
        self.events: list = []
        self.console = None
        self.query_cleared = False
        self.timers: list[threading.Timer] = []

    def later(self, delay_s, function, *args):
        timer = threading.Timer(delay_s, function, args)
        self.timers.append(timer)
        timer.start()

    def stop(self):
        """Cancel pending acks, so none lands in the next test's flash."""
        for timer in self.timers:
            timer.cancel()
            timer.join()

    # -- what cflib's DeckMemoryManager does -------------------------------
    def query_decks(self, on_success, on_failure=None):
        from cflib.crazyflie.mem.deck_memory import DeckMemory

        self.events.append("query")
        if not self.answer_queries:
            return
        deck = DeckMemory(self, self.COMMAND_BASE)
        deck.name = self.name
        deck._base_address = self.DATA_BASE
        deck._bit_field1 = (
            DeckMemory.MASK_IS_VALID | DeckMemory.MASK_IS_STARTED | DeckMemory.MASK_SUPPORTS_WRITE
            | (DeckMemory.MASK_SUPPORTS_UPGRADE if self.supports_upgrade else 0)
            | (DeckMemory.MASK_BOOTLOADER_ACTIVE if self.bootloader else 0)
        )
        deck._bit_field2 = DeckMemory.MASK_SUPPORTS_RESET_TO_BOOTLOADER if self.supports_reset else 0
        on_success({0: deck})

    def _clear_query_cb(self):
        self.query_cleared = True

    def _write(self, base, address, data, complete_cb, failed_cb, progress_cb):
        from cflib.crazyflie.mem import _WriteRequest
        from cflib.crazyflie.mem.deck_memory import DeckMemory

        if base == self.COMMAND_BASE:
            if address == DeckMemory.ADR_COMMAND_BIT_FIELD:
                self.events.append("reset_to_bootloader")
                self.bootloader = self.enter_bootloader
            elif address == DeckMemory.ADR_FW_NEW_FLASH:
                self.events.append(("size", struct.unpack("<L", bytes(data))[0]))
            if self.ack_commands:
                complete_cb(address)
            return

        self.events.append(("write", address))
        if self.interrupt:
            raise KeyboardInterrupt  # Ctrl-C while the main thread streams the image
        if address == 0:
            self.flash = bytearray()
        sent: list = []
        request = _WriteRequest(self, base + address, data, SimpleNamespace(
            send_packet=lambda pk, expected_reply=None, timeout=None: sent.append(bytes(pk.data))
        ), progress_cb)
        request.start()
        chunks = 0
        done = False
        while sent and not done:
            packet = sent.pop(0)
            addr = struct.unpack("<BI", packet[:5])[1]
            if self.attempt in self.stall_attempts and chunks >= self.stall_after_chunks:
                return  # this chunk's ack is lost and nothing resends it
            offset = addr - base
            self.flash[offset:offset + len(packet) - 5] = packet[5:]
            chunks += 1
            if not sent and len(request._data) == 0 and self.final_ack_delay_s:
                # The last chunk: the firmware holds its ack while the GAP8 hashes.
                self.later(self.final_ack_delay_s, self._finish,
                           request, addr, complete_cb, failed_cb, address)
                return
            # Through the class attribute, which is where _hardened_cflib hooks acks.
            done = _WriteRequest.write_done(request, addr)
        self._finish(None, None, complete_cb, failed_cb, address)

    def _finish(self, request, addr, complete_cb, failed_cb, address):
        from cflib.crazyflie.mem import _WriteRequest

        if request is not None:
            _WriteRequest.write_done(request, addr)
        if self.corrupt_offset is not None:
            self.flash[self.corrupt_offset] ^= 0xFF
        if self.fail_write:
            failed_cb(address)
            return
        if self.report_md5 and self.console is not None:
            digest = hashlib.md5(bytes(self.flash)).hexdigest()
            line = f"GAP8 md5: {digest} len={self.md5_reply_len}\n"
            for start in range(0, len(line), 20):  # cflib delivers console text in fragments
                self.console(line[start:start + 20])
        self.completed.append(bytes(self.flash))
        complete_cb(address)

    # -- the radio around it -------------------------------------------------
    def connect(self, cf, uri, *, timeout_s=None, retry=False):
        self.events.append("connect")
        if self.connect_timeouts:
            self.connect_timeouts -= 1
            raise TimeoutError(f"no answer from {uri}")

        def settle():
            self.events.append("params_settled")
            cf.fully_connected.call(uri)

        if self.params_delay_s:
            self.later(self.params_delay_s, settle)
        else:
            settle()


class FakeCrazyflie:
    def __init__(self, deck: FakeDeck):
        from cflib.utils.callbacks import Caller

        self._deck = deck
        deck.attempt += 1
        self.console = SimpleNamespace(receivedChar=Caller())
        self.fully_connected = Caller()
        self.mem = SimpleNamespace(get_mems=lambda _type: [deck])
        deck.console = self.console.receivedChar.call
        if deck.on_attempt is not None:
            deck.on_attempt(deck.attempt)

    def close_link(self):
        self._deck.events.append("close_link")


class FakePowerSwitch:
    def __init__(self, deck: FakeDeck):
        self._deck = deck

    def reboot_to_fw(self):
        self._deck.events.append("reboot_to_fw")

    def close(self):
        pass


@pytest.fixture
def fake_deck(monkeypatch, tmp_path):
    """A drone with an AI deck that flashes cleanly; tests switch on its faults."""
    import tempfile

    import cflib.crazyflie
    import cflib.crtp
    import cflib.utils.power_switch

    from cf_lab.aideck import flash
    from cf_lab.backends import crazyflie as backend

    deck = FakeDeck()
    # The flasher's private snapshot of the image lands here, not in the system temp dir.
    deck.snapshots = tmp_path / "snapshots"
    deck.snapshots.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(deck.snapshots))
    monkeypatch.setattr(cflib.crazyflie, "Crazyflie", lambda rw_cache=None: FakeCrazyflie(deck))
    monkeypatch.setattr(cflib.crtp, "init_drivers", lambda *args, **kwargs: None)
    monkeypatch.setattr(cflib.utils.power_switch, "PowerSwitch", lambda uri: FakePowerSwitch(deck))
    monkeypatch.setattr(backend, "connect", deck.connect)
    monkeypatch.setattr(backend, "toc_cache", lambda: None)
    monkeypatch.setattr(backend, "device_type_name", lambda uri, timeout_s=3.0: deck.device_type)
    monkeypatch.setattr(flash, "RETRY_SETTLE_S", 0.0)
    monkeypatch.setattr(flash, "MD5_LINE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(flash, "POST_FLASH_SETTLE_S", 0.0)
    monkeypatch.setattr(flash, "DECK_READY_TIMEOUT_S", 0.0)
    deck.alive = [True]
    deck.liveness_checks = 0

    def gap8_is_alive(uri, **_kwargs):
        deck.liveness_checks += 1
        alive = deck.alive.pop(0) if len(deck.alive) > 1 else deck.alive[0]
        return alive, [10, 20] if alive else [10, 10]

    monkeypatch.setattr(flash, "gap8_is_alive", gap8_is_alive)
    monkeypatch.setattr(flash, "_cold_boot", lambda uri, **_kwargs: deck.events.append("cold_boot"))
    yield deck
    deck.stop()


@pytest.fixture
def cf_lab_root(tmp_path, monkeypatch):
    """A minimal cf-lab checkout, selected through $CF_LAB."""
    root = tmp_path / "cf-lab"
    (root / "lib").mkdir(parents=True)
    (root / "external" / "crazyflie-firmware").mkdir(parents=True)
    (root / "external" / "crazyflie-firmware" / "Makefile").write_text("all:\n")
    monkeypatch.setenv("CF_LAB", str(root))
    return root
