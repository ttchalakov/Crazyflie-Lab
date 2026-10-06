"""Flash a GAP8 or ESP32 firmware image to the AI deck over Crazyradio.

The image goes through the running STM32 firmware's deck-memory subsystem:
connect, reset the deck processor into its bootloader, stream the image, then
reboot the drone once so the deck starts it. cflib's own flash_full() also
takes the STM32 through its bootloader twice and reconnects a second time, none
of which a deck flash needs; each of those steps is one more place to fail, and
a failure inside the bootloader leaves the drone listening on a radio address
nothing else uses. Here the STM32 never leaves its firmware.

The drone must be on, in radio range, and running firmware with the bcAI deck
detected. Expect a few minutes: the radio uplink moves roughly 1-2 kB/s.

cflib waits forever on a lost radio packet in several places, so every wait is
bounded here -- see _hardened_cflib().
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import Event

# A healthy flash acks a 24-byte chunk many times a second, so a stall this long
# means the deck has stopped acknowledging writes and will never resume on its own.
DEFAULT_STALL_TIMEOUT_S = 30.0
# One attempt plus one retry. The retry is gated on FlashStalled in _flash_write(),
# and that failure has a known cause: a dropped radio packet that cflib never
# retransmits. Stalls are uncommon, so one retry is enough; anything that is NOT a
# stall surfaces immediately.
DEFAULT_ATTEMPTS = 2
# Bound on how long we wait for the connection's parameter download to finish.
PARAM_SETTLE_TIMEOUT_S = 20.0
# The GAP8 hashes the whole region it wrote before the last chunk is acked. The
# firmware allows 5 s + 5 us/byte for it (aideck.c), and so does the stall check
# for that one ack.
GAP8_VERIFY_S_PER_BYTE = 5e-6
# One deck query is a single memory read; the answer takes milliseconds.
DECK_QUERY_TIMEOUT_S = 5.0
# How long a deck may take to report started / bootloader active. cflib allows
# 4 s plus 5 s of AI-deck boot delay.
DECK_READY_TIMEOUT_S = 9.0
# The firmware prints the GAP8 MD5 before it acks the last write chunk, so the
# line is normally there already; this only covers console buffering.
MD5_LINE_TIMEOUT_S = 3.0
# How long the drone may take to answer again after the closing reboot.
REBOOT_TIMEOUT_S = 15.0
# Let cflib release the Crazyradio USB handle before the next attempt reopens it.
RETRY_SETTLE_S = 3.0

# Printed by cf-lab's crazyflie-firmware (aideck.c) at the end of a GAP8 flash: the
# MD5 the GAP8 bootloader computed over the region it just wrote. The reply is the
# command byte plus the 16-byte digest, so a valid one has len=17. The firmware's
# printf renders %x in upper case.
RE_GAP8_MD5 = re.compile(r"GAP8 md5: ([0-9a-fA-F]{32}) len=(\d+)")
GAP8_MD5_REPLY_LEN = 17

# Post-flash boot verification (GAP8 only), the fallback when the firmware printed
# no MD5 (stock Bitcraze firmware discards it). cflib's deck flash confirms only
# that the STM32 acked the bytes, so without the digest a silently-corrupt write
# reports success.
#
# It never re-flashes. Liveness is inferred from the ESP32's rxp counter climbing,
# which cannot distinguish "the GAP8 is dead" from "the GAP8 is running an app that
# does not send anything" -- examples/idle is exactly that. Verify, report
# honestly, and leave the reflash to the user.
#
# The AI-deck ESP firmware logs its GAP8<->ESP SPI stats every 5 s; a window a bit
# over 2 intervals guarantees at least two rxp samples to compare.
BOOT_LIVENESS_WINDOW_S = 16.0
# Give the STM32 + deck time to boot the freshly-flashed app before we listen.
POST_FLASH_SETTLE_S = 8.0
# Cold-boot hold-off and the settle after it before the bootloader handshake, for
# flash_deck(cold_boot=True). The cold boot that recovers a deck that did not come
# up after a flash uses the hold-off too.
COLD_BOOT_POWER_DOWN_S = 4.0
COLD_BOOT_SETTLE_S = 7.0


class FlashStalled(RuntimeError):
    """The deck stopped acking memory writes; the transfer cannot resume in place."""


class DeckFlashCorrupt(RuntimeError):
    """The GAP8 reported an MD5 for the region it wrote that does not match the image."""


class DeckBootFailed(RuntimeError):
    """The image flashed and cflib reported success, but the GAP8 never started.

    Distinct from FlashStalled (the transfer itself stalled). Raised only when the
    firmware reported no MD5 (stock firmware discards it), so a silently-corrupt
    write -- or a hard GAP8 wedge that only a physical power cycle clears -- looks
    exactly like a successful flash to cflib, and boot is the only signal left.
    """


DECK_TARGETS = {
    "gap8": "bcAI:gap8",
    "esp": "bcAI:esp",
}


@contextmanager
def _hardened_cflib(stall_timeout_s: float) -> Iterator[None]:
    """Patch around the cflib defect that makes a lost radio packet fatal.

    Nothing retransmits a dropped packet. The radio driver sets
    `needs_resending = not self._has_safelink`, and safelink is on, so cflib
    arms no retry timer. The deck query, the reset/size command path and
    DeckMemory.write_sync() then wait on an Event with no timeout, so the
    flash blocks forever. We bound all three instead. The write counts as
    stalled when no 24-byte chunk has been acked for stall_timeout_s: cflib's
    progress callback only fires when the whole percentage rises, and 1% of a
    6.6 MB GAP8 image is 66 KB, so timing that would fail large images. The
    last chunk also gets the GAP8's verify time (GAP8_VERIFY_S_PER_BYTE).

    The deck's write handler (gap8DeckFlasherWrite in the STM32 firmware) is a
    stateful sequential buffer keyed on memAddr == 0, so a stalled transfer cannot
    be resumed mid-stream -- it has to restart from address 0. That is why the
    retry unit is the whole flash, not the chunk.
    """
    from cflib.crazyflie.mem import _WriteRequest
    from cflib.crazyflie.mem.deck_memory import DeckMemory, SyncDeckMemoryManager

    last_ack = [time.monotonic()]
    last_chunk_sent = [False]

    def chunk_acked(self, addr):
        last_ack[0] = time.monotonic()
        done = original_chunk_acked(self, addr)
        last_chunk_sent[0] = len(self._data) == 0
        return done

    def bounded_write_sync(self, address, data, progress_cb=None):
        done = Event()
        succeeded = False
        last_ack[0] = time.monotonic()
        last_chunk_sent[0] = False
        verify_s = len(data) * GAP8_VERIFY_S_PER_BYTE

        def on_success(*_args) -> None:
            nonlocal succeeded
            succeeded = True
            done.set()

        def on_failure(*_args) -> None:
            done.set()

        self.write(address, data, on_success, write_failed_cb=on_failure, progress_cb=progress_cb)

        while not done.wait(0.5):
            stalled_for = time.monotonic() - last_ack[0]
            if stalled_for > stall_timeout_s + (verify_s if last_chunk_sent[0] else 0.0):
                raise FlashStalled(
                    f"deck memory write made no progress for {stalled_for:.0f}s "
                    f"(a lost radio packet, which cflib never resends, or the drone rebooted)"
                )
        return succeeded

    def bounded_command_write(self, address, data):
        if not self.is_started:
            raise RuntimeError("Deck not ready")

        done = Event()
        succeeded = False

        def on_success(*_args) -> None:
            nonlocal succeeded
            succeeded = True
            done.set()

        def on_failure(*_args) -> None:
            done.set()

        self._deck_memory_manager._write(
            self._command_base_address,
            address,
            data,
            on_success,
            on_failure,
            None,
        )
        if not done.wait(stall_timeout_s):
            raise FlashStalled(
                f"deck command write made no progress for {stall_timeout_s:.0f}s "
                f"(a lost radio packet, which cflib never resends, or the drone rebooted)"
            )
        return succeeded

    def bounded_query_decks(self):
        done = Event()
        result: dict = {}

        def on_success(decks) -> None:
            result["decks"] = decks
            done.set()

        def on_failure(message) -> None:
            result["error"] = message
            done.set()

        manager = self._deck_memory_manager
        manager.query_decks(on_success, on_failure)
        if not done.wait(DECK_QUERY_TIMEOUT_S):
            # Clear the pending query, or the next one raises "Query ongoing".
            manager._clear_query_cb()
            raise FlashStalled(
                f"the deck query got no answer for {DECK_QUERY_TIMEOUT_S:.0f}s "
                f"(a lost radio packet, which cflib never resends, or the drone rebooted)"
            )
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["decks"]

    original_write_sync = DeckMemory.write_sync
    original_command_write = DeckMemory._write_command_data
    original_query_decks = SyncDeckMemoryManager.query_decks
    original_chunk_acked = _WriteRequest.write_done
    DeckMemory.write_sync = bounded_write_sync
    DeckMemory._write_command_data = bounded_command_write
    SyncDeckMemoryManager.query_decks = bounded_query_decks
    _WriteRequest.write_done = chunk_acked
    try:
        yield
    finally:
        DeckMemory.write_sync = original_write_sync
        DeckMemory._write_command_data = original_command_write
        SyncDeckMemoryManager.query_decks = original_query_decks
        _WriteRequest.write_done = original_chunk_acked


def _cold_boot(uri: str, *, power_down_s: float, settle_s: float, log=None) -> None:
    """Power-cycle the drone and wait for the deck to come back up.

    For recovering a wedged deck only; a normal flash does not need it (see
    flash_deck's cold_boot).
    """
    from cf_lab.backends.crazyflie import power_cycle

    power_cycle(uri, power_down_s=power_down_s, log=log)
    time.sleep(settle_s)


def gap8_is_alive(uri: str):
    """App-agnostic GAP8 liveness over the CPX console -- no WiFi association needed.

    The AI-deck ESP firmware periodically logs ``WIFI: spi: ... rxp N``, where N
    counts CPX packets received from the GAP8 over SPI. N climbs only while a GAP8
    app is actually running; a GAP8 that crashed at boot (``pmsis_exit``) halts and
    N stays flat. Returns ``(alive, rxp_samples)``.
    """
    from cflib.crazyflie import Crazyflie

    from cf_lab.backends.crazyflie import connect, toc_cache

    rxps: list[int] = []
    buf = {"text": ""}

    def on_console(text: str) -> None:
        buf["text"] += text
        while "\n" in buf["text"]:
            line, buf["text"] = buf["text"].split("\n", 1)
            match = re.search(r"rxp (\d+)", line)
            if match is not None:
                rxps.append(int(match.group(1)))

    cf = Crazyflie(rw_cache=toc_cache())
    try:
        # The radio is not reachable immediately after a power cycle.
        connect(cf, uri, timeout_s=15.0, retry=True)
        cf.console.receivedChar.add_callback(on_console)
        time.sleep(BOOT_LIVENESS_WINDOW_S)
    except (TimeoutError, ConnectionError):
        return False, rxps
    finally:
        cf.close_link()

    alive = len(rxps) >= 2 and rxps[-1] > rxps[0]
    return alive, rxps


def _connect(cf, uri: str) -> None:
    """Connect, retrying while the drone boots, then let the params settle.

    Deck writes wait for the parameter download because they contend with it for
    the radio: cflib's own SyncCrazyflie.wait_for_params() docstring warns that
    "radio bandwidth will be limited while parameters are downloaded", and a write
    chunk (or its ack) dropped in that window stalls the flash. A connect that
    times out is a lost packet, so it becomes FlashStalled and gets the flash's
    one retry.
    """
    from cf_lab.backends.crazyflie import connect

    settled = Event()
    cf.fully_connected.add_callback(lambda _uri: settled.set())
    try:
        connect(cf, uri, retry=True)
    except TimeoutError as exc:
        raise FlashStalled(str(exc)) from exc
    settled.wait(PARAM_SETTLE_TIMEOUT_S)


def _wait_for_deck(manager, name: str, ready: Callable, what: str):
    """Re-query the drone's decks until ``name`` satisfies ``ready``; return it."""
    deadline = time.monotonic() + DECK_READY_TIMEOUT_S
    while True:
        decks = {deck.name: deck for deck in manager.query_decks().values()}
        deck = decks.get(name)
        if deck is None:
            found = ", ".join(sorted(decks)) or "none"
            raise RuntimeError(f"the drone reports no {name} deck (found: {found})")
        if ready(deck):
            return deck
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{name} did not {what} within {DECK_READY_TIMEOUT_S:.0f}s")
        time.sleep(0.5)


def _reboot_to_firmware(uri: str, note: Callable[[str], None]) -> None:
    """Restart the drone into firmware, which also resets both deck processors.

    The nRF51 handles this command, so it works even when the STM32 is wedged.
    Best effort: a failure is reported, not raised, since it usually runs while
    another error is already on its way out.
    """
    from cflib.utils.power_switch import PowerSwitch

    try:
        switch = PowerSwitch(uri)
        try:
            switch.reboot_to_fw()
        finally:
            switch.close()
    except Exception as exc:  # noqa: BLE001 - report, the caller's error matters more
        note(f"could not reboot the drone ({exc}); turn it off and on before the next flash")


def _write_deck(
    *,
    uri: str,
    processor: str,
    data: bytes,
    console: list[str],
    progress_cb: Callable[[str, int], None] | None,
    note: Callable[[str], None],
) -> None:
    """Write ``data`` to one AI-deck processor, then reboot the drone.

    Runs inside _hardened_cflib, so the cflib calls below are bounded. The
    reboot runs whatever happens, Ctrl-C included: on success it starts the new
    image, and on failure it leaves the STM32 and the deck clean for a retry.
    """
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.mem import MemoryElement
    from cflib.crazyflie.mem.deck_memory import SyncDeckMemoryManager

    from cf_lab.backends.crazyflie import toc_cache

    name = DECK_TARGETS[processor]
    cf = Crazyflie(rw_cache=toc_cache())
    cf.console.receivedChar.add_callback(console.append)
    try:
        _connect(cf, uri)
        mems = cf.mem.get_mems(MemoryElement.TYPE_DECK_MEMORY)
        if not mems:
            raise RuntimeError("the drone's firmware has no deck memory, so it cannot flash decks")
        manager = SyncDeckMemoryManager(mems[0])
        deck = _wait_for_deck(manager, name, lambda d: d.is_started, "start")
        if not deck.supports_fw_upgrade:
            raise RuntimeError(f"the drone's firmware cannot update {name}")
        if deck.supports_reset_to_bootloader:
            deck.reset_to_bootloader()
            deck = _wait_for_deck(manager, name, lambda d: d.is_bootloader_active, "enter its bootloader")
        deck.set_fw_new_flash_size(len(data))
        if not deck.write_sync(0, data, progress_cb):
            raise RuntimeError(f"the drone reported a failed write to {name}")
        if processor == "gap8":
            deadline = time.monotonic() + MD5_LINE_TIMEOUT_S
            while RE_GAP8_MD5.search("".join(console)) is None and time.monotonic() < deadline:
                time.sleep(0.1)
    finally:
        cf.close_link()
        _reboot_to_firmware(uri, note)


def _flash_write(
    *,
    artifact: Path,
    uri: str,
    processor: str,
    progress_cb: Callable[[str, int], None] | None,
    attempts: int,
    stall_timeout_s: float,
) -> str:
    """Stream the image to the deck, retrying the whole transfer on a stall.

    Returns the console text of the successful attempt; for the GAP8 it carries
    the MD5 line that gap8_reported_md5() reads.
    """
    from cf_lab.backends.crazyflie import device_type_name

    def note(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message, 0)

    data = artifact.read_bytes()
    stall: FlashStalled | None = None
    for attempt in range(1, attempts + 1):
        console: list[str] = []
        try:
            with _hardened_cflib(stall_timeout_s):
                _write_deck(uri=uri, processor=processor, data=data, console=console,
                            progress_cb=progress_cb, note=note)
        except FlashStalled as exc:
            stall = exc
            note(f"stalled on attempt {attempt}/{attempts}: {exc}")
            if attempt < attempts:
                time.sleep(RETRY_SETTLE_S)
            continue
        if device_type_name(uri, timeout_s=REBOOT_TIMEOUT_S) is None:
            note(f"the drone did not answer within {REBOOT_TIMEOUT_S:.0f}s of the closing reboot")
        return "".join(console)

    raise FlashStalled(f"AI-deck {processor} flash stalled on all {attempts} attempts: {stall}")


def gap8_reported_md5(console: str) -> str | None:
    """The MD5 the GAP8 reported for the region it wrote, or None if it sent none."""
    match = RE_GAP8_MD5.search(console)
    if match is None or int(match.group(2)) != GAP8_MD5_REPLY_LEN:
        return None
    return match.group(1).lower()


def verify_gap8_md5(image: Path, console: str) -> str | None:
    """Compare the GAP8's reported MD5 with the image's; return it, or None if absent.

    Raises DeckFlashCorrupt on a mismatch. A missing digest is inconclusive, not a
    failure: stock firmware prints none.
    """
    reported = gap8_reported_md5(console)
    if reported is None:
        return None
    expected = hashlib.md5(image.read_bytes()).hexdigest()
    if reported != expected:
        raise DeckFlashCorrupt(
            f"the GAP8 reported md5 {reported} for the region it wrote, but the image "
            f"hashes to {expected}: the bytes on the deck are not the bytes we sent"
        )
    return reported


# --------------------------------------------------------------------------
# GAP8 flash-image repair: drop trailing zero-length areas
# --------------------------------------------------------------------------
# A GAP8 flash image starts with a 16-byte header -- partition-table offset,
# area count, entry point, load base -- followed by that many 16-byte area
# descriptors (flash offset, L2 address, size, block count). The ROM bootloader
# walks the areas, DMAs each one from HyperFlash into L2, and jumps to the entry
# point.
#
# For some section layouts gapy emits a final area with **size 0**. The ROM still
# processes it, and a zero-length HyperBus transfer leaves the controller's read
# address unprogrammed -- so every pi_flash_read() the application later performs
# ignores its address argument and streams sequentially from wherever the ROM left
# off. pi_fs_mount() reads the partition table into that garbage and returns -1,
# and since DORY's mem_init() reports failures with printf() -- to UART pins this
# deck does not expose -- the app dies mid-boot in total silence.
#
# Whether an image ends with an empty area depends on its section layout, so a
# build change that looks unrelated, such as an extra driver, can add or remove it.
#
# Dropping the descriptor loads nothing, so the repair is content-preserving: the
# area moves 0 bytes by definition. Only *trailing* empty areas are removed, and
# only when the header parses cleanly; anything unexpected is left alone, because
# flashing a mangled image costs a battery pull to recover.
GAP8_IMAGE_HEADER_SIZE = 16
GAP8_IMAGE_AREA_SIZE = 16
GAP8_IMAGE_MAX_AREAS = 64


def _u32(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off:off + 4], "little")


def repair_gap8_image(data: bytes) -> tuple[bytes, int]:
    """Return (image, n_dropped) with trailing zero-length areas removed."""
    if len(data) < GAP8_IMAGE_HEADER_SIZE:
        return data, 0
    n_areas = _u32(data, 4)
    if not 1 <= n_areas <= GAP8_IMAGE_MAX_AREAS:
        return data, 0
    end = GAP8_IMAGE_HEADER_SIZE + n_areas * GAP8_IMAGE_AREA_SIZE
    if end > len(data):
        return data, 0

    def area_size(i: int) -> int:
        return _u32(data, GAP8_IMAGE_HEADER_SIZE + i * GAP8_IMAGE_AREA_SIZE + 8)

    kept = n_areas
    while kept > 1 and area_size(kept - 1) == 0:
        kept -= 1
    if kept == n_areas:
        return data, 0
    return data[:4] + kept.to_bytes(4, "little") + data[8:], n_areas - kept


@contextmanager
def _snapshot_artifact(artifact: Path, processor: str,
                       note: Callable[[str], None]) -> Iterator[Path]:
    """Yield a private snapshot of ``artifact``, repaired if it is a GAP8 image.

    The snapshot is taken once, before the first attempt, and both the flash and
    the MD5 check read it, so a rebuild landing in the build directory mid-flash
    cannot change the bytes either one sees.
    """
    import shutil
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="cf-lab-flash-"))
    snapshot = tmpdir / artifact.name
    if processor == "gap8":
        fixed, dropped = repair_gap8_image(artifact.read_bytes())
        if dropped:
            note(f"dropping {dropped} zero-length area(s) from the flash image -- "
                 f"they leave the GAP8's HyperBus read address unprogrammed, which "
                 f"breaks pi_fs_mount() and every later pi_flash_read()")
        snapshot.write_bytes(fixed)
    else:
        shutil.copyfile(artifact, snapshot)
    try:
        yield snapshot
    finally:
        snapshot.unlink(missing_ok=True)
        tmpdir.rmdir()


def flash_deck(
    *,
    artifact: Path,
    uri: str,
    processor: str,
    progress_cb: Callable[[str, int], None] | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
    cold_boot: bool = False,
    verify_boot: bool | None = None,
) -> None:
    """Flash a deck image and, for the GAP8, verify it by MD5 or by boot.

    For the GAP8, the firmware prints the MD5 the bootloader computed over the
    region it wrote; a match proves the deck holds the image, and a mismatch raises
    :class:`DeckFlashCorrupt`. Only when no digest arrives (stock firmware) does
    it fall back to ``verify_boot``.

    ``cold_boot`` defaults to **off**: a warm reflash writes the image correctly,
    as the GAP8 bootloader's MD5 of the written region confirms, so no power
    cycle is needed. Pass ``cold_boot=True`` to recover a deck that is already
    wedged; do not pay for it on every flash. examples/flash_bench measures warm
    and cold flashes.

    ``verify_boot`` (default: on for the GAP8) confirms the GAP8 came alive after
    flashing. It checks after the flash's own restart and, if that shows nothing,
    once more after one cold boot, then raises :class:`DeckBootFailed` -- it
    does not re-flash, because the signal it is built on is not reliable enough
    to act on automatically (see below).

    Caveat worth knowing: liveness is inferred from the ESP32's ``rxp`` counter
    climbing, so an app that legitimately sends nothing -- ``examples/idle`` is
    the example -- reads as dead. The MD5 check above does not have that problem.
    """
    import cflib.crtp

    if processor not in DECK_TARGETS:
        choices = ", ".join(sorted(DECK_TARGETS))
        raise ValueError(f"unknown AI-deck processor {processor!r}; choose {choices}")
    if not artifact.is_file():
        raise FileNotFoundError(f"AI-deck {processor} flash artifact not found: {artifact}")

    if verify_boot is None:
        verify_boot = processor == "gap8"

    cflib.crtp.init_drivers()

    def note(message: str, percent: int = 0) -> None:
        if progress_cb is not None:
            progress_cb(message, percent)

    def note_log(message: str) -> None:
        note(message)

    with ExitStack() as stack:
        # Snapshot once, then flash the snapshot on every attempt: it pins the
        # bytes against a concurrent rebuild, and for the GAP8 it also applies
        # repair_gap8_image() -- a trailing zero-length area in the image breaks
        # every pi_flash_read() the app later performs. Repaired here rather than
        # in each app's build, because it is a property of the image gapy produced
        # and every GAP8 app that reads its own flash is exposed to it.
        image = stack.enter_context(_snapshot_artifact(artifact, processor, note_log))

        if cold_boot:
            note(f"cold boot before flash ({COLD_BOOT_POWER_DOWN_S:.0f}s off)")
            _cold_boot(uri, power_down_s=COLD_BOOT_POWER_DOWN_S, settle_s=COLD_BOOT_SETTLE_S,
                       log=note_log)

        console = _flash_write(
            artifact=image,
            uri=uri,
            processor=processor,
            progress_cb=progress_cb,
            attempts=attempts,
            stall_timeout_s=stall_timeout_s,
        )

        if processor == "gap8":
            digest = verify_gap8_md5(image, console)
            if digest is not None:
                note(f"GAP8 md5 verified ({digest})", 100)
                return
            note("the firmware reported no GAP8 md5, so the flash cannot be verified by digest")

        if not verify_boot:
            return

        # Prefer the flash's own restart (no extra power cycle, which can itself
        # wedge the GAP8); only cold-boot if that did not bring the app up.
        note("verifying the GAP8 started (via the post-flash restart)")
        time.sleep(POST_FLASH_SETTLE_S)
        alive, rxps = gap8_is_alive(uri)
        if not alive:
            note("no liveness after the flash restart; trying one cold boot")
            _cold_boot(uri, power_down_s=COLD_BOOT_POWER_DOWN_S, settle_s=POST_FLASH_SETTLE_S,
                       log=note_log)
            alive, rxps = gap8_is_alive(uri)

        if alive:
            note(f"GAP8 booted OK (ESP32 rxp {rxps[0]}->{rxps[-1]})", 100)
            return
        note(f"GAP8 did NOT start after flash (rxp flat: {rxps})", 0)

    raise DeckBootFailed(
        f"AI-deck {processor} flashed and cflib reported success, but no GAP8 activity "
        f"was seen afterwards (ESP32 rxp stayed flat: {rxps}).\n"
        "\n"
        "This check watches the ESP32's rxp counter climb, so it CANNOT tell a dead "
        "GAP8 from an app that simply never sends anything -- examples/idle looks "
        "identical to a failed flash here. If the image you just flashed is a quiet "
        "app, this is a false alarm: pass verify_boot=False.\n"
        "\n"
        "The firmware on the drone reported no GAP8 md5, so the flash could not be "
        "verified by digest. Flash a controller built from this cf-lab checkout, "
        "whose firmware reports it. To recover now: physically power-cycle the "
        "drone (unplug the battery ~10 s) and retry."
    )


def flash_gap8(**kwargs) -> None:
    flash_deck(processor="gap8", **kwargs)
