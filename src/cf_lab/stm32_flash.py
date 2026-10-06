"""Flash STM32 firmware to the Crazyflie over the Crazyradio.

The cfloader CLI ships with cfclient, which cf_lab does not depend on (it is only
in the optional dev extra), so this module drives the same cflib Bootloader path
directly: warm-reboot to the bootloader, stream the firmware, and restart.

Radio bootloader entry is handled by the nRF51, not the STM32, so no STM32
image can lock it out; the guards here are about never leaving the drone
somewhere the next flash cannot find it.
"""

from __future__ import annotations

import contextlib
import io
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

# How long the new firmware may take to answer after cflib's closing restart,
# which already waits 6 s.
BOOT_TIMEOUT_S = 15.0


class PlatformMismatch(RuntimeError):
    """The image was not built for the drone it is about to be flashed to."""


class FirmwareNotAnswering(RuntimeError):
    """The flash finished, but no firmware answers afterwards."""


class SelfTestFailed(RuntimeError):
    """The new firmware booted, but its boot self-test did not pass."""


def stm32_target():
    from cflib.bootloader import Target

    return Target("cf2", "stm32", "fw", [], [])


def check_platform(image: bytes, device: str, name: str) -> None:
    """Raise PlatformMismatch unless ``image`` supports the drone's device type.

    A firmware image carries the device type names of the boards it supports
    (platform_*.c), NUL-terminated, so the drone's name must appear in it. The
    terminator keeps "Crazyflie 2.1" from matching "Crazyflie 2.1 Brushless".
    """
    if device.encode() + b"\0" not in image:
        raise PlatformMismatch(f"the drone is a {device}, and {name} was not built for it")


def _recover(bootloader, uri: str, note: Callable[[str], None]) -> None:
    """After a failed flash, get the drone back where the next flash can reach it.

    Left in its bootloader, the drone listens on a radio address only the
    interrupted session knew, and looks dead. Restarting it to firmware brings
    the nRF51 back to the usual URI even if the STM32 image is now incomplete,
    and the nRF51 can still start the bootloader from there.
    """
    from cf_lab.backends.crazyflie import device_type_name

    try:
        bootloader.reset_to_firmware()
    except Exception:  # noqa: BLE001 - the check below says whether it worked
        pass
    if device_type_name(uri, timeout_s=BOOT_TIMEOUT_S) is None:
        note(f"the drone does not answer on {uri}. Run the flash again; if that cannot "
             f"reach the bootloader, turn the drone off and on first.")


def flash_stm32(
    *,
    artifact: Path,
    uri: str,
    progress_cb: Callable[[str, int], None] | None = None,
) -> None:
    import cflib.crtp
    from cflib.bootloader import Bootloader

    from cf_lab.backends.crazyflie import device_type_name

    if not artifact.is_file():
        raise FileNotFoundError(f"Firmware artifact not found: {artifact}")
    if artifact.suffix != ".bin" or zipfile.is_zipfile(artifact):
        # cflib flashes every target in a zip, which it tells by content, not name,
        # and a release zip can update the nRF51's bootloader and softdevice: the
        # one radio flash that can brick the drone, since only a debugger recovers
        # a failed one.
        raise ValueError(f"{artifact.name} is not an STM32 .bin image; use cfclient for "
                         f"release zips (whatever their name), which can update the nRF51")

    def note(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message, 0)

    cflib.crtp.init_drivers()
    device = device_type_name(uri)
    if device is None:
        note("no firmware answered, so the platform check is skipped "
             "(expected when the current firmware does not boot)")
    else:
        check_platform(artifact.read_bytes(), device, artifact.name)

    # cflib prints nRF51 version details and a stray newline around the transfer;
    # keep them out of the way unless the flash fails.
    cflib_output = io.StringIO()
    bootloader = Bootloader(uri)
    try:
        with contextlib.redirect_stdout(cflib_output):
            bootloader.flash_full(
                filename=str(artifact),
                warm=True,
                targets=[stm32_target()],
                progress_cb=progress_cb,
            )
    except BaseException:
        sys.stderr.write(cflib_output.getvalue())
        _recover(bootloader, uri, note)
        raise
    finally:
        bootloader.close()

    if device_type_name(uri, timeout_s=BOOT_TIMEOUT_S) is None:
        raise FirmwareNotAnswering(
            f"the flash finished, but no firmware answered within {BOOT_TIMEOUT_S:.0f}s: "
            f"the new image may not boot. The bootloader is still reachable over the "
            f"radio, so flash a known-good controller."
        )
    # Answering is not enough: firmware that hangs or asserts at boot is reset by
    # its watchdog, and the next boot fails the self-test and never starts the app
    # but keeps answering. Costs about 2 s (5 s when the TOC changed, which also
    # fills the cache for the next connection).
    if not _self_test_passed(uri):
        raise SelfTestFailed(
            "the new firmware failed its boot self-test. A hang or assert at boot "
            "looks like this after the watchdog reset; now and then it is a sensor "
            "that did not come up, which `cf-lab reboot` clears. If it keeps "
            "failing, flash a known-good controller."
        )


def _self_test_passed(uri: str) -> bool:
    from cflib.crazyflie import Crazyflie

    from cf_lab.backends.crazyflie import connect, toc_cache
    from cf_lab.crazyflie.health import read_self_test

    cf = Crazyflie(rw_cache=toc_cache())
    try:
        connect(cf, uri, retry=True)
        return read_self_test(cf).passed
    finally:
        cf.close_link()
