"""STM32 radio flashing: what may be flashed, to which drone, and how a failure is left."""
from __future__ import annotations

import io
import zipfile

import pytest

from cf_lab import stm32_flash
from cf_lab.stm32_flash import check_platform, flash_stm32

URI = "radio://0/80/2M/E7E7E7E7E7"
CF21BL = "Crazyflie 2.1 Brushless"


class FakeBootloader:
    def __init__(self, uri, fail=None, reset_fails=False):
        self.uri = uri
        self.fail = fail
        self.reset_fails = reset_fails
        self.calls: list = []

    def flash_full(self, **kwargs):
        self.calls.append(("flash_full", kwargs))
        print("nRF51 version 2025.02")
        if self.fail is not None:
            raise self.fail

    def reset_to_firmware(self):
        self.calls.append(("reset_to_firmware",))
        if self.reset_fails:
            raise TimeoutError("bootloader link lost")

    def close(self):
        self.calls.append(("close",))


@pytest.fixture
def drone(monkeypatch):
    """The radio side of flash_stm32: a drone that answers before and after the flash."""
    import cflib.bootloader
    import cflib.crtp

    from cf_lab.backends import crazyflie as backend

    state = {"before": CF21BL, "after": CF21BL, "self_test": True, "fail": None, "reset_fails": False,
             "bootloaders": []}

    def bootloader(uri):
        instance = FakeBootloader(uri, state["fail"], state["reset_fails"])
        state["bootloaders"].append(instance)
        return instance

    answers = []

    def device_type_name(uri, timeout_s=3.0):
        answers.append(uri)
        return state["before"] if len(answers) == 1 else state["after"]

    monkeypatch.setattr(cflib.crtp, "init_drivers", lambda *args, **kwargs: None)
    monkeypatch.setattr(cflib.bootloader, "Bootloader", bootloader)
    monkeypatch.setattr(backend, "device_type_name", device_type_name)
    monkeypatch.setattr(stm32_flash, "_self_test_passed", lambda uri: state["self_test"])
    return state


def firmware(tmp_path, platforms=(CF21BL,), name="cf21bl.bin"):
    path = tmp_path / name
    path.write_bytes(b"\x00\x10\x00\x20" + b"".join(p.encode() + b"\0" for p in platforms) + b"\xff" * 256)
    return path


def test_good_image_is_flashed_warm_to_the_stm32_only(drone, tmp_path):
    flash_stm32(artifact=firmware(tmp_path), uri=URI)

    (bootloader,) = drone["bootloaders"]
    (_, kwargs), = [call for call in bootloader.calls if call[0] == "flash_full"]
    assert kwargs["warm"] is True
    assert [target.target for target in kwargs["targets"]] == ["stm32"]
    assert bootloader.calls[-1] == ("close",)


@pytest.mark.parametrize("image_platforms, device, ok", [
    ((CF21BL,), CF21BL, True),
    (("Crazyflie 2.0", "Crazyflie 2.1"), "Crazyflie 2.1", True),
    (("Crazyflie 2.0", "Crazyflie 2.1"), CF21BL, False),
    ((CF21BL,), "Crazyflie 2.1", False),
], ids=["cf21bl", "cf2-on-cf21", "cf2-on-brushless", "brushless-on-cf21"])
def test_check_platform_matches_whole_device_names(image_platforms, device, ok):
    image = b"".join(p.encode() + b"\0" for p in image_platforms)
    if ok:
        check_platform(image, device, "fw.bin")
    else:
        with pytest.raises(RuntimeError, match=device):
            check_platform(image, device, "fw.bin")


def test_image_for_another_platform_is_never_flashed(drone, tmp_path):
    with pytest.raises(RuntimeError, match="not built for it"):
        flash_stm32(artifact=firmware(tmp_path, platforms=("Crazyflie 2.1",)), uri=URI)

    assert drone["bootloaders"] == []


def test_drone_without_working_firmware_can_still_be_flashed(drone, tmp_path):
    drone["before"] = None

    flash_stm32(artifact=firmware(tmp_path, platforms=("something else",)), uri=URI)

    assert len(drone["bootloaders"]) == 1


def test_release_zip_is_refused_before_the_radio(drone, tmp_path):
    archive = tmp_path / "firmware-cf21bl-2025.02.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", "{}")

    with pytest.raises(ValueError, match="nRF51"):
        flash_stm32(artifact=archive, uri=URI)

    assert drone["bootloaders"] == []


def test_release_zip_with_a_bin_name_is_refused(drone, tmp_path):
    # cflib picks zip vs bin by content (zipfile.is_zipfile), so the suffix is no guard:
    # with no firmware answering, the platform check is skipped too, and cflib would
    # flash every target in the archive, nRF51 bootloader+softdevice included.
    drone["before"] = None
    archive = tmp_path / "cf21bl.bin"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("manifest.json", '{"version": 2, "files": {}}')
    archive.write_bytes(buffer.getvalue())

    with pytest.raises(ValueError):
        flash_stm32(artifact=archive, uri=URI)

    assert drone["bootloaders"] == []


def test_failed_flash_restarts_the_drone_to_firmware_and_reraises(drone, tmp_path, capsys):
    drone["fail"] = OSError("radio lost")

    with pytest.raises(OSError, match="radio lost"):
        flash_stm32(artifact=firmware(tmp_path), uri=URI)

    (bootloader,) = drone["bootloaders"]
    assert ("reset_to_firmware",) in bootloader.calls
    assert bootloader.calls[-1] == ("close",)
    # cflib's own output is shown when the flash fails, and only then.
    assert "nRF51 version" in capsys.readouterr().err


def test_failed_restart_after_a_failed_flash_keeps_the_original_error(drone, tmp_path):
    drone["fail"] = OSError("radio lost")
    drone["reset_fails"] = True

    with pytest.raises(OSError, match="radio lost"):
        flash_stm32(artifact=firmware(tmp_path), uri=URI)

    assert drone["bootloaders"][0].calls[-1] == ("close",)


def test_interrupted_flash_also_restarts_the_drone(drone, tmp_path):
    drone["fail"] = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        flash_stm32(artifact=firmware(tmp_path), uri=URI)

    assert ("reset_to_firmware",) in drone["bootloaders"][0].calls


def test_firmware_that_does_not_answer_after_the_flash_is_an_error(drone, tmp_path):
    drone["after"] = None

    with pytest.raises(RuntimeError, match="no firmware answered"):
        flash_stm32(artifact=firmware(tmp_path), uri=URI)


def test_firmware_that_fails_its_self_test_is_an_error(drone, tmp_path, capsys):
    drone["self_test"] = False

    with pytest.raises(RuntimeError, match="self-test"):
        flash_stm32(artifact=firmware(tmp_path), uri=URI)

    assert "nRF51 version" not in capsys.readouterr().err
