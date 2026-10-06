from __future__ import annotations

import time
from typing import Any

from cf_lab.mocap.types import PoseSample
from cf_lab.sdk import SdkError, sdk_root


class CrazyflieExtposeBackend:
    """Small adapter around cflib's extpose API.

    ``position_only`` streams position (``send_extpos``) instead of full pose
    (``send_extpose``): the EKF then fuses only position and keeps its own
    IMU-based attitude. That is far more robust than fusing an external
    orientation, which can lag or flip (e.g. symmetric mocap markers) and
    destabilize attitude control.
    """

    def __init__(self, crazyflie: Any, *, position_only: bool = False) -> None:
        self._cf = crazyflie
        self._position_only = position_only

    def send_extpose(self, pose: PoseSample) -> None:
        x, y, z = pose.position_m
        if self._position_only:
            self._cf.extpos.send_extpos(x, y, z)
            return
        q = pose.orientation.normalized()
        self._cf.extpos.send_extpose(x, y, z, q.qx, q.qy, q.qz, q.qw)


# Opening a link includes the log and param TOC download: about 1 s from the
# cache, 4 s without it. Retries while a drone boots fit in this too.
CONNECT_TIMEOUT_S = 30.0


def toc_cache() -> str | None:
    """cflib's log/param TOC cache (``rw_cache``): artifacts/cache/ in the cf-lab
    checkout, wherever the script was started. A Python-only install has no
    checkout, so it goes without and downloads the TOCs on every connect.
    """
    try:
        return str(sdk_root() / "artifacts" / "cache")
    except SdkError:
        return None


def connect(cf, uri: str, *, timeout_s: float = CONNECT_TIMEOUT_S, retry: bool = False) -> None:
    """Open ``cf``'s link to ``uri`` and wait for its TOCs, with a deadline.

    cflib's SyncCrazyflie.open_link() waits forever when one radio packet of the
    TOC download is lost; here that raises TimeoutError. With ``retry``, a failed
    open is retried until the deadline, for a drone that is still booting;
    otherwise it raises ConnectionError at once. Parameter values may still be
    downloading on return, as with SyncCrazyflie.
    """
    from threading import Event

    connected = Event()
    failures: list[str] = []

    def on_connected(_uri: str) -> None:
        connected.set()

    def on_failed(_uri: str, message: str) -> None:
        failures.append(message)
        connected.set()

    cf.connected.add_callback(on_connected)
    cf.connection_failed.add_callback(on_failed)
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            connected.clear()
            failures.clear()
            cf.open_link(uri)
            if not connected.wait(max(0.0, deadline - time.monotonic())):
                cf.close_link()
                raise TimeoutError(f"connecting to {uri} did not finish within {timeout_s:.0f}s")
            if not failures:
                return
            cf.close_link()
            if not retry or time.monotonic() >= deadline:
                raise ConnectionError(f"could not connect to {uri}: {failures[-1]}")
            time.sleep(0.5)
    finally:
        cf.connected.remove_callback(on_connected)
        cf.connection_failed.remove_callback(on_failed)


class CrazyflieRadioBackend(CrazyflieExtposeBackend):
    """Own a cflib radio link and expose only the extpose operation.

    If ``console_sink`` is given, the firmware console is captured. The firmware
    emits console packets over the radio whenever it prints (DEBUG_PRINT),
    regardless of whether the host listens, so capturing costs no extra link
    bandwidth — cflib delivers the text in arbitrary chunks, which we reassemble
    into whole lines and hand to ``console_sink`` one completed line at a time.
    """

    def __init__(self, uri: str, *, console_sink=None, position_only: bool = False) -> None:
        self._uri = uri
        self._console_sink = console_sink
        self._console_buf = ""
        super().__init__(crazyflie=None, position_only=position_only)

    def open(self) -> None:
        import cflib.crtp
        from cflib.crazyflie import Crazyflie

        cflib.crtp.init_drivers()
        cf = Crazyflie(rw_cache=toc_cache())
        connect(cf, self._uri)
        self._cf = cf
        if self._console_sink is not None:
            self._cf.console.receivedChar.add_callback(self._on_console)

    def _on_console(self, text: str) -> None:
        self._console_buf += text
        while "\n" in self._console_buf:
            line, self._console_buf = self._console_buf.split("\n", 1)
            self._console_sink(line)

    def close(self) -> None:
        try:
            if self._console_sink is not None and self._console_buf:
                self._console_sink(self._console_buf)  # flush trailing partial line
                self._console_buf = ""
        finally:  # a failing sink must not keep the Crazyradio held
            if self._cf is not None:
                self._cf.close_link()
                self._cf = None

    @property
    def crazyflie(self) -> Any:
        if self._cf is None:
            raise RuntimeError("Crazyflie link is not open.")
        return self._cf

    def __enter__(self) -> "CrazyflieRadioBackend":
        self.open()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


# Bits of the firmware "supervisor.info" log variable (see supervisor.c), by name.
SUPERVISOR_INFO_BITS = {
    "canArm": 0x0001,
    "ARMED": 0x0002,
    "autoArm": 0x0004,
    "canFly": 0x0008,
    "flying": 0x0010,
    "TUMBLED": 0x0020,
    "LOCKED": 0x0040,
    "crashed": 0x0080,
}


# cflib's PowerSwitch.stm_power_cycle() holds the STM (and expansion-deck) power
# off for only 1 s, which is too short to reliably reset the AI-deck: the GAP8
# intermittently comes back wedged — the STM re-enumerates the deck (bcAI=1) but
# the GAP8 app never resumes streaming frames. Holding power off longer fully
# drains the deck so it cold-boots cleanly every time.
DECK_POWER_DOWN_S = 4.0


def power_cycle(uri: str, *, power_down_s: float = DECK_POWER_DOWN_S, log=None) -> bool:
    """Power-cycle the drone's STM32 and expansion decks over the radio.

    Power is held off for ``power_down_s`` (longer than cflib's built-in 1 s) so
    the AI-deck GAP8 reliably cold-boots instead of coming back wedged. A power
    cycle also clears the firmware emergency-stop deadman, which otherwise stays
    armed until reboot. ``cf-lab reboot`` uses it (through
    :func:`power_cycle_until_healthy`), and the flash and bench tools use it to
    cold-boot the deck. Failures are reported through ``log`` (if given) and
    return ``False`` rather than raising, so callers that require a confirmed
    cold boot can stop safely.
    """
    from cflib.utils.power_switch import PowerSwitch

    if log is not None:
        log(f"Power-cycling the drone ({power_down_s:.0f}s off so the deck cold-boots "
            "cleanly; firmware deadman stays armed until reboot).")
    try:
        switch = PowerSwitch(uri)
        try:
            # Equivalent to stm_power_cycle() but with a longer, fully-draining
            # power-down so the AI-deck does not come back wedged.
            switch.stm_power_down()
            try:
                time.sleep(power_down_s)
            finally:
                # Even on Ctrl-C: a drone left with its STM32 off answers only
                # the nRF51's power commands and looks dead to everything else.
                switch.stm_power_up()
        finally:
            switch.close()
    except Exception as exc:
        if log is not None:
            log(f"Power-cycle failed ({exc}); power-cycle the drone before the next run.")
        return False
    return True


# CRTP platform port, version channel: platformservice.c answers with the
# running firmware's device type name, e.g. "Crazyflie 2.1 Brushless".
PLATFORM_VERSION_CHANNEL = 1
GET_DEVICE_TYPE_NAME = 0x02


def device_type_name(uri: str, *, timeout_s: float = 3.0) -> str | None:
    """The running firmware's device type name, or None if no firmware answers.

    A bare link and one request, with no TOC download, so it is quick. A drone in
    its bootloader, or one whose firmware does not boot, returns None.
    """
    import cflib.crtp
    from cflib.crtp.crtpstack import CRTPPacket, CRTPPort

    cflib.crtp.init_drivers()
    link = cflib.crtp.get_link_driver(uri)
    if link is None:
        return None
    request = CRTPPacket()
    request.set_header(CRTPPort.PLATFORM, PLATFORM_VERSION_CHANNEL)
    request.data = (GET_DEVICE_TYPE_NAME,)
    try:
        deadline = time.monotonic() + timeout_s
        next_send = 0.0
        while time.monotonic() < deadline:
            if time.monotonic() >= next_send:
                link.send_packet(request)
                next_send = time.monotonic() + 0.5
            reply = link.receive_packet(0.1)
            if (reply is not None and reply.port == CRTPPort.PLATFORM
                    and reply.channel == PLATFORM_VERSION_CHANNEL
                    and len(reply.data) > 1 and reply.data[0] == GET_DEVICE_TYPE_NAME):
                return bytes(reply.data[1:]).split(b"\0")[0].decode(errors="replace")
    finally:
        link.close()
    return None


#  A power-cycled drone intermittently comes back with its IMU unenumerated:
#  the STM32, radio, parameters and log subsystem are all fine, but acc/gyro/baro
#  read exactly 0.0, `system.selftestPassed` latches 0, and the supervisor then
#  refuses to arm for the life of that boot. Another power cycle clears it. The
#  cause is not established -- a plausible suspect is that stm_power_down() leaves
#  the nRF51 running and may not fully reset the expansion decks, so a deck can
#  hold the sensor bus while the STM32 re-inits, but that is untested.
SELF_TEST_ATTEMPTS = 3
SELF_TEST_SETTLE_S = 6.0


def power_cycle_until_healthy(
    uri: str,
    *,
    attempts: int = SELF_TEST_ATTEMPTS,
    settle_s: float = SELF_TEST_SETTLE_S,
    power_down_s: float = DECK_POWER_DOWN_S,
    log=None,
) -> bool:
    """Power-cycle until the firmware's boot self-test passes.

    Returns False if every attempt came back with a failed self-test, which
    points at hardware rather than the intermittent init race this retries.
    """
    import cflib.crtp
    from cflib.crazyflie import Crazyflie

    from cf_lab.crazyflie.health import read_self_test

    # Registering the radio driver is idempotent, and this is the first place in
    # the reboot path that opens a full link rather than a bare PowerSwitch.
    cflib.crtp.init_drivers()

    self_test_read = False
    for attempt in range(1, attempts + 1):
        if not power_cycle(uri, power_down_s=power_down_s, log=log):
            return False
        if log is not None:
            log(f"Holding {settle_s:.0f}s for onboard gyro calibration. Do not move the drone.")
        time.sleep(settle_s)
        try:
            cf = Crazyflie(rw_cache=toc_cache())
            try:
                connect(cf, uri)
                result = read_self_test(cf)
            finally:
                cf.close_link()
        except Exception as exc:  # noqa: BLE001 - an unreachable drone is a failed check
            result = None
            if log is not None:
                log(f"Could not connect to read the self-test back ({type(exc).__name__}: {exc}).")
        if result is not None and result.passed:
            if log is not None:
                log(f"Boot self-test passed (attempt {attempt}/{attempts}).")
            return True
        if result is not None and result.error is None:
            self_test_read = True
        if log is not None:
            if result is None:
                detail = "unreachable"
            else:
                detail = result.error or f"selftestPassed={result.raw}"
            log(f"Boot self-test did not pass ({detail}) on attempt {attempt}/{attempts}.")
    if log is not None:
        if self_test_read:
            log(
                f"Self-test still failing after {attempts} power-cycles. This is no longer "
                "the intermittent init race; suspect hardware and try the physical power "
                "button before flying."
            )
        else:
            log(
                f"Never read the self-test in {attempts} attempts. Check the radio dongle, "
                "the URI, and that the drone is powered before concluding anything about "
                "its sensors."
            )
    return False


def power_off(uri: str, *, log=None) -> bool:
    """Shut the drone fully down over the radio — nRF51 included.

    Unlike ``power_cycle``, which only drops the STM and decks and leaves the
    nRF51 listening, this is the equivalent of holding the power button: nothing
    stays awake on the radio afterwards, so the drone can only be brought back
    by a physical button press. Use it to end a session, never mid-experiment.
    Failures are reported through ``log`` (if given) and return ``False`` rather
    than raising, so callers can decide whether an un-shut-down drone matters.
    """
    from cflib.utils.power_switch import PowerSwitch

    if log is not None:
        log("Shutting the drone down (nRF51 included; only the power button "
            "brings it back).")
    try:
        switch = PowerSwitch(uri)
        try:
            switch.platform_power_down()
        finally:
            switch.close()
    except Exception as exc:
        if log is not None:
            log(f"Shutdown failed ({exc}); turn the drone off with the power button.")
        return False
    return True
