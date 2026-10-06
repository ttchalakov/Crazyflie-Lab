"""Radio power control: power-cycle, shutdown, reboot-until-healthy and the firmware liveness probe."""
import cflib.crazyflie
import cflib.crtp
import cflib.utils.power_switch
import pytest
from cflib.crtp.crtpstack import CRTPPacket, CRTPPort

import cf_lab.backends.crazyflie as backend_mod
from cf_lab.backends.crazyflie import (
    GET_DEVICE_TYPE_NAME,
    PLATFORM_VERSION_CHANNEL,
    device_type_name,
    power_cycle,
    power_cycle_until_healthy,
    power_off,
)
from cf_lab.crazyflie.health import SELF_TEST_PARAMETER

URI = "radio://0/80/2M/E7E7E7E7E7"


@pytest.fixture
def switch(monkeypatch):
    """Fake PowerSwitch; ``switch.fail[method] = exc`` makes that call raise."""
    events: list[str] = []
    fail: dict = {}

    class FakeSwitch:
        def __init__(self, uri):
            if "init" in fail:
                raise fail["init"]
            events.append(f"open {uri}")

        def __getattr__(self, name):
            def call():
                events.append(name)
                if name in fail:
                    raise fail[name]

            return call

    monkeypatch.setattr(cflib.utils.power_switch, "PowerSwitch", FakeSwitch)
    FakeSwitch.events = events
    FakeSwitch.fail = fail
    return FakeSwitch


def test_power_cycle_holds_power_off_then_restores_it(switch, clock, monkeypatch):
    monkeypatch.setattr(backend_mod, "time", clock.module())

    assert power_cycle(URI) is True

    assert switch.events == [f"open {URI}", "stm_power_down", "stm_power_up", "close"]
    assert clock.sleeps == [4.0]  # longer than cflib's 1 s so the AI-deck cold-boots


def test_power_cycle_restores_power_when_interrupted_during_the_hold(switch, monkeypatch):
    def interrupted(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(backend_mod, "time", type("T", (), {"sleep": staticmethod(interrupted)}))

    with pytest.raises(KeyboardInterrupt):
        power_cycle(URI)

    assert switch.events[-2:] == ["stm_power_up", "close"]


@pytest.mark.parametrize("failing", ["init", "stm_power_down", "stm_power_up"])
def test_power_cycle_failure_returns_false_and_releases_the_radio(switch, clock, monkeypatch, failing):
    monkeypatch.setattr(backend_mod, "time", clock.module())
    switch.fail[failing] = OSError("Crazyradio not found")
    lines = []

    assert power_cycle(URI, log=lines.append) is False

    assert lines and "failed" in lines[-1]
    if failing != "init":
        assert switch.events[-1] == "close"


@pytest.mark.parametrize("failing", [None, "platform_power_down"])
def test_power_off_reports_failure_and_always_releases_the_radio(switch, failing):
    if failing:
        switch.fail[failing] = OSError("no ack")

    assert power_off(URI) is (failing is None)

    assert switch.events[-2:] == ["platform_power_down", "close"]


@pytest.fixture
def reboot(monkeypatch, clock, fake_cf_class, fake_param_class):
    """power_cycle_until_healthy with radio, clock and Crazyflie faked; ``boots`` scripts each boot.

    Each entry of ``boots`` is the self-test value the next boot latches, an
    exception that connecting raises, or ``{"param_error": exc}`` for a link that
    connects but whose parameter read fails.
    """
    state = {"boots": [], "cycles": 0, "cycle_ok": True, "cfs": []}
    monkeypatch.setattr(backend_mod, "time", clock.module())
    monkeypatch.setattr(cflib.crtp, "init_drivers", lambda *a, **k: None)

    def fake_power_cycle(uri, **_kwargs):
        state["cycles"] += 1
        return state["cycle_ok"]

    def make_cf(rw_cache=None):
        boot = state["boots"][state["cycles"] - 1]
        if isinstance(boot, dict):
            param = fake_param_class(error=boot["param_error"])
        else:
            param = fake_param_class({SELF_TEST_PARAMETER: boot})
        cf = fake_cf_class(rw_cache, param=param)
        cf.boot = boot
        state["cfs"].append(cf)
        return cf

    def fake_connect(cf, uri, **_kwargs):
        if isinstance(cf.boot, Exception):
            raise cf.boot

    monkeypatch.setattr(backend_mod, "power_cycle", fake_power_cycle)
    monkeypatch.setattr(backend_mod, "connect", fake_connect)
    monkeypatch.setattr(cflib.crazyflie, "Crazyflie", make_cf)
    return state


@pytest.mark.parametrize(
    ("boots", "healthy", "cycles"),
    [
        (["1"], True, 1),
        (["0", "1"], True, 2),
        ([TimeoutError("no TOC"), "1"], True, 2),
        (["0", "0", "0"], False, 3),
        ([ConnectionError("no ack")] * 3, False, 3),
    ],
)
def test_reboot_repeats_power_cycles_until_the_self_test_passes(reboot, boots, healthy, cycles):
    reboot["boots"] = boots
    lines = []

    assert power_cycle_until_healthy(URI, log=lines.append) is healthy

    assert reboot["cycles"] == cycles
    assert all(cf.closed == 1 for cf in reboot["cfs"])  # every probe link is released


@pytest.mark.parametrize(
    ("boots", "advice"),
    [(["0", "0", "0"], "suspect hardware"),
     ([ConnectionError("no ack")] * 3, "Never read the self-test")],
)
def test_reboot_failure_advice_separates_bad_sensors_from_an_unreachable_drone(reboot, boots, advice):
    reboot["boots"] = boots
    lines = []

    power_cycle_until_healthy(URI, log=lines.append)

    assert advice in lines[-1]


def test_reboot_does_not_blame_hardware_when_the_self_test_was_never_read(reboot):
    reboot["boots"] = [{"param_error": TimeoutError("param values never arrived")}] * 3
    lines = []

    assert power_cycle_until_healthy(URI, log=lines.append) is False

    assert "suspect hardware" not in lines[-1]


def test_reboot_stops_at_once_when_the_power_cycle_fails(reboot):
    reboot["boots"] = ["1"]
    reboot["cycle_ok"] = False

    assert power_cycle_until_healthy(URI) is False

    assert reboot["cycles"] == 1
    assert reboot["cfs"] == []


def test_reboot_waits_for_gyro_calibration_before_reading(reboot, clock):
    reboot["boots"] = ["1"]

    power_cycle_until_healthy(URI, settle_s=6.0)

    assert clock.sleeps == [6.0]


def _packet(port, channel, data):
    packet = CRTPPacket()
    packet.set_header(port, channel)
    packet.data = data
    return packet


class _Link:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.closed = False

    def send_packet(self, packet):
        self.sent.append(packet)

    def receive_packet(self, _timeout):
        return self.replies.pop(0) if self.replies else None

    def close(self):
        self.closed = True


@pytest.fixture
def link(monkeypatch):
    holder = {}
    monkeypatch.setattr(cflib.crtp, "init_drivers", lambda *a, **k: None)
    monkeypatch.setattr(cflib.crtp, "get_link_driver", lambda uri: holder.get("link"))
    return holder


def test_device_type_name_skips_unrelated_packets(link):
    name = b"Crazyflie 2.1 Brushless\0garbage"
    link["link"] = _Link([
        None,
        _packet(CRTPPort.LOGGING, 2, (GET_DEVICE_TYPE_NAME, 0x41)),  # other port
        _packet(CRTPPort.PLATFORM, 0, (GET_DEVICE_TYPE_NAME, 0x41)),  # other channel
        _packet(CRTPPort.PLATFORM, PLATFORM_VERSION_CHANNEL, (0x01, 0x41)),  # other command
        _packet(CRTPPort.PLATFORM, PLATFORM_VERSION_CHANNEL, (GET_DEVICE_TYPE_NAME,)),  # empty answer
        _packet(CRTPPort.PLATFORM, PLATFORM_VERSION_CHANNEL, (GET_DEVICE_TYPE_NAME, *name)),
    ])

    assert device_type_name(URI, timeout_s=1.0) == "Crazyflie 2.1 Brushless"
    assert link["link"].closed
    assert tuple(link["link"].sent[0].data) == (GET_DEVICE_TYPE_NAME,)


def test_device_type_name_is_none_when_no_firmware_answers(link):
    link["link"] = _Link([])

    assert device_type_name(URI, timeout_s=0.05) is None
    assert link["link"].closed


def test_device_type_name_is_none_without_a_link_driver(link):
    assert device_type_name(URI, timeout_s=0.05) is None
