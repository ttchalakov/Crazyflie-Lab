"""Radio sessions and the emergency-stop deadman heartbeat, against a fake Crazyflie."""
import time
import types

import pytest

import cf_lab.runner.session as session_mod
from cf_lab.runner.control import RadioLinkLost, poll_kill
from cf_lab.runner.session import FlightRadioSession, RadioSessionBase
from cf_lab.runner.watchdog import WatchdogHeartbeat

PERIOD_S = 0.01


def _wait_for(predicate, timeout_s=1.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


@pytest.fixture
def cf(fake_cf_class):
    return fake_cf_class()


@pytest.fixture
def heartbeats():
    """Every heartbeat a session starts; all are stopped after the test, even leaked ones."""
    started: list[WatchdogHeartbeat] = []
    yield started
    for heartbeat in started:
        heartbeat.stop()


@pytest.fixture
def flight(cf, heartbeats, monkeypatch):
    """A FlightRadioSession with instant sleeps and a fast heartbeat."""

    def make_heartbeat(ping):
        heartbeat = WatchdogHeartbeat(ping, period_s=PERIOD_S)
        heartbeats.append(heartbeat)
        return heartbeat

    monkeypatch.setattr(session_mod, "time", types.SimpleNamespace(sleep=lambda _s: None))
    monkeypatch.setattr(session_mod, "WatchdogHeartbeat", make_heartbeat)
    session = FlightRadioSession(types.SimpleNamespace(crazyflie=cf))
    yield session
    session.stop_heartbeat()


def _pings(cf):
    return cf.names().count("supervisor.send_emergency_stop_watchdog")


def test_radio_loss_callback_trips_the_kill_check(cf):
    session = RadioSessionBase(types.SimpleNamespace(crazyflie=cf))
    keyboard = types.SimpleNamespace(pressed=lambda: "")
    streamer = types.SimpleNamespace(error=None, check_poses=lambda: None)
    poll_kill(keyboard, streamer, session)

    cf.connection_lost.call("radio://0/80/2M/E7", "Too many packets lost")

    with pytest.raises(RadioLinkLost):
        poll_kill(keyboard, streamer, session)


def test_prepare_selects_kalman_and_pulses_its_reset(cf, flight):
    flight.prepare()

    assert cf.param.writes == [
        ("stabilizer.estimator", "2"),
        ("kalman.resetEstimation", "1"),
        ("kalman.resetEstimation", "0"),
    ]


def test_arm_requests_arming_and_pings_the_deadman_before_returning(cf, flight):
    flight.arm()

    names = cf.names("supervisor.")
    assert names[0] == "supervisor.send_arming_request"
    assert cf.calls[0][1] == (True,)
    assert names[1] == "supervisor.send_emergency_stop_watchdog"  # first ping arms the deadman


def test_heartbeat_keeps_the_deadman_fed_until_stopped(cf, flight):
    flight.arm()
    assert _wait_for(lambda: _pings(cf) >= 4)
    heartbeat = flight._heartbeat

    flight.stop_heartbeat()
    heartbeat.join(1.0)
    stopped_at = _pings(cf)
    time.sleep(3 * PERIOD_S)

    assert not heartbeat.is_alive()
    assert _pings(cf) == stopped_at


def test_arming_twice_leaves_no_heartbeat_running_after_stop(flight, heartbeats):
    flight.arm()
    flight.arm()  # e.g. re-arming after the supervisor auto-disarmed

    flight.stop_heartbeat()

    for heartbeat in heartbeats:
        heartbeat.join(0.2)
    assert not any(heartbeat.is_alive() for heartbeat in heartbeats)


def test_emergency_stop_cuts_motors_then_sends_a_stop_setpoint(cf, flight):
    flight.emergency_stop()

    assert cf.names() == ["supervisor.send_emergency_stop", "commander.send_stop_setpoint"]


def test_emergency_stop_survives_a_failing_stop_setpoint(cf, flight):
    cf.failures["send_stop_setpoint"] = OSError("link closed")

    flight.emergency_stop()

    assert cf.names()[0] == "supervisor.send_emergency_stop"


def test_unlock_sends_the_zero_thrust_setpoint(cf, flight):
    flight.unlock()

    name, (roll, pitch, yawrate, thrust) = cf.calls[0]
    assert name == "commander.send_setpoint"
    assert thrust == 0


def test_heartbeat_pings_on_its_period_until_stopped():
    pings = []
    heartbeat = WatchdogHeartbeat(lambda: pings.append(time.monotonic()), period_s=PERIOD_S)
    heartbeat.start()
    assert _wait_for(lambda: len(pings) >= 5)

    heartbeat.stop()
    heartbeat.join(1.0)
    count = len(pings)
    time.sleep(3 * PERIOD_S)

    assert not heartbeat.is_alive()
    assert len(pings) == count
    gaps = [b - a for a, b in zip(pings, pings[1:])]
    assert min(gaps) >= PERIOD_S * 0.5  # never a burst that would hide a stalled link


def test_heartbeat_ends_when_a_ping_fails_so_the_onboard_deadman_takes_over():
    calls = []

    def ping():
        calls.append(1)
        if len(calls) == 2:
            raise OSError("radio gone")

    heartbeat = WatchdogHeartbeat(ping, period_s=PERIOD_S)
    heartbeat.start()
    heartbeat.join(1.0)

    assert not heartbeat.is_alive()
    assert len(calls) == 2
