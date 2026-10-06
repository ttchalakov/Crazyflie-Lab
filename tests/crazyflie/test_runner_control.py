"""Runner stop conditions, kill-aware waits, ramped setpoint streaming, the mocap gate and the run log."""
import types

import pytest

import cf_lab.runner.control as control
from cf_lab.mocap import MocapBodyLost
from cf_lab.runner.control import (
    ExtposeStreamFailed,
    KillSwitchEngaged,
    RadioLinkLost,
    kill_aware_wait,
    make_run_logger,
    mocap_gate_required,
    mocap_settle_and_gate,
    poll_kill,
    stream_segment,
    tag_console_line,
)


class Keyboard:
    def __init__(self, keys=""):
        self.keys = keys

    def pressed(self):
        return self.keys


def _streamer(error=None, sent=0, lost=None):
    def check_poses():
        if lost is not None:
            raise lost

    return types.SimpleNamespace(error=error, sent=sent, check_poses=check_poses)


def _session(link_lost=False):
    return types.SimpleNamespace(link_lost=link_lost)


@pytest.fixture
def fake_time(clock, monkeypatch):
    monkeypatch.setattr(control, "time", clock.module())
    return clock


@pytest.mark.parametrize(
    ("keys", "error", "link_lost", "raised"),
    [
        (" ", RuntimeError("mocap died"), True, KillSwitchEngaged),  # operator first
        ("x \n", None, False, KillSwitchEngaged),  # SPACE among other keys
        ("", RuntimeError("mocap died"), True, ExtposeStreamFailed),
        ("", None, True, RadioLinkLost),
        ("q", None, False, None),
    ],
)
def test_poll_kill_raises_the_highest_priority_stop(keys, error, link_lost, raised):
    if raised is None:
        poll_kill(Keyboard(keys), _streamer(error), _session(link_lost))
        return
    with pytest.raises(raised):
        poll_kill(Keyboard(keys), _streamer(error), _session(link_lost))


def test_lost_mocap_is_raised_after_the_radio_check():
    # With the radio gone the drone cannot land on command, so radio loss wins.
    lost = MocapBodyLost("no pose for 1.2 s: the mocap system is streaming, but the rigid body is not tracked")

    with pytest.raises(RadioLinkLost):
        poll_kill(Keyboard(), _streamer(lost=lost), _session(link_lost=True))
    with pytest.raises(MocapBodyLost):
        poll_kill(Keyboard(), _streamer(lost=lost), _session())


def test_stream_failure_chains_the_streamer_error():
    cause = ConnectionResetError("NatNet gone")

    with pytest.raises(ExtposeStreamFailed) as info:
        poll_kill(Keyboard(), _streamer(cause), _session())

    assert info.value.__cause__ is cause


def test_kill_aware_wait_waits_the_full_duration(fake_time):
    start = fake_time.now

    kill_aware_wait(Keyboard(), _streamer(), _session(), 2.0)

    assert fake_time.now - start == pytest.approx(2.0, abs=0.06)


def test_kill_aware_wait_raises_as_soon_as_the_link_drops(clock, monkeypatch):
    session = _session()
    start = clock.now

    def sleep(seconds):
        clock.sleep(seconds)
        session.link_lost = clock.now - start >= 0.5

    monkeypatch.setattr(control, "time", types.SimpleNamespace(monotonic=clock.monotonic, sleep=sleep))

    with pytest.raises(RadioLinkLost):
        kill_aware_wait(Keyboard(), _streamer(), session, 10.0)

    assert clock.now - start < 0.6


def test_stream_segment_ramps_from_zero_to_exactly_one(fake_time):
    fractions = []

    stream_segment(1.0, fractions.append, keyboard=Keyboard(), streamer=_streamer(),
                   session=_session(), rate_hz=50.0)

    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions)
    assert len(fractions) == pytest.approx(51, abs=1)


@pytest.mark.parametrize("duration_s", [0.0, -1.0])
def test_stream_segment_without_duration_sends_one_final_setpoint(fake_time, duration_s):
    fractions = []

    stream_segment(duration_s, fractions.append, keyboard=Keyboard(), streamer=_streamer(),
                   session=_session(), rate_hz=50.0)

    assert fractions == [1.0]


def test_stream_segment_sends_nothing_after_space(fake_time):
    keyboard = Keyboard()
    fractions = []

    def command(fraction):
        fractions.append(fraction)
        if len(fractions) == 5:
            keyboard.keys = " "

    with pytest.raises(KillSwitchEngaged):
        stream_segment(2.0, command, keyboard=keyboard, streamer=_streamer(),
                       session=_session(), rate_hz=100.0)

    assert len(fractions) == 5


def test_stream_segment_holds_its_rate_when_sending_takes_time(fake_time):
    fractions = []

    def command(fraction):
        fractions.append(fraction)
        fake_time.now += 0.004  # a 4 ms setpoint send at 100 Hz

    stream_segment(1.0, command, keyboard=Keyboard(), streamer=_streamer(),
                   session=_session(), rate_hz=100.0)

    assert len(fractions) >= 95


@pytest.mark.parametrize(
    ("settle_s", "rate_hz", "required"),
    [(2.0, 100.0, 50), (3.0, 120.0, 90), (0.5, 20.0, 10), (0.0, 100.0, 10)],
)
def test_mocap_gate_needs_a_quarter_of_the_nominal_samples_and_at_least_ten(settle_s, rate_hz, required):
    assert mocap_gate_required(settle_s, rate_hz) == required


@pytest.mark.parametrize(("sent", "passed"), [(0, False), (49, False), (50, True), (400, True)])
def test_mocap_settle_gate_blocks_arming_without_enough_samples(fake_time, sent, passed):
    cfg = types.SimpleNamespace(mocap=types.SimpleNamespace(stream_rate_hz=100.0))
    lines = []

    result = mocap_settle_and_gate(_streamer(sent=sent), keyboard=Keyboard(), session=_session(),
                                   cfg=cfg, settle_s=2.0, log=lines.append)

    assert result is passed
    assert lines[-1].startswith("Mocap gate passed" if passed else "ARMING BLOCKED")


def test_mocap_settle_gate_lets_space_abort_the_settle(fake_time):
    cfg = types.SimpleNamespace(mocap=types.SimpleNamespace(stream_rate_hz=100.0))

    with pytest.raises(KillSwitchEngaged):
        mocap_settle_and_gate(_streamer(sent=1000), keyboard=Keyboard(" "), session=_session(),
                              cfg=cfg, settle_s=2.0, log=lambda _line: None)


@pytest.mark.parametrize(
    ("line", "tagged"),
    [
        ("CPX: GAP8: frame 12 sent", ("GAP8", "frame 12 sent")),
        ("CPX: ESP32: wifi up", ("ESP32", "wifi up")),
        ("SYS: Crazyflie 2.1 is up", ("STM32", "SYS: Crazyflie 2.1 is up")),
        ("CPX: other: text", ("STM32", "CPX: other: text")),
        ("", ("STM32", "")),
    ],
)
def test_console_lines_are_tagged_by_their_source_chip(line, tagged):
    assert tag_console_line(line) == tagged


def test_run_log_records_host_and_firmware_lines_in_order(tmp_path, capsys):
    log, console_sink, close = make_run_logger(tmp_path / "run")
    log("arming")
    console_sink("CPX: GAP8: inference 3 ms")
    console_sink("SUP: armed")
    close()

    lines = (tmp_path / "run" / "debug_log.txt").read_text().splitlines()
    tagged = [line.split("] ", 1)[1] for line in lines[1:]]
    assert tagged == ["[HOST] arming", "[GAP8] inference 3 ms", "[STM32] SUP: armed"]
    printed = capsys.readouterr().out
    assert "arming" in printed and "inference" not in printed  # firmware lines stay in the file


def test_run_log_echoes_firmware_lines_when_asked(tmp_path, capsys):
    _log, console_sink, close = make_run_logger(tmp_path, echo_console=True)
    console_sink("CPX: ESP32: wifi up")
    close()

    assert "[ESP32] wifi up" in capsys.readouterr().out
