"""KeyboardMonitor: the SPACE kill switch reads a real terminal and is inert without one."""
import io
import os
import termios
import time

import pytest

from cf_lab.runner.keyboard import KeyboardMonitor


def test_monitor_is_inert_without_a_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(" "))

    with KeyboardMonitor() as keyboard:
        assert keyboard.pressed() == ""


@pytest.fixture
def terminal(monkeypatch):
    """stdin replaced by the slave side of a pseudo-terminal; yields the master fd."""
    master, slave = os.openpty()
    stdin = os.fdopen(slave, "r", buffering=1)
    monkeypatch.setattr("sys.stdin", stdin)
    yield master, slave
    stdin.close()
    os.close(master)


def _poll(keyboard, want):
    keys = ""
    for _ in range(50):
        keys += keyboard.pressed()
        if len(keys) >= want:
            break
        time.sleep(0.002)
    return keys


def test_monitor_reads_space_without_enter_and_restores_the_terminal(terminal):
    master, slave = terminal
    before = termios.tcgetattr(slave)

    with KeyboardMonitor() as keyboard:
        assert termios.tcgetattr(slave)[3] & termios.ICANON == 0  # cbreak: no Enter needed
        assert keyboard.pressed() == ""
        os.write(master, b" ")
        assert _poll(keyboard, 1) == " "
        assert keyboard.pressed() == ""

    assert termios.tcgetattr(slave) == before


def test_monitor_sees_space_typed_right_after_another_key(terminal):
    master, _slave = terminal

    with KeyboardMonitor() as keyboard:
        os.write(master, b"a ")
        assert " " in _poll(keyboard, 2)
