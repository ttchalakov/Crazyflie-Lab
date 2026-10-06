"""Fakes for the cflib Crazyflie object, the radio power switch and the host clock."""
from __future__ import annotations

import threading
import types

import pytest
from cflib.utils.callbacks import Caller


class FakeParam:
    is_updated = True  # cflib's flag: every parameter value has arrived

    def __init__(self, values=None, error: Exception | None = None):
        self.values = dict(values or {})
        self.error = error
        self.writes: list[tuple[str, str]] = []

    def get_value(self, name, timeout=None):
        if self.error is not None:
            raise self.error
        return self.values[name]

    def set_value(self, name, value):
        self.writes.append((name, str(value)))


class Recorder:
    """Records every method call in one shared, ordered, thread-safe log."""

    def __init__(self, log: list, prefix: str, fail: dict | None = None):
        self._log = log
        self._prefix = prefix
        self._fail = fail if fail is not None else {}
        self._lock = threading.Lock()

    def __getattr__(self, name):
        def call(*args):
            with self._lock:
                self._log.append((f"{self._prefix}.{name}", args))
            error = self._fail.get(name)
            if error is not None:
                raise error

        return call


class FakeCrazyflie:
    """The parts of cflib's Crazyflie the area touches; link events are driven by ``on_open``.

    ``on_open(cf, uri)`` runs inside ``open_link`` and decides what the link reports,
    e.g. ``cf.connected.call(uri)`` or ``cf.connection_failed.call(uri, "no ack")``.
    """

    def __init__(self, rw_cache=None, *, on_open=None, param=None):
        self.rw_cache = rw_cache
        self.on_open = on_open
        self.connected = Caller()
        self.connection_failed = Caller()
        self.connection_lost = Caller()
        self.console = types.SimpleNamespace(receivedChar=Caller())
        self.param = param if param is not None else FakeParam()
        self.calls: list = []
        self.failures: dict = {}
        self.supervisor = Recorder(self.calls, "supervisor", self.failures)
        self.commander = Recorder(self.calls, "commander", self.failures)
        self.extpos = Recorder(self.calls, "extpos", self.failures)
        self.opened: list[str] = []
        self.closed = 0

    def open_link(self, uri):
        self.opened.append(uri)
        if self.on_open is not None:
            self.on_open(self, uri)

    def close_link(self):
        self.closed += 1

    def names(self, prefix: str = "") -> list[str]:
        return [name for name, _ in self.calls if name.startswith(prefix)]


class FakeClock:
    """``time.monotonic``/``time.sleep`` stand-in: sleeping advances the clock instantly."""

    def __init__(self, start: float = 1000.0):
        self.now = start
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def module(self):
        return types.SimpleNamespace(monotonic=self.monotonic, sleep=self.sleep)


@pytest.fixture
def fake_cf_class():
    return FakeCrazyflie


@pytest.fixture
def fake_param_class():
    return FakeParam


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def connected_on_open():
    def on_open(cf, uri):
        cf.connected.call(uri)

    return on_open
