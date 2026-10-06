"""capture_log_window: one firmware log block for a short window, always removed from the drone."""
import types

import cflib.crazyflie.log
import pytest

from cf_lab.crazyflie import capture_log_window


def _drone(monkeypatch, samples, start_error=None):
    """A connected cf whose log blocks deliver ``samples`` on start; returns (cf, blocks)."""
    blocks = []

    class LogConfig:
        def __init__(self, name, period_in_ms):
            self.period_in_ms = period_in_ms
            self.variables, self._callbacks = [], []
            self.data_received_cb = types.SimpleNamespace(add_callback=self._callbacks.append)
            self.deleted = False
            blocks.append(self)

        def add_variable(self, name, cflib_type):
            self.variables.append((name, cflib_type))

        def start(self):
            if start_error is not None:
                raise start_error
            for data in samples:
                for callback in self._callbacks:
                    callback(0, data, self)

        def stop(self):
            pass

        def delete(self):
            self.deleted = True

    monkeypatch.setattr(cflib.crazyflie.log, "LogConfig", LogConfig)
    return types.SimpleNamespace(log=types.SimpleNamespace(add_config=lambda config: None)), blocks


def test_capture_returns_the_requested_variables_and_deletes_its_block(monkeypatch):
    cf, blocks = _drone(monkeypatch, [{"pm.vbat": 3.9, "gyro.z": 0.1, "other": 1}] * 2)

    rows = capture_log_window(cf, [("pm.vbat", "float"), ("gyro.z", "float")], seconds=0, period_ms=20)

    assert rows == ({"pm.vbat": 3.9, "gyro.z": 0.1},) * 2
    (block,) = blocks
    assert block.period_in_ms == 20
    assert block.variables == [("pm.vbat", "float"), ("gyro.z", "float")]
    assert block.deleted


def test_capture_deletes_its_block_when_starting_it_fails(monkeypatch):
    cf, blocks = _drone(monkeypatch, [], start_error=AttributeError("link is down"))

    with pytest.raises(AttributeError):
        capture_log_window(cf, [("pm.vbat", "float")], seconds=0, period_ms=10)

    assert blocks[0].deleted
