"""The boot self-test gate and the rate PID parameter snapshot."""
import threading
import types

import pytest
from cflib.crazyflie.param import Param
from cflib.utils.callbacks import Caller

from cf_lab.crazyflie.health import SELF_TEST_PARAMETER, read_self_test
from cf_lab.crazyflie.parameters import RATE_PID_PARAMETER_NAMES, read_rate_pid_parameters


@pytest.mark.parametrize(
    ("raw", "passed"),
    [("1", True), (" 1 ", True), ("1\n", True), (1, True), ("0", False), ("", False), (None, False)],
)
def test_self_test_passes_only_on_a_latched_one(fake_cf_class, fake_param_class, raw, passed):
    cf = fake_cf_class(param=fake_param_class({SELF_TEST_PARAMETER: raw}))

    result = read_self_test(cf)

    assert result.passed is passed
    assert result.error is None


@pytest.mark.parametrize("error", [TimeoutError("no answer"), KeyError(SELF_TEST_PARAMETER)])
def test_unreadable_self_test_fails_closed_with_the_reason(fake_cf_class, fake_param_class, error):
    cf = fake_cf_class(param=fake_param_class(error=error))

    result = read_self_test(cf)

    assert result.passed is False
    assert result.raw is None
    assert type(error).__name__ in result.error


@pytest.mark.parametrize("timeout_s", [0.0, -1.0])
def test_self_test_rejects_a_non_positive_timeout(fake_cf_class, timeout_s):
    with pytest.raises(ValueError):
        read_self_test(fake_cf_class(), timeout_s=timeout_s)


def test_rate_pid_snapshot_reads_every_gain_as_float(fake_param_class):
    # cflib returns parameter values as strings.
    values = {name: f"{index}.5" for index, name in enumerate(RATE_PID_PARAMETER_NAMES)}

    gains = read_rate_pid_parameters(fake_param_class(values))

    assert gains == {name: index + 0.5 for index, name in enumerate(RATE_PID_PARAMETER_NAMES)}


@pytest.fixture
def downloading():
    """A Crazyflie with cflib's real Param, connected but still downloading parameter values."""
    cf = types.SimpleNamespace(
        disconnected=Caller(),
        connection_requested=Caller(),
        is_called_by_incoming_handler_thread=lambda: False,
    )
    cf.param = Param(cf)
    yield cf
    cf.param._initialized.set()  # release a read still blocked inside cflib


def test_self_test_read_gives_up_at_its_timeout_while_values_download(downloading):
    done = threading.Event()

    def read():
        read_self_test(downloading, timeout_s=0.05)
        done.set()

    threading.Thread(target=read, daemon=True).start()

    assert done.wait(0.5)

