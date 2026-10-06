"""Reusable building blocks for the example host runners (examples/*/host/run.py).

Import everything from here:

    from cf_lab.runner import KeyboardMonitor, WatchdogHeartbeat, poll_kill
"""

from .control import (
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
from .keyboard import KeyboardMonitor
from .session import (
    FakeSessionBase,
    FlightFakeSession,
    FlightRadioSession,
    RadioSessionBase,
)
from .watchdog import WatchdogHeartbeat

__all__ = [
    "ExtposeStreamFailed",
    "FakeSessionBase",
    "FlightFakeSession",
    "FlightRadioSession",
    "KeyboardMonitor",
    "KillSwitchEngaged",
    "RadioLinkLost",
    "RadioSessionBase",
    "WatchdogHeartbeat",
    "kill_aware_wait",
    "make_run_logger",
    "mocap_gate_required",
    "mocap_settle_and_gate",
    "poll_kill",
    "stream_segment",
    "tag_console_line",
]
