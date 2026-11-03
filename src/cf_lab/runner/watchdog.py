from __future__ import annotations

import sys
import threading
from collections.abc import Callable


class WatchdogHeartbeat(threading.Thread):
    """Feed the firmware emergency-stop watchdog (a deadman switch).

    Once the first ping reaches the drone, the supervisor demands a fresh one
    every second or it cuts the motors onboard — so the drone stops itself when
    the radio link dies, with no command from the host. The firmware keeps the
    deadman armed until reboot.

    ``rearm`` is an optional second callback run on the same cadence: a runner that
    parks the drone armed-but-idle (motors at 0, never "flying") can use it to
    re-send the arming request, because the supervisor otherwise auto-disarms a
    not-flying drone after its preflight/landed timeout.
    """

    def __init__(
        self,
        ping: Callable[[], None],
        *,
        rearm: Callable[[], None] | None = None,
        period_s: float = 0.25,
    ) -> None:
        super().__init__(daemon=True, name="estop-watchdog-heartbeat")
        self._ping = ping
        self._rearm = rearm
        self._period_s = period_s
        self._stop_event = threading.Event()
        self.pings = 0

    def run(self) -> None:
        while not self._stop_event.wait(self._period_s):
            try:
                self._ping()
                if self._rearm is not None:
                    self._rearm()
            except Exception as exc:
                # The deadman cuts the motors a second later; leave a record of why.
                print(f"[heartbeat] stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
                return
            self.pings += 1

    def stop(self) -> None:
        self._stop_event.set()
