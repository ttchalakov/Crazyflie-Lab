"""Base drone-session classes shared by the example runners.

A "session" wraps the few drone operations a runner needs (prepare the
estimator, emergency-stop) behind one object so the flight logic reads the same
whether it drives a real radio link or the in-memory fake backend. Each demo
subclasses these to add its own verbs (arm, goto, set_led/set_box).

The classes expose *independent verbs* only — they never chain them into a flight
sequence. The runner's author decides when (or whether) to arm, take off, stream
setpoints, land, or disarm; nothing here assumes the usual liftoff->run->land
order.
"""

from __future__ import annotations

import time

from .watchdog import WatchdogHeartbeat


class RadioSessionBase:
    """Drive a real Crazyflie over the open cflib radio link.

    Tracks radio-link loss (``link_lost``) and provides the estimator setup that
    every mocap demo needs. Subclasses add the demo's own commands and extend
    ``emergency_stop`` to clean up demo state (LEDs restored, ...).
    """

    def __init__(self, backend) -> None:
        self._cf = backend.crazyflie
        self.link_lost = False
        self._cf.connection_lost.add_callback(self._on_link_lost)

    def _on_link_lost(self, _uri: str, message: str) -> None:
        self.link_lost = True
        print(f"[link] lost: {message}", flush=True)

    def prepare(self) -> None:
        # Extpose is only fused by the Kalman estimator; the default
        # complementary estimator reports barometric altitude as z. Select
        # Kalman, then reset it once extpose is already flowing so it converges
        # onto the zeroed mocap frame before the drone is used.
        self._cf.param.set_value("stabilizer.estimator", "2")
        time.sleep(0.3)
        self._cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self._cf.param.set_value("kalman.resetEstimation", "0")
        time.sleep(1.5)

    def emergency_stop(self) -> None:
        self._cf.supervisor.send_emergency_stop()


class FlightRadioSession(RadioSessionBase):
    """A ``RadioSessionBase`` plus the arm/deadman/setpoint verbs flying demos share.

    Adds the boilerplate every flying runner needs — arming the onboard radio-loss
    deadman, the zero-thrust unlock, stop/disarm, and the heartbeat lifecycle.
    These are independent verbs the runner calls in whatever order its flight
    logic needs; this class imposes no sequence.

    ``estimator`` selects how :meth:`prepare` sets up state estimation:
    ``"kalman"`` (default, fuses extpose for flights with mocap) or
    ``"complementary"`` (baro z, no horizontal position — for flights without mocap).
    """

    def __init__(self, backend, *, estimator: str = "kalman") -> None:
        super().__init__(backend)
        self._estimator = estimator
        self._heartbeat: WatchdogHeartbeat | None = None

    def prepare(self) -> None:
        if self._estimator == "complementary":
            # Fuses baro into z and does NOT estimate horizontal position, so
            # altitude hold can never get x/y feedback (no flyaway) — the drone
            # just drifts. No extpose reset is needed (there is no mocap).
            self._cf.param.set_value("stabilizer.estimator", "1")
            time.sleep(2.0)
            return
        super().prepare()

    def arm(self) -> None:
        self.stop_heartbeat()  # arming again must not leave the old heartbeat running
        self._cf.supervisor.send_arming_request(True)
        # Arm the onboard deadman: from the first ping on, one second of radio
        # silence makes the drone cut its own motors.
        self._cf.supervisor.send_emergency_stop_watchdog()
        self._heartbeat = WatchdogHeartbeat(self._cf.supervisor.send_emergency_stop_watchdog)
        self._heartbeat.start()
        time.sleep(0.5)

    def unlock(self) -> None:
        # The firmware ignores thrust until it has seen a zero-thrust setpoint;
        # send one so the first real setpoint is honored.
        self._cf.commander.send_setpoint(0.0, 0.0, 0, 0)

    def stop(self) -> None:
        self._cf.commander.send_stop_setpoint()

    def disarm(self) -> None:
        self._cf.supervisor.send_arming_request(False)

    def stop_heartbeat(self) -> None:
        if self._heartbeat is not None:
            self._heartbeat.stop()
            self._heartbeat = None

    def emergency_stop(self) -> None:
        super().emergency_stop()
        try:
            self._cf.commander.send_stop_setpoint()
        except Exception:
            pass


class FakeSessionBase:
    """Print-only session backed by the in-memory fake backend."""

    def __init__(self, backend) -> None:
        self.backend = backend
        self.link_lost = False

    def prepare(self) -> None:
        pass

    def emergency_stop(self) -> None:
        self.backend.emergency_stop()


class FlightFakeSession(FakeSessionBase):
    """Print-only counterpart to :class:`FlightRadioSession` for dry runs.

    Mirrors the flight verbs as no-ops (apart from arm/disarm/emergency-stop,
    which drive the in-memory backend so the gate/estop logic stays observable).
    Demos subclass this and add their own no-op verbs (goto, ...).
    """

    def arm(self) -> None:
        self.backend.arm()

    def unlock(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def disarm(self) -> None:
        self.backend.disarm()

    def stop_heartbeat(self) -> None:
        pass
