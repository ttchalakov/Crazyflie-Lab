from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from cf_lab.mocap.types import PoseSample


class MocapSampleSource(Protocol):
    def samples(self, *, start_s: float = 0.0, dt_s: float = 0.01) -> Iterator[PoseSample]: ...


class ExtposeBackend(Protocol):
    def send_extpose(self, pose: PoseSample) -> None: ...


class MocapLost(Exception):
    """No new pose has reached the drone for the pose watch's timeout.

    ExtposeStreamer.check_poses() raises it, through a runner's poll_kill, once
    the runner has started the watch; the runner decides what follows.
    """


class MocapBodyLost(MocapLost):
    """The mocap system is still streaming, but the rigid body is not tracked."""


class MocapConnectionLost(MocapLost):
    """Nothing arrives from the mocap system: it stopped, the network dropped, or
    the source's receive thread died."""


@dataclass(frozen=True)
class MocapObservation:
    """One pose observed at the live source and at the radio uplink.

    ``PoseSample.timestamp_s`` belongs to the mocap source and may use a clock
    unrelated to the host. These two timestamps deliberately use the host's
    monotonic clock, so freshness and forwarding latency can be checked without
    guessing how the source clock is synchronized. ``forwarded_monotonic_s``
    is ``None`` when this source pose was intentionally decimated by the
    extpose rate cap.
    """

    pose: PoseSample
    received_monotonic_s: float
    forwarded_monotonic_s: float | None


def iter_pose_samples(
    source: MocapSampleSource | Iterable[PoseSample],
    *,
    start_s: float,
    dt_s: float,
) -> Iterator[PoseSample]:
    """Adapt a sample source or plain iterable into one pose iterator."""
    if hasattr(source, "samples"):
        return source.samples(start_s=start_s, dt_s=dt_s)
    return iter(source)


def first_pose(source: MocapSampleSource, *, timeout_s: float) -> PoseSample | None:
    """The first pose ``source`` delivers within ``timeout_s``, else None.

    A live source blocks until its rigid body is tracked (NatNet skips untracked
    bodies), so the read runs on a helper thread; on a timeout the source is
    closed to release it. Errors the source raises while connecting, such as an
    unknown rigid-body name, are raised here.
    """
    result: dict[str, object] = {}
    done = threading.Event()

    def read() -> None:
        try:
            result["pose"] = next(iter_pose_samples(source, start_s=0.0, dt_s=0.01), None)
        except BaseException as exc:  # handed to the caller
            result["error"] = exc
        done.set()

    threading.Thread(target=read, name="mocap-first-pose", daemon=True).start()
    if not done.wait(timeout_s):
        close = getattr(source, "close", None)
        if close is not None:
            close()
        return None
    if "error" in result:
        raise result["error"]  # type: ignore[misc]
    return result["pose"]  # type: ignore[return-value]


class ExtposeStreamer(threading.Thread):
    """Forward pose samples to an extpose backend from a background thread.

    The uplink never exceeds max_rate_hz. Synthetic sources (pace=True) are
    paced by sleeping between samples so their profile time tracks wall time;
    live sources (pace=False) are decimated latest-wins, so a mocap system
    running faster than the cap cannot flood the radio link — stale samples
    are dropped and only the freshest pose is forwarded.
    """

    def __init__(
        self,
        *,
        source: MocapSampleSource | Iterable[PoseSample],
        backend: ExtposeBackend,
        max_rate_hz: float,
        pace: bool,
        transform: Callable[[PoseSample], PoseSample] | None = None,
    ) -> None:
        super().__init__(daemon=True, name="extpose-streamer")
        self._source = source
        self._backend = backend
        self._dt_s = 1.0 / max_rate_hz
        self._pace = pace
        self._transform = transform
        self._stop = threading.Event()
        self.sent = 0
        self.dropped = 0
        self.latest: PoseSample | None = None
        self.error: Exception | None = None
        self.latest_received_monotonic_s: float | None = None
        self.latest_forwarded_monotonic_s: float | None = None
        self._pose_timeout_s: float | None = None
        self.observer_errors = 0
        self._observer_lock = threading.Lock()
        self._observers: list[Callable[[MocapObservation], None]] = []

    def add_observer(
        self, observer: Callable[[MocapObservation], None]
    ) -> Callable[[], None]:
        """Observe received poses without opening or consuming the source.

        The callback runs on the streamer thread and therefore must return
        promptly.  The returned function is an idempotent unsubscribe handle.
        This separate observer path keeps the stream's source single-consumer
        contract intact while allowing diagnostics to attach after ``start``.
        Observations identify poses that the rate cap intentionally did not
        forward with a ``None`` forwarding timestamp.
        """

        with self._observer_lock:
            self._observers.append(observer)

        def unsubscribe() -> None:
            with self._observer_lock:
                try:
                    self._observers.remove(observer)
                except ValueError:
                    pass

        return unsubscribe

    def _publish_observation(self, observation: MocapObservation) -> None:
        with self._observer_lock:
            observers = tuple(self._observers)
        for observer in observers:
            try:
                observer(observation)
            except Exception:
                # A diagnostic must never be able to kill the extpose uplink.
                self.observer_errors += 1

    def run(self) -> None:
        try:
            next_allowed_s = 0.0
            for pose in iter_pose_samples(self._source, start_s=0.0, dt_s=self._dt_s):
                source_pose = pose
                received_s = time.monotonic()
                self.latest_received_monotonic_s = received_s
                if self._stop.is_set():
                    return
                if not self._pace and time.monotonic() < next_allowed_s:
                    self.dropped += 1
                    self._publish_observation(
                        MocapObservation(
                            pose=source_pose,
                            received_monotonic_s=received_s,
                            forwarded_monotonic_s=None,
                        )
                    )
                    continue
                # Step along a fixed grid, so a source only a little faster than the
                # cap still gets the full rate; after a gap, start a new grid rather
                # than burst to catch up.
                now_s = time.monotonic()
                on_grid = now_s < next_allowed_s + self._dt_s
                next_allowed_s = (next_allowed_s if on_grid else now_s) + self._dt_s
                if self._transform is not None:
                    pose = self._transform(pose)
                self.latest = pose
                self._backend.send_extpose(pose)
                self.sent += 1
                forwarded_s = time.monotonic()
                self.latest_forwarded_monotonic_s = forwarded_s
                self._publish_observation(
                    MocapObservation(
                        pose=source_pose,
                        received_monotonic_s=received_s,
                        forwarded_monotonic_s=forwarded_s,
                    )
                )
                if self._pace and self._stop.wait(self._dt_s):
                    return
        except Exception as exc:  # surfaced by callers polling .error
            self.error = exc

    def stop(self) -> None:
        self._stop.set()

    def start_pose_watch(self, timeout_s: float) -> None:
        """From now on, check_poses() raises MocapLost once no pose has been
        forwarded for ``timeout_s``.

        A runner starts the watch once the drone is armed and stops it when a
        landing starts, so the loss a landing answers cannot interrupt it.
        Shorter gaps, such as a body briefly hidden from the cameras, are left to
        the drone's estimator.
        """
        self._pose_timeout_s = timeout_s

    def stop_pose_watch(self) -> None:
        self._pose_timeout_s = None

    def check_poses(self) -> None:
        """Raise MocapLost if the pose watch is on and its timeout passed without a pose.

        MocapBodyLost while the source still hears from the mocap system (its
        ``connected(within_s)``), MocapConnectionLost otherwise.
        """
        timeout_s = self._pose_timeout_s
        last_s = self.latest_forwarded_monotonic_s
        gap_s = None if last_s is None else time.monotonic() - last_s
        if timeout_s is None or (gap_s is not None and gap_s <= timeout_s):
            return
        gap = "no pose yet" if gap_s is None else f"no pose for {gap_s:.1f} s"
        connected = getattr(self._source, "connected", None)
        if connected is not None and connected(timeout_s):
            raise MocapBodyLost(
                f"{gap}: the mocap system is streaming, but the rigid body is not tracked"
            )
        raise MocapConnectionLost(f"{gap}: nothing is arriving from the mocap system")
