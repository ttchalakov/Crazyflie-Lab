"""Fakes at the hardware boundary for the preflight checks: cflib params/log blocks, a clocked mocap stream, a camera feed."""
from __future__ import annotations

import csv
import sys
import threading
import time
import types
from pathlib import Path

import cflib.crazyflie.log
import pytest

import cf_lab.mocap.stream as stream_module
import cf_lab.preflight.camera as camera_checks
import cf_lab.preflight.crazyflie as crazyflie_checks
import cf_lab.preflight.mocap as mocap_checks
from cf_lab.backends.fake import FakeCrazyflieBackend
from cf_lab.camera.ext import CAMERA_STREAM_EXT, decode_deck_ext
from cf_lab.camera.feed import CapturedFrame
from cf_lab.imgstream import CF_IMG_MAGIC, CF_IMG_TYPE_JPEG, CF_IMG_TYPE_RAW_GRAY, ImgHeader
from cf_lab.mocap.stream import ExtposeStreamer, MocapObservation
from cf_lab.mocap.types import PoseSample, Quaternion

DATA = Path(__file__).parent / "data"
IMU_VARIABLES = ("acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z")
# canArm plus "high-level commander idle": what a healthy, disarmed drone on the pad logs.
SUPERVISOR_READY = 0x0401


def read_csv(name: str) -> list[dict[str, float]]:
    with (DATA / name).open(newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


# --- cflib Crazyflie --------------------------------------------------------


class _Caller:
    def __init__(self) -> None:
        self.callbacks = []

    def add_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def call(self, *args) -> None:
        for callback in self.callbacks:
            callback(*args)


class FakeLogConfig:
    """Stands in for cflib's LogConfig; the owning FakeLog streams rows on start()."""

    def __init__(self, name: str, period_in_ms: int) -> None:
        self.name = name
        self.period_in_ms = period_in_ms
        self.variables: list[str] = []
        self.data_received_cb = _Caller()
        self.log: FakeLog | None = None
        self.started = self.stopped = self.deleted = False

    def add_variable(self, name: str, fetch_as: str | None = None) -> None:
        self.variables.append(name)

    def start(self) -> None:
        self.started = True
        self.log.stream(self)

    def stop(self) -> None:
        self.stopped = True

    def delete(self) -> None:
        self.deleted = True


class FakeLog:
    """cf.log: a started block delivers one row per index of ``series`` (variable -> values)."""

    def __init__(self, series: dict[str, list], start_error: Exception | None) -> None:
        self.series = series
        self.start_error = start_error
        self.configs: list[FakeLogConfig] = []

    def add_config(self, config: FakeLogConfig) -> None:
        for variable in config.variables:
            if variable not in self.series:
                raise KeyError(f"Variable {variable} not in TOC")
        config.log = self
        self.configs.append(config)

    def stream(self, config: FakeLogConfig) -> None:
        if self.start_error is not None:
            raise self.start_error
        count = min(len(self.series[variable]) for variable in config.variables)
        for index in range(count):
            row = {variable: self.series[variable][index] for variable in config.variables}
            config.data_received_cb.call(index * config.period_in_ms, row, config)


class FakeParam:
    """cf.param with only reads: a check that tried to write would hit AttributeError."""

    is_updated = True  # cflib's flag: every parameter value has arrived

    def __init__(self, values: dict[str, object]) -> None:
        self.values = dict(values)

    def get_value(self, name: str, timeout: float | None = None):
        if name not in self.values:
            raise KeyError(f"{name} not in param TOC")
        value = self.values[name]
        if isinstance(value, BaseException):
            raise value
        return value


class FakeCrazyflie:
    def __init__(self, params, series, start_error) -> None:
        self.param = FakeParam(params)
        self.log = FakeLog(series, start_error)


def imu_series(rows, supervisor_info=SUPERVISOR_READY) -> dict[str, list]:
    """Log series for the IMU block from rows of acc/gyro values."""
    series = {name: [row[name] for row in rows] for name in IMU_VARIABLES}
    if isinstance(supervisor_info, int):
        supervisor_info = [supervisor_info] * len(rows)
    series["supervisor.info"] = list(supervisor_info)
    return series


def still_imu_rows(count: int = 75, *, acc=(0.0, 0.0, 1.0)) -> list[dict[str, float]]:
    return [dict(zip(IMU_VARIABLES, (*acc, 0.0, 0.0, 0.0))) for _ in range(count)]


def battery_series(voltages, *, level: int = 80, state: int = 0) -> dict[str, list]:
    return {
        "pm.vbat": list(voltages),
        "pm.batteryLevel": [level] * len(voltages),
        "pm.state": [state] * len(voltages),
    }


@pytest.fixture
def make_cf(monkeypatch):
    """Build a FakeCrazyflie; cflib's LogConfig is swapped in and the capture sleeps are skipped."""
    monkeypatch.setattr(cflib.crazyflie.log, "LogConfig", FakeLogConfig)
    monkeypatch.setattr(
        crazyflie_checks,
        "time",
        types.SimpleNamespace(sleep=lambda _s: None, perf_counter=time.perf_counter),
    )

    def build(params=None, series=None, *, start_error=None) -> FakeCrazyflie:
        return FakeCrazyflie(params or {}, series or {}, start_error)

    return build


# --- mocap ------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.on_wait = self.advance

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Wait:
    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock

    def wait(self, timeout: float) -> bool:
        self._clock.on_wait(timeout)
        return False


@pytest.fixture
def mocap_clock(monkeypatch) -> FakeClock:
    """A host clock for check_mocap_quality; its observation window advances only this clock."""
    clock = FakeClock()
    monkeypatch.setattr(mocap_checks, "time", types.SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(
        mocap_checks,
        "threading",
        types.SimpleNamespace(Lock=threading.Lock, Event=lambda: _Wait(clock)),
    )
    return clock


class ReplayStreamer:
    """An ExtposeStreamer stand-in that replays (received_s, forwarded_s | None, pose) during the window.

    Offsets are seconds after the window opens; a negative ``latest_*`` offset is
    a pose the streamer saw before the check attached.
    """

    def __init__(
        self,
        clock: FakeClock,
        events,
        *,
        error: Exception | None = None,
        latest_received_s: float | None = None,
        latest_forwarded_s: float | None = None,
    ) -> None:
        self.clock = clock
        self.events = list(events)
        self.error = error
        self.sent = sum(forwarded is not None for _, forwarded, _ in self.events)
        self.dropped = len(self.events) - self.sent
        self.observer_errors = 0
        self.latest_received_monotonic_s = (
            None if latest_received_s is None else clock.now + latest_received_s
        )
        self.latest_forwarded_monotonic_s = (
            None if latest_forwarded_s is None else clock.now + latest_forwarded_s
        )
        self.observer = None
        self.unsubscribed = False

    def add_observer(self, observer):
        self.observer = observer
        self.clock.on_wait = self.play
        return self.unsubscribe

    def unsubscribe(self) -> None:
        self.observer = None
        self.unsubscribed = True

    def play(self, duration_s: float) -> None:
        start = self.clock.now
        for received, forwarded, pose in self.events:
            if received > duration_s:
                break
            self.clock.now = start + received
            if self.observer is not None:
                self.observer(
                    MocapObservation(
                        pose=pose,
                        received_monotonic_s=start + received,
                        forwarded_monotonic_s=None if forwarded is None else start + forwarded,
                    )
                )
        self.clock.now = start + duration_s


def pose(
    position=(0.43, -0.44, 0.06),
    q=(0.0, 0.0, 0.0, 1.0),
    *,
    seq: int | None = None,
    body: str = "cf21bl_01",
) -> PoseSample:
    return PoseSample(
        timestamp_s=0.0,
        position_m=tuple(position),
        orientation=Quaternion(*q),
        rigid_body=body,
        source_seq=seq,
    )


def steady_stream(*, rate_hz: float = 100.0, duration_s: float = 1.0, poses=None):
    """Events for a stream forwarding every pose at ``rate_hz``; ``poses(i)`` builds pose i."""
    count = int(round(rate_hz * duration_s))
    build = poses or (lambda index: pose(seq=index))
    return [((index + 1) / rate_hz - 1e-4,) * 2 + (build(index),) for index in range(count)]


def recorded_poses(name: str):
    """(host seconds, pose) pairs of a recorded NatNet window."""
    return [
        (
            row["host_ms"] / 1000.0,
            pose(
                (row["x"], row["y"], row["z"]),
                (row["qx"], row["qy"], row["qz"], row["qw"]),
                seq=int(row["source_seq"]),
            ),
        )
        for row in read_csv(name)
    ]


@pytest.fixture
def through_uplink(monkeypatch):
    """Replay events from the real ExtposeStreamer's latest-wins cap, fed (host_s, pose) on a fake clock."""

    def run(timed_poses, *, cap_hz: float = 100.0):
        now = [0.0]

        def source():
            for host_s, sample in timed_poses:
                now[0] = host_s
                yield sample

        streamer = ExtposeStreamer(
            source=source(), backend=FakeCrazyflieBackend(), max_rate_hz=cap_hz, pace=False
        )
        events = []
        streamer.add_observer(
            lambda o: events.append((o.received_monotonic_s, o.forwarded_monotonic_s, o.pose))
        )
        with monkeypatch.context() as patch:
            patch.setattr(stream_module, "time", types.SimpleNamespace(monotonic=lambda: now[0]))
            streamer.run()  # synchronously, on this thread
        assert streamer.error is None
        return events

    return run


# --- camera -----------------------------------------------------------------


class NsClock:
    def __init__(self) -> None:
        self.ns = 5_000_000_000

    def monotonic_ns(self) -> int:
        return self.ns


@pytest.fixture
def camera_clock(monkeypatch) -> NsClock:
    """The receipt clock check_camera_stream stamps frames with."""
    clock = NsClock()
    monkeypatch.setattr(
        camera_checks,
        "time",
        types.SimpleNamespace(monotonic=time.monotonic, monotonic_ns=clock.monotonic_ns),
    )
    return clock


def camera_ext(frame_id: int, *, camera_frame_id: int | None = None, drops=(0, 0, 0)) -> bytes:
    camera_id = frame_id if camera_frame_id is None else camera_frame_id
    return CAMERA_STREAM_EXT.pack(frame_id, (1000 + frame_id) & 0xFFFFFFFF, 3, 4, 1, 2, camera_id, 990, *drops)


def make_frame(
    frame_id: int = 0,
    *,
    width: int = 320,
    height: int = 240,
    jpeg: bool = False,
    payload: bytes | None = None,
    ext: bytes | None = None,
) -> CapturedFrame:
    if payload is None:
        payload = bytes(width * height) if not jpeg else b"\xff\xd8\xff\xd9"
    ext = camera_ext(frame_id) if ext is None else ext
    header = ImgHeader(
        magic=CF_IMG_MAGIC,
        type=CF_IMG_TYPE_JPEG if jpeg else CF_IMG_TYPE_RAW_GRAY,
        depth=1,
        ext_len=len(ext),
        width=width,
        height=height,
        size=len(payload),
    )
    return CapturedFrame(header=header, payload=payload, ext=decode_deck_ext(ext), raw_ext=ext, host_ns=0)


class FakeFeed:
    """A CameraFeed stand-in. A live feed delivers when observed, an owned one when started."""

    def __init__(
        self,
        frames,
        *,
        clock: NsClock | None = None,
        fps: float = 30.0,
        error: BaseException | None = None,
        start_error: Exception | None = None,
        owned: bool = False,
    ) -> None:
        self.frames = list(frames)
        self.clock = clock
        self.fps = fps
        self.error = error
        self.start_error = start_error
        self.owned = owned
        self.observers = []
        self.started = self.stopped = False
        self.frames_received = 0
        self.sink_errors = 0
        self.observer_errors = 0

    def add_observer(self, observer):
        self.observers.append(observer)
        if not self.owned:
            self._deliver()

        def unsubscribe() -> None:
            if observer in self.observers:
                self.observers.remove(observer)

        return unsubscribe

    def start(self) -> None:
        self.started = True
        if self.start_error is not None:
            raise self.start_error
        self._deliver()

    def stop(self) -> None:
        self.stopped = True

    def _deliver(self) -> None:
        for frame in self.frames:
            if self.clock is not None:
                self.clock.ns += int(1e9 / self.fps)
            self.frames_received += 1
            for observer in list(self.observers):
                observer.offer(frame)


@pytest.fixture
def fakes():
    """This module's builders; a conftest is not importable under --import-mode=importlib."""
    return sys.modules[__name__]
