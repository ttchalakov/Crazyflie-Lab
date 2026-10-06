"""TelemetryRecorder lifecycle: disabled no-op, start/record/close, mocap pump shutdown, TOC capture."""
import datetime
import json
import threading
import warnings
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
from cflib.crazyflie.log import LogTocElement
from cflib.crazyflie.param import ParamTocElement
from cflib.crazyflie.toc import Toc

from cf_lab.backends.fake import FakeCrazyflieBackend
from cf_lab.config import CameraConfig, LabConfig, TelemetryConfig
from cf_lab.mocap.fake import BoxSweepSource
from cf_lab.mocap.types import PoseSample, Quaternion
from cf_lab.telemetry import recorder as recorder_module
from cf_lab.telemetry.recorder import TelemetryRecorder
from cf_lab.telemetry.sync import load_run
from cf_lab.telemetry.toc import snapshot_crazyflie_toc


@pytest.fixture(autouse=True)
def no_git(monkeypatch):
    monkeypatch.setattr(recorder_module, "_git", lambda args: "")


def _close(recorder, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recorder.close(**kwargs)


def _manifest(run_dir):
    return json.loads((run_dir / "manifest.json").read_text())


def test_disabled_recorder_is_a_silent_no_op(tmp_path):
    recorder = TelemetryRecorder(TelemetryConfig(enabled=False), run_root=tmp_path / "runs")
    recorder.start()
    source, backend = BoxSweepSource(), FakeCrazyflieBackend()
    assert recorder.wrap_pose_source(source) is source
    assert recorder.wrap_extpose_backend(backend) is backend
    recorder.record_setpoint(0.0, 0.0, 1.0, 0.0)
    recorder.event("takeoff")
    recorder.attach_stm32(SimpleNamespace())  # never touched
    assert recorder.attach_mocap(source) is False
    recorder.close()
    assert recorder.run_dir is None
    assert recorder.stats() == {}
    assert not (tmp_path / "runs").exists()


def test_recorder_flight_lifecycle_produces_a_clean_loadable_run(tmp_path):
    cfg = TelemetryConfig(enabled=True, state_hz=0.0, controls_hz=0.0)
    lab = LabConfig(telemetry=cfg, camera=CameraConfig(ssid="lab", key="dummy-key-123"))
    recorder = TelemetryRecorder(cfg, run_root=tmp_path / "runs", lab_config=lab, example="unit")
    recorder.start()
    run_dir = recorder.run_dir
    assert run_dir.parent == tmp_path / "runs"
    assert _manifest(run_dir)["status"] == "recording"

    poses = recorder.wrap_pose_source(BoxSweepSource()).samples(start_s=0.0, dt_s=0.01)
    backend = FakeCrazyflieBackend()
    uplink = recorder.wrap_extpose_backend(backend)
    for _ in range(25):
        uplink.send_extpose(next(poses))
    for i in range(10):
        recorder.record_setpoint(0.0, 0.0, 0.1 * i, 0.0)
    recorder.event("takeoff", "go", height_m=1.0)
    recorder.close()
    recorder.close()  # finally blocks may close twice

    assert backend.extpose_count == 25
    manifest = _manifest(run_dir)
    assert manifest["status"] == "clean"
    assert manifest["example"] == "unit"
    assert manifest["config"]["telemetry"]["enabled"] is True
    assert manifest["config"]["camera"]["ssid"] == "lab"
    assert "key" not in manifest["config"]["camera"]  # the WiFi password stays out
    assert manifest["empty_streams"] == []
    run = load_run(run_dir)
    assert {name: t.num_rows for name, t in run.streams.items()} == {
        "mocap": 25, "extpose": 25, "setpoints": 10, "events": 1}
    event = run.streams["events"].to_pylist()[0]
    assert (event["kind"], json.loads(event["data_json"])) == ("takeoff", {"height_m": 1.0})


def test_disabled_streams_are_not_recorded_and_do_not_raise(tmp_path):
    cfg = TelemetryConfig(enabled=True, state_hz=0.0, controls_hz=0.0, mocap=False,
                          extpose=False, setpoints=False)
    recorder = TelemetryRecorder(cfg, run_root=tmp_path)
    recorder.start()
    source = BoxSweepSource()
    assert recorder.wrap_pose_source(source) is source
    recorder.record_setpoint(1.0, 2.0, 3.0, 4.0)  # setpoints off: must not KeyError mid-flight
    recorder.event("abort")
    recorder.close(status="aborted")
    manifest = _manifest(recorder.run_dir)
    assert set(manifest["streams"]) == {"events"}
    assert manifest["status"] == "aborted"


def test_runs_started_in_the_same_second_get_separate_directories(tmp_path, monkeypatch):
    fixed = datetime.datetime(2026, 10, 3, 12, 0, 0)
    monkeypatch.setattr(recorder_module, "datetime",
                        SimpleNamespace(datetime=SimpleNamespace(now=lambda: fixed)))
    cfg = TelemetryConfig(enabled=True, state_hz=0.0, controls_hz=0.0)
    first = TelemetryRecorder(cfg, run_root=tmp_path)
    second = TelemetryRecorder(cfg, run_root=tmp_path)
    first.start()
    second.start()
    try:
        assert first.run_dir != second.run_dir
        assert first.run_dir.name == "run_20261003T120000"
    finally:
        _close(first)
        _close(second)


class _BlockingSource:
    """A tracked-then-lost rigid body: yields a few poses, then blocks until closed."""

    def __init__(self, n):
        self.n = n
        self.closed = threading.Event()

    def samples(self, *, start_s=0.0, dt_s=0.01):
        for i in range(self.n):
            yield PoseSample(timestamp_s=i * dt_s, position_m=(0.0, 0.0, 0.0),
                             orientation=Quaternion.identity(), rigid_body="cf",
                             source_seq=i + 1)
        self.closed.wait(10.0)

    def close(self):
        self.closed.set()


def test_mocap_pump_records_every_pose_and_close_unblocks_it(tmp_path, wait_for):
    cfg = TelemetryConfig(enabled=True, state_hz=0.0, controls_hz=0.0)
    recorder = TelemetryRecorder(cfg, run_root=tmp_path)
    recorder.start()
    source = _BlockingSource(40)
    assert recorder.attach_mocap(source) is True
    pump = recorder._mocap_pump
    wait_for(lambda: recorder.mocap_stats == {"recorded": 40})
    _close(recorder)  # must not hang on the blocked read
    _close(recorder)
    assert source.closed.is_set()
    assert not pump.is_alive()
    mocap = pq.read_table(recorder.run_dir / "mocap.parquet")
    assert mocap.column("source_seq").to_pylist() == list(range(1, 41))
    assert _mocap_recorded(recorder) == [{"poses": 40, "error": None}]


def _mocap_recorded(recorder):
    events = pq.read_table(recorder.run_dir / "events.parquet").to_pylist()
    return [json.loads(e["data_json"]) for e in events if e["kind"] == "mocap_recorded"]


class _FailingSource:
    """Yields a few poses, then the transport fails."""

    def __init__(self, n):
        self.n = n

    def samples(self, *, start_s=0.0, dt_s=0.01):
        for i in range(self.n):
            yield PoseSample(timestamp_s=i * dt_s, position_m=(0.0, 0.0, 0.0),
                             orientation=Quaternion.identity(), rigid_body="cf",
                             source_seq=i + 1)
        raise OSError("NatNet socket closed")


def test_a_mocap_recording_failure_is_recorded_not_dropped(tmp_path, wait_for):
    recorder = TelemetryRecorder(TelemetryConfig(enabled=True, state_hz=0.0, controls_hz=0.0),
                                 run_root=tmp_path)
    recorder.start()
    recorder.attach_mocap(_FailingSource(3))
    wait_for(lambda: not recorder._mocap_pump.is_alive())
    _close(recorder)
    assert _mocap_recorded(recorder) == [{"poses": 3, "error": "NatNet socket closed"}]


def _log_element(ident, group, name, type_id):
    return LogTocElement(ident, bytes([type_id]) + f"{group}\0{name}\0".encode())


def _param_element(ident, group, name, metadata, persistent=False):
    element = ParamTocElement(ident, bytes([metadata]) + f"{group}\0{name}\0".encode())
    if persistent:
        element.mark_persistent()
    return element


def _fake_cf():
    log, param = Toc(), Toc()
    for element in (_log_element(0, "stateEstimate", "z", 0x07),
                    _log_element(1, "stateEstimate", "x", 0x07),
                    _log_element(2, "motor", "m1", 0x02)):
        log.add_element(element)
    for element in (_param_element(0, "stabilizer", "estimator", 0x08),
                    _param_element(1, "deck", "bcAI", 0x08 | 0x40),
                    _param_element(2, "pid_rate", "roll_kp", 0x06 | 0x10, persistent=True)):
        param.add_element(element)
    return SimpleNamespace(log=SimpleNamespace(toc=log), param=SimpleNamespace(toc=param))


def test_toc_snapshot_lists_every_entry_with_types_and_access():
    snap = snapshot_crazyflie_toc(_fake_cf())
    json.dumps(snap)  # must be JSON-safe for the run directory
    assert snap["log"]["count"] == 3
    assert [e["full_name"] for e in snap["log"]["entries"]] == [
        "motor.m1", "stateEstimate.x", "stateEstimate.z"]
    assert snap["log"]["entries"][1]["ctype"] == "float"
    params = {e["full_name"]: e for e in snap["parameter"]["entries"]}
    assert params["deck.bcAI"]["access"] == "RO"
    assert params["stabilizer.estimator"]["access"] == "RW"
    assert params["pid_rate.roll_kp"]["persistent"] is True
    assert params["pid_rate.roll_kp"]["extended"] is True
    assert params["pid_rate.roll_kp"]["ctype"] == "float"


def _stm32_recorder(tmp_path):
    cfg = TelemetryConfig(enabled=True, state_hz=100.0, controls_hz=0.0, mocap=False,
                          extpose=False, setpoints=False)
    recorder = TelemetryRecorder(cfg, run_root=tmp_path)
    recorder.start()
    cf = _fake_cf()
    started = []

    def add_config(lc):
        lc.cf = SimpleNamespace(link=None)  # LogConfig.start is a no-op without a link
        started.append(lc)

    cf.log.add_config = add_config
    return recorder, cf, started


def test_attach_stm32_archives_toc_once_and_starts_log_blocks(tmp_path):
    recorder, cf, started = _stm32_recorder(tmp_path)
    recorder.attach_stm32(cf)
    recorder.capture_crazyflie_toc(cf)  # a second capture is a no-op
    started[0].data_received_cb.call(1234, {f"stateEstimate.{k}": 0.5 for k in
                                            ("x", "y", "z", "vx", "vy", "vz")}, started[0])
    recorder.detach_stm32()
    _close(recorder)

    run_dir = recorder.run_dir
    assert sorted(lc.name for lc in started) == ["tel_state_att", "tel_state_pos"]
    toc = json.loads((run_dir / "crazyflie_toc.json").read_text())
    assert toc["parameter"]["count"] == 3
    kinds = pq.read_table(run_dir / "events.parquet").column("kind").to_pylist()
    assert kinds.count("crazyflie_toc_saved") == 1
    assert pq.read_table(run_dir / "state_pos.parquet").column("stm32_ms").to_pylist() == [1234]


def test_close_stops_the_stm32_log_blocks(tmp_path):
    recorder, cf, started = _stm32_recorder(tmp_path)
    recorder.attach_stm32(cf)
    sent = []
    for block_id, lc in enumerate(started):  # the link is up now: stop() sends CMD_STOP_LOGGING
        lc.id = block_id
        lc.cf = SimpleNamespace(link=object(), send_packet=lambda pk, expected_reply: sent.append(pk.data))
    _close(recorder)  # an error path that never reached detach_stm32()
    assert len(sent) == len(started) == 2
