"""Source taps: STM32 log callbacks, mocap pose tap, extpose uplink tap and the camera blob recorder."""
import errno
import json
import random
import threading
import warnings
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from cf_lab.backends.fake import FakeCrazyflieBackend
from cf_lab.camera.ext import SCHED_EXT, decode_deck_ext
from cf_lab.camera.feed import CapturedFrame
from cf_lab.config import CustomStreamConfig
from cf_lab.imgstream import CF_IMG_MAGIC, CF_IMG_TYPE_JPEG, CF_IMG_TYPE_RAW_GRAY, ImgHeader
from cf_lab.mocap.fake import BoxSweepSource
from cf_lab.mocap.types import PoseSample, Quaternion
from cf_lab.telemetry import schema
from cf_lab.telemetry.camera import CameraRecorder
from cf_lab.telemetry.mocap import TappedExtposeBackend, TappedPoseSource
from cf_lab.telemetry.stm32 import Stm32LogTap
from cf_lab.telemetry.writer import RunWriter


@pytest.fixture
def open_writer(tmp_path):
    """Start a RunWriter for some specs; closed (quietly) at teardown if the test did not."""
    writers = []

    def start(*specs):
        writer = RunWriter(tmp_path / "run", [*specs, schema.events_spec()], queue_size=100_000,
                           flush_interval_s=60.0)
        writer.start()
        writers.append(writer)
        return writer

    yield start
    for writer in writers:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            writer.close()


def _close(writer):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        writer.close()
    return writer.run_dir


class _FakeLog:
    """cf.log at the cflib boundary: accepts LogConfigs, never touches a link."""

    def __init__(self):
        self.configs = []

    def add_config(self, lc):
        lc.cf = SimpleNamespace(link=None)  # LogConfig.start/stop are no-ops without a link
        self.configs.append(lc)


def test_stm32_tap_registers_one_block_per_stream_with_snapped_period(open_writer):
    custom = schema.custom_spec(CustomStreamConfig(
        name="tba", period_ms=40, variables=("tba.tick:uint32_t", "tba.sin1:float")))
    specs = [schema.state_pos_spec(30.0), custom, schema.setpoints_spec()]
    writer = open_writer(*specs)
    cf = SimpleNamespace(log=_FakeLog())
    Stm32LogTap(writer, specs).start(cf)

    blocks = {lc.name: lc for lc in cf.log.configs}
    assert set(blocks) == {"tel_state_pos", "tel_tba"}  # setpoints is host-side
    assert blocks["tel_state_pos"].period_in_ms == 30
    assert blocks["tel_tba"].period_in_ms == 40
    assert [(v.name, v.fetch_as_string) for v in blocks["tel_tba"].variables] == [
        ("tba.tick", "uint32_t"), ("tba.sin1", "float")]


def test_stm32_callback_records_timestamp_and_values_in_column_order(open_writer):
    custom = schema.custom_spec(CustomStreamConfig(
        name="tba", period_ms=10, variables=("tba.tick:uint32_t", "tba.sin1:float", "tba.saw:float")))
    writer = open_writer(custom)
    cf = SimpleNamespace(log=_FakeLog())
    Stm32LogTap(writer, [custom]).start(cf)
    lc = cf.log.configs[0]
    # cflib hands over a dict; its order is not the schema's.
    lc.data_received_cb.call(5610, {"tba.saw": 0.91, "tba.tick": 3910, "tba.sin1": -0.5}, lc)
    lc.data_received_cb.call(16_777_210, {"tba.saw": 0.92, "tba.tick": 3920, "tba.sin1": -0.25}, lc)
    rows = pq.read_table(_close(writer) / "tba.parquet").to_pylist()
    assert [(r["stm32_ms"], r["tba.tick"], r["tba.sin1"], r["tba.saw"]) for r in rows] == [
        (5610, 3910, -0.5, pytest.approx(0.91)), (16_777_210, 3920, -0.25, pytest.approx(0.92))]


def test_stm32_tap_stop_survives_a_dead_link(open_writer):
    spec = schema.state_pos_spec(100.0)
    writer = open_writer(spec)
    cf = SimpleNamespace(log=_FakeLog())
    tap = Stm32LogTap(writer, [spec])
    tap.start(cf)
    cf.log.configs[0].cf = None  # link torn down: LogConfig.stop raises AttributeError
    tap.stop()
    tap.stop()


class _ArrivalStampedSource:
    """A finite NatNet-like source whose poses carry their own arrival stamp and latency."""

    def __init__(self, n):
        self.poses = [
            PoseSample(
                timestamp_s=10.0 + i / 360.0,
                position_m=(0.1 * i, -0.2, 0.3),
                orientation=Quaternion(qx=0.0, qy=0.0, qz=0.6, qw=0.8),
                rigid_body="cf",
                source_seq=1000 + i,
                host_ns=7_000_000_000 + 2_777_778 * i,
                latency_s=0.0042,
            )
            for i in range(n)
        ]
        self.closed = False

    def samples(self, *, start_s=0.0, dt_s=0.01):
        yield from self.poses

    def close(self):
        self.closed = True


def test_pose_tap_records_raw_pose_with_arrival_stamp_and_latency(open_writer):
    writer = open_writer(schema.mocap_spec("natnet"))
    source = _ArrivalStampedSource(3)
    tapped = TappedPoseSource(source, writer)
    assert list(tapped.samples()) == source.poses  # consumer sees the poses unchanged
    tapped.close()
    assert source.closed

    rows = pq.read_table(_close(writer) / "mocap.parquet").to_pylist()
    assert [r["host_ns"] for r in rows] == [p.host_ns for p in source.poses]
    assert [r["source_seq"] for r in rows] == [1000, 1001, 1002]
    assert rows[1]["x"] == pytest.approx(0.1)
    assert (rows[1]["qz"], rows[1]["qw"]) == (0.6, 0.8)
    assert rows[0]["latency_us"] == pytest.approx(4200.0)


def test_pose_tap_without_frame_numbers_marks_them_absent(open_writer):
    writer = open_writer(schema.mocap_spec("vrpn"))
    samples = TappedPoseSource(BoxSweepSource(), writer).samples(start_s=0.0, dt_s=0.01)
    poses = [next(samples) for _ in range(4)]
    rows = pq.read_table(_close(writer) / "mocap.parquet").to_pylist()
    assert [r["source_ts_s"] for r in rows] == [p.timestamp_s for p in poses]
    assert {r["source_seq"] for r in rows} == {-1}
    assert {r["latency_us"] for r in rows} == {-1.0}


def test_extpose_tap_records_and_forwards_every_pose(open_writer):
    writer = open_writer(schema.extpose_spec())
    backend = FakeCrazyflieBackend()
    tapped = TappedExtposeBackend(backend, writer)
    pose = PoseSample(timestamp_s=1.5, position_m=(0.0, 0.1, 0.2),
                      orientation=Quaternion.identity(), rigid_body="cf",
                      source_seq=42)
    tapped.send_extpose(pose)
    tapped.emergency_stop()  # everything else, safety calls included, reaches the real backend
    assert backend.extpose_count == 1
    assert backend.emergency_stopped
    row = pq.read_table(_close(writer) / "extpose.parquet").to_pylist()[0]
    assert (row["source_seq"], row["y"], row["z"], row["qw"]) == (42, 0.1, 0.2, 1.0)


def _frame(rng, frame_id, *, ext=True, jpeg=True):
    payload = rng.randbytes(rng.randint(1, 5000))
    raw_ext = SCHED_EXT.pack(2, 0, 100, frame_id, 50_000 + 16 * frame_id, 3,
                             9_000 + 16 * frame_id, 49_990 + 16 * frame_id) if ext else b""
    header = ImgHeader(magic=CF_IMG_MAGIC, type=CF_IMG_TYPE_JPEG if jpeg else CF_IMG_TYPE_RAW_GRAY,
                       depth=1, ext_len=len(raw_ext), width=324, height=244, size=len(payload))
    return CapturedFrame(header=header, payload=payload, ext=decode_deck_ext(raw_ext),
                         raw_ext=raw_ext, host_ns=1_000_000 * (frame_id + 1))


def test_camera_index_rows_point_at_their_exact_bytes_in_the_blob(open_writer):
    rng = random.Random(7)
    writer = open_writer(schema.camera_spec())
    recorder = CameraRecorder(writer)
    recorder.start()
    frames = [_frame(rng, i, ext=i != 3, jpeg=i % 2 == 0) for i in range(20)]
    for frame in frames:
        recorder.offer(frame)
    recorder.stop()
    run_dir = _close(writer)

    blob = (run_dir / schema.CAMERA_BLOB_FILE).read_bytes()
    rows = pq.read_table(run_dir / "camera.parquet").to_pylist()
    assert len(rows) == 20
    for row, frame in zip(rows, frames):
        assert blob[row["blob_offset"]:row["blob_offset"] + row["size"]] == frame.payload
        assert row["host_ns"] == frame.host_ns
        assert row["pixel_format"] == frame.header.type
        assert row["ext"] == frame.raw_ext
    assert len(blob) == sum(len(f.payload) for f in frames)
    assert (rows[5]["frame_id"], rows[5]["deck_ms"]) == (5, 50_080)
    assert (rows[5]["stm32_anchor_ms"], rows[5]["deck_anchor_ms"]) == (9_080, 50_070)
    no_ext = rows[3]
    assert (no_ext["frame_id"], no_ext["deck_ms"], no_ext["stm32_anchor_ms"]) == (-1, -1, -1)


def test_camera_full_queue_drops_frames_and_reports_them(open_writer):
    rng = random.Random(8)
    writer = open_writer(schema.camera_spec())
    recorder = CameraRecorder(writer, queue_size=3)
    for i in range(8):  # writer thread not running yet: the queue fills
        recorder.offer(_frame(rng, i))
    assert recorder.dropped == 5
    recorder.start()
    recorder.stop()
    recorder.stop()  # idempotent
    run_dir = _close(writer)

    assert pq.read_table(run_dir / "camera.parquet").column("frame_id").to_pylist() == [0, 1, 2]
    events = pq.read_table(run_dir / "events.parquet").to_pylist()
    recorded = [json.loads(e["data_json"]) for e in events if e["kind"] == "camera_recorded"]
    assert len(recorded) == 1
    assert recorded[0]["frames"] == 3
    assert recorded[0]["dropped"] == 5
    assert recorded[0]["error"] is None


class _FullDisk:
    def write(self, data):
        raise OSError(errno.ENOSPC, "No space left on device")

    def close(self):
        pass


def test_camera_stop_returns_after_a_disk_error(open_writer, wait_for):
    rng = random.Random(9)
    writer = open_writer(schema.camera_spec())
    recorder = CameraRecorder(writer, queue_size=2)
    recorder.start()
    recorder._blob.close()
    recorder._blob = _FullDisk()  # the disk fills up mid-flight
    recorder.offer(_frame(rng, 0))
    wait_for(lambda: recorder.error is not None)
    for i in range(1, 5):  # the feed keeps delivering into the dead recorder
        recorder.offer(_frame(rng, i))

    stopper = threading.Thread(target=recorder.stop, daemon=True)
    stopper.start()
    stopper.join(1.0)
    assert not stopper.is_alive(), "stop() hung, so TelemetryRecorder.close() never seals the run"
    events = pq.read_table(_close(writer) / "events.parquet").to_pylist()
    recorded = next(json.loads(e["data_json"]) for e in events if e["kind"] == "camera_recorded")
    assert "No space left" in recorded["error"]
