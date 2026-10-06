"""RunWriter: Parquet round trip, bounded non-blocking queues, drops in the manifest, close and crash paths."""
import json
import shutil
import threading
import warnings

import pyarrow.parquet as pq
import pytest

from cf_lab.config import CustomStreamConfig
from cf_lab.telemetry import schema
from cf_lab.telemetry.sync import load_run
from cf_lab.telemetry.writer import RunWriter


def _manifest(run_dir):
    return json.loads((run_dir / "manifest.json").read_text())


def _close_quietly(writer, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        writer.close(**kwargs)


def test_round_trip_writes_typed_parquet_per_stream(tmp_path, write_run):
    custom = schema.custom_spec(
        CustomStreamConfig(name="tba", variables=("tba.tick:uint32_t", "tba.sin:float"))
    )
    specs = [custom, schema.setpoints_spec(), schema.events_spec()]
    rows = {
        "tba": [(1_000 + i, (5000 + 10 * i, 3900 + 10 * i, 0.25 * i)) for i in range(5)],
        "setpoints": [(2_000 + i, (0.1 * i, -0.2, 1.0, 90.0)) for i in range(3)],
        "events": [(3_000, ("takeoff", "go", json.dumps({"z": 1.0})))],
    }
    run_dir = write_run(tmp_path / "run", specs, rows)

    tba = pq.read_table(run_dir / "tba.parquet")
    assert tba.schema.names == ["seq", "host_ns", "stm32_ms", "tba.tick", "tba.sin"]
    assert [str(t) for t in tba.schema.types] == ["int64", "int64", "int64", "uint32", "float"]
    assert tba.column("seq").to_pylist() == [1, 2, 3, 4, 5]
    assert tba.column("host_ns").to_pylist() == [1000, 1001, 1002, 1003, 1004]
    assert tba.column("tba.tick").to_pylist() == [3900, 3910, 3920, 3930, 3940]
    assert tba.column("tba.sin").to_pylist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    setpoints = pq.read_table(run_dir / "setpoints.parquet").to_pylist()
    assert setpoints[2] == {"seq": 3, "host_ns": 2002, "x": 0.2, "y": -0.2, "z": 1.0, "yaw_deg": 90.0}
    assert not list(run_dir.glob("*.arrow"))

    manifest = _manifest(run_dir)
    assert manifest["status"] == "clean"
    assert manifest["error"] is None
    assert {name: entry["rows"] for name, entry in manifest["streams"].items()} == {
        "tba": 5, "setpoints": 3, "events": 1,
    }
    assert manifest["streams"]["tba"]["file"] == "tba.parquet"
    assert manifest["streams"]["tba"]["clock"] == "stm32_ms"


def test_full_queue_drops_and_counts_instead_of_blocking(tmp_path):
    writer = RunWriter(tmp_path / "run", [schema.setpoints_spec(), schema.events_spec()],
                       queue_size=4, flush_interval_s=60.0)
    writer.start()

    def flood():
        for i in range(1000):
            writer.append("setpoints", (float(i), 0.0, 0.0, 0.0))

    producer = threading.Thread(target=flood, daemon=True)
    producer.start()
    producer.join(timeout=2.0)
    assert not producer.is_alive(), "append blocked on a full queue"
    assert writer.stats()["setpoints"] == {"rows": 0, "dropped": 996, "queued": 4}

    _close_quietly(writer)
    run_dir = tmp_path / "run"
    table = pq.read_table(run_dir / "setpoints.parquet")
    assert table.column("seq").to_pylist() == [1, 2, 3, 4]
    assert table.column("x").to_pylist() == [0.0, 1.0, 2.0, 3.0]
    entry = _manifest(run_dir)["streams"]["setpoints"]
    assert (entry["rows"], entry["dropped"]) == (4, 996)


def test_manifest_says_recording_until_close_seals_it(tmp_path):
    writer = RunWriter(tmp_path / "run", [schema.setpoints_spec(), schema.events_spec()])
    writer.start()
    try:
        manifest = _manifest(tmp_path / "run")
        assert manifest["status"] == "recording"
        assert manifest["streams"]["setpoints"]["file"] == "setpoints.arrow"
    finally:
        writer.append("setpoints", (0.0, 0.0, 0.0, 0.0))
        writer.event("note")
        writer.close()
    assert _manifest(tmp_path / "run")["status"] == "clean"


def test_close_is_idempotent(tmp_path):
    writer = RunWriter(tmp_path / "run", [schema.setpoints_spec(), schema.events_spec()])
    writer.start()
    writer.append("setpoints", (0.0, 0.0, 0.0, 0.0))
    writer.event("note")
    writer.close()
    first = (tmp_path / "run" / "manifest.json").read_text()
    writer.close(status="error")
    assert (tmp_path / "run" / "manifest.json").read_text() == first


def test_close_without_start_writes_nothing(tmp_path):
    writer = RunWriter(tmp_path / "run", [schema.setpoints_spec(), schema.events_spec()])
    writer.close()
    assert not (tmp_path / "run").exists()


def test_status_passed_to_close_reaches_the_manifest(tmp_path):
    writer = RunWriter(tmp_path / "run", [schema.events_spec()])
    writer.start()
    writer.event("abort", "operator stop")
    writer.close(status="aborted")
    assert _manifest(tmp_path / "run")["status"] == "aborted"


def test_duplicate_stream_names_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        RunWriter(tmp_path / "run", [schema.events_spec(), schema.events_spec()])


def test_empty_enabled_stream_is_reported_not_silent(tmp_path):
    writer = RunWriter(tmp_path / "run", [schema.mocap_spec("natnet"), schema.setpoints_spec(),
                                          schema.events_spec()])
    writer.start()
    writer.append("setpoints", (0.0, 0.0, 0.0, 0.0))
    with pytest.warns(UserWarning, match="mocap"):
        writer.close()
    run_dir = tmp_path / "run"
    manifest = _manifest(run_dir)
    assert manifest["status"] == "clean"
    assert manifest["empty_streams"] == ["mocap"]  # events being empty is not reported
    events = pq.read_table(run_dir / "events.parquet").to_pylist()
    assert [e["kind"] for e in events] == ["streams_empty"]
    assert json.loads(events[0]["data_json"]) == {"streams": ["mocap"]}
    assert pq.read_table(run_dir / "mocap.parquet").num_rows == 0


def test_writer_thread_failure_marks_run_error_but_keeps_other_data(tmp_path, wait_for):
    writer = RunWriter(tmp_path / "run", [schema.setpoints_spec(), schema.events_spec()],
                       flush_interval_s=0.005)
    writer.start()
    writer.event("before")
    wait_for(lambda: writer.stats()["events"]["rows"] == 1)
    writer.append("setpoints", ("not a number", 0.0, 0.0, 0.0))
    wait_for(lambda: writer.error is not None)
    writer.event("after")  # queued after the thread died; close still drains it
    _close_quietly(writer)

    run_dir = tmp_path / "run"
    manifest = _manifest(run_dir)
    assert manifest["status"] == "error"
    assert "not a number" in manifest["error"]
    kinds = pq.read_table(run_dir / "events.parquet").column("kind").to_pylist()
    assert kinds[:2] == ["before", "after"]


def test_close_with_malformed_record_still_seals_the_run(tmp_path):
    writer = RunWriter(tmp_path / "run", [schema.setpoints_spec(), schema.events_spec()],
                       flush_interval_s=60.0)
    writer.start()
    writer.event("before")
    writer.append("setpoints", (1.0, 2.0, 3.0, 4.0))
    writer.append("setpoints", ("oops", 2.0, 3.0, 4.0))
    try:
        _close_quietly(writer)
    finally:
        manifest = _manifest(tmp_path / "run")
    assert manifest["status"] == "error"
    assert (tmp_path / "run" / "events.parquet").exists()


def test_two_producers_on_one_stream_keep_seq_in_file_order(tmp_path, monkeypatch):
    writer = RunWriter(tmp_path / "run", [schema.events_spec()], flush_interval_s=60.0)
    writer.start()
    q = writer._streams["events"].queue
    real_put, other = q.put_nowait, []

    def put_nowait(item):
        if not other:  # the GIL hands over to the camera thread right after seq was taken
            other.append(threading.Thread(target=writer.event, args=("deck_status",)))
            other[0].start()
            other[0].join(0.5)  # the writer holds its lock here, so this times out
        real_put(item)

    monkeypatch.setattr(q, "put_nowait", put_nowait)
    writer.event("takeoff")
    other[0].join(2.0)
    _close_quietly(writer)
    seq = pq.read_table(tmp_path / "run" / "events.parquet").column("seq").to_pylist()
    assert seq == [1, 2]  # validate_run fails "sequences strictly increasing" otherwise


def _snapshot(src, dst):
    """What a SIGKILL leaves behind: the files as they are on disk right now."""
    return shutil.copytree(src, dst)


def test_crashed_run_loads_complete_batches_and_ignores_a_torn_tail(tmp_path, wait_for):
    writer = RunWriter(tmp_path / "live", [schema.setpoints_spec(), schema.events_spec()],
                       flush_interval_s=0.005)
    writer.start()
    try:
        writer.event("start")
        for i in range(5):
            writer.append("setpoints", (float(i), 0.0, 0.0, 0.0))
        wait_for(lambda: writer.stats()["setpoints"]["rows"] == 5)
        for i in range(5, 8):
            writer.append("setpoints", (float(i), 0.0, 0.0, 0.0))
        wait_for(lambda: writer.stats()["setpoints"]["rows"] == 8
                 and writer.stats()["events"]["rows"] == 1)
        crashed = _snapshot(tmp_path / "live", tmp_path / "crashed")
    finally:
        _close_quietly(writer)

    run = load_run(crashed)
    assert run.manifest["status"] == "recording"
    assert run.streams["setpoints"].column("x").to_pylist() == [float(i) for i in range(8)]

    arrow = crashed / "setpoints.arrow"
    arrow.write_bytes(arrow.read_bytes()[:-24])  # power lost mid-batch
    run = load_run(crashed)
    assert run.streams["setpoints"].column("x").to_pylist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_crashed_run_with_a_still_empty_stream_loads(tmp_path, wait_for):
    writer = RunWriter(tmp_path / "live", [schema.setpoints_spec(), schema.mocap_spec("natnet"),
                                           schema.events_spec()], flush_interval_s=0.005)
    writer.start()
    try:
        writer.event("start")
        writer.append("setpoints", (1.0, 0.0, 0.0, 0.0))
        wait_for(lambda: writer.stats()["setpoints"]["rows"] == 1
                 and writer.stats()["events"]["rows"] == 1)
        crashed = _snapshot(tmp_path / "live", tmp_path / "crashed")
    finally:
        _close_quietly(writer)
    run = load_run(crashed)  # mocap never got a row before the crash
    assert run.streams["setpoints"].num_rows == 1
    assert run.streams["mocap"].num_rows == 0


def test_manifest_lists_every_log_variable_of_a_stream(tmp_path, write_run):
    spec = schema.custom_spec(
        CustomStreamConfig(name="tba", variables=("tba.tick:uint32_t", "tba.sin:float"))
    )
    run_dir = write_run(tmp_path / "run", [spec, schema.events_spec()],
                        {"tba": [(1, (10, 1, 0.5))]})
    meta = _manifest(run_dir)["streams"]["tba"]["meta"]
    assert "tba.tick:uint32_t" in json.dumps(meta)
    assert "tba.sin:float" in json.dumps(meta)


def test_schema_types_survive_extreme_values(tmp_path, write_run):
    spec = schema.custom_spec(CustomStreamConfig(
        name="ints", variables=("a.u8:uint8_t", "a.i8:int8_t", "a.u32:uint32_t", "a.i32:int32_t")))
    values = (2**24 - 1, 255, -128, 2**32 - 1, -(2**31))
    run_dir = write_run(tmp_path / "run", [spec, schema.events_spec()], {"ints": [(5, values)]})
    row = pq.read_table(run_dir / "ints.parquet").to_pylist()[0]
    assert (row["stm32_ms"], row["a.u8"], row["a.i8"], row["a.u32"], row["a.i32"]) == values
