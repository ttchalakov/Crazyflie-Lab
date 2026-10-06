"""validate_run on a real recorded bench run, then on copies with one defect injected each."""
import json
import math

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from cf_lab.mocap.relative import OriginZeroer
from cf_lab.mocap.types import PoseSample, Quaternion
from cf_lab.telemetry import schema
from cf_lab.telemetry.integrity import validate_run

FULL_CHECKS = dict(
    required_streams=("state_pos", "tba", "mocap", "extpose", "setpoints"),
    expected_rows={"state_pos": 260, "tba": 260},
    maximum_rows={"setpoints": 200},
    required_events=("origin_zeroed",),
    source_sequence_streams=("mocap",),
    recreate_extpose=True,
    clock_rms_limits_ns={"stm32_ms": 5e6},
)


def _edit_table(run_dir, name, edit):
    path = run_dir / f"{name}.parquet"
    pq.write_table(edit(pq.read_table(path)), path)


def _edit_manifest(run_dir, edit):
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))


def _labels(report):
    return sorted(check.label for check in report.failures)


def test_real_bench_run_passes_every_check(bench_run):
    run, report = validate_run(bench_run, **FULL_CHECKS)
    assert report.failures == ()
    assert report.passed
    assert run.fits["stm32_ms"].rms_ns < 2e6  # 0.99 ms on this recording
    # Every STM32 log block and the extpose check actually ran.
    labels = {check.label for check in report.checks}
    assert "state_pos stream records until the run ends" in labels
    assert "tba stream records until the run ends" in labels
    assert "extpose recreated from mocap + origin event" in labels


def _unclean(run_dir):
    _edit_manifest(run_dir, lambda m: m.update(status="recording"))


def _missing_state_pos(run_dir):
    (run_dir / "state_pos.parquet").unlink()


def _queue_drops(run_dir):
    _edit_manifest(run_dir, lambda m: m["streams"]["mocap"].update(dropped=12))


def _seq_out_of_order(run_dir):
    def swap(table):
        idx = list(range(table.num_rows))
        idx[10], idx[11] = idx[11], idx[10]
        return table.take(idx)
    _edit_table(run_dir, "setpoints", swap)


def _source_seq_backwards(run_dir):
    def rewind(table):
        seq = table.column("source_seq").to_pylist()
        seq[100] = seq[98]
        return table.set_column(table.schema.get_field_index("source_seq"), "source_seq",
                                pa.array(seq, pa.int64()))
    _edit_table(run_dir, "mocap", rewind)


def _log_block_dies(run_dir):
    def cut(table):
        host = table.column("host_ns")
        stop = pc.min(host).as_py() + int(0.2e9)
        return table.filter(pc.less_equal(host, stop))
    _edit_table(run_dir, "tba", cut)


def _extpose_corrupted(run_dir):
    def nudge(table):
        x = table.column("x").to_pylist()
        x[50] += 1e-4
        return table.set_column(table.schema.get_field_index("x"), "x", pa.array(x, pa.float64()))
    _edit_table(run_dir, "extpose", nudge)


def _origin_event_lost(run_dir):
    _edit_table(run_dir, "events", lambda t: t.filter(pc.not_equal(t.column("kind"), "origin_zeroed")))


def _half_delivered(run_dir):
    _edit_table(run_dir, "state_pos", lambda t: t.take(list(range(0, t.num_rows, 2))))


@pytest.mark.parametrize(
    "inject, failures",
    [
        (_unclean, ["manifest closed clean"]),
        (_missing_state_pos, ["state_pos delivery >= 90%", "state_pos stream contains rows",
                              "state_pos stream records until the run ends"]),
        (_queue_drops, ["no writer-queue drops"]),
        (_seq_out_of_order, ["recorded stream sequences strictly increasing"]),
        (_source_seq_backwards, ["mocap source_seq strictly increasing"]),
        (_log_block_dies, ["tba delivery >= 90%", "tba stream records until the run ends"]),
        (_extpose_corrupted, ["extpose recreated from mocap + origin event"]),
        (_origin_event_lost, ["extpose recreated from mocap + origin event",
                              "origin_zeroed event recorded"]),
        (_half_delivered, ["state_pos delivery >= 90%"]),
    ],
    ids=lambda v: v.__name__.strip("_") if callable(v) else None,
)
def test_injected_defect_fails_exactly_its_check(bench_run, inject, failures):
    inject(bench_run)
    _, report = validate_run(bench_run, **FULL_CHECKS)
    assert _labels(report) == sorted(failures)
    assert not report.passed


def test_mocap_frame_gaps_are_counted_but_not_failed(bench_run):
    _edit_table(bench_run, "mocap", lambda t: t.take([i for i in range(t.num_rows) if i % 100 != 7]))
    _, report = validate_run(bench_run, source_sequence_streams=("mocap",))
    check = next(c for c in report.checks if c.label == "mocap source_seq strictly increasing")
    assert check.status == "PASS"
    assert check.detail == "50 frame(s) in gaps"


def test_skipped_stream_is_not_failed_for_being_empty(bench_run):
    _missing_state_pos(bench_run)
    _, report = validate_run(
        bench_run, required_streams=("state_pos", "mocap"),
        skipped_streams={"state_pos": "fake backend has no STM32"},
    )
    assert report.passed
    assert [c.label for c in report.skipped] == ["state_pos stream"]


def test_clock_limits_fail_on_jitter_and_on_a_missing_fit(bench_run):
    _, report = validate_run(bench_run, clock_rms_limits_ns={"stm32_ms": 0.1e6, "gap8_ms": 5e6})
    assert _labels(report) == ["gap8_ms clock fit available", "stm32_ms clock fit rms <= 0.1 ms"]


def test_row_cap_catches_an_overproducing_stream(bench_run):
    _, report = validate_run(bench_run, maximum_rows={"setpoints": 100})
    assert _labels(report) == ["setpoints respects row cap"]


def _zeroed_uplink_run(run_dir, write_run):
    """Raw mocap plus the extpose OriginZeroer really produced from it."""
    zeroer = OriginZeroer()
    mocap_rows, extpose_rows, origin = [], [], None
    for i in range(50):
        yaw = 0.3 + 0.01 * i
        pose = PoseSample(
            timestamp_s=1.79e9 + 0.01 * i,
            position_m=(1.0 + 0.01 * i, -0.5, 0.02 * i),
            orientation=Quaternion(qx=0.0, qy=0.0, qz=math.sin(yaw / 2), qw=math.cos(yaw / 2)),
            rigid_body="cf",
        )
        sent = zeroer.zero(pose)
        if origin is None:
            origin = {"x": pose.position_m[0], "y": pose.position_m[1], "z": pose.position_m[2],
                      "yaw_rad": pose.euler_rad.yaw}
        seq = i + 1
        q, sq = pose.orientation, sent.orientation
        host = 1_000_000 * (i + 1)
        mocap_rows.append((host, (pose.timestamp_s, seq, *pose.position_m, q.qx, q.qy, q.qz, q.qw, -1.0)))
        extpose_rows.append((host + 1, (pose.timestamp_s, seq, *sent.position_m, sq.qx, sq.qy, sq.qz, sq.qw)))
    return write_run(
        run_dir,
        [schema.mocap_spec("natnet"), schema.extpose_spec(), schema.events_spec()],
        {"mocap": mocap_rows, "extpose": extpose_rows,
         "events": [(0, ("origin_zeroed", "", json.dumps(origin)))]},
    )


def test_extpose_recreation_passes_for_natnet_mocap(tmp_path, write_run):
    _, report = validate_run(_zeroed_uplink_run(tmp_path / "run", write_run),
                             recreate_extpose=True)
    assert report.passed, report.format()
