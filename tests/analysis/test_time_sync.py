"""analyze_time_sync: drift, delay floor and timing recovered from device clocks."""
import numpy as np
import pytest

from cf_lab.analysis import analyze_time_sync
from cf_lab.analysis.time_sync import time_sync_figure
from cf_lab.telemetry.sync import load_run

STM32_DRIFT = -366e-6  # measured STM32 drift vs the host
STM32_EPOCH_S = 0.75  # host time of STM32 tick 0, after the run anchor


def stm32_host_s(ms):
    """True host time (s after the anchor) at which the STM32 measured ``ms``."""
    return STM32_EPOCH_S + np.asarray(ms, dtype=float) / 1000.0 * (1.0 + STM32_DRIFT)


def radio_delays_s(rng, n, floor_s=0.0):
    """One-sided link delay: a hard floor plus exponential jitter (mean 1.5 ms)."""
    return floor_s + rng.exponential(1.5e-3, n)


def state_pos(runs, ms, arrival_s):
    return {
        "host_ns": [runs.host_ns_at(t) for t in arrival_s],
        "stm32_ms": [int(m) for m in ms],
        **{f"stateEstimate.{k}": [0.0] * len(ms) for k in ("x", "y", "z", "vx", "vy", "vz")},
    }


def test_stm32_fit_recovers_drift_and_measurement_time_under_one_sided_jitter(tmp_path, runs):
    rng = np.random.default_rng(1)
    ms = np.arange(2000, 62000, 10)
    true_s = stm32_host_s(ms)
    run = runs.write_run(tmp_path, {
        "state_pos": state_pos(runs, ms, true_s + radio_delays_s(rng, ms.size)),
    })

    (sync,) = analyze_time_sync(run)

    assert sync.mapping == "stm32_ms"
    assert sync.drift_ppm == pytest.approx(STM32_DRIFT * 1e6, abs=1.0)
    assert np.max(np.abs(sync.t_s - true_s)) < 0.1e-3
    summary = sync.summary()
    assert summary["below_line"] == 0
    assert 0.0 <= summary["min_delay_ms"] < 0.1
    assert summary["median_delay_ms"] == pytest.approx(1.5 * np.log(2), abs=0.1)
    assert summary["drift_ms"] == pytest.approx(366e-6 * 60e3, abs=0.2)


def test_a_sample_arriving_before_it_was_measured_is_flagged_and_does_not_move_the_line(
    tmp_path, runs
):
    rng = np.random.default_rng(5)
    ms = np.arange(2000, 22000, 10)
    true_s = stm32_host_s(ms)
    arrival = true_s + radio_delays_s(rng, ms.size)
    arrival[700] = true_s[700] - 20e-3  # a host clock glitch
    run = runs.write_run(tmp_path, {"state_pos": state_pos(runs, ms, arrival)})

    (sync,) = analyze_time_sync(run)

    assert sync.summary()["below_line"] == 1
    assert sync.delay_ms[700] == pytest.approx(-20.0, abs=0.1)
    assert sync.drift_ppm == pytest.approx(STM32_DRIFT * 1e6, abs=2.0)
    assert np.max(np.abs(sync.t_s - true_s)) < 0.1e-3


def test_stm32_streams_share_one_line_so_a_slower_stream_floors_above_zero(tmp_path, runs):
    rng = np.random.default_rng(2)
    ms = np.arange(2000, 22000, 10)
    true_s = stm32_host_s(ms)
    ctrl_ms = ms + 3
    run = runs.write_run(tmp_path, {
        "state_pos": state_pos(runs, ms, true_s + radio_delays_s(rng, ms.size)),
        "ctrl": {
            "host_ns": [runs.host_ns_at(t) for t in
                        stm32_host_s(ctrl_ms) + radio_delays_s(rng, ms.size, floor_s=3e-3)],
            "stm32_ms": [int(m) for m in ctrl_ms],
            **{f"controller.cmd_{k}": [0.0] * ms.size for k in ("thrust", "roll", "pitch", "yaw")},
        },
    })

    pos, ctrl = analyze_time_sync(run, ["state_pos", "ctrl"])

    assert pos.drift_ppm == ctrl.drift_ppm
    assert pos.summary()["min_delay_ms"] < 0.1
    assert ctrl.summary()["min_delay_ms"] == pytest.approx(3.0, abs=0.1)


def test_natnet_fit_lands_frames_at_mid_exposure_after_removing_motive_latency(tmp_path, runs):
    rng = np.random.default_rng(3)
    frames = np.arange(10_000, 10_000 + 360 * 20)
    host_drift = 20e-6
    true_s = 0.4 + (frames - frames[0]) / 360.0 * (1.0 + host_drift)
    latency_us = rng.uniform(3000.0, 6000.0, frames.size)
    arrival = true_s + latency_us * 1e-6 + radio_delays_s(rng, frames.size)
    zeros = [0.0] * frames.size
    run = runs.write_run(tmp_path, {"mocap": {
        "host_ns": [runs.host_ns_at(t) for t in arrival],
        "source_ts_s": zeros, "source_seq": frames.tolist(),
        "x": zeros, "y": zeros, "z": zeros, "qx": zeros, "qy": zeros, "qz": zeros,
        "qw": [1.0] * frames.size, "latency_us": latency_us.tolist(),
    }})

    (sync,) = analyze_time_sync(run, ["mocap"])

    assert np.max(np.abs(sync.t_s - true_s)) < 0.1e-3
    assert sync.drift_ppm == pytest.approx(host_drift * 1e6, abs=2.0)
    assert sync.summary()["min_delay_ms"] < 0.1


def camera(runs, arrival_s, deck_ms, stm32_anchor_ms, deck_anchor_ms):
    n = len(arrival_s)
    zeros = [0] * n
    return {
        "host_ns": [runs.host_ns_at(t) for t in arrival_s],
        "frame_id": list(range(n)), "deck_ms": np.round(deck_ms).astype(int).tolist(),
        "width": [162] * n, "height": [122] * n, "pixel_format": zeros, "depth": [1] * n,
        "size": zeros, "blob_offset": zeros, "ext": [b""] * n,
        "stm32_anchor_ms": np.asarray(stm32_anchor_ms).astype(int).tolist(),
        "deck_anchor_ms": np.asarray(deck_anchor_ms).astype(int).tolist(),
    }


def test_anchored_camera_is_timed_through_the_stm32_clock_and_floors_at_the_wifi_delay(
    tmp_path, runs
):
    rng = np.random.default_rng(4)
    ms = np.arange(2000, 32000, 10)
    wifi_floor_s = 13e-3

    def gap8_ms(stm32_ms):
        return 7000.0 + np.asarray(stm32_ms, dtype=float) * (1.0 + 50e-6)

    frame_ms = np.arange(2105, 31000, 33.0)  # frame complete, on the STM32 clock
    anchor_ms = (frame_ms // 10) * 10 - 10  # newest state message the deck had received
    deck_anchor_ms = np.floor(gap8_ms(anchor_ms) + rng.uniform(0.2, 1.0, frame_ms.size))
    arrival = stm32_host_s(frame_ms) + radio_delays_s(rng, frame_ms.size, wifi_floor_s)
    run = runs.write_run(tmp_path, {
        "state_pos": state_pos(runs, ms, stm32_host_s(ms) + radio_delays_s(rng, ms.size)),
        "camera": camera(runs, arrival, gap8_ms(frame_ms), anchor_ms, deck_anchor_ms),
    })

    (sync,) = analyze_time_sync(run, ["camera"])

    assert sync.mapping == "gap8_ms -> stm32_ms"
    assert np.max(np.abs(sync.t_s - stm32_host_s(frame_ms))) < 2e-3
    assert sync.summary()["min_delay_ms"] == pytest.approx(13.0, abs=1.5)


def test_unanchored_camera_keeps_its_own_wifi_fit_with_the_link_floor_in_its_times(
    tmp_path, runs
):
    # camera_stream sends no STM32 anchors (-1): frames are fitted on the GAP8
    # clock over WiFi, so the link's minimum delay stays in their timestamps.
    rng = np.random.default_rng(6)
    gap8_drift = 80e-6
    true_s = 0.3 + np.arange(0, 30.0, 1 / 30.0)
    deck_ms = 7000.0 + (true_s - 0.3) * 1000.0 / (1.0 + gap8_drift)
    arrival = true_s + radio_delays_s(rng, true_s.size, floor_s=13e-3)
    none = [-1] * true_s.size
    run = runs.write_run(tmp_path, {"camera": camera(runs, arrival, deck_ms, none, none)})

    (sync,) = analyze_time_sync(run, ["camera"])

    assert sync.mapping == "gap8_ms"
    assert sync.drift_ppm == pytest.approx(gap8_drift * 1e6, abs=20.0)
    # deck_ms counts whole ms, so the line may sit up to ~1 ms under the true floor.
    assert np.median(sync.t_s - true_s) == pytest.approx(12.5e-3, abs=0.6e-3)
    assert 0.5 < sync.summary()["median_delay_ms"] < 2.5


def test_recorded_flight_has_one_stm32_drift_and_no_sample_below_its_line(flight_run):
    syncs = {sync.stream: sync for sync in analyze_time_sync(flight_run)}

    assert set(syncs) == {"state_pos", "state_att", "ctrl", "mocap"}
    stm32 = [syncs[name] for name in ("state_pos", "state_att", "ctrl")]
    assert all(sync.drift_ppm == pytest.approx(stm32[0].drift_ppm, abs=1e-6) for sync in stm32)
    assert -400.0 < stm32[0].drift_ppm < -330.0
    assert abs(syncs["mocap"].drift_ppm) < 50.0
    for sync in syncs.values():
        summary = sync.summary()
        assert summary["below_line"] == 0, sync.stream
        assert summary["min_delay_ms"] < 0.5, sync.stream
        assert summary["median_delay_ms"] < 5.0, sync.stream


def test_time_sync_figure_draws_offset_and_delay_per_stream(flight_run):
    syncs = analyze_time_sync(flight_run, ["state_pos", "mocap"])

    figure = time_sync_figure(syncs)

    markers = [trace for trace in figure.data if trace.mode == "markers"]
    assert len(markers) == 4
    for row, sync in enumerate(syncs):
        offset, delay = markers[2 * row], markers[2 * row + 1]
        assert np.array_equal(offset.y, sync.offset_ms)
        assert np.array_equal(delay.y, sync.delay_ms)


@pytest.mark.parametrize("empty", ["ctrl", "camera"])
def test_analyze_time_sync_default_skips_enabled_streams_that_recorded_nothing(
    tmp_path, runs, empty
):
    ms = np.arange(2000, 4000, 10)
    columns = {
        "ctrl": ["host_ns", "stm32_ms", "controller.cmd_thrust", "controller.cmd_roll",
                 "controller.cmd_pitch", "controller.cmd_yaw"],
        "camera": ["host_ns", "frame_id", "deck_ms", "width", "height", "pixel_format", "depth",
                   "size", "blob_offset", "ext", "stm32_anchor_ms", "deck_anchor_ms"],
    }[empty]
    run = runs.write_run(tmp_path, {
        "state_pos": state_pos(runs, ms, stm32_host_s(ms) + 1e-3),
        empty: {name: [] for name in columns},
    })
    assert load_run(run).manifest["empty_streams"] == [empty]

    syncs = analyze_time_sync(run)

    assert [sync.stream for sync in syncs] == ["state_pos"]
