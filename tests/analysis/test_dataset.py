"""load_dataset and TelemetryDataset: units, alignment grid, interpolation and stream checks."""
import math

import numpy as np
import pytest

from cf_lab.analysis import TelemetryDataset, load_dataset
from cf_lab.telemetry.sync import load_run

POS = ("x", "y", "z", "vx", "vy", "vz")
ATT = ("roll", "pitch", "yaw")
GYRO = ("x", "y", "z")
CMDS = ("thrust", "roll", "pitch", "yaw")
RADIO_DELAY_S = 0.002


def stm32_rows(runs, ms, columns):
    """STM32 log rows stamped at ``ms`` that arrive RADIO_DELAY_S later."""
    return {
        "host_ns": [runs.host_ns_at(m / 1000 + RADIO_DELAY_S) for m in ms],
        "stm32_ms": list(ms),
        **{name: list(values) for name, values in columns.items()},
    }


def state_pos(runs, ms, **values):
    return stm32_rows(runs, ms, {
        f"stateEstimate.{k}": values.get(k, [0.0] * len(ms)) for k in POS
    })


def state_att(runs, ms, *, roll=None, pitch=None, yaw=None, gyro=(None, None, None)):
    zeros = [0.0] * len(ms)
    columns = {f"stateEstimate.{k}": v or zeros for k, v in zip(ATT, (roll, pitch, yaw))}
    columns.update({f"gyro.{k}": v or zeros for k, v in zip(GYRO, gyro)})
    return stm32_rows(runs, ms, columns)


def ctrl(runs, ms, thrust):
    zeros = [0.0] * len(ms)
    return stm32_rows(runs, ms, {f"controller.cmd_{k}": thrust if k == "thrust" else zeros
                                 for k in CMDS})


def test_load_dataset_converts_firmware_units_and_signs(tmp_path, runs):
    ms = range(1000, 1500, 10)
    n = len(ms)
    run = runs.write_run(tmp_path, {
        "state_pos": state_pos(runs, ms, x=[1.0] * n, y=[2.0] * n, z=[0.5] * n,
                               vx=[0.1] * n, vy=[0.2] * n, vz=[0.3] * n),
        "state_att": state_att(runs, ms, roll=[10.0] * n, pitch=[20.0] * n, yaw=[30.0] * n,
                               gyro=([90.0] * n, [-45.0] * n, [180.0] * n)),
    })

    data = load_dataset(run)

    expected = {"x": 1.0, "y": 2.0, "z": 0.5, "vx": 0.1, "vy": 0.2, "vz": 0.3,
                "roll": math.radians(10), "pitch": -math.radians(20), "yaw": math.radians(30),
                "p": math.radians(90), "q": math.radians(-45), "r": math.pi}
    assert data.state_names == tuple(expected)
    for name, value in expected.items():
        assert data[name] == pytest.approx([value] * len(data)), name
    assert data.units["pitch"] == "rad" and data.units["q"] == "rad/s"


def test_load_dataset_unwraps_yaw_across_180_degrees_before_interpolating(tmp_path, runs):
    # Attitude and position blocks log 5 ms apart, so every grid point falls
    # halfway between two yaw samples, including the one straddling +-180.
    att_ms = list(range(1000, 1200, 10))
    unwrapped_deg = [170.0 + 2.0 * i for i in range(len(att_ms))]
    wrapped_deg = [(d + 180.0) % 360.0 - 180.0 for d in unwrapped_deg]
    pos_ms = [m + 5 for m in att_ms[:-1]]
    run = runs.write_run(tmp_path, {
        "state_pos": state_pos(runs, pos_ms),
        "state_att": state_att(runs, att_ms, yaw=wrapped_deg),
    })

    data = load_dataset(run)

    true_yaw = np.radians(np.interp(np.array(data.t_s) - data.t_s[0] + 0.005,
                                    (np.array(att_ms) - 1000) / 1000, unwrapped_deg))
    assert np.allclose(data["yaw"], true_yaw, atol=1e-6)


def test_load_dataset_holds_the_previous_control_command(tmp_path, runs):
    state_ms = list(range(1005, 1205, 10))
    ctrl_ms = list(range(1000, 1210, 10))
    thrust = [float(30000 + 1000 * (i % 2)) for i in range(len(ctrl_ms))]
    run = runs.write_run(tmp_path, {
        "state_pos": state_pos(runs, state_ms),
        "state_att": state_att(runs, state_ms),
        "ctrl": ctrl(runs, ctrl_ms, thrust),
    })

    data = load_dataset(run)

    assert data.input_names == ("thrust_cmd", "roll_cmd", "pitch_cmd", "yaw_cmd")
    # Each grid point sits 5 ms after a command: it must read that command, not a blend.
    previous = [thrust[(round((t - data.t_s[0]) * 1000) + 5) // 10] for t in data.t_s]
    assert list(data["thrust_cmd"]) == previous


def test_load_dataset_grid_covers_only_the_common_span_at_the_slowest_state_rate(tmp_path, runs):
    pos_ms = list(range(1000, 3001, 10))  # 100 Hz over 1.0..3.0 s
    att_ms = list(range(1500, 2501, 20))  # 50 Hz over 1.5..2.5 s
    run = runs.write_run(tmp_path, {
        "state_pos": state_pos(runs, pos_ms),
        "state_att": state_att(runs, att_ms),
    })
    att_t = load_run(run).streams["state_att"]["t_s"].to_pylist()

    data = load_dataset(run)

    assert data.metadata["sample_hz"] == pytest.approx(50.0, rel=1e-6)
    assert data.t_s[0] == pytest.approx(att_t[0], abs=1e-9)
    assert data.t_s[-1] == pytest.approx(att_t[-1], abs=1e-6)
    assert len(data) == 51
    assert np.allclose(np.diff(data.t_s), 0.02)


def test_load_dataset_grid_rate_is_not_lowered_by_the_held_setpoints(flight_run):
    state_only = load_dataset(flight_run)

    with_setpoints = load_dataset(flight_run, include_setpoints=True)

    assert state_only.metadata["sample_hz"] == pytest.approx(100.0, rel=0.01)
    assert with_setpoints.metadata["sample_hz"] == pytest.approx(
        state_only.metadata["sample_hz"])


def test_load_dataset_leaves_out_an_enabled_but_empty_control_stream(tmp_path, runs):
    ms = list(range(1000, 1500, 10))
    run = runs.write_run(tmp_path, {
        "state_pos": state_pos(runs, ms),
        "state_att": state_att(runs, ms),
        "ctrl": ctrl(runs, [], []),
    })

    data = load_dataset(run)

    assert data.input_names == ()
    assert "thrust_cmd" not in data.signals
    assert len(data) == len(ms)


def test_load_dataset_leaves_out_unrecorded_mocap_position_and_setpoints(tmp_path, runs):
    # A run that ends before takeoff has an enabled but empty setpoints stream.
    ms = list(range(1000, 1500, 10))
    run = runs.write_run(tmp_path, {
        "state_pos": state_pos(runs, ms),
        "state_att": state_att(runs, ms),
        "setpoints": {"host_ns": [], "x": [], "y": [], "z": [], "yaw_deg": []},
    })

    data = load_dataset(run, include_mocap_position=True, include_setpoints=True)

    assert not {"mocap_x", "setpoint_x"} & set(data.signals)
    assert not data.metadata["mocap_position_included"]
    assert not data.metadata["setpoints_included"]
    assert len(data) == len(ms)


@pytest.mark.parametrize("streams, kwargs, match", [
    (("state_pos",), {}, "state_att"),
    (("state_pos", "empty_state_att"), {}, "state_att"),
])
def test_load_dataset_names_the_stream_a_run_is_missing(tmp_path, runs, streams, kwargs, match):
    ms = list(range(1000, 1500, 10))
    available = {
        "state_pos": state_pos(runs, ms),
        "state_att": state_att(runs, ms),
        "empty_state_att": state_att(runs, []),
    }
    run = runs.write_run(tmp_path, {
        name.removeprefix("empty_"): available[name] for name in streams
    })

    with pytest.raises(ValueError, match=match):
        load_dataset(run, **kwargs)


def test_load_dataset_on_a_recorded_flight_tracks_raw_columns_and_mocap(flight_run):
    data = load_dataset(flight_run, include_mocap_position=True, include_setpoints=True)

    t = np.array(data.t_s)
    assert np.allclose(np.diff(t), 1.0 / data.metadata["sample_hz"])
    att = flight_run.streams["state_att"]
    raw_t = att["t_s"].to_numpy()
    raw_pitch = att["stateEstimate.pitch"].to_numpy().astype(float)
    raw_gyro_y = att["gyro.y"].to_numpy().astype(float)
    assert np.allclose(data["pitch"], -np.radians(np.interp(t, raw_t, raw_pitch)), atol=1e-6)
    assert np.allclose(data["q"], np.radians(np.interp(t, raw_t, raw_gyro_y)), atol=1e-6)
    # The EKF fuses the zeroed mocap: in the air the two agree to a few centimetres.
    airborne = np.array(data["z"]) > 0.3
    assert airborne.sum() > 500
    for axis in "xyz":
        error = np.abs(np.array(data[axis]) - np.array(data[f"mocap_{axis}"]))[airborne]
        assert np.median(error) < 0.03, axis


@pytest.mark.parametrize("t_s, signals, match", [
    ([0.0, 0.1, 0.1], {"x": [1.0, 2.0, 3.0]}, "strictly increasing"),
    ([0.0, 0.2, 0.1], {"x": [1.0, 2.0, 3.0]}, "strictly increasing"),
    ([0.0, 0.1, 0.2], {"x": [1.0, 2.0]}, "samples"),
    ([0.0, 0.1, 0.2], {"x": [1.0, math.nan, 3.0]}, "non-finite"),
])
def test_telemetry_dataset_rejects_misaligned_or_non_finite_samples(t_s, signals, match):
    with pytest.raises(ValueError, match=match):
        TelemetryDataset(t_s, signals)
