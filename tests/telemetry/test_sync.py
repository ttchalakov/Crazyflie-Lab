"""Clock alignment: lower-envelope fits, counter unwrap, and load_run's per-stream measurement times."""
import json
import random

import pytest

from cf_lab.telemetry import schema
from cf_lab.telemetry.sync import STM32_MS_WRAP, fit_affine, load_run, unwrap_wrapping_counter

ANCHOR_NS = 0  # replaces the manifest's created_monotonic_ns, so t_s is true host time in seconds
DRIFT = -366e-6  # measured STM32 drift vs the host
SLOPE = 1e6 * (1.0 + DRIFT)  # host ns per STM32 ms
OFFSET_NS = 2.5e14


def _true_host_ns(stm32_ms):
    return SLOPE * stm32_ms + OFFSET_NS


def _stm32_samples(rng, n=3000, start_ms=5000, mean_delay_ns=2e6):
    device = [start_ms + 10 * i for i in range(n)]
    host = [int(_true_host_ns(d) + rng.expovariate(1.0 / mean_delay_ns)) for d in device]
    return device, host


def test_fit_affine_recovers_drift_and_offset_along_fastest_arrivals():
    rng = random.Random(1)
    device, host = _stm32_samples(rng)
    fit = fit_affine(device, host)
    assert fit.slope == pytest.approx(SLOPE, rel=5e-6)
    mid = device[len(device) // 2]
    # A fit through the middle would sit ~2 ms (the mean delay) late.
    assert abs(fit.to_host_ns(mid) - _true_host_ns(mid)) < 0.1e6
    assert fit.n == len(device)
    assert 1.0e6 < fit.rms_ns < 3.0e6  # spread of a 2 ms exponential delay


def test_fit_affine_ignores_an_impossibly_early_glitch():
    rng = random.Random(2)
    device, host = _stm32_samples(rng)
    host[1500] -= int(50e6)  # a clock glitch: arrived 50 ms before it was measured
    fit = fit_affine(device, host)
    mid = device[len(device) // 2]
    assert abs(fit.to_host_ns(mid) - _true_host_ns(mid)) < 0.1e6
    assert fit.slope == pytest.approx(SLOPE, rel=5e-6)
    assert fit.n == len(device) - 1


def test_fit_affine_on_exact_points_is_exact():
    fit = fit_affine([0, 10, 20, 30], [1000, 1010, 1020, 1030])
    assert (fit.slope, fit.offset_ns, fit.rms_ns) == (1.0, 1000.0, 0.0)


@pytest.mark.parametrize(
    "device, host",
    [([5], [1]), ([5, 5, 5], [1, 2, 3]), ([1, 2], [1])],
)
def test_fit_affine_rejects_unfittable_input(device, host):
    with pytest.raises(ValueError):
        fit_affine(device, host)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([10, 20, 30], [10, 20, 30]),
        ([STM32_MS_WRAP - 20, STM32_MS_WRAP - 10, 0, 10],
         [STM32_MS_WRAP - 20, STM32_MS_WRAP - 10, STM32_MS_WRAP, STM32_MS_WRAP + 10]),
        ([100, 90, 110], [100, 90, 110]),  # reordered packet, not a wrap
    ],
)
def test_unwrap_wrapping_counter(values, expected):
    assert unwrap_wrapping_counter(values, STM32_MS_WRAP) == expected


def _write(write_run, run_dir, specs, rows):
    run_dir = write_run(run_dir, specs, rows)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["created_monotonic_ns"] = ANCHOR_NS
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


def test_load_run_puts_stm32_rows_at_measurement_time_across_counter_wrap(tmp_path, write_run):
    rng = random.Random(3)
    unwrapped = [STM32_MS_WRAP - 5000 + 10 * i for i in range(1000)]  # wraps at i = 500
    pos = [(int(_true_host_ns(d) + rng.expovariate(1 / 3e6)), (d % STM32_MS_WRAP, 0, 0, 0, 0, 0, 0))
           for d in unwrapped]
    att = [(int(_true_host_ns(d + 3) + rng.expovariate(1 / 3e6)),
            ((d + 3) % STM32_MS_WRAP, 0, 0, 0, 0, 0, 0)) for d in unwrapped]
    run_dir = _write(write_run, tmp_path / "run",
                     [schema.state_pos_spec(100), schema.state_att_spec(100), schema.events_spec()],
                     {"state_pos": pos, "state_att": att})
    run = load_run(run_dir)

    fit = run.fits["stm32_ms"]
    assert fit.n == 2000  # both STM32 streams share one fit
    assert fit.slope == pytest.approx(SLOPE, rel=2e-5)
    t_s = run.streams["state_pos"].column("t_s").to_pylist()
    truth = [_true_host_ns(d) / 1e9 for d in unwrapped]
    assert max(abs(a - b) for a, b in zip(t_s, truth)) < 0.3e-3
    t_host = run.streams["state_pos"].column("t_host_s").to_pylist()
    assert t_host == [h / 1e9 for h, _ in pos]


def test_load_run_takes_motive_latency_off_natnet_mocap(tmp_path, write_run):
    rng = random.Random(4)
    frame_ns = 1e9 / 360.0
    exposure_ns = [3e14 + frame_ns * f for f in range(2000)]
    latency_us = 4000.0
    rows = [
        (int(t + latency_us * 1e3 + rng.expovariate(1 / 0.3e6)),
         (0.0, f, 0, 0, 0, 0, 0, 0, 1, latency_us))
        for f, t in enumerate(exposure_ns)
    ]
    run_dir = _write(write_run, tmp_path / "run",
                     [schema.mocap_spec("natnet"), schema.events_spec()], {"mocap": rows})
    t_s = load_run(run_dir).streams["mocap"].column("t_s").to_pylist()
    errors = [a - t / 1e9 for a, t in zip(t_s, exposure_ns)]
    assert max(abs(e) for e in errors) < 0.05e-3  # mid-exposure, not 4 ms later at transmit


def test_load_run_maps_vrpn_ros_stamps_at_epoch_scale(tmp_path, write_run):
    rng = random.Random(5)
    stamps = [1.79e9 + 0.01 * i for i in range(1000)]  # ROS header stamps, epoch seconds
    true_ns = [4e13 + (s - stamps[0]) * 1e9 * (1 + 20e-6) for s in stamps]
    rows = [(int(t + rng.expovariate(1 / 1e6)), (s, -1, 0, 0, 0, 0, 0, 0, 1, -1.0))
            for s, t in zip(stamps, true_ns)]
    run_dir = _write(write_run, tmp_path / "run",
                     [schema.mocap_spec("vrpn"), schema.events_spec()], {"mocap": rows})
    t_s = load_run(run_dir).streams["mocap"].column("t_s").to_pylist()
    assert max(abs(a - t / 1e9) for a, t in zip(t_s, true_ns)) < 0.1e-3


def _camera_row(frame_id, deck_ms, stm32_anchor, deck_anchor):
    return (frame_id, deck_ms, 324, 244, 1, 1, 100, 100 * frame_id, b"", stm32_anchor, deck_anchor)


@pytest.mark.parametrize("tick_base", [0, 3 * STM32_MS_WRAP], ids=["fresh_boot", "uptime_over_4h40"])
def test_load_run_maps_camera_frames_through_stm32_anchors(tmp_path, write_run, tick_base):
    """A frame's time comes from the onboard anchors, skipping the 13 ms+ WiFi delay.

    Anchors carry the full 32-bit STM32 tick; log blocks only its low 24 bits.
    """
    rng = random.Random(6)

    def deck_from_stm32(ms):  # the GAP8 clock: own epoch, 200 ppm fast
        return 1.0002 * ms + 777_000.0

    stm32_ms = [10_000 + 10 * i for i in range(1500)]
    pos = [(int(_true_host_ns(d) + rng.expovariate(1 / 2e6)), (d, 0, 0, 0, 0, 0, 0))
           for d in stm32_ms]
    frames, truth = [], []
    for k in range(200):
        done_stm32 = 10_500 + 70.0 * k + rng.random()  # frame complete, on the STM32 clock
        stamp = int(done_stm32) - rng.randint(1, 9)  # newest STM32 message the deck had
        arrived = stamp + rng.expovariate(1 / 1.0)  # its arrival over SPI/UART
        host = _true_host_ns(done_stm32) + 13e6 + rng.expovariate(1 / 5e6)  # WiFi
        frames.append((int(host), _camera_row(k, round(deck_from_stm32(done_stm32)),
                                              tick_base + stamp, int(deck_from_stm32(arrived)))))
        truth.append(_true_host_ns(done_stm32) / 1e9)
    run_dir = _write(write_run, tmp_path / "run",
                     [schema.state_pos_spec(100), schema.camera_spec(), schema.events_spec()],
                     {"state_pos": pos, "camera": frames})
    run = load_run(run_dir)

    assert "gap8_ms_from_stm32_ms" in run.fits
    t_s = run.streams["camera"].column("t_s").to_pylist()
    assert max(abs(a - b) for a, b in zip(t_s, truth)) < 2.5e-3


def test_load_run_camera_without_extension_keeps_arrival_time(tmp_path, write_run):
    frames = [(10_000_000 * k, _camera_row(-1, -1, -1, -1)) for k in range(5)]
    run_dir = _write(write_run, tmp_path / "run", [schema.camera_spec(), schema.events_spec()],
                     {"camera": frames})
    camera = load_run(run_dir).streams["camera"]
    assert camera.column("t_s").to_pylist() == camera.column("t_host_s").to_pylist()


def test_load_run_host_native_streams_keep_arrival_time(tmp_path, write_run):
    rows = {"setpoints": [(1_500_000_000, (0, 0, 1, 0))], "events": [(2_000_000_000, ("x", "", "{}"))]}
    run_dir = _write(write_run, tmp_path / "run", [schema.setpoints_spec(), schema.events_spec()], rows)
    run = load_run(run_dir)
    assert run.fits == {}
    assert run.streams["setpoints"].column("t_s").to_pylist() == [1.5]
    assert run.streams["events"].column("t_host_s").to_pylist() == [2.0]
