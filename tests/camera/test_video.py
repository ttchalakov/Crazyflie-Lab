"""Rendering a recorded camera run: hold timing, lossless VP9 exactness, deck fps, extraction."""

import re
import subprocess
import time

import cv2
import numpy as np
import pytest

from cf_lab.camera.ext import DeckLine, decode_camera_stream
from cf_lab.camera.feed import CapturedFrame
from cf_lab.camera.video import (
    extract_frames,
    ffmpeg_executable,
    hold_schedule,
    load_frames,
    render_video,
)
from cf_lab.imgstream import ImgHeader
from cf_lab.telemetry.camera import CameraRecorder
from cf_lab.telemetry.schema import camera_spec, events_spec
from cf_lab.telemetry.writer import RunWriter


def _record(run_dir, frames, *, deck_fps=None):
    """A run directory as the live recorder writes it.

    ``frames`` is [(t_s, payload, width, height, is_jpeg)]; no extension, so t_s is arrival.
    """
    writer = RunWriter(run_dir, [camera_spec(), events_spec()], flush_interval_s=0.01)
    writer.start()
    recorder = CameraRecorder(writer)
    recorder.start()
    if deck_fps is not None:
        recorder.offer_console(DeckLine("config", f"camera config: fps={deck_fps}",
                                        {"fps": deck_fps}))
    base_ns = time.monotonic_ns()
    for t_s, payload, width, height, is_jpeg in frames:
        header = ImgHeader(0xBC, int(is_jpeg), 1, 0, width, height, len(payload))
        recorder.offer(CapturedFrame(header, payload, None, b"", base_ns + int(t_s * 1e9)))
    recorder.stop()
    writer.close()
    return run_dir


def _decode_video(path):
    """Every frame of ``path`` as (Y, U, V) planes, with no colour conversion."""
    out = subprocess.run(
        [ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        check=True, capture_output=True,
    ).stdout
    probe = subprocess.run([ffmpeg_executable(), "-hide_banner", "-i", str(path)],
                           capture_output=True, text=True).stderr
    width, height = map(int, re.search(r", (\d+)x(\d+)", probe).groups())
    luma, chroma = width * height, (width // 2) * (height // 2)
    step = luma + 2 * chroma
    planes = []
    for start in range(0, len(out), step):
        frame = np.frombuffer(out[start:start + step], np.uint8)
        planes.append((frame[:luma].reshape(height, width), frame[luma:luma + chroma],
                       frame[luma + chroma:]))
    return planes, probe


@pytest.mark.parametrize("timestamps, fps, expected", [
    ([0.0, 0.1, 0.2, 0.3], 10, [0, 1, 2, 3]),                # deck rate == playback rate
    ([0.0, 1 / 30, 2 / 30, 3 / 30], 30, [0, 1, 2, 3]),       # on-grid ties within an ULP
    ([0.0, 0.1, 0.45], 10, [0, 1, 1, 1, 1, 2]),              # a gap holds the last frame
    ([0.0, 0.01, 0.02, 0.1], 10, [0, 3]),                    # a burst shows only its newest
    ([0.0, 0.31], 10, [0, 0, 0, 0, 1]),                      # last slot still reaches the end
    ([5.0], 30, [0]),
    ([], 30, []),
])
def test_hold_schedule_shows_the_frame_current_at_each_slot(timestamps, fps, expected):
    assert hold_schedule(timestamps, fps) == expected


@pytest.mark.parametrize("fps", [0, -30])
def test_hold_schedule_rejects_a_non_positive_rate(fps):
    with pytest.raises(ValueError):
        hold_schedule([0.0, 1.0], fps)


def test_lossless_render_is_pixel_exact_full_range_at_native_size(tmp_path, qvga_frames):
    ramp = np.tile(np.arange(256, dtype=np.uint8), 300)[:320 * 240].reshape(240, 320)
    images = [ramp, *qvga_frames]
    # 0.0 .. 0.4 at 10 FPS with one gap: slot 3 holds the frame from 0.2.
    times = [0.0, 0.1, 0.2, 0.4, 0.5]
    run = _record(tmp_path / "run", [(t, im.tobytes(), 320, 240, False)
                                     for t, im in zip(times, images)])
    out = tmp_path / "lossless.mp4"

    stats = render_video(run, out, fps=10, lossless=True)
    planes, probe = _decode_video(out)

    shown = [images[i] for i in (0, 1, 2, 2, 3, 4)]
    assert (stats.output_frames, stats.held, stats.skipped) == (6, 1, 0)
    assert len(planes) == 6
    for (y, u, v), expected in zip(planes, shown):
        assert np.array_equal(y, expected)
        assert (u == 128).all() and (v == 128).all()
    # Chromium honours full range only with an explicit matrix tag.
    assert re.search(r"yuv420p\(pc, bt470bg", probe), probe


def test_lossless_render_of_real_jpeg_frames_matches_their_decode(tmp_path, deck_frames):
    frames = deck_frames[:3]
    run = _record(tmp_path / "run", [(i / 60, f["payload"], 320, 240, True)
                                     for i, f in enumerate(frames)])
    out = tmp_path / "jpeg.mp4"

    render_video(run, out, fps=60, lossless=True)
    planes, _ = _decode_video(out)

    expected = [cv2.imdecode(np.frombuffer(f["payload"], np.uint8), cv2.IMREAD_GRAYSCALE)
                for f in frames]
    assert len(planes) == 3
    assert all(np.array_equal(y, e) for (y, _, _), e in zip(planes, expected))


def test_loaded_frames_decode_to_the_recorded_pixels(tmp_path, deck_frames, qvga_frames):
    raw, jpeg = qvga_frames[0], deck_frames[0]["payload"]
    run = _record(tmp_path / "run", [(0.0, raw.tobytes(), 320, 240, False),
                                     (0.1, jpeg, 320, 240, True)])

    frames = load_frames(run)

    assert np.array_equal(frames[0].decode(), raw)
    assert np.array_equal(frames[1].decode(),
                          cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE))


def test_render_defaults_to_the_fps_the_deck_reported(tmp_path, qvga_frames):
    run = _record(tmp_path / "run", [(i * 0.05, qvga_frames[i % 4].tobytes(), 320, 240, False)
                                     for i in range(5)], deck_fps=20)
    out = tmp_path / "camera.mp4"

    stats = render_video(run, out)
    planes, probe = _decode_video(out)

    assert stats.fps == 20 and stats.output_frames == 5 and len(planes) == 5
    assert ", 20 fps" in probe and planes[0][0].shape == (240, 320)


def test_render_without_a_deck_config_uses_the_default_rate(tmp_path, qvga_frames):
    run = _record(tmp_path / "run", [(0.0, qvga_frames[0].tobytes(), 320, 240, False),
                                     (0.1, qvga_frames[1].tobytes(), 320, 240, False)])

    stats = render_video(run, tmp_path / "camera.mp4")

    assert stats.fps == 30.0 and stats.output_frames == 4


def test_render_refuses_a_run_that_mixes_frame_sizes(tmp_path):
    run = _record(tmp_path / "run", [(0.0, bytes(320 * 240), 320, 240, False),
                                     (0.1, bytes(160 * 120), 160, 120, False)])

    with pytest.raises(ValueError, match="mixes frame sizes"):
        render_video(run, tmp_path / "camera.mp4", fps=10)
    assert not (tmp_path / "camera.mp4").exists()


def test_undecodable_frame_holds_the_previous_image(tmp_path, deck_frames):
    corrupt = b"\x00\x00" + deck_frames[1]["payload"][2:]   # SOI marker lost
    payloads = [deck_frames[0]["payload"], corrupt, deck_frames[2]["payload"]]
    run = _record(tmp_path / "run", [(i * 0.1, p, 320, 240, True)
                                     for i, p in enumerate(payloads)])
    out = tmp_path / "camera.mp4"

    stats = render_video(run, out, fps=10, lossless=True)
    planes, _ = _decode_video(out)

    assert len(planes) == stats.output_frames == 3


def test_extract_writes_jpeg_verbatim_and_raw_as_lossless_png(tmp_path, deck_frames, qvga_frames):
    run = _record(tmp_path / "run", [(0.0, deck_frames[0]["payload"], 320, 240, True),
                                     (0.1, qvga_frames[0].tobytes(), 320, 240, False)])
    out = tmp_path / "frames"

    assert extract_frames(run, out) == 2
    jpg, png = sorted(out.iterdir())

    assert jpg.read_bytes() == deck_frames[0]["payload"]
    assert np.array_equal(cv2.imread(str(png), cv2.IMREAD_UNCHANGED), qvga_frames[0])


def test_frame_times_follow_the_decks_capture_clock_not_arrival(tmp_path, deck_frames):
    # Real frames whose arrivals are jittered by up to 40 ms, as WiFi delivers them.
    writer = RunWriter(tmp_path / "run", [camera_spec(), events_spec()], flush_interval_s=0.01)
    writer.start()
    recorder = CameraRecorder(writer)
    recorder.start()
    exts = [decode_camera_stream(f["ext"]) for f in deck_frames]
    rng = np.random.default_rng(7)
    base_ns = time.monotonic_ns()
    for f, ext in zip(deck_frames, exts):
        late_ns = int(rng.integers(0, 40)) * 1_000_000
        arrival = base_ns + (ext.camera_done_ms - exts[0].camera_done_ms) * 1_000_000 + late_ns
        header = ImgHeader(0xBC, 1, 1, len(f["ext"]), 320, 240, len(f["payload"]))
        recorder.offer(CapturedFrame(header, f["payload"], ext, f["ext"], arrival))
    recorder.stop()
    writer.close()

    times = [frame.t_s for frame in load_frames(tmp_path / "run")]

    # An affine image of the deck clock: none of the arrival jitter survives. (The fitted
    # rate itself is telemetry.sync's business; on a run this short it can be a few % off.)
    deck_s = np.array([e.camera_done_ms for e in exts]) / 1000
    residual = times - np.polyval(np.polyfit(deck_s, times, 1), deck_s)
    assert np.abs(residual).max() < 1e-4
