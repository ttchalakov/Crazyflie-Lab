"""FeedStats: received and sensor frame rates, bitrate, link jitter, deck drop counters."""

import pytest

from cf_lab.camera.ext import CameraStreamExt, SchedFrameExt, decode_camera_stream
from cf_lab.camera.feed import CapturedFrame
from cf_lab.camera.stats import FeedStats, stats_line
from cf_lab.imgstream import ImgHeader

T0_NS = 7_000_000_000_000  # a realistic monotonic clock, ~2 h after boot


def _frame(host_ns, size=1000, ext=None, jpeg=True):
    header = ImgHeader(0xBC, 1 if jpeg else 0, 1, 36 if ext else 0, 320, 240, size)
    return CapturedFrame(header, bytes(size), ext, b"", host_ns)


def _ext(frame_id, deck_ms, camera_frame_id, done_ms, drops=(0, 0, 0)):
    return CameraStreamExt(frame_id, deck_ms, 21, 17, 0, 4, camera_frame_id, done_ms, *drops)


def test_received_fps_and_bitrate_over_the_window():
    stats = FeedStats(window_s=1.0)
    for i in range(31):  # 30 FPS for one second, 2000 B frames
        stats.offer(_frame(T0_NS + i * 1_000_000_000 // 30, size=2000))

    snap = stats.snapshot()

    assert snap.frames == 31
    assert snap.fps == pytest.approx(30.0, rel=1e-6)
    assert snap.bitrate_kbps == pytest.approx(30 * 2000 * 8 / 1000, rel=1e-6)


def test_rates_forget_frames_older_than_the_window():
    stats = FeedStats(window_s=1.0)
    for i in range(11):                      # 10 FPS ...
        stats.offer(_frame(T0_NS + i * 100_000_000))
    for i in range(1, 41):                   # ... then 2 s at 20 FPS
        stats.offer(_frame(T0_NS + 1_000_000_000 + i * 50_000_000))

    assert stats.snapshot().fps == pytest.approx(20.0, rel=1e-6)


def test_first_frame_reports_zero_rates_not_a_division_error():
    stats = FeedStats()
    stats.offer(_frame(T0_NS))

    snap = stats.snapshot()

    assert (snap.frames, snap.fps, snap.bitrate_kbps, snap.camera_fps) == (1, 0.0, 0.0, 0.0)


def test_camera_fps_counts_sensor_frames_not_frames_sent():
    # The deck streams every other capture of a 50 FPS sensor: 25 received, 50 captured.
    stats = FeedStats(window_s=1.0)
    for i in range(26):
        done_ms = 500_000 + i * 40
        ext = _ext(i, done_ms - 20, 1000 + 2 * i, done_ms)
        stats.offer(_frame(T0_NS + i * 40_000_000, ext=ext))

    snap = stats.snapshot()

    assert snap.fps == pytest.approx(25.0, rel=1e-6)
    assert snap.camera_fps == pytest.approx(50.0, rel=1e-6)


def test_link_jitter_is_arrival_delay_above_the_fastest_frame():
    stats = FeedStats()
    delays_ms = [30, 12, 12, 47]
    for i, delay in enumerate(delays_ms):
        deck_ms = 80_000 + 20 * i
        stats.offer(_frame(T0_NS + (deck_ms + delay) * 1_000_000, ext=_ext(i, deck_ms, i + 1,
                                                                             deck_ms)))

    assert stats.snapshot().link_jitter_ms == pytest.approx(47 - 12, abs=1e-6)


def test_snapshot_carries_the_decks_timings_and_cumulative_drops(deck_frames):
    stats = FeedStats()
    last = deck_frames[-1]
    ext = decode_camera_stream(last["ext"])
    stats.offer(_frame(last["host_ns"], ext=ext))

    snap = stats.snapshot()

    assert snap.has_ext
    assert (snap.deck_ms, snap.capture_ms, snap.jpeg_ms) == (ext.encode_ms, ext.capture_ms,
                                                             ext.jpeg_ms)
    assert (snap.dropped_capture, snap.dropped_ready, snap.dropped_streaming) == (
        ext.dropped_capture, ext.dropped_ready, ext.dropped_streaming)
    assert "drops 0/83/0" in stats_line(snap)


def test_real_flight_rates_read_the_60_fps_build(deck_frames):
    # The recorded build captured and sent at 60 FPS; WiFi delivered every frame.
    stats = FeedStats(window_s=10.0)
    for f in deck_frames:
        stats.offer(_frame(f["host_ns"], size=len(f["payload"]),
                           ext=decode_camera_stream(f["ext"])))

    snap = stats.snapshot()

    assert snap.fps == pytest.approx(60, rel=0.03)
    assert snap.camera_fps == pytest.approx(60, rel=0.03)
    assert snap.link_jitter_ms < 50


def test_sched_extension_gives_jitter_without_sensor_rate():
    # hover_neural_network / scheduler_bench frames carry SchedFrameExt: no camera_frame_id
    # or timings.
    stats = FeedStats()
    for i, delay in enumerate([9, 4, 30]):
        deck_ms = 50_000 + 20 * i
        ext = SchedFrameExt(2, 1, 0, i, deck_ms, 0, -1, -1)
        stats.offer(_frame(T0_NS + (deck_ms + delay) * 1_000_000, ext=ext))

    snap = stats.snapshot()

    assert snap.has_ext and snap.camera_fps == 0.0 and snap.capture_ms == 0
    assert snap.link_jitter_ms == pytest.approx(30 - 4, abs=1e-6)


def test_stats_recover_after_the_deck_reboots_mid_feed():
    stats = FeedStats(window_s=1.0)
    host_ns = T0_NS
    for i in range(60):                      # deck up for 10 minutes, 30 FPS
        done_ms = 600_000 + i * 33
        stats.offer(_frame(host_ns, ext=_ext(18_000 + i, done_ms, 18_000 + i, done_ms)))
        host_ns += 33_000_000
    host_ns += 15_000_000_000                # reboot; UDP re-announce brings frames back
    for i in range(90):                      # fresh deck clock and frame ids
        done_ms = 12_000 + i * 33
        stats.offer(_frame(host_ns, ext=_ext(i + 1, done_ms, i + 1, done_ms)))
        host_ns += 33_000_000

    snap = stats.snapshot()

    assert snap.camera_fps == pytest.approx(1000 / 33, rel=0.05)
    assert snap.link_jitter_ms < 100         # the viewer turns > 100 ms into an alert
