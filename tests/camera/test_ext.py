"""App extension decoders and the deck's console lines."""

import pytest

from cf_lab.camera.ext import (
    CAMERA_STREAM_EXT,
    SCHED_EXT,
    CameraStreamExt,
    SchedFrameExt,
    decode_camera_stream,
    decode_deck_ext,
    decode_sched,
    parse_deck_line,
)


def _le(value, width):
    return value.to_bytes(width, "little")


def test_real_camera_stream_ext_decodes_consistently(deck_frames):
    exts = [decode_camera_stream(f["ext"]) for f in deck_frames]

    # The recorder's own columns from the flight.
    assert [(e.frame_id, e.deck_ms) for e in exts] == [(f["frame_id"], f["deck_ms"])
                                                      for f in deck_frames]
    for prev, ext in zip(exts, exts[1:]):
        # A 60 FPS build sending every capture: both counters step by one, both clocks ~16.7 ms.
        assert ext.frame_id - prev.frame_id == 1 and ext.camera_frame_id - prev.camera_frame_id == 1
        assert 16 <= ext.deck_ms - prev.deck_ms <= 17
        assert 16 <= ext.camera_done_ms - prev.camera_done_ms <= 17
    for ext in exts:
        # camera_stream.c fills encode_ms as capture + resize + jpeg.
        assert ext.encode_ms == ext.capture_ms + ext.preprocess_ms + ext.jpeg_ms
        assert 0 <= ext.done_ms - ext.deck_ms <= 40


def test_camera_stream_ext_layout_matches_the_c_struct():
    # camera_stream_frame_ext_t (packed): u32 frame_id, timestamp_ms; u16 encode,
    # capture, preprocess, jpeg; u32 camera_frame_id, camera_done_ms, three drop counters.
    raw = b"".join([_le(7, 4), _le(1000, 4), _le(21, 2), _le(17, 2), _le(0, 2), _le(4, 2),
                    _le(99, 4), _le(1017, 4), _le(1, 4), _le(2, 4), _le(3, 4)])

    ext = decode_camera_stream(raw)

    assert ext == CameraStreamExt(7, 1000, 21, 17, 0, 4, 99, 1017, 1, 2, 3)
    assert ext.done_ms == 1017  # buffer complete, not queue time


def test_sched_ext_layout_matches_the_c_struct():
    # sched_frame_ext_t (packed): u8 version, kind; u16 encode_us/8; u32 frame_id,
    # captured_ms, inferences, anchor_remote_ms, anchor_local_ms.
    raw = b"".join([_le(2, 1), _le(1, 1), _le(100, 2), _le(42, 4), _le(3000, 4), _le(9, 4),
                    _le(5000, 4), _le(4200, 4)])

    ext = decode_sched(raw)

    assert (ext.kind, ext.frame_id, ext.done_ms) == (1, 42, 3000)
    assert (ext.stm32_anchor_ms, ext.deck_anchor_ms) == (5000, 4200)


@pytest.mark.parametrize("anchors, expected", [((0, 4200), (0, 4200)),
                                               ((0, 0), (-1, -1))])
def test_sched_ext_reports_missing_anchors_as_minus_one(anchors, expected):
    ext = decode_sched(SCHED_EXT.pack(2, 1, 100, 42, 3000, 9, *anchors))

    assert (ext.stm32_anchor_ms, ext.deck_anchor_ms) == expected


def test_sched_ext_of_another_version_is_not_decoded():
    assert decode_sched(SCHED_EXT.pack(1, 1, 100, 42, 3000, 9, 1, 1)) is None


@pytest.mark.parametrize("raw, kind", [
    (CAMERA_STREAM_EXT.pack(*range(11)), CameraStreamExt),
    (SCHED_EXT.pack(2, 0, 0, 1, 2, 3, 4, 5), SchedFrameExt),
    (b"", type(None)),
    (bytes(50), type(None)),   # someone else's extension
    (bytes(35), type(None)),
])
def test_decode_deck_ext_tells_apps_apart_by_size(raw, kind):
    assert type(decode_deck_ext(raw)) is kind


def test_config_line_fields_parse_with_numbers_as_ints():
    text = ("camera config: sensor=qvga60 capture=324x244 out=320x240 fit=native "
            "fps=60 sensor_fps=60 format=jpeg jpeg=cluster quality=80 policy=exact "
            "buffers=2 flip=1 window=80 exposure=auto flicker=off transport=tcp channel=1 log=0")

    line = parse_deck_line(text)

    assert line.kind == "config"
    assert line.fields["fps"] == 60 and line.fields["buffers"] == 2
    assert line.fields["out"] == "320x240" and line.fields["transport"] == "tcp"


def test_config_line_negative_window_parses_as_an_int():
    # A negative value is a number like any other.
    line = parse_deck_line("camera config: fps=60 window=-1")

    assert line.fields["window"] == -1


def test_status_line_is_parsed_and_other_text_is_plain_console():
    status = parse_deck_line("status frames=120 faulted=0 spi_hard=2 tx_free=7")
    other = parse_deck_line("camera_stream: WiFi session open failed (status 3)")

    assert status.kind == "status" and status.fields["spi_hard"] == 2
    assert (other.kind, other.fields) == ("console", {})
