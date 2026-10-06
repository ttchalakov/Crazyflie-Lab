"""The viewer's Qt-free half: newest-frame decode slot and the side panel's alert rows."""

import threading
import time

import cv2
import numpy as np
import pytest

from cf_lab.camera.ext import parse_deck_line
from cf_lab.camera.feed import CapturedFrame
from cf_lab.camera.stats import StatsSnapshot
from cf_lab.camera.viewer import Viewer, _sections
from cf_lab.imgstream import ImgHeader


def _jpeg(payload, n=0):
    return CapturedFrame(ImgHeader(0xBC, 1, 1, 0, 320, 240, len(payload)), payload, None, b"", n)


@pytest.fixture
def decoding_viewer():
    viewer = Viewer()
    thread = threading.Thread(target=viewer._decode_loop, daemon=True)
    thread.start()
    yield viewer
    viewer._stop.set()
    thread.join(2.0)


def _wait_for_seq(viewer, seq, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while True:
        latest = viewer.latest_image()
        if latest is not None and latest[0] >= seq:
            return latest
        assert time.monotonic() < deadline, "decode thread never produced the frame"
        time.sleep(0.005)


def test_viewer_decodes_the_offered_frame_and_counts_it(decoding_viewer, deck_frames):
    payload = deck_frames[0]["payload"]

    decoding_viewer.offer(_jpeg(payload, n=1))
    _seq, image, frame = _wait_for_seq(decoding_viewer, 1)

    assert np.array_equal(image, cv2.imdecode(np.frombuffer(payload, np.uint8),
                                              cv2.IMREAD_GRAYSCALE))
    assert frame.host_ns == 1
    assert decoding_viewer.stats.snapshot().frames == 1


def test_an_undecodable_frame_keeps_the_last_good_image(decoding_viewer, deck_frames):
    decoding_viewer.offer(_jpeg(deck_frames[0]["payload"], n=1))
    _wait_for_seq(decoding_viewer, 1)

    decoding_viewer.offer(_jpeg(b"\x00" * 100, n=2))
    decoding_viewer.offer(_jpeg(deck_frames[1]["payload"], n=3))
    seq, _image, frame = _wait_for_seq(decoding_viewer, 2)

    assert (seq, frame.host_ns) == (2, 3)


def test_status_line_problems_are_flagged_as_alerts():
    healthy = parse_deck_line("status frames=9 faulted=0 watchdog_restarts=0 encode_failures=0 "
                              "tx_free=7 spi_soft=3 spi_hard=0").fields
    faulted = parse_deck_line("status frames=9 faulted=1 watchdog_restarts=2 encode_failures=0 "
                              "tx_free=0 spi_soft=3 spi_hard=1").fields

    def alerts(status):
        sections = dict(_sections(StatsSnapshot(), {}, status))
        return {label for label, _value, alert in sections["Deck status"] if alert}

    assert alerts(healthy) == set()
    assert alerts(faulted) == {"Camera faulted", "Watchdog restarts", "SPI recoveries"}


@pytest.mark.parametrize("jitter_ms, alert", [(12.0, False), (180.0, True)])
def test_link_jitter_above_100_ms_is_an_alert(jitter_ms, alert):
    sections = dict(_sections(StatsSnapshot(has_ext=True, link_jitter_ms=jitter_ms), {}, {}))

    assert dict((label, a) for label, _, a in sections["Deck"])["Link jitter"] is alert


def test_view_without_ip_or_config_reports_an_error(tmp_path, monkeypatch, capsys):
    from argparse import Namespace

    from cf_lab.camera import find_deck, viewer

    def no_deck(uri, timeout_s=0.0):
        raise OSError(f"no deck answers for {uri}")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(find_deck, "resolve_deck", no_deck)  # never reach the network
    args = Namespace(config=tmp_path / "missing.yaml", ip=None, port=None, transport=None,
                     snapshot_dir=tmp_path)

    assert viewer.run(args) == 1
    assert "error" in capsys.readouterr().err
