"""Camera preflight: the AP/mocap shared-NIC gate and the frame-stream check over a faked feed."""
import math

import numpy as np
import pytest

import cf_lab.camera.preflight as route_module
import cf_lab.preflight.camera as camera_checks
from cf_lab.camera.ext import SCHED_EXT, SCHED_EXT_VERSION
from cf_lab.config import CameraConfig, LabConfig, MocapConfig, NatNetConfig
from cf_lab.preflight.camera import check_camera_mocap_coexistence, check_camera_stream

DECK_AP_IP = "192.168.4.1"
MOCAP_IP = "192.0.2.10"
WIRED = "192.0.2.50"  # this host's address on the mocap network
WIFI = "192.168.4.2"  # this host's address on the deck's access point


def _cfg(mode="access_point", *, ssid="dummy-camera-ap"):
    return LabConfig(
        mocap=MocapConfig(transport="natnet", natnet=NatNetConfig(server_ip=MOCAP_IP)),
        camera=CameraConfig(ssid=ssid, mode=mode),
    )


@pytest.fixture
def routes(monkeypatch):
    """Route the deck and the mocap server from given host source addresses, without the network."""

    def install(*, deck_source, mocap_source=WIRED, deck_error=None):
        def resolve_deck(uri, timeout_s=0.0):
            if deck_error is not None:
                raise deck_error
            return DECK_AP_IP

        sources = {DECK_AP_IP: deck_source, MOCAP_IP: mocap_source}
        monkeypatch.setattr(route_module, "resolve_deck", resolve_deck)
        monkeypatch.setattr(route_module, "source_address_for", lambda ip, port=9: sources[ip])

    return install


# --- camera / mocap coexistence ---------------------------------------------


def test_access_point_on_the_mocap_interface_fails(routes):
    # One NIC: the kernel would reach the deck and Motive from the same address.
    routes(deck_source=WIRED)

    result = check_camera_mocap_coexistence(_cfg())

    assert not result.passed
    assert result.summary.startswith("Camera AP and mocap resolve through the same host interface")
    assert result.details["shared_interface"] is True


def test_access_point_on_a_second_interface_passes(routes):
    routes(deck_source=WIFI)

    assert check_camera_mocap_coexistence(_cfg()).passed


def test_station_mode_may_share_the_mocap_interface(routes):
    routes(deck_source=WIRED)

    assert check_camera_mocap_coexistence(_cfg(mode="station")).passed


def test_disabled_camera_is_not_applicable(routes):
    routes(deck_source=WIRED, deck_error=AssertionError("must not look the deck up"))

    result = check_camera_mocap_coexistence(_cfg(ssid=""))

    assert result.passed
    assert result.summary == "Camera/mocap route coexistence is not applicable."


def test_unresolvable_deck_fails_instead_of_raising(routes):
    routes(deck_source=WIRED, deck_error=OSError("no deck answers to cf-80-2m-e7e7e7e7e7.local"))

    result = check_camera_mocap_coexistence(_cfg())

    assert not result.passed
    assert result.error.startswith("OSError: no deck answers")


# --- camera stream: live feed -----------------------------------------------


def _frames(fakes, count=8, **kwargs):
    return [fakes.make_frame(index, **kwargs) for index in range(count)]


def test_live_feed_with_good_frames_passes_without_being_started_or_stopped(fakes, camera_clock):
    feed = fakes.FakeFeed(_frames(fakes), clock=camera_clock, fps=30.0)

    result = check_camera_stream(feed=feed, minimum_fps=25.0, expected_width=320,
                                 expected_height=240, expected_format="raw",
                                 require_extension=True, max_new_drops=0)

    assert result.passed, result.summary
    assert result.summary == "Camera stream passed: 8 frame(s), 30.0 FPS, 320x240 raw."
    # The experiment owns this connection: the check only observed it.
    assert not feed.started and not feed.stopped
    assert feed.observers == []


def test_too_few_frames_before_the_timeout_fails(fakes, camera_clock):
    feed = fakes.FakeFeed(_frames(fakes, count=3), clock=camera_clock)

    result = check_camera_stream(feed=feed, timeout_s=0.05)

    assert not result.passed
    assert result.details["failures"] == ("received 3 frame(s), need 8",)


@pytest.mark.parametrize(("fps", "passed"), [(9.5, False), (10.5, True)])
def test_minimum_fps_is_measured_from_frame_receipt_times(fakes, camera_clock, fps, passed):
    feed = fakes.FakeFeed(_frames(fakes), clock=camera_clock, fps=fps)

    result = check_camera_stream(feed=feed, minimum_fps=10.0)

    assert result.passed is passed
    assert result.details["observed_fps"] == pytest.approx(fps, rel=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "failure"),
    [
        ({"expected_width": 324}, "width(s) [320], expected 324"),
        ({"expected_height": 244}, "height(s) [240], expected 244"),
        ({"expected_format": "jpeg"}, "format(s) ['raw'], expected jpeg"),
    ],
)
def test_frames_must_match_the_expected_build(fakes, camera_clock, kwargs, failure):
    feed = fakes.FakeFeed(_frames(fakes), clock=camera_clock)

    result = check_camera_stream(feed=feed, **kwargs)

    assert not result.passed
    assert result.details["failures"] == (failure,)


def test_mixed_resolutions_fail_an_expected_width(fakes, camera_clock):
    frames = _frames(fakes, count=7) + [fakes.make_frame(7, width=162, height=122)]
    feed = fakes.FakeFeed(frames, clock=camera_clock)

    assert not check_camera_stream(feed=feed, expected_width=320).passed


def test_missing_extension_fails_only_when_required(fakes, camera_clock):
    frames = _frames(fakes, count=7) + [fakes.make_frame(7, ext=b"")]

    lenient = check_camera_stream(feed=fakes.FakeFeed(frames, clock=camera_clock))
    strict = check_camera_stream(
        feed=fakes.FakeFeed(frames, clock=camera_clock), require_extension=True
    )

    assert lenient.passed
    assert strict.details["failures"] == ("camera extension decoded for 7/8 frame(s)",)


@pytest.mark.parametrize(
    ("ids", "passed"),
    [
        ([0, 1, 2, 3, 4, 5, 6, 7], True),
        ([0, 2, 5, 6, 9, 10, 11, 15], True),
        ([2**32 - 3, 2**32 - 2, 2**32 - 1, 0, 1, 2, 3, 4], True),
        ([0, 1, 2, 3, 3, 4, 5, 6], False),
        ([500, 501, 502, 503, 0, 1, 2, 3], False),
    ],
    ids=["consecutive", "gaps", "uint32_wrap", "duplicate", "deck_rebooted"],
)
def test_application_frame_ids_must_strictly_advance(fakes, camera_clock, ids, passed):
    feed = fakes.FakeFeed([fakes.make_frame(i) for i in ids], clock=camera_clock)

    result = check_camera_stream(feed=feed)

    assert result.passed is passed
    if not passed:
        assert "application frame IDs are not strictly advancing" in result.details["failures"]


@pytest.mark.parametrize(
    ("ids", "passed"), [(list(range(8)), True), ([0, 1, 2, 2, 3, 4, 5, 6], False)]
)
def test_hover_nn_scheduler_frames_are_checked_by_their_own_frame_id(
    fakes, camera_clock, ids, passed
):
    # hover_neural_network sends JPEGs with lib/gap8/sched's extension: no camera id and no
    # drop counters.
    frames = [
        fakes.make_frame(
            i, jpeg=True, ext=SCHED_EXT.pack(SCHED_EXT_VERSION, 0, 10, i, 1000 + 17 * n, 0, 0, 0)
        )
        for n, i in enumerate(ids)
    ]

    result = check_camera_stream(
        feed=fakes.FakeFeed(frames, clock=camera_clock),
        expected_width=320,
        expected_height=240,
        expected_format="jpeg",
        require_extension=True,
    )

    assert result.passed is passed, result.summary
    assert result.details["drop_deltas"]["dropped_capture"] is None


def test_repeated_camera_frame_id_fails(fakes, camera_clock):
    # The app sent a new frame id around a camera buffer it had already sent.
    frames = [
        fakes.make_frame(i, ext=fakes.camera_ext(i, camera_frame_id=min(i, 4))) for i in range(8)
    ]

    result = check_camera_stream(feed=fakes.FakeFeed(frames, clock=camera_clock))

    assert result.details["failures"] == ("camera frame IDs are not strictly advancing",)


def _with_drops(fakes, cumulative):
    return [
        fakes.make_frame(i, ext=fakes.camera_ext(i, drops=(value, 0, 1)))
        for i, value in enumerate(cumulative)
    ]


@pytest.mark.parametrize(
    ("cumulative", "limit", "passed", "new_drops"),
    [
        ([4, 4, 4, 5, 5, 6, 6, 6], 2, True, 2),
        ([4, 4, 4, 5, 5, 6, 6, 7], 2, False, 3),
        ([4, 4, 4, 5, 5, 6, 6, 7], None, True, 3),
        ([2**32 - 1, 0, 0, 1, 1, 1, 1, 1], 2, True, 2),
    ],
    ids=["within_limit", "over_limit", "no_limit", "uint32_wrap"],
)
def test_new_drops_are_counted_from_the_cumulative_counters(
    fakes, camera_clock, cumulative, limit, passed, new_drops
):
    feed = fakes.FakeFeed(_with_drops(fakes, cumulative), clock=camera_clock)

    result = check_camera_stream(feed=feed, max_new_drops=limit)

    assert result.passed is passed
    assert result.details["new_drops"] == new_drops


def test_drop_counter_regression_fails_even_without_a_limit(fakes, camera_clock):
    feed = fakes.FakeFeed(_with_drops(fakes, [9, 9, 9, 9, 0, 0, 0, 0]), clock=camera_clock)

    result = check_camera_stream(feed=feed)

    assert result.details["failures"] == ("camera drop counters regressed 1 time(s)",)


def test_feed_receive_error_fails_even_with_enough_frames(fakes, camera_clock):
    feed = fakes.FakeFeed(
        _frames(fakes), clock=camera_clock, error=ConnectionError("image stream closed")
    )

    result = check_camera_stream(feed=feed)

    assert not result.passed
    assert result.error == "ConnectionError: image stream closed"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_s": math.nan},
        {"sample_frames": 1},
        {"minimum_fps": math.nan},
        {"maximum_dark_fraction": 1.5},
        {"minimum_mean": math.nan},
        {"max_new_drops": -1},
    ],
    ids=["nan_timeout", "one_frame", "nan_fps", "fraction_over_one", "nan_mean", "negative_drops"],
)
def test_invalid_thresholds_fail_without_observing(fakes, camera_clock, kwargs):
    feed = fakes.FakeFeed(_frames(fakes), clock=camera_clock)

    result = check_camera_stream(feed=feed, **kwargs)

    assert not result.passed
    assert result.summary == "Invalid camera-stream check configuration."
    assert feed.observers == [] and feed.frames_received == 0


# --- camera stream: owned feed ----------------------------------------------


@pytest.fixture
def owned_feed(monkeypatch, fakes, camera_clock):
    """Make check_camera_stream's own CameraFeed.from_config(cfg) return a FakeFeed."""

    def install(feed=None, *, from_config_error=None):
        class _Factory:
            @staticmethod
            def from_config(cfg, **kwargs):
                if from_config_error is not None:
                    raise from_config_error
                return feed

        monkeypatch.setattr(camera_checks, "CameraFeed", _Factory)
        return feed

    return install


def test_owned_feed_is_started_and_always_stopped(fakes, camera_clock, owned_feed):
    feed = owned_feed(fakes.FakeFeed(_frames(fakes), clock=camera_clock, owned=True))

    result = check_camera_stream(_cfg(mode="station"))

    assert result.passed, result.summary
    assert feed.started and feed.stopped


def test_owned_feed_that_cannot_connect_fails_and_is_still_stopped(fakes, camera_clock, owned_feed):
    feed = owned_feed(
        fakes.FakeFeed([], clock=camera_clock, owned=True, start_error=ConnectionRefusedError(111))
    )

    result = check_camera_stream(_cfg(mode="station"))

    assert not result.passed
    assert result.error.startswith("ConnectionRefusedError")
    assert feed.stopped


def test_disabled_camera_fails_without_connecting(owned_feed):
    owned_feed(from_config_error=AssertionError("must not connect"))

    result = check_camera_stream(_cfg(ssid=""))

    assert not result.passed
    assert result.summary == "Camera stream is disabled in the supplied configuration."


def test_owned_feed_whose_deck_cannot_be_found_fails_with_the_find_deck_advice(owned_feed):
    owned_feed(from_config_error=OSError("no deck answers"))

    result = check_camera_stream(_cfg(mode="station"))

    assert not result.passed
    assert result.error == "OSError: no deck answers"
    assert "cf-lab find-deck" in result.remediation


# --- camera stream: pixel analysis ------------------------------------------


def _known_frame(fakes, frame_id):
    # 10% black, 5% saturated, the rest mid-gray, in a seeded random layout.
    pixels = np.full(320 * 240, 128, dtype=np.uint8)
    pixels[:7680] = 0
    pixels[7680:7680 + 3840] = 255
    np.random.default_rng(frame_id).shuffle(pixels)
    return fakes.make_frame(frame_id, payload=pixels.tobytes())


def test_pixel_statistics_are_exact_on_a_known_histogram(fakes, camera_clock):
    feed = fakes.FakeFeed([_known_frame(fakes, i) for i in range(8)], clock=camera_clock)

    result = check_camera_stream(feed=feed, analyze_pixels=True)

    details = result.details
    assert result.passed
    assert details["pixel_count"] == 8 * 320 * 240
    assert details["pixel_mean"] == pytest.approx((3840 * 255 + 65280 * 128) / 76800)
    assert details["dark_fraction"] == pytest.approx(0.10)
    assert details["clipped_fraction"] == pytest.approx(0.05)
    assert (details["pixel_min"], details["pixel_median"], details["pixel_max"]) == (0, 128, 255)
    assert details["pixel_p10"] == 0


@pytest.mark.parametrize(
    ("kwargs", "failure"),
    [
        ({"minimum_mean": 130.0}, "pixel mean 121.5 is below 130.0"),
        ({"maximum_mean": 100.0}, "pixel mean 121.5 exceeds 100.0"),
        ({"maximum_dark_fraction": 0.09}, "dark fraction 0.100 exceeds 0.090"),
        ({"maximum_clipped_fraction": 0.04}, "clipped fraction 0.050 exceeds 0.040"),
    ],
)
def test_pixel_thresholds_fail_the_stream(fakes, camera_clock, kwargs, failure):
    feed = fakes.FakeFeed([_known_frame(fakes, i) for i in range(8)], clock=camera_clock)

    result = check_camera_stream(feed=feed, **kwargs)

    assert result.details["failures"] == (failure,)


def test_jpeg_frames_are_decoded_for_pixel_analysis(fakes, camera_clock):
    cv2 = pytest.importorskip("cv2")
    gradient = np.tile(np.linspace(40, 200, 320, dtype=np.uint8), (240, 1))
    ok, encoded = cv2.imencode(".jpg", gradient, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    frames = [fakes.make_frame(i, jpeg=True, payload=encoded.tobytes()) for i in range(8)]

    result = check_camera_stream(
        feed=fakes.FakeFeed(frames, clock=camera_clock), expected_format="jpeg", analyze_pixels=True
    )

    assert result.passed, result.summary
    assert result.details["pixel_mean"] == pytest.approx(gradient.mean(), abs=1.0)


def test_undecodable_frame_fails_pixel_analysis(fakes, camera_clock):
    # A raw frame whose payload is shorter than its header's width x height.
    frames = _frames(fakes, count=7) + [fakes.make_frame(7, payload=bytes(1000))]

    result = check_camera_stream(feed=fakes.FakeFeed(frames, clock=camera_clock), analyze_pixels=True)

    assert not result.passed
    assert result.details["failures"][0].startswith("pixel analysis failed: ValueError")
