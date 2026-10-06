"""ExtposeStreamer rate cap, decimation accounting, freshness, failure surfacing, and first_pose."""
import threading

import pytest

from cf_lab.mocap.fake import BoxSweepSource
from cf_lab.mocap.relative import OriginZeroer
from cf_lab.mocap.stream import ExtposeStreamer, first_pose


def _run_live(kit, clock, arrivals, *, rate_hz=100.0, transform=None):
    source = kit.Source([(t, kit.pose((t, 0.0, 0.0), seq=i, t=t)) for i, t in enumerate(arrivals)], clock)
    backend = kit.Backend()
    streamer = ExtposeStreamer(source=source, backend=backend, max_rate_hz=rate_hz, pace=False, transform=transform)
    observations = []
    streamer.add_observer(observations.append)
    streamer.run()
    return streamer, backend, observations


def _max_per_second(times):
    return max(sum(1 for u in times if t <= u < t + 1.0) for t in times)


def test_paced_source_forwards_every_sample_transformed_and_in_order(kit):
    poses = [kit.pose((1.0 + 0.1 * i, 2.0, 0.0), seq=i) for i in range(5)]
    source = kit.Source(poses)
    backend = kit.Backend()
    streamer = ExtposeStreamer(
        source=source, backend=backend, max_rate_hz=1000.0, pace=True, transform=OriginZeroer().zero
    )
    streamer.run()
    assert [p.source_seq for p in backend.poses] == [0, 1, 2, 3, 4]
    assert [p.position_m[0] for p in backend.poses] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert (streamer.sent, streamer.dropped, streamer.error) == (5, 0, None)
    assert streamer.latest == backend.poses[-1]
    # A synthetic profile advances one uplink period per sample, so it tracks wall time.
    assert source.calls == [{"start_s": 0.0, "dt_s": pytest.approx(0.001)}]


def test_live_source_faster_than_the_cap_never_exceeds_it(kit, fake_clock):
    arrivals = [k * 1e-3 for k in range(2000)]  # 1 kHz for 2 s
    streamer, backend, observations = _run_live(kit, fake_clock, arrivals)
    forwarded = [o.forwarded_monotonic_s for o in observations if o.forwarded_monotonic_s is not None]
    assert len(forwarded) == streamer.sent == len(backend.poses)
    assert _max_per_second(forwarded) <= 101
    assert streamer.sent + streamer.dropped == len(arrivals) == len(observations)


def test_live_source_slower_than_the_cap_is_forwarded_whole(kit, fake_clock):
    arrivals = [k / 50.0 for k in range(100)]  # 50 Hz under a 100 Hz cap
    streamer, backend, _ = _run_live(kit, fake_clock, arrivals)
    assert (streamer.sent, streamer.dropped) == (100, 0)


def test_observations_carry_the_raw_pose_and_mark_decimated_ones(kit, fake_clock):
    arrivals = [0.0, 0.002, 0.004, 0.011]
    streamer, backend, observations = _run_live(
        kit, fake_clock, [t + 5.0 for t in arrivals], transform=OriginZeroer().zero
    )
    assert [o.forwarded_monotonic_s is not None for o in observations] == [True, False, False, True]
    assert [o.received_monotonic_s for o in observations] == pytest.approx([t + 5.0 for t in arrivals])
    assert observations[3].pose.position_m[0] == pytest.approx(5.011)  # raw, not zeroed
    assert backend.poses[1].position_m[0] == pytest.approx(0.011)


def test_freshness_tracks_dropped_samples_too(kit, fake_clock):
    streamer, _, _ = _run_live(kit, fake_clock, [10.0, 10.001, 10.002])
    assert streamer.latest_received_monotonic_s == pytest.approx(10.002)
    assert streamer.latest_forwarded_monotonic_s == pytest.approx(10.0)


def test_backend_failure_is_captured_and_stops_the_uplink(kit):
    backend = kit.Backend(fail_on=2)
    streamer = ExtposeStreamer(
        source=kit.Source([kit.pose(seq=i) for i in range(10)]), backend=backend, max_rate_hz=1000.0, pace=True
    )
    streamer.run()
    assert isinstance(streamer.error, ConnectionError)
    assert len(backend.poses) == 2 and streamer.sent == 2


def test_failing_observer_cannot_kill_the_uplink(kit, fake_clock):
    source = kit.Source([(k * 0.02, kit.pose(seq=k)) for k in range(5)], fake_clock)
    backend = kit.Backend()
    streamer = ExtposeStreamer(source=source, backend=backend, max_rate_hz=100.0, pace=False)
    seen = []
    streamer.add_observer(lambda o: 1 / 0)
    streamer.add_observer(seen.append)
    streamer.run()
    assert (streamer.sent, streamer.observer_errors, len(seen), streamer.error) == (5, 5, 5, None)


def test_unsubscribed_observer_stops_receiving_and_unsubscribe_is_idempotent(kit, fake_clock):
    source = kit.Source([(k * 0.02, kit.pose(seq=k)) for k in range(4)], fake_clock)
    streamer = ExtposeStreamer(source=source, backend=kit.Backend(), max_rate_hz=100.0, pace=False)
    seen = []
    unsubscribe = streamer.add_observer(lambda o: (seen.append(o), unsubscribe()))
    streamer.run()
    unsubscribe()
    assert len(seen) == 1


@pytest.mark.parametrize("pace", [True, False])
def test_stop_ends_an_endless_stream(kit, pace):
    backend = kit.Backend(stop_after=3)
    streamer = ExtposeStreamer(source=BoxSweepSource(), backend=backend, max_rate_hz=200.0, pace=pace)
    backend.streamer = streamer
    streamer.start()
    streamer.join(2.0)
    assert not streamer.is_alive()
    assert streamer.sent == 3


def test_real_360hz_natnet_arrivals_respect_the_100hz_cap(kit, fake_clock, flight_sample):
    arrivals = [row["host_ns"] / 1e9 for row in flight_sample.mocap if row["segment"] == "flight"]
    streamer, _, observations = _run_live(kit, fake_clock, arrivals)
    forwarded = [o.forwarded_monotonic_s for o in observations if o.forwarded_monotonic_s is not None]
    assert _max_per_second(forwarded) <= 101
    assert streamer.sent + streamer.dropped == len(arrivals)


@pytest.mark.parametrize("source", ["real_360hz_flight", "synthetic_120hz"])
def test_live_uplink_reaches_the_cap_when_the_source_is_faster(kit, fake_clock, flight_sample, source):
    if source == "real_360hz_flight":
        arrivals = [row["host_ns"] / 1e9 for row in flight_sample.mocap if row["segment"] == "flight"]
    else:
        arrivals = [k / 120.0 for k in range(240)]
    streamer, _, _ = _run_live(kit, fake_clock, arrivals)
    duration_s = arrivals[-1] - arrivals[0]
    assert streamer.sent / duration_s >= 95.0


# -- first_pose -------------------------------------------------------------------


class _BlockingSource:
    def __init__(self, error=None):
        self.closed = threading.Event()
        self.error = error

    def samples(self, *, start_s=0.0, dt_s=0.01):
        if self.error is not None:
            raise self.error
        self.closed.wait(5.0)
        return
        yield


def test_first_pose_returns_the_first_sample(kit):
    poses = [kit.pose(seq=1), kit.pose(seq=2)]
    assert first_pose(kit.Source(poses), timeout_s=2.0).source_seq == 1


def test_first_pose_times_out_to_none_and_closes_the_source():
    source = _BlockingSource()
    source.close = source.closed.set
    assert first_pose(source, timeout_s=0.05) is None
    assert source.closed.is_set()


def test_first_pose_raises_connect_errors_instead_of_timing_out():
    source = _BlockingSource(error=ValueError("Motive rigid body 'cf21bl_01' must match exactly one asset"))
    with pytest.raises(ValueError, match="exactly one asset"):
        first_pose(source, timeout_s=2.0)
