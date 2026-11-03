"""cf_lab.timing: the clock map from ping-pong exchanges, on a simulated link with drift and asymmetric delays."""

import numpy as np

from cf_lab.timing import fit_clock


def simulated_exchanges(rng, n=600, drift=420e-6, offset=5e6, up=600.0, down=1500.0, span_s=20.0):
    """(t1, T2, T3, t4) in microseconds: drone clock T = (1 + drift) t + offset, minimum one-way delays up and down
    plus exponential queueing, 50 us between reception and reply on the drone."""
    t1 = np.sort(np.concatenate([rng.uniform(0, 1e6, n // 2), rng.uniform(span_s * 1e6, (span_s + 1) * 1e6, n // 2)]))
    clock = lambda t: (1.0 + drift) * t + offset
    arrive = t1 + up + rng.exponential(300.0, n)
    reply = arrive + 50.0
    t4 = reply + down + rng.exponential(1000.0, n)
    return [tuple(map(int, e)) for e in zip(t1, clock(arrive), clock(reply), t4)], clock


def test_every_drone_event_is_placed_within_the_stated_uncertainty():
    rng = np.random.default_rng(0)
    exchanges, clock = simulated_exchanges(rng)
    m = fit_clock(exchanges)
    for t in rng.uniform(0, 21e6, 200):
        drone = clock(t)
        lo, hi = m.bounds(drone)
        assert lo - 20 <= t <= hi + 20  # the true host time lies between the bounds (integer rounding aside)
        assert abs(m.to_host_us(drone) - t) <= m.uncertainty_us(drone) + 20
    assert abs(m.up_slope - 1.00042) < 2e-5
    assert 900 < m.uncertainty_us(clock(10e6)) < 1300  # about half of the shortest round trip, (600 + 1500) / 2
