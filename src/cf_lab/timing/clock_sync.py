"""Map the STM32's microsecond clock onto the host's, by ping-pong over the radio (lib/stm32/clock_sync).

Each exchange gives four stamps: host send t1, STM32 reception T2, STM32 send T3, host reception t4. A packet arrives
after it is sent, so every drone event T happened at a host time between the latest send that can reach it (the
uplink's lower envelope, T2 against t1) and the earliest reception that can follow it (the downlink's lower envelope,
t4 against T3). ``ClockMap`` returns the middle of that interval, and ``uncertainty_us`` its half-width: the bound on
the error of any one-way latency computed with the map, whatever the asymmetry of the link. Both envelopes are lines,
so the clocks' relative drift is fitted as well (cf_lab.telemetry.sync.fit_affine).
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass

from cf_lab.telemetry.sync import fit_affine

PING = struct.Struct("<HI")
PONG = struct.Struct("<HIII")


def host_us() -> int:
    """The host clock the map refers to: the monotonic performance counter, in microseconds."""
    return time.perf_counter_ns() // 1000


@dataclass(frozen=True)
class ClockMap:
    up_slope: float  # T2 ~ up_slope * t1 + up_offset along the fastest uplink packets
    up_offset: float
    down_slope: float  # t4 ~ down_slope * T3 + down_offset along the fastest downlink packets
    down_offset: float
    exchanges: int
    rtt_min_us: float

    def bounds(self, drone_us: float) -> tuple[float, float]:
        """Host times between which a drone event at drone_us happened."""
        sent = (drone_us - self.up_offset) / self.up_slope
        received = self.down_slope * drone_us + self.down_offset
        return sent, received

    def to_host_us(self, drone_us: float) -> float:
        lo, hi = self.bounds(drone_us)
        return 0.5 * (lo + hi)

    def uncertainty_us(self, drone_us: float) -> float:
        lo, hi = self.bounds(drone_us)
        return 0.5 * (hi - lo)


def measure_clock(cf, port: int = 0x0B, n: int = 300, period_s: float = 0.003,
                  timeout_s: float = 3.0) -> tuple[ClockMap, list[tuple[int, int, int, int]]]:
    """Ping ``n`` times on ``port`` (channel 0) through an open cflib Crazyflie; returns the map and the exchanges
    (t1, T2, T3, t4) in microseconds, host stamps from ``host_us`` and drone stamps unwrapped from 32 bits."""
    from cflib.crtp.crtpstack import CRTPPacket

    sent: dict[int, int] = {}
    got: list[tuple[int, int, int, int]] = []
    lock = threading.Lock()

    def on_packet(pk) -> None:
        t4 = host_us()
        if pk.channel != 0 or len(pk.data) < PONG.size:
            return
        seq, _, rx, tx = PONG.unpack(bytes(pk.data[: PONG.size]))
        with lock:
            if seq in sent:
                got.append((sent.pop(seq), rx, tx, t4))

    cf.add_port_callback(port, on_packet)
    try:
        for seq in range(n):
            pk = CRTPPacket()
            pk.set_header(port, 0)
            t1 = host_us()
            pk.data = PING.pack(seq, t1 & 0xFFFFFFFF)
            with lock:
                sent[seq] = t1
            cf.send_packet(pk)
            time.sleep(period_s)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and len(got) < n and sent:
            time.sleep(0.01)
    finally:
        cf.remove_port_callback(port, on_packet)
    if len(got) < 10:
        raise RuntimeError(f"only {len(got)} of {n} clock pings were answered; is the responder on port {port}?")
    got.sort()
    # unwrap the 32-bit drone microseconds (71 minutes per wrap)
    base, last, unwrapped = 0, None, []
    for t1, rx, tx, t4 in got:
        if last is not None and rx < last - (1 << 31):
            base += 1 << 32
        last = rx
        unwrapped.append((t1, rx + base, tx + base + ((1 << 32) if tx < rx - (1 << 31) else 0), t4))
    return fit_clock(unwrapped), unwrapped


def fit_clock(exchanges: list[tuple[int, int, int, int]]) -> ClockMap:
    """The map from exchanges (t1, T2, T3, t4). The drift is a slope, so exchanges spread over the time the map is
    used for (pings before and after a run, joined) pin it down; a map fitted over one second and used for a minute
    carries its slope error times the minute."""
    up = fit_affine([e[0] for e in exchanges], [e[1] for e in exchanges], slack=1000.0)
    down = fit_affine([e[2] for e in exchanges], [e[3] for e in exchanges], slack=1000.0)
    rtt = min((t4 - t1) - (tx - rx) for t1, rx, tx, t4 in exchanges)
    return ClockMap(up.slope, up.offset_ns, down.slope, down.offset_ns, len(exchanges), float(rtt))
