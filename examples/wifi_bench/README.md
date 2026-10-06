# wifi_bench

The regression bench for the AI-deck's WiFi link, GAP8 → ESP32 → host and
back. A bench GAP8 app sends synthetic, checksummed CPX traffic over TCP or UDP;
the host checks every packet, exercises reconnects and malformed input, and
records the ESP32's heap and send errors from the radio console. No camera, and
nothing flies.

```bash
python examples/wifi_bench/host/bench.py                    # the common set: every test, TCP and UDP
python examples/wifi_bench/host/bench.py --set long         # a one-hour soak per transport
python examples/wifi_bench/host/bench.py --set integration  # the load beside STM32 telemetry
python examples/wifi_bench/host/bench.py --case reconnect-tcp
python examples/wifi_bench/host/bench.py --list
python examples/wifi_bench/host/bench.py --record-baseline  # also save passing results as the baseline
```

It needs the drone on (its radio address comes from `configs/drone.yaml`, or
`--config`), a Crazyradio, and a WiFi adapter on this computer. It leaves wired
networking alone: the first time it runs it creates a NetworkManager profile,
`wifi-bench`, for the deck's access point that never takes the default route or
DNS and has power saving off. The ESP32 firmware is used as flashed.
`wifi.channel` in `host/bench.yaml` (or `--channel`) sets the access point's
channel.

## Cases

Each test in `host/suite.py` runs on each transport it applies to, named
`<test>-<transport>`, for example `stream-1020x1000-udp`:

| Test | Transports | Checks |
| --- | --- | --- |
| `stream-64x50`, `stream-512x500`, `stream-1020x1000` | both | The GAP8 sends packets of that size at that rate per second for 3 s. Every packet's checksum and pattern must hold; TCP must deliver every packet the GAP8 queued, once and in order; UDP loss is reported (advisory above 2 %). |
| `echo` | both | 10 round trips of a 64-byte request. |
| `host-exit` | both | A host process dies mid-stream without closing its socket; the next client must be served. |
| `stalled-reader` | both | A client stops reading during a 2000 packet/s stream; a new client must be served while the stalled socket stays open. |
| `fragmented` | TCP | A request split over four segments arrives whole. |
| `incomplete-header`, `incomplete-payload` | TCP | A request cut short by the host's FIN closes the connection, and the server recovers. |
| `short-length`, `long-length` | both | A request with an impossible length is rejected, and the server recovers. |
| `truncated-datagram`, `oversized-datagram` | UDP | A datagram shorter or longer than its length field is rejected. |
| `spi-invalid-length` | both | The GAP8 sends a malformed SPI frame; the ESP32's `spiRej` counter must rise by one. Needs the radio console. |
| `wifi-rejoin` | both | The host leaves and rejoins the access point mid-stream; the server recovers. |
| `reconnect` | both | 15 connect, echo and close cycles, closing alternately gracefully and with a TCP reset. Runs idle, and last, because a wedged server fails everything after it. |

The **common** set is all of them (27 cases, a few minutes). **long** is
`soak-tcp` and `soak-udp`: 1020-byte packets at 1000 per second for an hour
each. **integration** is the same load for 30 s while cflib records the STM32's
state and control telemetry at 100 Hz, which must pass `cf_lab.telemetry`'s
integrity check.

## What a run does

1. Puts the **idle controller** on the STM32 (`--keep-controller` skips it).
2. For each transport, **builds** `gap8/` for the bench's access point and that
   transport, **flashes** it and **power-cycles** the drone, opens the radio
   console, joins the access point, and **asks the deck** which transport and
   channel it runs. A wrong image fails here.
3. Runs that transport's cases in the order above.

The radio console stays open throughout, since the ESP32's status lines are
part of the measurement; an active radio link costs some WiFi throughput.
`--no-console` closes it, which skips `spi-invalid-length` and the ESP32 checks
and cannot run the integration set.

## Checks

A case fails on any check of its test (table above). Three more are advisory:
reported, but they do not fail the case.

| Check | Passes when |
| --- | --- |
| `esp_heap` | the ESP32's least free heap during the case, from its 5 s status lines, is at least 4 KiB |
| `mbps_vs_baseline` | a stream's throughput is at least 90 % of the baseline's |
| `rtt_vs_baseline` | the echo median round trip is at most 1.5 times the baseline's plus 2 ms |

## Output

Every run writes `artifacts/bench_<stamp>/`:

```text
run.json                       the cases, access point, channel and ESP32 checkout
results.jsonl                  one line per case: summary, checks, result
tcp-build.log, udp-build.log   the GAP8 builds
tcp-console.log, ...           the radio console from power-up
<case>/samples.jsonl           a stream's packets and bytes each second
<case>/telemetry_integrity.txt an integration case's telemetry check
```

Each case's summary carries `esp`: the least free heap seen during the case
(`heap_free_min`), the ESP32's low-water mark since boot (`heap_watermark`),
and counts of its `Send timeout`, `Send stalled`, send-error and client
disconnect lines. `host_rejoined_ap` is true when this computer had dropped the
deck's access point before the case and the bench joined it again.

## Wire format

Each CPX packet carries one record (`host/link.py`, `gap8/wifi_bench.c`): a
magic, kind, size, session, sequence, GAP8 timestamp and CRC32, then a
deterministic pattern. The host asks for a stream with a rate, size and
duration; rates are start-to-start and not guaranteed, and the GAP8 counts
packets its CPX queue refused (`gap8_queue_drops`) separately from packets lost
after it queued them (`missing`).

## The baseline

`host/baseline.json` holds each stream's throughput and the echo round trip
from a run recorded with `--record-baseline`, which merges only passing cases.
Throughput follows the radio conditions around the bench as much as the build.
