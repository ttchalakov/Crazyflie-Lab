# camera_bench

The regression bench for [camera_stream](../camera_stream). For each case it
builds a camera_stream configuration, flashes it over the radio, boots it,
streams from it, and checks the result against the case and against a recorded
baseline.

```bash
python examples/camera_bench/host/bench.py                     # the common set (9 cases)
python examples/camera_bench/host/bench.py --set exhaustive    # every combination (212 cases, hours)
python examples/camera_bench/host/bench.py --case qvga60-native-jpegcl-preview-tcp
python examples/camera_bench/host/bench.py --list
python examples/camera_bench/host/bench.py --record-baseline   # also save passing results as the baseline
python examples/camera_bench/host/plot.py examples/camera_bench/artifacts/bench_<stamp>
```

It needs the drone on (its radio address comes from `configs/drone.yaml`, or
`--config`), a Crazyradio, and a WiFi adapter on this computer. It leaves wired
networking alone: the first time it runs it creates a NetworkManager profile for
the deck's access point that never takes the default route and has power saving
off.

## Cases

Cases come from a matrix in `host/bench.yaml`: every sensor mode, output size,
format, capture policy and transport, named from their settings as
`sensor-size-format-policy-transport`, for example
`qvga60-96crop-jpegcl-exact-udp`:

| Axis | Values |
| --- | --- |
| sensor | `qvga30`, `qvga60`, `qqvga30`, `qqvga60`, `qqvga120` |
| size | `native`, `96sq` (96x96 squashed), `96crop` (96x96 cropped) |
| format | `raw`, `jpegfc` (JPEG on the fabric controller), `jpegcl` (on the cluster) |
| policy | `preview`, `exact`, `exact3` (exact with three camera buffers; native size only) |
| transport | `tcp`, `udp` |

Three special cases run on one of those builds: `reconnect` disconnects and
reconnects 10 times, each connection needing 5 frames within 15 s; `lifecycle`
reads the camera start/stop/resume/close self-test the deck ran at boot; `soak`
measures for 10 minutes.

The **common** set, the default, is what people actually run: live video,
reconnecting to it, tear-free recording, video beside a DORY network (FC JPEG),
lossless raw frames, 120 FPS, a network-input size, UDP, and the self-test. The
**exhaustive** set is every matrix case plus `reconnect` and `lifecycle`; **long**
is the soak.

## What a run does

1. Puts the **idle controller** on the STM32, so nothing but the bench talks to
   the deck (`--keep-controller` skips it).
2. For each case, **builds** `gap8/`, the bench's GAP8 app, with the case's make
   variables and the bench's access point, then **flashes** it (MD5 verified)
   and **power-cycles** the drone so the case starts from a cold deck.
   Consecutive cases with the same build skip both.
3. **Connects**: joins the deck's access point and connects to its stream. The
   radio console is captured from power-up until then, and closed before
   measuring: a Crazyradio link disturbs the deck's WiFi.
4. **Checks the build**: the deck's config line, sent before its first frame,
   must report every setting the case asked for. A wrong image fails here.
5. **Measures** for 20 s after a 5 s warm-up (or runs the special case's test).

The bench GAP8 app is separate code from camera_stream's but streams exactly the
same way, from the same camera settings and libraries. On top it sends two
status lines every second (camera and transport counters, then the stream
task's stage, free L2 and each task's stack headroom), and has the self-test and
a per-frame trace (`LIFECYCLE=1`, `TRACE=1`).

## Checks

| Check | Passes when |
| --- | --- |
| build | the deck reports the case's settings |
| `camera_fps` | the camera captured within 2 % of the sensor's rate |
| `tcp_frames_lost` | over TCP, every frame the deck sent arrived |
| `stream` | the stream never went silent for 10 s |
| `deck_watchdog_restarts`, `deck_encode_failures`, `deck_spi_hard`, `camera_faulted` | none happened while measuring |
| `deck_preprocess_ms`, `deck_jpeg_ms` | the deck's resize and JPEG times are within the baseline's + 10 % (or + 1 ms) |
| `received_fps` | advisory: at least 90 % of the baseline's received rate |

The received rate is advisory because it follows the WiFi around the bench as
much as the build. Everything the deck measures about itself is a hard check. A
build that does not fit in the GAP8's L2 memory reports it on the radio console
at boot and is counted as **unsupported**, not failed: that is a property of the
configuration, not a regression. QVGA with three exact buffers and cluster JPEG
is one (three 77 KB camera buffers, a whole-frame transmit queue and 128 KB of
encoder buffers). A run's exit status is 0 when no case failed.

## The access point channel

`wifi.channel` in `bench.yaml` (or `--channel`) sets the deck's access-point
channel, which needs the ESP32 firmware from this repository. Choose the least
crowded one (`nmcli device wifi list`; channels 12 and 13 are not allowed in the
US): next to strong networks on a nearby channel, a stream can break that runs
cleanly a few channels away.

## Output

Each run gets its own folder, `examples/camera_bench/artifacts/bench_<stamp>/`:

```text
run.json                     the set, cases, channel and start time
results.jsonl                one line per case: result, the deck's config, the summary, the checks
controller.log               the idle controller's build and flash
<case>/build.log             the GAP8 build
<case>/console.log           the radio console from power-up to connection
<case>/frames.csv            every measured frame: arrival time, size and the deck's timings
<case>/deck.jsonl            the deck's status lines while measuring
plots/                       written by host/plot.py
```

`host/plot.py` draws one run into its `plots/` folder: `results.md`, the table
of every case, and high-resolution PNGs of received against captured frame rate
(`rate.png`), the received rate over time (`timeline.png`), the deck's time per
frame (`deck_time.png`), how late frames arrive (`jitter.png`) and the lowest
free L2 (`l2.png`). A small run gets a bar per case. From 30 measured cases on
(the exhaustive set), each plot becomes a grid of sensor and size against
format and policy, one panel per transport, with the value in each cell; failed
cases are red and cases that do not fit in L2 gray. The rate over time shows at
most 12 cases: the least steady ones, by their worst second.

The deck times its stages in whole milliseconds, so a stage that takes less
than one (cluster JPEG of a 96x96 frame) averages somewhere between 0 and 1.
Capture time runs from when a buffer is queued to when its frame is complete:
how long each policy takes to deliver a whole frame.

The stage in the deck's status line names where its stream task is: 1 waiting
for a camera frame, 2 waiting for a client, 3 encoding, 4 waiting for transmit
queue space, 5 sending, 6 pacing. A stage that stops changing names the call that
hung.

## The baseline

`host/baseline.json` holds the last recorded measurement of each case, with the
date and git commit it came from; `--record-baseline` replaces the entries for
the cases that passed. Record a new baseline after an intended performance
change, never to make a regression pass.

## Inspecting frames

`host/capture_frames.py` saves a spread of frames and a montage from whatever
build is streaming, for looking at image quality (tearing, orientation,
exposure).

## Open items

1. **Re-record the baseline and rerun the exhaustive set.** `host/baseline.json`
   comes from commit `6e46d67`, before the current resize, pacing and centred
   QQVGA window, which change the frames and so the JPEG sizes.
2. **Size the exact transmit queue to the largest frame.** It is fixed at 96
   1 KiB slots; the largest JPEG needs about 65 and a 96x96 raw frame about 10,
   which would free 30 to 85 KiB of L2.
3. **Give `camera_preview` an output size** by building it on
   `camera_pipeline`; the camera_stream demo then need not carry its own copy of
   the stream loop.
4. **Decide how one frame reaches two consumers**, for an onboard network
   beside a video stream: camera_stream and camera_pipeline allow one lease at a
   time.
5. **QQVGA auto exposure meters the whole sensor**, not the window shown, so a
   bright light above the window darkens the frame. The sensor may offer an
   exposure window; not investigated.
6. **The sensor test pattern**: a register that makes the Himax send a known
   pattern would let the host check pixels exactly (the register is not yet
   verified).
