# camera_stream

The AI-deck camera as a WiFi video server. The GAP8 app captures the Himax
camera, optionally resizes and JPEG-encodes each frame, and streams it to a host
in the cf_lab image format. `host/run.py` shows the stream and records it as
telemetry, with motion-capture poses alongside when you want them.

```text
Himax -> lib/gap8/camera_stream -> resize / JPEG -> CPX -> ESP32 WiFi -> host/run.py
```

You choose the camera configuration when you build; nothing in the code needs
editing. [camera_bench](../camera_bench) builds, flashes and measures every
configuration against its tracked baseline.

## Quick start

```bash
# 1. Build and flash, from the repository root. WiFi comes from configs/drone.yaml.
cf-lab flash gap8 --example camera_stream --make-args "SENSOR=QVGA_60 FORMAT=JPEG JPEG=CLUSTER"

# 2. Put this computer on the deck's network (access point mode: join its SSID).

# 3. Watch and record.
python examples/camera_stream/host/run.py
```

The deck streams a few seconds after the flash finishes; `run.py` keeps retrying
until it answers.

## Build settings

Camera settings are make variables passed with `--make-args`. Each becomes a
`-D` flag: the stream settings for
[gap8/camera_stream_config.h](gap8/camera_stream_config.h), which holds their
defaults, and `FLIP` and the exposure settings for the camera library's own
defaults (`CAMERA_STREAM_DEFAULT_*` in
[camera_stream.h](../../lib/gap8/camera_stream/inc/camera_stream.h)). Leave a
variable out to keep its default. Values are not case-sensitive, and a value that is not one of the
choices stops the build with the list.

| Variable | Choices | Default |
| --- | --- | --- |
| `SENSOR` | `QVGA_30` `QVGA_60` (320x240), `QQVGA_BIN2_30` `QQVGA_BIN2_60` `QQVGA_BIN2_120` (160x120, 2x2 binned) | `QVGA_60` |
| `SIZE` | output `WxH`, resized on the deck, e.g. `96x96` | the capture size |
| `FIT` | `SQUASH` (whole view, aspect distorted), `CROP` (aspect kept, view cropped) | `SQUASH` |
| `FORMAT` | `RAW`, `JPEG` | `RAW` |
| `JPEG` | `FC` (fabric controller), `CLUSTER` (8 cluster cores) | `FC` |
| `QUALITY` | JPEG quality, 1..100 | 50 |
| `POLICY` | `PREVIEW`, `EXACT` | `PREVIEW` |
| `BUFFERS` | `2`, `3` camera buffers (with `POLICY=EXACT`) | `2` |
| `FPS` | a frame-rate cap below the sensor's | the sensor's |
| `FLIP` | 0..3: bit 0 mirrors, bit 1 flips vertically | 3 (upright for the deck's mounting) |
| `WINDOW` | first sensor row shown, 0..84 (42 = centred; see below) | QVGA: the sensor's QVGA window; QQVGA: centred |
| `EXPOSURE` | `AUTO`, `MANUAL` | `AUTO` |
| `AE_TARGET` | auto exposure's target brightness, 0..255 | 60 |
| `AE_MAX_EXPOSURE_US` `AE_MAX_ANALOG_GAIN` `AE_MAX_DIGITAL_GAIN` | what auto exposure may use: µs (0 = the frame period), `1` `2` `4` `8`, 256..1023 (256 = 1x) | 0, 8, 768 |
| `EXPOSURE_US` `ANALOG_GAIN` `DIGITAL_GAIN` | manual exposure: µs, `1` `2` `4` `8`, 1..1023 (256 = 1x) | 4000, 1, 256 |
| `FLICKER` | `OFF`, `50_HZ`, `60_HZ` (auto exposure only; not at 120 FPS) | `OFF` |
| `LOG` | `1` sends a status line every second (see [Deck messages](#deck-messages)) | 0 |

`cf-lab build` and `cf-lab flash` rebuild from clean by default, so a changed
setting never mixes with objects from the previous build.

Choosing:

- **Raw or JPEG.** Raw QVGA is 76.8 KB a frame, more than WiFi carries at the
  sensor's rate; JPEG reaches it. JPEG is recorded as sent, never re-encoded.
- **FC or cluster JPEG.** Both produce byte-identical output. `CLUSTER` is many
  times faster, but it takes all 8 cluster cores and more L2, so it cannot run
  beside a neural network on the cluster.
- **Preview or exact.** `PREVIEW` keeps one camera buffer in a continuous
  overwrite loop: the least memory and latency, but a frame can tear while it is
  being sent. `EXACT` hands out whole frames that nothing overwrites while they
  are sent. The newest finished frame replaces an unconsumed older one, so the
  camera always keeps the sensor's rate. With a `SIZE`, a frame is resized only
  when the stream takes it, and its camera buffer goes straight back to the
  camera: a fast sensor costs no extra work.
- **Two or three exact buffers.** A native frame held longer than one sensor
  period (FC JPEG, say) leaves two buffers nowhere to keep the next finished
  frame, so the stream then waits for the frame in flight and its rate rounds
  down to whole sensor periods (FC JPEG at `QQVGA_BIN2_60`: 30 FPS).
  `BUFFERS=3` keeps the newest frame ready instead, for one more camera buffer.
  A resized frame never needs it.

Not every combination fits in the GAP8's L2 memory. A native QVGA stream with
`POLICY=EXACT BUFFERS=3` holds three 77 KB camera buffers and a whole-frame
transmit queue, which leaves no room for `JPEG=CLUSTER`'s 128 KB of encoder
buffers. Such a build compiles, then reports "out of L2 memory" on the radio
console at boot and sends no frames; use `JPEG=FC`, `BUFFERS=2`, a QQVGA sensor
or a smaller `SIZE`.

The sensor has 324 rows; `WINDOW` picks which 240 of them (120 binned rows in
QQVGA) the frame shows, counted in full-resolution rows so one value gives the
same view in QVGA and QQVGA. QVGA defaults to the sensor's own window, which
sits low in the view with the default `FLIP=3` and is the only one `QVGA_60` has
time to read; any other QVGA window needs `QVGA_30`. QQVGA defaults to the
centre. To make QQVGA show exactly what QVGA does, pass the row the config line
reports for the QVGA build (`window=`).

For photogrammetry or Gaussian splatting, lock the exposure so brightness never
drifts between frames:

```bash
--make-args "SENSOR=QVGA_30 POLICY=EXACT EXPOSURE=MANUAL EXPOSURE_US=16000 ANALOG_GAIN=8"
```

## WiFi

The WiFi settings come from the `camera:` section of the drone config
(`configs/drone.yaml`, or `--config`), so the build and the host always agree:

| `camera.mode` | The deck | Deck address | This computer |
| --- | --- | --- | --- |
| `access_point` | is the network `access_point_ssid` | `192.168.4.1` | joins that network |
| `station` | joins `station_ssid` | from DHCP | stays on its network |

The host does not need the address: the build names the deck after `radio.uri`
(`cf-80-2m-e7e7e7e7e7.local` over mDNS), and `run.py` looks it up by that name in
either mode, and again before each retry: for about 15 s after the deck reboots,
avahi can still answer with its old address, and in station mode the deck may come
back on a new one. `cf-lab find-deck` prints it.

In access point mode, `access_point_channel` picks the 2.4 GHz channel (1..13,
1..11 in the US). Choose one away from strong neighbouring networks, which
`nmcli device wifi list` shows; it needs the ESP32 firmware from this repository
(`cf-lab flash esp`), and older firmware stays on channel 6.

Override any of them for one build on the command line:

```bash
cf-lab flash gap8 --example camera_stream --wifi-mode access_point --ssid my-deck --channel 1 --transport udp
```

`run.py` reads the same config, so change `drone.yaml` instead when the change
should stick.

- The ESP32 is 2.4 GHz only: in station mode, use a 2.4 GHz-only SSID. A network
  that steers clients between bands (Google and Nest WiFi do by default) never
  associates.
- An access-point key must be empty or at least 8 characters; for a shorter key
  the ESP32 starts no access point at all.
- The name needs the ESP32 firmware from this repository (`cf-lab flash esp`);
  older firmware keeps its own `aideck-XXXXXX` name and the lookup fails. Rebuild
  the GAP8 app after changing `radio.uri`.
- Joining the deck's access point takes a single-network computer off the
  motion-capture network. Use station mode, or a second network interface, to
  record mocap beside the camera.

`camera.transport` picks TCP (every byte arrives, the rate dips under
interference) or UDP (no retransmits; a frame with a lost packet is dropped and
the stream resumes at the next one). Use TCP for recordings you analyse.

The received frame rate depends heavily on the WiFi around you, while the deck
keeps capturing at the sensor's rate. Two things matter on the host side:

- Turn off WiFi power saving for the deck's network, which Ubuntu enables by
  default: `nmcli connection modify <profile> 802-11-wireless.powersave 2`.
- An active Crazyradio link to the same drone disturbs its WiFi, since the two
  radios sit centimetres apart. Expect some of that while flying.

## Watching and recording

```bash
python examples/camera_stream/host/run.py                  # window + recording
python examples/camera_stream/host/run.py --no-display     # metrics in the terminal instead
python examples/camera_stream/host/run.py --duration 30    # stop after 30 s
python examples/camera_stream/host/run.py --no-render      # skip the video at the end
```

`run.py` reads everything from the drone config: `camera:` says how to reach the
deck, and `telemetry:` what to record (`telemetry.camera` for frames,
`telemetry.mocap` for the raw global pose of `mocap.rigid_body`). Before it
records anything it checks that it has what it needs, and stops if:

- no frames arrive from the deck, or
- `telemetry.mocap` is on and no pose of the rigid body arrives (untracked in
  Motive, wrong name, or the Motive PC unreachable).

With `telemetry.mocap` off it records the camera alone. When it stops (Ctrl-C,
closing the window, or `--duration`) it renders the frames to `camera.mp4` at the
frame rate the deck was built for.

Runs go to `artifacts/run_<timestamp>/` (git ignores `artifacts/`):

```text
camera_frames.bin    every frame's bytes as received, concatenated
camera.parquet       one row per frame: offset and size in the blob, deck frame id and clock, extension
mocap.parquet        raw global poses, when telemetry.mocap is on
events.parquet       run events, including the deck's config and status lines
manifest.json        status, per-stream counts
camera.mp4           the rendered video
```

To render again at another rate, losslessly, or to extract the frames as files:

```bash
python -m cf_lab.camera.video examples/camera_stream/artifacts/run_<stamp> --fps 30
python -m cf_lab.camera.video examples/camera_stream/artifacts/run_<stamp> --lossless
python -m cf_lab.camera.video examples/camera_stream/artifacts/run_<stamp> --extract frames/
```

`camera.mp4` is H.264: small, but its compression smears the fine detail of a
low-resolution frame. `--lossless` writes `camera_lossless.mp4` instead, VP9 in
lossless mode at the recorded resolution: every decoded pixel equals the recorded
one (for a JPEG build, the frame as the host decoded it). It plays in VSCode,
Chrome, Edge, Firefox and the Slack desktop app, and is much larger than
`camera.mp4`.

To just look at the stream, without recording, use `cf-lab view`, which finds
the deck the same way (`--ip` and `--transport` override it). In the window:
F fullscreen, Space pause, S snapshot, 1 actual size, M smoothing, P side panel,
Q quit.

## Deck messages

Besides frames, the deck sends console lines on the same connection, which the
host records in `events.parquet`:

- `camera config: sensor=QVGA_60 capture=320x240 ... exposure=auto ae_target=60 ...
  out=320x240 fps=60 format=jpeg ...` once per connection, before the first
  frame: every camera setting the deck runs with (exposure and gains included),
  then the stream's. This is how `run.py` knows the frame rate, the viewer shows
  the build, and a run's telemetry records the camera setup.
- `status frames=... captures=... dropped_capture=... watchdog_restarts=...
  tx_free=... spi_hard=... exposure_us=... analog_gain=... digital_gain=...`
  every second with `LOG=1`: the camera's and the transport's counters, for
  when a stream misbehaves, and the exposure the sensor is using.

Boot progress and errors go to the radio console, where cfclient's console tab
shows them.

## Wire format

Frames use the shared cf_lab image format: `lib/shared/include/cf_lab/img_protocol.h`
on the deck, `cf_lab.imgstream` on the host.

```text
[ one CPX packet: 12-byte header | 36-byte camera_stream extension ]
[ CPX packets carrying the image bytes ]
```

The header is fixed and self-describing, so any cf_lab viewer decodes any cf_lab
app's stream. The extension carries the frame id, the deck's capture timestamp,
the capture, resize and JPEG times, and the deck's cumulative drop counters
(`cf_lab.camera.ext`). Console lines travel as CPX console packets on the same
socket.
