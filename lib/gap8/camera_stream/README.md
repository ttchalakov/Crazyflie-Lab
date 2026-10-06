# GAP8 camera stream

This directory contains the transport-independent Himax continuous-capture
driver: the camera lifecycle, raw L2 buffers, ownership policies, completion
worker, and watchdog.

```make
include $(CF_LAB)/lib/gap8/camera_stream/camera_stream.mk
```

The core never calls CPX, Wi-Fi, JPEG, or a logger. Its cumulative counters and
raw-buffer addresses are available through `camera_stream_get_stats()` and
`camera_stream_get_info()`; they are useful for production health checks, not
just debugging. `lib/gap8/camera_stream_log` formats them, with the transport's
counters, into an optional status line, and `camera_stream_describe()` writes
every setting the stream runs with as one `k=v ...` string.

## Usage

```c
static camera_stream_t camera;           /* caller-owned, valid until close() */
camera_stream_config_t config;

camera_stream_config_init(&config);      /* the defaults below */
config.policy = CAMERA_STREAM_POLICY_EXACT_FRAME;

camera_stream_init(&camera, &config);    /* validate and copy only */
camera_stream_open(&camera);             /* the Himax first, then L2 buffers and the worker task */
camera_stream_start(&camera);

camera_stream_frame_t frame;
if (camera_stream_acquire(&camera, &frame, CAMERA_STREAM_WAIT_FOREVER) == CAMERA_STREAM_OK) {
  use_pixels(frame.data, frame.width, frame.height, frame.stride);
  camera_stream_release(&camera, &frame); /* with the token unchanged */
}

camera_stream_stop(&camera);             /* soft: capture keeps running, frames are discarded */
camera_stream_close(&camera);            /* hard: returns BUSY while a frame is leased */
```

At most one task may wait in `acquire()`, and at most one frame may be leased at
a time. `camera_stream_memory_plan()` reports a configuration's L2 needs before
`open()`. Open the camera before other large L2 users, as the examples do.

| Field | Values | Default |
| --- | --- | --- |
| `sensor_mode` | `CAMERA_STREAM_SENSOR_QVGA_30`, `_QVGA_60` (320x240); `_QQVGA_BIN2_30`, `_60`, `_120` (160x120, 2x2 binned) | `QVGA_60` |
| `policy` | `CAMERA_STREAM_POLICY_PREVIEW_LIVE`: one continuously overwritten buffer, so acquired pixels can tear. `CAMERA_STREAM_POLICY_EXACT_FRAME`: a completed buffer stays unchanged until `release()`, and the newest completed frame replaces an unconsumed older one, so the camera always runs at the sensor's rate | `PREVIEW_LIVE` |
| `capture_buffer_count` | 0 for the policy's own (preview 1, exact 2); exact also takes 3, which keeps the newest frame ready while a lease is held longer than one sensor period | 0 |
| `watchdog_timeout_ms` | a capture outstanding this long is restarted (counted in `watchdog_restarts`); 0 disables it | 2000 |

Three build-time switches trim the library: `CAMERA_STREAM_MAX_CAPTURE_BUFFERS`
(1..3, default 3) bounds the buffer count, and
`CAMERA_STREAM_SPECIALIZE_SENSOR_MODE` and `CAMERA_STREAM_SPECIALIZE_POLICY`
(default -1, any mode) fix the sensor mode or policy so the others' code is left
out; `init()` then rejects any other value.

## Orientation, window, and exposure

`camera_stream_config_init()` supplies the defaults below; each can also be
chosen at GAP8 compile time with the macro beside it.

| Compile-time default | Configuration field | Default |
| --- | --- | --- |
| `CAMERA_STREAM_DEFAULT_ORIENTATION` | `orientation` | `3` (horizontal + vertical flip) |
| | `window_top` | `CAMERA_STREAM_WINDOW_DEFAULT` |
| `CAMERA_STREAM_DEFAULT_EXPOSURE_MODE` | `exposure_mode` | `CAMERA_STREAM_EXPOSURE_AUTO` |
| `CAMERA_STREAM_DEFAULT_AE_TARGET_MEAN` | `ae_target_mean` | `0x3c` |
| `CAMERA_STREAM_DEFAULT_AE_MAX_EXPOSURE_US` | `ae_max_exposure_us` | `0` (the frame period) |
| `CAMERA_STREAM_DEFAULT_AE_MAX_ANALOG_GAIN` | `ae_max_analog_gain` | `CAMERA_STREAM_ANALOG_GAIN_8X` |
| `CAMERA_STREAM_DEFAULT_AE_MAX_DIGITAL_GAIN` | `ae_max_digital_gain` | `768` (3x) |
| `CAMERA_STREAM_DEFAULT_FLICKER_MODE` | `flicker_mode` | `CAMERA_STREAM_FLICKER_DISABLED` |
| `CAMERA_STREAM_DEFAULT_MANUAL_EXPOSURE_US` | `manual_exposure_us` | `4000` |
| `CAMERA_STREAM_DEFAULT_MANUAL_ANALOG_GAIN` | `manual_analog_gain` | `CAMERA_STREAM_ANALOG_GAIN_1X` |
| `CAMERA_STREAM_DEFAULT_MANUAL_DIGITAL_GAIN` | `manual_digital_gain` | `256` (1x) |
| `CAMERA_STREAM_DEFAULT_SENSOR_SETTLE_MS` | `sensor_settle_ms` | `200` ms |

**Orientation** uses the HM01B0 `IMG_ORIENTATION` bits: bit 0 mirrors
horizontally and bit 1 flips vertically; 3 is upright on the AI-deck. It is
programmed with the sensor timing before the first physical `START`. After that
`START`, completed frames are discarded for `sensor_settle_ms` so the new
settings get real frame boundaries in which to take effect; a soft
`stop()`/`start()` does not repeat it.

**The window** chooses which of the sensor's 324 rows the frame shows:
`window_top` is its first row, counted from the top of the image as delivered,
0 to 84. It is in full-resolution rows in every mode, so the same value frames
the same view in QVGA and in 2x2-binned QQVGA (which uses rows in pairs). 42 is
centred. The default, `CAMERA_STREAM_WINDOW_DEFAULT`, is the sensor's own QVGA
window for QVGA, the only window `QVGA_60` has time to read, and the centre for
QQVGA. Any other QVGA window reads the whole sensor, which only `QVGA_30` can;
`init()` rejects it at 60 FPS. `camera_stream_get_info()` reports the row the
default resolved to.

**Exposure** times are in microseconds and are limited by the frame period;
analog gain is `CAMERA_STREAM_ANALOG_GAIN_1X` to `_8X`; digital gain is in 1/256
steps (256 is 1x, 1023 the most). Automatic exposure drives the sensor's own loop
towards `ae_target_mean`, within `ae_max_exposure_us` (0 for the whole frame),
`ae_max_analog_gain` and `ae_max_digital_gain`; raising the gain limits trades
noise for brightness in the dark. `flicker_mode` (50 or 60 Hz, automatic only)
steps the exposure with the mains period so lights do not flicker; it does not
correct an unrelated LED PWM frequency, and `QQVGA_BIN2_120` has no room for it.
Manual exposure switches the loop off and uses `manual_exposure_us`,
`manual_analog_gain` and `manual_digital_gain`. `init()` rejects a value out of
range for the sensor mode.

The exposure fields can also change while the camera streams:

```c
camera_stream_info_t info;
camera_stream_get_info(&stream, &info);          /* the current settings */
info.config.exposure_mode = CAMERA_STREAM_EXPOSURE_MANUAL;
info.config.manual_exposure_us = 5000;
info.config.manual_analog_gain = CAMERA_STREAM_ANALOG_GAIN_2X;
camera_stream_set_exposure(&stream, &info.config); /* from the next frame on */

camera_stream_exposure_t now;
camera_stream_get_exposure(&stream, &now);       /* what the sensor uses now */
```

`camera_stream_set_exposure()` applies only the exposure fields, all in one
frame, and changes nothing when a value is out of range.
`camera_stream_get_exposure()` reads the sensor's exposure time and gains over
I2C, including the automatic loop's latest choice; `camera_stream_log` puts them
on its status line.
