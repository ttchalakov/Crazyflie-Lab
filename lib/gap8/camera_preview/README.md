# GAP8 Camera Preview Facade

`camera_preview` is the small, opinionated entry point for applications that
want a QVGA 60 FPS cluster-JPEG preview stream without rebuilding
the camera/JPEG/Wi-Fi orchestration themselves. The lower-level libraries remain
available for DORY input leases, resizing, exact ownership, custom encoding, and
manual transmission control.

```c
#include "pmsis.h"
#include "bsp/bsp.h"
#include "camera_preview.h"

static camera_preview_t preview;

static void app_task(void *argument) {
  (void)argument;
  pi_bsp_init();

  if (camera_preview_start_default(
          &preview, CAMERA_PREVIEW_TRANSPORT_TCP) != CAMERA_PREVIEW_OK) {
    pmsis_exit(-1);
  }

  /* Run the experiment. Preview capture/JPEG/transmission is asynchronous. */

  camera_preview_close(&preview); /* JPEG/cluster off, physical camera STOP */
  pmsis_exit(0);
}

int main(void) { return pmsis_kickoff((void *)app_task); }
```

Include the umbrella Make fragment instead of listing each dependency:

```make
PMSIS_OS = freertos
CF_LAB ?= $(abspath $(CURDIR)/../cf-lab)
include $(CF_LAB)/lib/gap8/camera_preview/camera_preview.mk
```

The umbrella selects FreeRTOS when `PMSIS_OS` is unset and rejects an explicitly
different OS. It has include guards in every dependency, so it may appear before
or after the individual component fragments without duplicating sources.

The default is an open `camera-stream` access point, TCP transport, QVGA at
60 FPS, one continuously overwritten camera buffer, quality-50 JPEG on eight
cluster cores, a 64 KiB JPEG output budget, and the camera demo's optional
36-byte timing/drop metadata. UDP is selected by changing the enum in the one
start call.

## GAP8 camera controls

The facade applies orientation, exposure, and flicker configuration while Himax
is still in standby, before its first physical stream start. It uses
`camera_stream`'s defaults, so the same GAP8 compiler flags set them:

```make
APP_CFLAGS += -DCAMERA_STREAM_DEFAULT_ORIENTATION=3
APP_CFLAGS += -DCAMERA_STREAM_DEFAULT_EXPOSURE_MODE=CAMERA_STREAM_EXPOSURE_AUTO
APP_CFLAGS += -DCAMERA_STREAM_DEFAULT_AE_TARGET_MEAN=0x3c
APP_CFLAGS += -DCAMERA_STREAM_DEFAULT_FLICKER_MODE=CAMERA_STREAM_FLICKER_60_HZ
APP_CFLAGS += -DCAMERA_STREAM_DEFAULT_SENSOR_SETTLE_MS=200
```

`CAMERA_STREAM_DEFAULT_ORIENTATION` is the two-bit HM01B0 orientation value:
bit 0 is horizontal mirror and bit 1 is vertical flip. The default is 3, which
applies both. The implementation trusts that board-specific setting; it never
examines image content to guess or switch orientation.

Automatic exposure uses `CAMERA_STREAM_DEFAULT_AE_TARGET_MEAN` as the sensor
loop's target brightness. For a fixed exposure, select
`CAMERA_STREAM_EXPOSURE_MANUAL` and also set:

```make
APP_CFLAGS += -DCAMERA_STREAM_DEFAULT_EXPOSURE_MODE=CAMERA_STREAM_EXPOSURE_MANUAL
APP_CFLAGS += -DCAMERA_STREAM_DEFAULT_MANUAL_EXPOSURE_US=5000
APP_CFLAGS += -DCAMERA_STREAM_DEFAULT_MANUAL_ANALOG_GAIN=CAMERA_STREAM_ANALOG_GAIN_2X
APP_CFLAGS += -DCAMERA_STREAM_DEFAULT_MANUAL_DIGITAL_GAIN=256
APP_CFLAGS += -DCAMERA_STREAM_DEFAULT_FLICKER_MODE=CAMERA_STREAM_FLICKER_DISABLED
```

Manual exposure is in microseconds, up to the frame period. Analog gain is
`CAMERA_STREAM_ANALOG_GAIN_1X` to `_8X`; digital gain is in 1/256 steps, 256
being 1x. Flicker mode is disabled, 50 Hz,
or 60 Hz and applies only to automatic exposure. It constrains auto-exposure to
mains-compatible integration steps; it cannot remove bands caused by an unrelated
PWM/LED frequency, and `QQVGA_BIN2_120` cannot accommodate either supported
flicker step.

The corresponding fields remain ordinary runtime configuration when using the
expanded facade API:

```c
camera_preview_config_t config;

camera_preview_config_init(&config);
config.camera.orientation = 0;
config.camera.exposure_mode = CAMERA_STREAM_EXPOSURE_AUTO;
config.camera.ae_target_mean = 0x30;
config.camera.flicker_mode = CAMERA_STREAM_FLICKER_60_HZ;
config.camera.sensor_settle_ms = 200;
```

On the first physical `START`, capture runs while publication and transmission are
suppressed for `sensor_settle_ms` (200 ms by default). This gives retimed sensor
registers several real frame boundaries in which to become active; a soft camera
stop/resume does not repeat it. These are firmware choices only: the facade adds no
Python, radio, Wi-Fi, or streaming-protocol command for changing them at runtime.

The fragment also specializes `camera_stream` to one QVGA60 preview buffer by
default, which reduces code and object size. Set
`CAMERA_PREVIEW_SPECIALIZE_CAMERA=0` before the include if the same binary also
creates another lower-level camera-stream profile. The make variable
`TXQ_SIZE` (default 8) sizes the process-wide CPX transmit queue; size it for
every CPX producer in the firmware, not only the preview.

The preset owns camera/JPEG/transport setup but deliberately does not change
application-wide PMU voltage or FC frequency. The camera_stream example runs at
1.2 V with the FC at 150 MHz for cluster JPEG; set the same in the application
to compare throughput with it. Owned cluster JPEG does set its configured
cluster frequency to 150 MHz. Lifecycle calls must run from a live PMSIS/FreeRTOS task,
after the application's normal BSP/board initialization.

## Cluster-free preset

The named default uses the cluster internally even though the calling
application does not manage it. If an experiment needs the cluster for DORY or
wants preview never to touch it, use the FC encoder preset:

```c
camera_preview_config_t config;
camera_preview_config_init(&config);
camera_preview_config_use_fc_jpeg(&config);
camera_preview_init(&preview, &config);
camera_preview_start(&preview);
```

FC JPEG leaves cluster L1 and the cluster device untouched, but QVGA encoding
is substantially slower and consumes FC time.

## Ownership and shutdown

- The one-call default initializes CPX exactly once, after Himax opens. This
  preserves the camera-before-transport allocation order. If the application
  already initialized CPX, use `camera_preview_config_use_existing_cpx()` before
  `camera_preview_init()`; never initialize CPX through both paths.
- CPX exposes one `CPX_F_WIFI_CTRL` queue and no queue teardown. The facade's
  internal Wi-Fi receiver therefore remains alive for the process lifetime.
  `close()` stops the preview worker, closes JPEG and any owned cluster, and
  physically closes Himax, but it does not shut down CPX or the ESP32 AP.
- A later start can reuse the same internal SSID/key/transport. Changing TCP to
  UDP or changing credentials after the internal session has opened requires a
  GAP8 reboot. Applications that already own Wi-Fi can supply an opened,
  connected `borrowed_wifi_session` after selecting existing CPX with
  `camera_preview_config_use_existing_cpx()`.
- `imgstream_send()` is intentionally frame-serial and blocking. `close()` waits
  for an in-flight frame send rather than force-deleting a task that may hold a
  CPX semaphore.
- The caller-owned `camera_preview_t` must be static, global, or otherwise remain
  valid until `camera_preview_close()` returns.

## Memory planning

Call `camera_preview_memory_plan()` before `camera_preview_init()` to inspect the
raw/JPEG bulk allocations, facade and internal-Wi-Fi objects, CPX/Wi-Fi queue
payloads, and the camera/CPX/Wi-Fi/preview task stacks. `known_l2_bytes` combines
those known quantities. It is intentionally not a linker-map replacement: RTOS
queue/control-block metadata, lower-level static BSS, platform driver state, and
cluster L1 are implementation-dependent and excluded. When CPX or Wi-Fi is
borrowed, their already-existing allocations are reported as zero incremental
cost so an application can avoid double-counting them.
