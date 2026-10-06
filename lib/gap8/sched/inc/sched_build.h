/*
 * sched_build.h -- the build-time settings of lib/gap8/sched, with defaults.
 *
 * sched.mk turns make variables into these (SENSOR=QVGA_60 FORMAT=JPEG ...), so
 * a level-1 application chooses its camera and stream without writing code. An
 * application may also define any of them itself before including sched.h.
 */
#ifndef CF_GAP8_SCHED_BUILD_H
#define CF_GAP8_SCHED_BUILD_H

#include "camera_stream.h"
#include "cpx_wifi_protocol.h"

/* -- Camera                                       make: SENSOR=QVGA_60 ... ---- */
#ifndef SCHED_SENSOR_MODE
#define SCHED_SENSOR_MODE CAMERA_STREAM_SENSOR_QVGA_60
#endif
/* Exact frames, so the network and the stream never see a frame being written.
 * 0 chooses: three buffers keep the newest frame ready while one is held; QVGA
 * gets two, as three (230 KB) leave a network almost no L2. make: BUFFERS=2|3 */
#ifndef SCHED_CAMERA_BUFFERS
#define SCHED_CAMERA_BUFFERS 0
#endif
#ifndef SCHED_CAMERA_ORIENTATION
#define SCHED_CAMERA_ORIENTATION CAMERA_STREAM_DEFAULT_ORIENTATION
#endif
#ifndef SCHED_CAMERA_WATCHDOG_MS
#define SCHED_CAMERA_WATCHDOG_MS 1000u
#endif

/* -- Video stream                          make: FORMAT=JPEG QUALITY=50 STREAM=1 -- */
#define SCHED_FORMAT_RAW 0
#define SCHED_FORMAT_JPEG 1 /* encoded on the cluster */
#ifndef SCHED_STREAM_FORMAT
#define SCHED_STREAM_FORMAT SCHED_FORMAT_JPEG
#endif
#ifndef SCHED_STREAM
#define SCHED_STREAM 1
#endif
#ifndef SCHED_JPEG_QUALITY
#define SCHED_JPEG_QUALITY 50
#endif
#ifndef SCHED_JPEG_CORES
#define SCHED_JPEG_CORES 8
#endif
/* Cluster jobs per encode; 0 chooses (sched_config_t.jpeg_slices). */
#ifndef SCHED_JPEG_SLICES
#define SCHED_JPEG_SLICES 0
#endif
/* The encoded frame's L2 buffer. 0 chooses a third of the raw frame, far above
 * a quality-50 frame; a frame that does not fit is not sent (encode_failures).
 * The worst case the encoder could produce is several times the raw frame. */
#ifndef SCHED_JPEG_BUDGET_BYTES
#define SCHED_JPEG_BUDGET_BYTES 0u
#endif

/* -- WiFi: the same macros every deck app takes (cf_lab's WIFI_DEFINES) ------ */
#ifndef WIFI_MODE
#define WIFI_MODE CPX_WIFI_MODE_ACCESS_POINT
#endif
#ifndef WIFI_SSID
#define WIFI_SSID "aideck-sched"
#endif
#ifndef WIFI_KEY
#define WIFI_KEY ""
#endif
#ifndef WIFI_TRANSPORT
#define WIFI_TRANSPORT CPX_WIFI_TRANSPORT_UDP
#endif
#ifndef WIFI_CHANNEL
#define WIFI_CHANNEL 0
#endif
/* The deck's mDNS hostname and TXT name; empty keeps the ESP32's aideck-XXXXXX. */
#ifndef WIFI_HOSTNAME
#define WIFI_HOSTNAME ""
#endif
#ifndef WIFI_NAME
#define WIFI_NAME ""
#endif

/* -- Schedulers                       make: SCHED=PIPELINED | SEQUENTIAL | BOTH -- */
#ifndef SCHED_BUILD_PIPELINED
#define SCHED_BUILD_PIPELINED 1
#endif
#ifndef SCHED_BUILD_SEQUENTIAL
#define SCHED_BUILD_SEQUENTIAL 0
#endif
#ifndef SCHED_DEFAULT_KIND
#define SCHED_DEFAULT_KIND (SCHED_BUILD_PIPELINED ? SCHED_PIPELINED : SCHED_SEQUENTIAL)
#endif

/* -- Clocks -- */
#ifndef SCHED_FC_HZ
#define SCHED_FC_HZ 150000000u
#endif
#ifndef SCHED_CLUSTER_HZ
#define SCHED_CLUSTER_HZ 150000000u
#endif

#ifndef SCHED_REPORT_MS
#define SCHED_REPORT_MS 1000u
#endif

#endif /* CF_GAP8_SCHED_BUILD_H */
