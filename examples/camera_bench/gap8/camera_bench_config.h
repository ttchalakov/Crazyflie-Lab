// camera_bench_config.h -- the settings of the camera_bench GAP8 app, with defaults.
//
// The camera settings it takes are camera_stream's (same names, same make
// variables; see the Makefile), so a bench case is a camera_stream build with the
// window and exposure at the camera library's defaults. The bench runner passes
// them from host/bench.yaml.
// WiFi is the bench's own access point, from bench.yaml. Its own instrumentation
// settings follow the camera settings.

#ifndef CAMERA_BENCH_CONFIG_H
#define CAMERA_BENCH_CONFIG_H

// -- WiFi ---------------------------------------------------------------------
// Access point: the deck is the network, at 192.168.4.1. Station: the deck joins
// WIFI_SSID and takes a DHCP address, printed as "WiFi up (a.b.c.d)" on the
// radio console. The ESP32 is 2.4 GHz only.
#ifndef WIFI_MODE
#define WIFI_MODE CPX_WIFI_MODE_ACCESS_POINT  // or CPX_WIFI_MODE_STATION
#endif
#ifndef WIFI_SSID
#define WIFI_SSID "camera-bench"
#endif
// Empty for an open network. An access point needs 8 or more characters or none:
// the ESP32 starts no access point at all for a shorter key.
#ifndef WIFI_KEY
#define WIFI_KEY ""
#endif
#ifndef WIFI_TRANSPORT
#define WIFI_TRANSPORT CPX_WIFI_TRANSPORT_TCP  // or CPX_WIFI_TRANSPORT_UDP
#endif
// Access point channel, 1..13 (in the US only 1..11 are allowed). 0 keeps the
// ESP32 firmware's default, 6. Pick one away from strong neighbouring networks.
#ifndef WIFI_CHANNEL
#define WIFI_CHANNEL 0
#endif

// -- Sensor mode                                   make: SENSOR=QVGA_60 ... ----
#define SENSOR_MODE_QVGA_30        0  // 320x240 at 30 FPS
#define SENSOR_MODE_QVGA_60        1  // 320x240 at 60 FPS
#define SENSOR_MODE_QQVGA_BIN2_30  2  // 160x120 at 30 FPS, 2x2 binned
#define SENSOR_MODE_QQVGA_BIN2_60  3  // 160x120 at 60 FPS, 2x2 binned
#define SENSOR_MODE_QQVGA_BIN2_120 4  // 160x120 at 120 FPS, 2x2 binned
#ifndef SENSOR_MODE
#define SENSOR_MODE SENSOR_MODE_QVGA_60
#endif

// -- Output size                                  make: SIZE=96x96 FIT=CROP --
// Frames are resized on the deck when the output is smaller than the capture.
// FIT_SQUASH keeps the whole view and distorts the aspect; FIT_CROP keeps the
// aspect and crops the view to it.
#define FIT_SQUASH 0
#define FIT_CROP   1
#ifndef OUT_FIT
#define OUT_FIT FIT_SQUASH
#endif
// OUT_WIDTH/OUT_HEIGHT default to the capture size, set below.

// -- Stream format                     make: FORMAT=JPEG JPEG=CLUSTER QUALITY=50 --
// Raw QVGA fits about 10 FPS over WiFi; JPEG reaches the sensor rate.
#define FORMAT_RAW  0
#define FORMAT_JPEG 1
#ifndef STREAM_FORMAT
#define STREAM_FORMAT FORMAT_RAW
#endif
// FC encodes on the fabric controller; CLUSTER spreads it over the 8 cluster cores.
#define JPEG_ENCODER_FC      0
#define JPEG_ENCODER_CLUSTER 1
#ifndef JPEG_ENCODER
#define JPEG_ENCODER JPEG_ENCODER_FC
#endif
#ifndef JPEG_QUALITY
#define JPEG_QUALITY 50  // 1..100
#endif

// -- Capture policy                               make: POLICY=EXACT BUFFERS=3 --
// PREVIEW keeps one camera buffer in a continuous overwrite loop: the lowest
// latency and memory, but a frame can tear while it is being sent. EXACT hands
// out whole frames, never overwritten while held, at the cost of more memory;
// the newest finished frame replaces an unconsumed older one. With two buffers,
// a frame held longer than one sensor period leaves the camera nowhere to keep
// the next, so the following acquire waits for the frame in flight. BUFFERS=3
// keeps the newest frame ready instead. It only helps at native size: a resized
// frame gives its camera buffer back as soon as it is resized.
#define CAPTURE_POLICY_PREVIEW 0
#define CAPTURE_POLICY_EXACT   1
#ifndef CAPTURE_POLICY
#define CAPTURE_POLICY CAPTURE_POLICY_PREVIEW
#endif
#ifndef EXACT_BUFFERS
#define EXACT_BUFFERS 2  // 2 or 3
#endif

// -- Frame rate cap                                              make: FPS=30 --
// TARGET_FPS defaults to the sensor mode's rate, set below.

// -- Orientation                                                  make: FLIP=3 --
// Sets lib/gap8/camera_stream's CAMERA_STREAM_DEFAULT_ORIENTATION: bit 0 mirrors
// horizontally, bit 1 flips vertically; its default 3 turns the image the right
// way up for the deck's mounting.

// -- Bench instrumentation -----------------------------------------------------
// The bench always sends the status lines (camera/transport counters, then the
// app stage, free L2 and task stack headroom) every BENCH_STATUS_INTERVAL_MS.
#define BENCH_STATUS_INTERVAL_MS 1000
// How long a status line may wait for transmit queue space before it is dropped.
#define STATUS_SEND_TIMEOUT_MS 100
// LIFECYCLE=1: instead of streaming, run the camera start/stop/resume/close
// self-test at boot and report the result to each client that connects.
#ifndef BENCH_LIFECYCLE
#define BENCH_LIFECYCLE 0
#endif
// TRACE=1: a "trace" line for each of the first BENCH_TRACE_FRAMES frames.
#ifndef BENCH_TRACE
#define BENCH_TRACE 0
#endif
#define BENCH_TRACE_FRAMES 8

// -- Tuning (no make variable) -------------------------------------------------
#define JPEG_CLUSTER_CORES 8
#define JPEG_OUT_BUDGET (64 * 1024)
// A new client gets this long to settle before the first frame.
#define STREAM_CLIENT_SETTLE_MS 150

// -- Derived; not settings ----------------------------------------------------
#if SENSOR_MODE == SENSOR_MODE_QVGA_30 || SENSOR_MODE == SENSOR_MODE_QVGA_60
#define CAM_WIDTH  320
#define CAM_HEIGHT 240
#else
#define CAM_WIDTH  160
#define CAM_HEIGHT 120
#endif
#if SENSOR_MODE == SENSOR_MODE_QVGA_30 || SENSOR_MODE == SENSOR_MODE_QQVGA_BIN2_30
#define SENSOR_FPS 30
#elif SENSOR_MODE == SENSOR_MODE_QVGA_60 || SENSOR_MODE == SENSOR_MODE_QQVGA_BIN2_60
#define SENSOR_FPS 60
#else
#define SENSOR_FPS 120
#endif
#ifndef OUT_WIDTH
#define OUT_WIDTH CAM_WIDTH
#endif
#ifndef OUT_HEIGHT
#define OUT_HEIGHT CAM_HEIGHT
#endif
#ifndef TARGET_FPS
#define TARGET_FPS SENSOR_FPS
#endif
#define OUT_IS_NATIVE (OUT_WIDTH == CAM_WIDTH && OUT_HEIGHT == CAM_HEIGHT)

// Camera buffers: preview shares one live buffer, exact keeps EXACT_BUFFERS.
#if CAPTURE_POLICY == CAPTURE_POLICY_PREVIEW
#define RAW_CAPTURE_BUFFER_COUNT 1
#elif EXACT_BUFFERS == 2 || EXACT_BUFFERS == 3
#define RAW_CAPTURE_BUFFER_COUNT EXACT_BUFFERS
#else
#error "EXACT_BUFFERS must be 2 or 3"
#endif

// Size lib/gap8/camera_stream for exactly this build (see camera_stream.h).
#define CAMERA_STREAM_MAX_CAPTURE_BUFFERS RAW_CAPTURE_BUFFER_COUNT
#define CAMERA_STREAM_SPECIALIZE_SENSOR_MODE SENSOR_MODE
#define CAMERA_STREAM_SPECIALIZE_POLICY CAPTURE_POLICY

// CPX transmit queue, in packets (lib/gap8/cpx). Preview sends from a small
// queue while the camera keeps capturing; exact admits a whole frame at once so
// it goes out contiguously.
#if CAPTURE_POLICY == CAPTURE_POLICY_PREVIEW
#define TXQ_SIZE 8
#else
#define TXQ_SIZE 96
#endif

#endif  // CAMERA_BENCH_CONFIG_H
