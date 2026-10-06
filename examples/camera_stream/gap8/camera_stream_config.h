// camera_stream_config.h -- every setting of the camera_stream demo, with its default.
//
// Nothing here needs editing. Each setting is an #ifndef default that the build
// overrides with a -D flag:
//
//   WiFi    comes from the camera: section of the drone config (configs/drone.yaml),
//           which `cf-lab build gap8` and `cf-lab flash gap8` turn into -D flags.
//           --wifi-mode, --ssid, --key and --transport override it.
//   Camera  comes from make variables, which the Makefile turns into -D flags:
//             cf-lab flash gap8 --example camera_stream --make-args "SENSOR=QVGA_60 FORMAT=JPEG"
//           The make variable for each setting is named beside it below.
//
// A default in a header plus a -D flag from the build is the pattern to copy for
// your own app's settings. A setting a library already defaults (orientation,
// exposure) goes straight to that library's hook instead. The Makefile
// force-includes this file into every source, the shared libraries included, so
// the derived values at the bottom size them for this build.

#ifndef CAMERA_STREAM_CONFIG_H
#define CAMERA_STREAM_CONFIG_H

// -- WiFi ---------------------------------------------------------------------
// Access point: the deck is the network, at 192.168.4.1. Station: the deck joins
// WIFI_SSID and takes a DHCP address, printed as "WiFi up (a.b.c.d)" on the
// radio console. The ESP32 is 2.4 GHz only.
#ifndef WIFI_MODE
#define WIFI_MODE CPX_WIFI_MODE_ACCESS_POINT  // or CPX_WIFI_MODE_STATION
#endif
#ifndef WIFI_SSID
#define WIFI_SSID "camera-stream"
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
// The deck's name on the network, which is how the host finds it in either mode:
// cf_lab derives both from the drone config's radio.uri (cf-80-2m-e7e7e7e7e7.local,
// TXT name 80/2M/E7E7E7E7E7). Empty keeps the ESP32's own aideck-XXXXXX.
#ifndef WIFI_HOSTNAME
#define WIFI_HOSTNAME ""
#endif
#ifndef WIFI_NAME
#define WIFI_NAME ""
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

// -- Window                                                    make: WINDOW=42 --
// The frame's first row among the sensor's 324, from the top of the image,
// 0..84, in full-resolution rows so one value frames the same view in QVGA and
// QQVGA; 42 is centred. By default QVGA uses the sensor's own QVGA window (the
// only one QVGA_60 has time for; other rows need QVGA_30) and QQVGA is centred.
// WINDOW_TOP defaults to camera_stream's own choice.

// -- Orientation and exposure   make: FLIP=3 EXPOSURE=MANUAL EXPOSURE_US=5000 ... --
// These set lib/gap8/camera_stream's own defaults (CAMERA_STREAM_DEFAULT_* in
// camera_stream.h, which documents them), so this app restates none of them.
// FLIP: bit 0 mirrors horizontally, bit 1 flips vertically; the default 3 turns
// the image the right way up for the deck's mounting. Exposure times are in
// microseconds, up to the frame period; digital gains in 1/256 steps (256 = 1x,
// at most 1023). AE_MAX_* bound what automatic exposure may use. FLICKER steps
// auto exposure with the mains frequency; not available in the 120 FPS mode.

// -- Status log                                                  make: LOG=1 --
// Sends a "status ..." line with camera and transport counters on the image
// socket every CAMERA_STREAM_LOG_INTERVAL_MS; the host records it in the run.
#ifndef CAMERA_STREAM_LOG
#define CAMERA_STREAM_LOG 0
#endif
#define CAMERA_STREAM_LOG_INTERVAL_MS 1000
#define STATUS_SEND_TIMEOUT_MS 100

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

#endif  // CAMERA_STREAM_CONFIG_H
