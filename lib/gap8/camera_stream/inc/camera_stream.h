#ifndef CF_GAP8_CAMERA_STREAM_H
#define CF_GAP8_CAMERA_STREAM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Build-time capability controls.  A specialization value of -1 keeps that
 * choice runtime-configurable.  Otherwise it must name the corresponding enum
 * value below and camera_stream_init() rejects configurations that do not
 * match it.  Specialization lets production builds dead-strip unused policy
 * branches while retaining one source/API.
 */
#ifndef CAMERA_STREAM_MAX_CAPTURE_BUFFERS
#define CAMERA_STREAM_MAX_CAPTURE_BUFFERS 3
#endif

#ifndef CAMERA_STREAM_SPECIALIZE_SENSOR_MODE
#define CAMERA_STREAM_SPECIALIZE_SENSOR_MODE (-1)
#endif

#ifndef CAMERA_STREAM_SPECIALIZE_POLICY
#define CAMERA_STREAM_SPECIALIZE_POLICY (-1)
#endif

#if CAMERA_STREAM_MAX_CAPTURE_BUFFERS < 1 || CAMERA_STREAM_MAX_CAPTURE_BUFFERS > 3
#error "CAMERA_STREAM_MAX_CAPTURE_BUFFERS must be in [1, 3]"
#endif

#define CAMERA_STREAM_WAIT_FOREVER UINT32_MAX

/* The stream's worker task: it drains capture completions, hands out frames
 * and runs the watchdog. */
#define CAMERA_STREAM_TASK_STACK_WORDS (configMINIMAL_STACK_SIZE * 2u)
#define CAMERA_STREAM_TASK_PRIORITY (tskIDLE_PRIORITY + 2u)

/* window_top: the sensor's own QVGA window for QVGA, the centre for QQVGA. */
#define CAMERA_STREAM_WINDOW_DEFAULT UINT16_MAX

/* This is an ABI size, not an allocation. camera_stream_t remains caller-owned. */
#define CAMERA_STREAM_OBJECT_BYTES \
  (384u + (192u * CAMERA_STREAM_MAX_CAPTURE_BUFFERS))

typedef enum {
  CAMERA_STREAM_OK = 0,
  CAMERA_STREAM_ERR_INVALID_ARGUMENT = -1,
  CAMERA_STREAM_ERR_INVALID_STATE = -2,
  CAMERA_STREAM_ERR_NO_MEMORY = -3,
  CAMERA_STREAM_ERR_CAMERA = -4,
  CAMERA_STREAM_ERR_RTOS = -5,
  CAMERA_STREAM_ERR_TIMEOUT = -6,
  CAMERA_STREAM_ERR_STOPPED = -7,
  CAMERA_STREAM_ERR_BUSY = -8,
  CAMERA_STREAM_ERR_STALE_FRAME = -9,
  CAMERA_STREAM_ERR_CLOSED = -10,
} camera_stream_status_t;

typedef enum {
  CAMERA_STREAM_SENSOR_QVGA_30 = 0,
  CAMERA_STREAM_SENSOR_QVGA_60 = 1,
  CAMERA_STREAM_SENSOR_QQVGA_BIN2_30 = 2,
  CAMERA_STREAM_SENSOR_QQVGA_BIN2_60 = 3,
  CAMERA_STREAM_SENSOR_QQVGA_BIN2_120 = 4,
} camera_stream_sensor_mode_t;

typedef enum {
  /* One continuously overwritten buffer. Acquired pixels may tear/change. */
  CAMERA_STREAM_POLICY_PREVIEW_LIVE = 0,
  /*
   * A completed buffer remains immutable until camera_stream_release(). The
   * newest completed frame supersedes an unconsumed older one, so the camera
   * always runs at the sensor rate.
   */
  CAMERA_STREAM_POLICY_EXACT_FRAME = 1,
} camera_stream_policy_t;

typedef enum {
  CAMERA_STREAM_EXPOSURE_AUTO = 0,
  CAMERA_STREAM_EXPOSURE_MANUAL = 1,
} camera_stream_exposure_mode_t;

enum {
  CAMERA_STREAM_FLICKER_DISABLED = 0,
  CAMERA_STREAM_FLICKER_50_HZ = 1,
  CAMERA_STREAM_FLICKER_60_HZ = 2,
};
typedef uint8_t camera_stream_flicker_mode_t;

enum {
  CAMERA_STREAM_ANALOG_GAIN_1X = 0,
  CAMERA_STREAM_ANALOG_GAIN_2X = 1,
  CAMERA_STREAM_ANALOG_GAIN_4X = 2,
  CAMERA_STREAM_ANALOG_GAIN_8X = 3,
};
typedef uint8_t camera_stream_analog_gain_t;

/*
 * Compile-time defaults used by camera_stream_config_init(). Applications may
 * still override the copied fields before camera_stream_init(). Keeping the
 * defaults here also lets small facade-only builds select one camera behavior
 * entirely through GAP8 compiler flags, without any host control protocol.
 */
#ifndef CAMERA_STREAM_DEFAULT_ORIENTATION
#define CAMERA_STREAM_DEFAULT_ORIENTATION 3u
#endif
#ifndef CAMERA_STREAM_DEFAULT_EXPOSURE_MODE
#define CAMERA_STREAM_DEFAULT_EXPOSURE_MODE CAMERA_STREAM_EXPOSURE_AUTO
#endif
#ifndef CAMERA_STREAM_DEFAULT_AE_TARGET_MEAN
#define CAMERA_STREAM_DEFAULT_AE_TARGET_MEAN 0x3cu
#endif
#ifndef CAMERA_STREAM_DEFAULT_AE_MAX_EXPOSURE_US
#define CAMERA_STREAM_DEFAULT_AE_MAX_EXPOSURE_US 0u
#endif
#ifndef CAMERA_STREAM_DEFAULT_AE_MAX_ANALOG_GAIN
#define CAMERA_STREAM_DEFAULT_AE_MAX_ANALOG_GAIN CAMERA_STREAM_ANALOG_GAIN_8X
#endif
#ifndef CAMERA_STREAM_DEFAULT_AE_MAX_DIGITAL_GAIN
#define CAMERA_STREAM_DEFAULT_AE_MAX_DIGITAL_GAIN 0x0300u
#endif
#ifndef CAMERA_STREAM_DEFAULT_MANUAL_EXPOSURE_US
#define CAMERA_STREAM_DEFAULT_MANUAL_EXPOSURE_US 4000u
#endif
#ifndef CAMERA_STREAM_DEFAULT_MANUAL_ANALOG_GAIN
#define CAMERA_STREAM_DEFAULT_MANUAL_ANALOG_GAIN CAMERA_STREAM_ANALOG_GAIN_1X
#endif
#ifndef CAMERA_STREAM_DEFAULT_MANUAL_DIGITAL_GAIN
#define CAMERA_STREAM_DEFAULT_MANUAL_DIGITAL_GAIN 0x0100u
#endif
#ifndef CAMERA_STREAM_DEFAULT_FLICKER_MODE
#define CAMERA_STREAM_DEFAULT_FLICKER_MODE CAMERA_STREAM_FLICKER_DISABLED
#endif
#ifndef CAMERA_STREAM_DEFAULT_SENSOR_SETTLE_MS
#define CAMERA_STREAM_DEFAULT_SENSOR_SETTLE_MS 200u
#endif

typedef struct {
  camera_stream_sensor_mode_t sensor_mode;
  camera_stream_policy_t policy;

  /*
   * 0 selects the policy default: preview=1, exact=2. Exact also takes 3.
   * With two, a frame completed while the other buffer is leased has nowhere
   * to wait, so it is overwritten and release() is followed by a wait for the
   * frame in flight. The third buffer keeps the newest frame ready while a
   * lease is held longer than one sensor period.
   */
  uint8_t capture_buffer_count;

  /* HM01B0 IMG_ORIENTATION: bit 0 horizontal, bit 1 vertical flip. */
  uint8_t orientation;

  /*
   * The rows the frame shows: its first row within the sensor's 324, counted
   * from the top of the image as delivered, 0..84. The unit is a full-resolution
   * row in every mode, so one value frames the same view in QVGA and QQVGA.
   * QVGA_60 has time only for the sensor's own QVGA window; other QVGA rows
   * read the whole sensor, which QVGA_30 can.
   */
  uint16_t window_top;

  /*
   * Exposure: every field below may also change while streaming, through
   * camera_stream_set_exposure(). Times are microseconds; digital gains are
   * in 1/256 steps (256 = 1x, at most 1023).
   */
  camera_stream_exposure_mode_t exposure_mode;
  /* Automatic: the target mean level, 0..255, and the limits it may use. */
  uint8_t ae_target_mean;
  uint32_t ae_max_exposure_us; /* 0: the whole frame period */
  camera_stream_analog_gain_t ae_max_analog_gain;
  uint16_t ae_max_digital_gain;
  /* Automatic only: step the exposure with the mains period. */
  camera_stream_flicker_mode_t flicker_mode;
  /* Manual. */
  uint32_t manual_exposure_us;
  camera_stream_analog_gain_t manual_analog_gain;
  uint16_t manual_digital_gain;

  /*
   * On the first physical START, capture runs but frames are discarded for
   * this interval. This gives CMU-retimed orientation/exposure settings frame
   * boundaries in which to become active before a frame can be acquired.
   * Soft stop/resume does not repeat the interval.
   */
  uint32_t sensor_settle_ms;

  /* Zero disables capture watchdog recovery. */
  uint32_t watchdog_timeout_ms;
} camera_stream_config_t;

typedef struct {
  uint16_t width;
  uint16_t height;
  uint16_t stride;
  uint8_t capture_buffer_count;
  size_t frame_size_bytes;
  size_t capture_l2_bytes;
} camera_stream_memory_plan_t;

typedef struct {
  uint8_t *data;
  size_t size_bytes;
  uint16_t width;
  uint16_t height;
  uint16_t stride;
  uint32_t capture_started_ms;
  uint32_t capture_completed_ms;
  uint32_t capture_duration_ms;
  uint32_t camera_frame_id;

  /* Opaque lease identity. Applications must preserve it until release(). */
  uintptr_t token;
} camera_stream_frame_t;

/* The exposure the sensor is using now: the automatic loop's latest choice, or
 * the manual setting. */
typedef struct {
  uint32_t exposure_us;
  camera_stream_analog_gain_t analog_gain;
  uint16_t digital_gain; /* 256 = 1x */
} camera_stream_exposure_t;

typedef struct {
  uint32_t capture_completions;
  uint32_t ready_frames;
  uint32_t dropped_capture_completions;
  uint32_t dropped_ready_frames;
  uint32_t dropped_while_leased;
  uint32_t sensor_settle_discards;
  uint32_t paused_discards;
  uint32_t watchdog_restarts;
} camera_stream_stats_t;

/*
 * The extension a camera stream sends after the 12-byte image header
 * (imgstream_frame_t.ext): the frame's capture and encode timings plus the
 * cumulative drop counters from camera_stream_get_stats(). The host decodes it
 * as cf_lab.camera.ext.CameraStreamExt; keep the two in step.
 */
typedef struct {
  uint32_t frame_id;      /* frames sent */
  uint32_t timestamp_ms;  /* capture_started_ms */
  uint16_t encode_ms;     /* capture + preprocess + JPEG */
  uint16_t capture_ms;
  uint16_t preprocess_ms; /* resize */
  uint16_t jpeg_ms;
  uint32_t camera_frame_id;
  uint32_t camera_done_ms; /* capture_completed_ms */
  uint32_t dropped_capture_completions;
  uint32_t dropped_ready_frames;
  uint32_t dropped_while_streaming; /* dropped_while_leased */
} __attribute__((packed)) camera_stream_frame_ext_t;

_Static_assert(sizeof(camera_stream_frame_ext_t) == 36,
               "the host decodes a 36-byte camera_stream_frame_ext_t");

typedef struct {
  camera_stream_config_t config;
  uint16_t width;
  uint16_t height;
  uint16_t stride;
  uint16_t named_fps;
  uint8_t capture_buffer_count;
  uint8_t is_open;
  uint8_t is_started;
  /* A fail-safe capture fault requires close() followed by open(). */
  uint8_t is_faulted;
  size_t frame_size_bytes;

  /*
   * Stable L2 allocation addresses, exposed for accelerator integration.
   * Ownership still follows acquire()/release(); writing an armed buffer or
   * retaining an exact buffer after release is undefined.
   */
  uint8_t *capture_buffers[CAMERA_STREAM_MAX_CAPTURE_BUFFERS];
} camera_stream_info_t;

/*
 * Opaque, caller-owned storage. It may be static/global/embedded in another
 * object, but must not be copied after camera_stream_init().
 */
typedef union {
  uint64_t _alignment;
  uint8_t _storage[CAMERA_STREAM_OBJECT_BYTES];
} camera_stream_t;

int camera_stream_config_init(camera_stream_config_t *config);
int camera_stream_init(camera_stream_t *stream,
                       const camera_stream_config_t *config);
int camera_stream_memory_plan(const camera_stream_config_t *config,
                              camera_stream_memory_plan_t *plan);

/* open() opens/configures Himax first, then allocates L2 and RTOS resources. */
int camera_stream_open(camera_stream_t *stream);
int camera_stream_start(camera_stream_t *stream);

/* At most one task may wait in acquire(), and at most one lease may exist. */
int camera_stream_acquire(camera_stream_t *stream,
                          camera_stream_frame_t *frame,
                          uint32_t timeout_ms);
int camera_stream_release(camera_stream_t *stream,
                          camera_stream_frame_t *frame);

/* stop() is soft: DMA remains active and captures are recycled/discarded. */
int camera_stream_stop(camera_stream_t *stream);
/* close() is hard and returns BUSY while any frame lease is outstanding. */
int camera_stream_close(camera_stream_t *stream);

int camera_stream_get_info(const camera_stream_t *stream,
                           camera_stream_info_t *info);
int camera_stream_get_stats(const camera_stream_t *stream,
                            camera_stream_stats_t *stats);

/*
 * Every setting the stream runs with, as "k=v ..." for a host to record:
 * sensor and capture, buffers and orientation, the window as resolved, and the
 * exposure settings. Always writes a NUL-terminated string into out (truncated
 * to size); returns get_info()'s status.
 */
int camera_stream_describe(const camera_stream_t *stream, char *out, size_t size);

/*
 * Applies the exposure fields of config (exposure_mode through
 * manual_digital_gain) from the next frame on, open or streaming; its other
 * fields are ignored. An invalid value changes nothing.
 */
int camera_stream_set_exposure(camera_stream_t *stream,
                               const camera_stream_config_t *config);
/* Reads the sensor's current exposure over I2C. */
int camera_stream_get_exposure(const camera_stream_t *stream,
                               camera_stream_exposure_t *exposure);

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_CAMERA_STREAM_H */
