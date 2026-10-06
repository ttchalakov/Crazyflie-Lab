#include "camera_stream.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "pmsis.h"
#include "bsp/camera/himax.h"

#if configUSE_MUTEXES != 1
#error "camera_stream requires configUSE_MUTEXES=1"
#endif

/* Himax timing for the two sensor families. QQVGA keeps the sensor's default
 * OSC_CLK_DIV. */
#define SENSOR_QVGA_LINE_LEN_PCK 0x0178
#define SENSOR_QVGA_OSC_CLK_DIV 0x0B
#define SENSOR_QVGA_ANA_REGISTER_17 0x00
#define SENSOR_QVGA_PIXEL_SHIFT 0x00
#define SENSOR_QQVGA_LINE_LEN_PCK 0x00BC
/* Digital gain enabled in every mode: without it the automatic loop has only
 * the 2x analog steps once exposure is at its limit, and swings between them. */
#define SENSOR_DGAIN_CONTROL 0x7F

/* One row's period with the timing above (frame_len_lines rows per frame);
 * converts exposure times to rows. */
#define SENSOR_QVGA_ROW_NS 64700u
#define SENSOR_QQVGA_ROW_NS 35610u

/*
 * Window geometry. The sensor reads 324 rows; its QVGA window is 244 of them,
 * at a position that depends on the vertical flip (in rows from the top of the
 * delivered image). Any other QVGA window reads all 324 rows, which needs a
 * frame of at least CS_FULL_READOUT_LINES.
 */
#define CS_SENSOR_ROWS 324u
#define CS_QVGA_WINDOW_ROWS 244u
#define CS_QVGA_WINDOW_TOP_FLIPPED 80u
#define CS_QVGA_WINDOW_TOP_UPRIGHT 0u
#define CS_FULL_READOUT_LINES 340u

/*
 * Flicker steps are exposure-row counts derived from the row rates of the two
 * sensor families. The GAP SDK's 0x0032/0x003c defaults describe its stock
 * timing and are not correct after our profile overrides.
 */
#define SENSOR_QVGA_FLICKER_50HZ_STEP 0x009A
#define SENSOR_QVGA_FLICKER_60HZ_STEP 0x0081
#define SENSOR_QQVGA_FLICKER_50HZ_STEP 0x0119
#define SENSOR_QQVGA_FLICKER_60HZ_STEP 0x00EA

/* C constant-expression checks also accept enum identifiers on a -D hook. */
typedef char cs_sensor_specialization_check[
    (CAMERA_STREAM_SPECIALIZE_SENSOR_MODE >= -1 &&
     CAMERA_STREAM_SPECIALIZE_SENSOR_MODE <= 4) ? 1 : -1];
typedef char cs_policy_specialization_check[
    (CAMERA_STREAM_SPECIALIZE_POLICY >= -1 &&
     CAMERA_STREAM_SPECIALIZE_POLICY <= 1) ? 1 : -1];
typedef char cs_default_orientation_check[
    ((long long)CAMERA_STREAM_DEFAULT_ORIENTATION >= 0 &&
     (long long)CAMERA_STREAM_DEFAULT_ORIENTATION <= 3) ? 1 : -1];
typedef char cs_default_exposure_mode_check[
    (CAMERA_STREAM_DEFAULT_EXPOSURE_MODE == CAMERA_STREAM_EXPOSURE_AUTO ||
     CAMERA_STREAM_DEFAULT_EXPOSURE_MODE == CAMERA_STREAM_EXPOSURE_MANUAL)
        ? 1
        : -1];
typedef char cs_default_ae_target_check[
    ((long long)CAMERA_STREAM_DEFAULT_AE_TARGET_MEAN >= 0 &&
     (long long)CAMERA_STREAM_DEFAULT_AE_TARGET_MEAN <= 0xff) ? 1 : -1];
typedef char cs_default_manual_exposure_check[
    ((long long)CAMERA_STREAM_DEFAULT_MANUAL_EXPOSURE_US > 0 &&
     (long long)CAMERA_STREAM_DEFAULT_MANUAL_EXPOSURE_US <= UINT32_MAX) ? 1 : -1];
typedef char cs_default_analog_gain_check[
    ((long long)CAMERA_STREAM_DEFAULT_MANUAL_ANALOG_GAIN >= 0 &&
     (long long)CAMERA_STREAM_DEFAULT_MANUAL_ANALOG_GAIN <= 3 &&
     (long long)CAMERA_STREAM_DEFAULT_AE_MAX_ANALOG_GAIN >= 0 &&
     (long long)CAMERA_STREAM_DEFAULT_AE_MAX_ANALOG_GAIN <= 3) ? 1 : -1];
typedef char cs_default_ae_max_digital_gain_check[
    ((long long)CAMERA_STREAM_DEFAULT_AE_MAX_DIGITAL_GAIN >= 0x100 &&
     (long long)CAMERA_STREAM_DEFAULT_AE_MAX_DIGITAL_GAIN <= 0x3ff) ? 1 : -1];
typedef char cs_default_digital_gain_check[
    ((long long)CAMERA_STREAM_DEFAULT_MANUAL_DIGITAL_GAIN > 0 &&
     (long long)CAMERA_STREAM_DEFAULT_MANUAL_DIGITAL_GAIN <= 0x3ff) ? 1 : -1];
typedef char cs_default_flicker_check[
    (CAMERA_STREAM_DEFAULT_FLICKER_MODE == CAMERA_STREAM_FLICKER_DISABLED ||
     CAMERA_STREAM_DEFAULT_FLICKER_MODE == CAMERA_STREAM_FLICKER_50_HZ ||
     CAMERA_STREAM_DEFAULT_FLICKER_MODE == CAMERA_STREAM_FLICKER_60_HZ)
        ? 1
        : -1];
typedef char cs_default_sensor_settle_check[
    ((long long)CAMERA_STREAM_DEFAULT_SENSOR_SETTLE_MS >= 0 &&
     (long long)CAMERA_STREAM_DEFAULT_SENSOR_SETTLE_MS <= INT32_MAX) ? 1 : -1];

/* HM01B0 registers (the SDK driver keeps these private). */
#define CS_HIMAX_MODE_SELECT       0x0100
#define CS_HIMAX_IMG_ORIENTATION   0x0101
#define CS_HIMAX_GRP_PARAM_HOLD    0x0104
#define CS_HIMAX_INTEGRATION_H     0x0202
#define CS_HIMAX_INTEGRATION_L     0x0203
#define CS_HIMAX_ANALOG_GAIN       0x0205
#define CS_HIMAX_DIGITAL_GAIN_H    0x020E
#define CS_HIMAX_DIGITAL_GAIN_L    0x020F
#define CS_HIMAX_FRAME_LEN_H       0x0340
#define CS_HIMAX_LINE_LEN_H        0x0342
#define CS_HIMAX_DGAIN_CONTROL     0x0350
#define CS_HIMAX_READOUT_X         0x0383
#define CS_HIMAX_READOUT_Y         0x0387
#define CS_HIMAX_BINNING_MODE      0x0390
#define CS_HIMAX_PIXEL_SHIFT       0x1012
#define CS_HIMAX_AE_CTRL           0x2100
#define CS_HIMAX_AE_TARGET_MEAN    0x2101
#define CS_HIMAX_MAX_INTG_H        0x2105
#define CS_HIMAX_MAX_AGAIN_FULL    0x2108
#define CS_HIMAX_MAX_AGAIN_BIN2    0x2109
#define CS_HIMAX_MAX_DGAIN         0x210B
#define CS_HIMAX_FS_CTRL           0x210E
#define CS_HIMAX_FS_60HZ_H         0x210F
#define CS_HIMAX_FS_50HZ_H         0x2111
#define CS_HIMAX_QVGA_WIN_EN       0x3010
#define CS_HIMAX_OSC_CLK_DIV       0x3060
#define CS_HIMAX_ANA_REGISTER_17   0x3067

#define CS_EVENT_CAPTURE_DONE (1u << 0)
#define CS_EVENT_FRAME_READY  (1u << 1)
#define CS_EVENT_CONTROL      (1u << 2)
#define CS_EVENT_SHUTDOWN     (1u << 3)
#define CS_EVENT_READER_WAKE  (1u << 4)

#define CS_MAGIC 0x43535452u /* CSTR */
#define CS_TASK_SLOTS_PER_BUFFER 2u

/* Never reset on close/reopen: copied stale leases cannot become valid again. */
static volatile uintptr_t cs_global_lease_token;

typedef enum {
  CS_BUFFER_FREE = 0,
  CS_BUFFER_FILLING,
  CS_BUFFER_READY,
  CS_BUFFER_LEASED,
} cs_buffer_state_t;

typedef struct {
  uint16_t width;
  uint16_t height;
  uint16_t named_fps;
  uint16_t frame_len_lines;
  uint16_t max_integration;
  uint16_t line_len_pck;
  uint16_t flicker_50hz_step;
  uint16_t flicker_60hz_step;
  uint32_t row_ns;
  uint8_t qvga;
} cs_sensor_profile_t;

/* Where the configured window comes from: the CPI crop's first row, and
 * whether a QVGA window needs the whole sensor read out. */
typedef struct {
  uint16_t crop_top;
  uint8_t full_readout;
} cs_window_t;

struct cs_impl;

typedef struct {
  struct cs_impl *stream;
  uint8_t index;
  uint8_t slot;
  uint32_t arm_generation;
} cs_callback_context_t;

typedef struct cs_impl {
  uint32_t magic;
  uintptr_t self;
  camera_stream_config_t config;
  cs_sensor_profile_t profile;
  cs_window_t window;
  size_t frame_size;

  struct pi_device camera;
  pi_task_t capture_tasks[CAMERA_STREAM_MAX_CAPTURE_BUFFERS]
                         [CS_TASK_SLOTS_PER_BUFFER];
  cs_callback_context_t callback_contexts[CAMERA_STREAM_MAX_CAPTURE_BUFFERS]
                                         [CS_TASK_SLOTS_PER_BUFFER];
  uint8_t *buffers[CAMERA_STREAM_MAX_CAPTURE_BUFFERS];
  uint32_t capture_started_ms[CAMERA_STREAM_MAX_CAPTURE_BUFFERS];
  uint32_t capture_completed_ms[CAMERA_STREAM_MAX_CAPTURE_BUFFERS];
  uint32_t capture_frame_id[CAMERA_STREAM_MAX_CAPTURE_BUFFERS];
  uint32_t arm_generation[CAMERA_STREAM_MAX_CAPTURE_BUFFERS];
  uint32_t completion_stop_epoch[CAMERA_STREAM_MAX_CAPTURE_BUFFERS];
  uint8_t completion_publishable[CAMERA_STREAM_MAX_CAPTURE_BUFFERS];
  uint8_t completion_settling[CAMERA_STREAM_MAX_CAPTURE_BUFFERS];
  uint8_t task_slot_active[CAMERA_STREAM_MAX_CAPTURE_BUFFERS]
                          [CS_TASK_SLOTS_PER_BUFFER];
  uint8_t buffer_state[CAMERA_STREAM_MAX_CAPTURE_BUFFERS];

  EventGroupHandle_t events;
  SemaphoreHandle_t lifecycle_lock;

  camera_stream_stats_t stats;
  volatile uint32_t completed_mask;
  volatile uint32_t armed_mask;
  volatile uint32_t next_camera_frame_id;
  volatile uint32_t next_arm_generation;
  volatile uint32_t latest_preview_frame_id;
  volatile uint32_t last_delivered_preview_frame_id;
  volatile uint32_t preview_capture_started_ms;
  volatile uint32_t preview_capture_completed_ms;
  volatile uint32_t stop_epoch;
  volatile uint32_t settle_until_ms;

  volatile int8_t ready_index;
  volatile int8_t lease_index;
  volatile uint8_t camera_open;
  volatile uint8_t open;
  volatile uint8_t started;
  volatile uint8_t hardware_streaming;
  volatile uint8_t settling;
  volatile uint8_t closing;
  volatile uint8_t worker_exited;
  volatile uint8_t reader_active;
  volatile uint8_t lease_active;
  volatile uint8_t faulted;
  volatile uint16_t lifecycle_api_users;
  volatile uint16_t active_callbacks;
  uintptr_t lease_token;
} cs_impl_t;

typedef char cs_object_size_check[
    (sizeof(cs_impl_t) <= CAMERA_STREAM_OBJECT_BYTES) ? 1 : -1];

static cs_impl_t *cs_impl(camera_stream_t *stream) {
  return (cs_impl_t *)(void *)stream->_storage;
}

static const cs_impl_t *cs_const_impl(const camera_stream_t *stream) {
  return (const cs_impl_t *)(const void *)stream->_storage;
}

static int cs_is_initialized(const cs_impl_t *stream) {
  return stream != NULL && stream->magic == CS_MAGIC &&
         stream->self == (uintptr_t)stream;
}

static camera_stream_sensor_mode_t cs_config_sensor_mode(
    const camera_stream_config_t *config) {
#if CAMERA_STREAM_SPECIALIZE_SENSOR_MODE >= 0
  (void)config;
  return (camera_stream_sensor_mode_t)CAMERA_STREAM_SPECIALIZE_SENSOR_MODE;
#else
  return config->sensor_mode;
#endif
}

static camera_stream_policy_t cs_config_policy(
    const camera_stream_config_t *config) {
#if CAMERA_STREAM_SPECIALIZE_POLICY >= 0
  (void)config;
  return (camera_stream_policy_t)CAMERA_STREAM_SPECIALIZE_POLICY;
#else
  return config->policy;
#endif
}

static camera_stream_policy_t cs_policy(const cs_impl_t *stream) {
  return cs_config_policy(&stream->config);
}

static int cs_sensor_is_qvga(const cs_impl_t *stream) {
#if CAMERA_STREAM_SPECIALIZE_SENSOR_MODE >= 0
  (void)stream;
  return (camera_stream_sensor_mode_t)CAMERA_STREAM_SPECIALIZE_SENSOR_MODE ==
             CAMERA_STREAM_SENSOR_QVGA_30 ||
         (camera_stream_sensor_mode_t)CAMERA_STREAM_SPECIALIZE_SENSOR_MODE ==
             CAMERA_STREAM_SENSOR_QVGA_60;
#else
  return stream->profile.qvga;
#endif
}

static uint32_t cs_now_ms(void) {
  return (uint32_t)xTaskGetTickCount() * (uint32_t)portTICK_PERIOD_MS;
}

static int cs_time_reached(uint32_t now_ms, uint32_t deadline_ms) {
  return (int32_t)(now_ms - deadline_ms) >= 0;
}

/* Caller holds the RTOS critical section. */
static void cs_begin_sensor_settle(cs_impl_t *stream) {
  stream->settling = stream->config.sensor_settle_ms != 0u;
  stream->settle_until_ms =
      cs_now_ms() + stream->config.sensor_settle_ms;
}

static TickType_t cs_ms_to_ticks(uint32_t timeout_ms) {
  TickType_t ticks = pdMS_TO_TICKS(timeout_ms);
  return (timeout_ms != 0 && ticks == 0) ? 1 : ticks;
}

static int cs_storage_looks_open(const camera_stream_t *storage) {
  const cs_impl_t *address = cs_const_impl(storage);
  const uint32_t expected_magic = CS_MAGIC;
  const uintptr_t expected_self = (uintptr_t)address;
  const uint8_t expected_open = 1;
  const uint8_t *bytes = storage->_storage;

  /* Byte-wise object-representation checks are defined even when caller-owned
   * automatic storage has never been initialized. Never form a typed read of
   * the tentative private object before camera_stream_init() establishes it. */
  return memcmp(bytes + offsetof(cs_impl_t, magic), &expected_magic,
                sizeof(expected_magic)) == 0 &&
         memcmp(bytes + offsetof(cs_impl_t, self), &expected_self,
                sizeof(expected_self)) == 0 &&
         memcmp(bytes + offsetof(cs_impl_t, open), &expected_open,
                sizeof(expected_open)) == 0;
}

/* Keep the dynamically allocated mutex alive while an API caller queues on it. */
static int cs_lifecycle_enter(cs_impl_t *stream, SemaphoreHandle_t *lock) {
  taskENTER_CRITICAL();
  if (!stream->open || stream->closing || stream->lifecycle_lock == NULL) {
    taskEXIT_CRITICAL();
    return CAMERA_STREAM_ERR_CLOSED;
  }
  stream->lifecycle_api_users++;
  *lock = stream->lifecycle_lock;
  taskEXIT_CRITICAL();
  xSemaphoreTake(*lock, portMAX_DELAY);
  return CAMERA_STREAM_OK;
}

static void cs_lifecycle_leave(cs_impl_t *stream, SemaphoreHandle_t lock) {
  xSemaphoreGive(lock);
  taskENTER_CRITICAL();
  stream->lifecycle_api_users--;
  taskEXIT_CRITICAL();
}

static int cs_close_api_quiescent(cs_impl_t *stream) {
  int quiescent;
  taskENTER_CRITICAL();
  quiescent = stream->worker_exited && !stream->reader_active &&
              stream->lifecycle_api_users == 0;
  taskEXIT_CRITICAL();
  return quiescent;
}

static int cs_callbacks_quiescent(cs_impl_t *stream) {
  int quiescent;
  taskENTER_CRITICAL();
  quiescent = stream->active_callbacks == 0;
  taskEXIT_CRITICAL();
  return quiescent;
}

static void cs_set_reg(struct pi_device *device, uint16_t address,
                       uint8_t value) {
  pi_camera_reg_set(device, address, &value);
}

static uint8_t cs_get_reg(struct pi_device *device, uint16_t address) {
  uint8_t value = 0;
  pi_camera_reg_get(device, address, &value);
  return value;
}

static void cs_set_reg16(struct pi_device *device, uint16_t high_address,
                         uint16_t value) {
  cs_set_reg(device, high_address, (uint8_t)(value >> 8));
  cs_set_reg(device, (uint16_t)(high_address + 1u), (uint8_t)value);
}

static cs_sensor_profile_t cs_sensor_profile(
    camera_stream_sensor_mode_t mode) {
  cs_sensor_profile_t profile;
  memset(&profile, 0, sizeof(profile));

  switch (mode) {
    case CAMERA_STREAM_SENSOR_QVGA_30:
      profile.width = 320;
      profile.height = 240;
      profile.named_fps = 30;
      profile.frame_len_lines = 0x0204;
      profile.max_integration = 0x0200;
      profile.line_len_pck = SENSOR_QVGA_LINE_LEN_PCK;
      profile.flicker_50hz_step = SENSOR_QVGA_FLICKER_50HZ_STEP;
      profile.flicker_60hz_step = SENSOR_QVGA_FLICKER_60HZ_STEP;
      profile.row_ns = SENSOR_QVGA_ROW_NS;
      profile.qvga = 1;
      break;
    case CAMERA_STREAM_SENSOR_QVGA_60:
      profile.width = 320;
      profile.height = 240;
      profile.named_fps = 60;
      profile.frame_len_lines = 0x0104;
      profile.max_integration = 0x0102;
      profile.line_len_pck = SENSOR_QVGA_LINE_LEN_PCK;
      profile.flicker_50hz_step = SENSOR_QVGA_FLICKER_50HZ_STEP;
      profile.flicker_60hz_step = SENSOR_QVGA_FLICKER_60HZ_STEP;
      profile.row_ns = SENSOR_QVGA_ROW_NS;
      profile.qvga = 1;
      break;
    case CAMERA_STREAM_SENSOR_QQVGA_BIN2_30:
      profile.width = 160;
      profile.height = 120;
      profile.named_fps = 30;
      profile.frame_len_lines = 0x03A8;
      profile.max_integration = 0x03A0;
      profile.line_len_pck = SENSOR_QQVGA_LINE_LEN_PCK;
      profile.flicker_50hz_step = SENSOR_QQVGA_FLICKER_50HZ_STEP;
      profile.flicker_60hz_step = SENSOR_QQVGA_FLICKER_60HZ_STEP;
      profile.row_ns = SENSOR_QQVGA_ROW_NS;
      break;
    case CAMERA_STREAM_SENSOR_QQVGA_BIN2_60:
      profile.width = 160;
      profile.height = 120;
      profile.named_fps = 60;
      profile.frame_len_lines = 0x01D4;
      profile.max_integration = 0x01D0;
      profile.line_len_pck = SENSOR_QQVGA_LINE_LEN_PCK;
      profile.flicker_50hz_step = SENSOR_QQVGA_FLICKER_50HZ_STEP;
      profile.flicker_60hz_step = SENSOR_QQVGA_FLICKER_60HZ_STEP;
      profile.row_ns = SENSOR_QQVGA_ROW_NS;
      break;
    case CAMERA_STREAM_SENSOR_QQVGA_BIN2_120:
      profile.width = 160;
      profile.height = 120;
      profile.named_fps = 120;
      profile.frame_len_lines = 0x00EA;
      profile.max_integration = 0x0078;
      profile.line_len_pck = SENSOR_QQVGA_LINE_LEN_PCK;
      profile.flicker_50hz_step = SENSOR_QQVGA_FLICKER_50HZ_STEP;
      profile.flicker_60hz_step = SENSOR_QQVGA_FLICKER_60HZ_STEP;
      profile.row_ns = SENSOR_QQVGA_ROW_NS;
      break;
    default:
      break;
  }

  return profile;
}

static uint32_t cs_exposure_rows(const cs_sensor_profile_t *profile,
                                 uint32_t exposure_us) {
  return (uint32_t)(((uint64_t)exposure_us * 1000u + profile->row_ns / 2u) /
                    profile->row_ns);
}

/* Validates the exposure fields and turns their times into sensor rows. */
static int cs_resolve_exposure(const camera_stream_config_t *config,
                               const cs_sensor_profile_t *profile,
                               uint16_t *manual_rows, uint16_t *ae_max_rows) {
  const uint32_t ae_max =
      config->ae_max_exposure_us == 0u
          ? profile->max_integration
          : cs_exposure_rows(profile, config->ae_max_exposure_us);
  const uint32_t manual = cs_exposure_rows(profile, config->manual_exposure_us);

  if (config->exposure_mode < CAMERA_STREAM_EXPOSURE_AUTO ||
      config->exposure_mode > CAMERA_STREAM_EXPOSURE_MANUAL ||
      config->flicker_mode > CAMERA_STREAM_FLICKER_60_HZ ||
      config->ae_max_analog_gain > CAMERA_STREAM_ANALOG_GAIN_8X ||
      config->manual_analog_gain > CAMERA_STREAM_ANALOG_GAIN_8X ||
      config->ae_max_digital_gain < 0x0100u ||
      config->ae_max_digital_gain > 0x03ffu ||
      config->manual_digital_gain > 0x03ffu || ae_max < 2u ||
      ae_max > profile->max_integration) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  if (config->exposure_mode == CAMERA_STREAM_EXPOSURE_MANUAL &&
      (manual < 2u || manual > profile->max_integration ||
       config->manual_digital_gain == 0)) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  if (config->flicker_mode != CAMERA_STREAM_FLICKER_DISABLED) {
    const uint16_t step =
        config->flicker_mode == CAMERA_STREAM_FLICKER_50_HZ
            ? profile->flicker_50hz_step
            : profile->flicker_60hz_step;
    if (config->exposure_mode != CAMERA_STREAM_EXPOSURE_AUTO || step > ae_max) {
      return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
    }
  }
  *manual_rows = (uint16_t)(manual < 2u ? 2u : manual > 0xffffu ? 0xffffu
                                                                 : manual);
  *ae_max_rows = (uint16_t)ae_max;
  return CAMERA_STREAM_OK;
}

/* Resolves window_top (the default, and binned rows in pairs) and where the
 * CPI crops it. */
static int cs_resolve_window(const camera_stream_config_t *config,
                             const cs_sensor_profile_t *profile,
                             uint16_t *top, cs_window_t *window) {
  const uint16_t rows = profile->qvga ? profile->height : 2u * profile->height;
  const uint16_t qvga_top = (config->orientation & 2u) != 0u
                                ? CS_QVGA_WINDOW_TOP_FLIPPED
                                : CS_QVGA_WINDOW_TOP_UPRIGHT;
  uint16_t row = config->window_top;

  if (row == CAMERA_STREAM_WINDOW_DEFAULT) {
    row = profile->qvga ? qvga_top : (uint16_t)((CS_SENSOR_ROWS - rows) / 2u);
  }
  if (row > CS_SENSOR_ROWS - rows) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  if (!profile->qvga) {
    row &= (uint16_t)~1u;
    window->crop_top = row / 2u;
    window->full_readout = 0;
  } else if (row >= qvga_top && row + rows <= qvga_top + CS_QVGA_WINDOW_ROWS) {
    window->crop_top = row - qvga_top;
    window->full_readout = 0;
  } else if (profile->frame_len_lines >= CS_FULL_READOUT_LINES) {
    window->crop_top = row;
    window->full_readout = 1;
  } else {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  *top = row;
  return CAMERA_STREAM_OK;
}

static int cs_resolve_config(const camera_stream_config_t *config,
                             camera_stream_config_t *resolved,
                             cs_sensor_profile_t *profile,
                             cs_window_t *window) {
  uint8_t buffer_count;
  uint16_t manual_rows;
  uint16_t ae_max_rows;
  uint16_t window_top;

  if (config == NULL || resolved == NULL || profile == NULL ||
      config->sensor_mode < CAMERA_STREAM_SENSOR_QVGA_30 ||
      config->sensor_mode > CAMERA_STREAM_SENSOR_QQVGA_BIN2_120 ||
      config->policy < CAMERA_STREAM_POLICY_PREVIEW_LIVE ||
      config->policy > CAMERA_STREAM_POLICY_EXACT_FRAME ||
      config->orientation > 3u ||
      config->sensor_settle_ms > (uint32_t)INT32_MAX) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }

#if CAMERA_STREAM_SPECIALIZE_SENSOR_MODE >= 0
  if ((int)config->sensor_mode != CAMERA_STREAM_SPECIALIZE_SENSOR_MODE) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
#endif
#if CAMERA_STREAM_SPECIALIZE_POLICY >= 0
  if ((int)config->policy != CAMERA_STREAM_SPECIALIZE_POLICY) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
#endif

  *profile = cs_sensor_profile(cs_config_sensor_mode(config));
  if (profile->width == 0 ||
      cs_resolve_exposure(config, profile, &manual_rows, &ae_max_rows) !=
          CAMERA_STREAM_OK ||
      cs_resolve_window(config, profile, &window_top, window) !=
          CAMERA_STREAM_OK) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }

  buffer_count = config->capture_buffer_count;
  if (buffer_count == 0) {
    buffer_count =
        cs_config_policy(config) == CAMERA_STREAM_POLICY_PREVIEW_LIVE ? 1 : 2;
  }

  if (buffer_count > CAMERA_STREAM_MAX_CAPTURE_BUFFERS ||
      (cs_config_policy(config) == CAMERA_STREAM_POLICY_PREVIEW_LIVE &&
       buffer_count != 1) ||
      (cs_config_policy(config) == CAMERA_STREAM_POLICY_EXACT_FRAME &&
       buffer_count != 2 && buffer_count != 3)) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }

  *resolved = *config;
  resolved->capture_buffer_count = buffer_count;
  resolved->window_top = window_top;
  return CAMERA_STREAM_OK;
}

/* The exposure registers; the caller holds the parameter group. */
static void cs_write_exposure(struct pi_device *device,
                              const camera_stream_config_t *config,
                              const cs_sensor_profile_t *profile,
                              uint16_t manual_rows, uint16_t ae_max_rows) {
  cs_set_reg16(device, CS_HIMAX_MAX_INTG_H, ae_max_rows);
  cs_set_reg(device, CS_HIMAX_MAX_AGAIN_FULL, config->ae_max_analog_gain);
  cs_set_reg(device, CS_HIMAX_MAX_AGAIN_BIN2, config->ae_max_analog_gain);
  /* The limit holds the top eight of the gain's ten bits. */
  cs_set_reg(device, CS_HIMAX_MAX_DGAIN,
             (uint8_t)(config->ae_max_digital_gain >> 2));
  if (config->exposure_mode == CAMERA_STREAM_EXPOSURE_AUTO) {
    cs_set_reg(device, CS_HIMAX_AE_CTRL, 0x01);
    cs_set_reg(device, CS_HIMAX_AE_TARGET_MEAN, config->ae_target_mean);
    cs_set_reg16(device, CS_HIMAX_FS_60HZ_H, profile->flicker_60hz_step);
    cs_set_reg16(device, CS_HIMAX_FS_50HZ_H, profile->flicker_50hz_step);
    cs_set_reg(device, CS_HIMAX_FS_CTRL,
               config->flicker_mode == CAMERA_STREAM_FLICKER_50_HZ   ? 0x01
               : config->flicker_mode == CAMERA_STREAM_FLICKER_60_HZ ? 0x03
                                                                     : 0x00);
    /* Deliberately skip PI_CAMERA_CMD_AEG_INIT: it destroys applied timing. */
  } else {
    cs_set_reg(device, CS_HIMAX_FS_CTRL, 0x00);
    cs_set_reg(device, CS_HIMAX_AE_CTRL, 0x00);
    cs_set_reg16(device, CS_HIMAX_INTEGRATION_H, manual_rows);
    cs_set_reg(device, CS_HIMAX_ANALOG_GAIN,
               (uint8_t)(config->manual_analog_gain << 4));
    cs_set_reg16(device, CS_HIMAX_DIGITAL_GAIN_H, config->manual_digital_gain);
  }
}

static int cs_apply_sensor_profile(cs_impl_t *stream) {
  struct pi_device *device = &stream->camera;
  const cs_sensor_profile_t *profile = &stream->profile;
  uint16_t manual_rows;
  uint16_t ae_max_rows;

  if (cs_resolve_exposure(&stream->config, profile, &manual_rows,
                          &ae_max_rows) != CAMERA_STREAM_OK) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }

  /* Standby + group hold before the first START is required for frame timing. */
  cs_set_reg(device, CS_HIMAX_MODE_SELECT, 0x00);
  pi_time_wait_us(1000u);
  cs_set_reg(device, CS_HIMAX_GRP_PARAM_HOLD, 0x01);
  if (cs_sensor_is_qvga(stream)) {
    cs_set_reg(device, CS_HIMAX_ANA_REGISTER_17,
               SENSOR_QVGA_ANA_REGISTER_17);
    cs_set_reg(device, CS_HIMAX_PIXEL_SHIFT, SENSOR_QVGA_PIXEL_SHIFT);
    cs_set_reg(device, CS_HIMAX_DGAIN_CONTROL, SENSOR_DGAIN_CONTROL);
    cs_set_reg(device, CS_HIMAX_QVGA_WIN_EN,
               stream->window.full_readout ? 0x00 : 0x01);
    cs_set_reg(device, CS_HIMAX_BINNING_MODE, 0x00);
    cs_set_reg(device, CS_HIMAX_READOUT_X, 0x01);
    cs_set_reg(device, CS_HIMAX_READOUT_Y, 0x01);
  } else {
    cs_set_reg(device, CS_HIMAX_DGAIN_CONTROL, SENSOR_DGAIN_CONTROL);
    cs_set_reg(device, CS_HIMAX_QVGA_WIN_EN, 0x00);
    cs_set_reg(device, CS_HIMAX_BINNING_MODE, 0x03);
    cs_set_reg(device, CS_HIMAX_READOUT_X, 0x03);
    cs_set_reg(device, CS_HIMAX_READOUT_Y, 0x03);
  }
  cs_set_reg16(device, CS_HIMAX_LINE_LEN_H, profile->line_len_pck);
  if (profile->qvga) {
    cs_set_reg(device, CS_HIMAX_OSC_CLK_DIV, SENSOR_QVGA_OSC_CLK_DIV);
  }
  cs_set_reg16(device, CS_HIMAX_FRAME_LEN_H, profile->frame_len_lines);
  /* IMG_ORIENTATION is CMU-retimed, so consume it atomically with timing. */
  cs_set_reg(device, CS_HIMAX_IMG_ORIENTATION, stream->config.orientation);
  cs_write_exposure(device, &stream->config, profile, manual_rows, ae_max_rows);

  cs_set_reg(device, CS_HIMAX_GRP_PARAM_HOLD, 0x00);
  /* Bus-integrity check only; active CMU state advances on frame boundaries. */
  if (cs_get_reg(device, CS_HIMAX_IMG_ORIENTATION) !=
      stream->config.orientation) {
    return CAMERA_STREAM_ERR_CAMERA;
  }
  return CAMERA_STREAM_OK;
}

static int cs_open_camera(cs_impl_t *stream) {
  struct pi_himax_conf config;
  pi_himax_conf_init(&config);
  config.format = cs_sensor_is_qvga(stream) ? PI_CAMERA_QVGA : PI_CAMERA_QQVGA;
  // The CPI slice window is NOT optional: the HM01B0 emits a 324-pixel line in
  // QVGA (the 324x244 array; QVGA_WIN_EN alone does not narrow the emitted
  // line). With slice_en=0 the CPI stores all 324 bytes per line into a buffer
  // laid out at `width` stride, which shears every frame by -4 px/row. The
  // slice window is what actually crops 324 -> 320, so keep it enabled.
  config.roi.slice_en = 1;
  config.roi.x = 0;
  config.roi.y = stream->window.crop_top;
  config.roi.w = stream->profile.width;
  config.roi.h = stream->profile.height;
  pi_open_from_conf(&stream->camera, &config);
  if (pi_camera_open(&stream->camera) != 0) {
    return CAMERA_STREAM_ERR_CAMERA;
  }
  stream->camera_open = 1;
  return cs_apply_sensor_profile(stream);
}

static void cs_capture_done(void *argument) {
  cs_callback_context_t *context = (cs_callback_context_t *)argument;
  cs_impl_t *stream = context->stream;
  const uint8_t index = context->index;
  const uint8_t slot = context->slot;
  const uint32_t arm_generation = context->arm_generation;
  uint8_t signal = 0;
  uint8_t accounted = 0;
  const uint32_t done_ms = cs_now_ms();

  taskENTER_CRITICAL();
  if (cs_is_initialized(stream)) {
    stream->active_callbacks++;
    accounted = 1;
  }
  if (!cs_is_initialized(stream) || index >= stream->config.capture_buffer_count ||
      slot >= CS_TASK_SLOTS_PER_BUFFER || !stream->camera_open ||
      stream->closing || !stream->task_slot_active[index][slot] ||
      stream->arm_generation[index] != arm_generation ||
      (stream->armed_mask & (1u << index)) == 0) {
    if (cs_is_initialized(stream)) {
      stream->stats.dropped_capture_completions++;
    }
  } else if ((stream->completed_mask & (1u << index)) != 0) {
    stream->stats.dropped_capture_completions++;
  } else {
    if (stream->settling &&
        cs_time_reached(done_ms, stream->settle_until_ms)) {
      stream->settling = 0;
    }
    stream->armed_mask &= ~(1u << index);
    stream->capture_completed_ms[index] = done_ms;
    stream->capture_frame_id[index] = ++stream->next_camera_frame_id;
    stream->completion_stop_epoch[index] = stream->stop_epoch;
    stream->completion_settling[index] = stream->settling;
    stream->completion_publishable[index] =
        stream->started && !stream->settling;
    stream->completed_mask |= (1u << index);
    stream->stats.capture_completions++;
    signal = 1;
  }
  /* All context fields needed below are now local, so releasing the slot here
   * is generation-safe. */
  if (cs_is_initialized(stream) &&
      index < stream->config.capture_buffer_count &&
      slot < CS_TASK_SLOTS_PER_BUFFER &&
      stream->callback_contexts[index][slot].arm_generation == arm_generation) {
    stream->task_slot_active[index][slot] = 0;
  }
  taskEXIT_CRITICAL();

  if (signal && stream->events != NULL) {
    xEventGroupSetBits(stream->events, CS_EVENT_CAPTURE_DONE);
  }

  taskENTER_CRITICAL();
  if (accounted) {
    stream->active_callbacks--;
  }
  taskEXIT_CRITICAL();
}

static int cs_queue_buffer(cs_impl_t *stream, uint8_t index) {
  int queue = 0;
  int eligible = 0;
  int slot = -1;
  uint32_t arm_generation = 0;
  taskENTER_CRITICAL();
  if (stream->open && !stream->closing &&
      index < stream->config.capture_buffer_count &&
      (stream->armed_mask & (1u << index)) == 0 &&
      (stream->completed_mask & (1u << index)) == 0 &&
      stream->buffer_state[index] != CS_BUFFER_LEASED) {
    eligible = 1;
    for (uint8_t candidate = 0; candidate < CS_TASK_SLOTS_PER_BUFFER;
         candidate++) {
      if (!stream->task_slot_active[index][candidate]) {
        slot = candidate;
        break;
      }
    }
    if (slot >= 0) {
      stream->next_arm_generation++;
      if (stream->next_arm_generation == 0) {
        stream->next_arm_generation++;
      }
      arm_generation = stream->next_arm_generation;
      stream->callback_contexts[index][slot].stream = stream;
      stream->callback_contexts[index][slot].index = index;
      stream->callback_contexts[index][slot].slot = (uint8_t)slot;
      stream->callback_contexts[index][slot].arm_generation = arm_generation;
      stream->task_slot_active[index][slot] = 1;
      stream->arm_generation[index] = arm_generation;
      stream->buffer_state[index] = CS_BUFFER_FILLING;
      stream->armed_mask |= (1u << index);
      stream->capture_started_ms[index] = cs_now_ms();
      queue = 1;
    }
  }
  taskEXIT_CRITICAL();

  if (!queue) {
    return eligible && slot < 0 ? CAMERA_STREAM_ERR_BUSY
                                : CAMERA_STREAM_ERR_INVALID_STATE;
  }
  pi_camera_capture_async(
      &stream->camera, stream->buffers[index], stream->frame_size,
      pi_task_callback(&stream->capture_tasks[index][slot], cs_capture_done,
                       &stream->callback_contexts[index][slot]));
  return CAMERA_STREAM_OK;
}

static int cs_queue_buffer_retry(cs_impl_t *stream, uint8_t index) {
  int status = cs_queue_buffer(stream, index);
  if (status == CAMERA_STREAM_ERR_BUSY) {
    /* A higher-priority worker can wake before the callback returns and frees
     * its task slot. One tick lets that callback finish; a still-busy pair is
     * genuinely unresolved across a watchdog generation. */
    vTaskDelay(1);
    status = cs_queue_buffer(stream, index);
  }
  return status;
}

static void cs_mark_capture_fault(cs_impl_t *stream, int sensor_is_stopped) {
  taskENTER_CRITICAL();
  if (stream->open && !stream->closing) {
    stream->faulted = 1;
    stream->started = 0;
    stream->settling = 0;
    stream->stop_epoch++;
    stream->stats.dropped_capture_completions++;
    if (sensor_is_stopped) {
      stream->hardware_streaming = 0;
    }
  }
  taskEXIT_CRITICAL();
  if (stream->events != NULL) {
    xEventGroupSetBits(stream->events,
                       CS_EVENT_CONTROL | CS_EVENT_READER_WAKE);
  }
}

static int cs_oldest_completed(cs_impl_t *stream) {
  int selected = -1;
  uint32_t oldest_id = UINT32_MAX;
  taskENTER_CRITICAL();
  for (uint8_t index = 0; index < stream->config.capture_buffer_count; index++) {
    if ((stream->completed_mask & (1u << index)) != 0 &&
        stream->capture_frame_id[index] < oldest_id) {
      oldest_id = stream->capture_frame_id[index];
      selected = index;
    }
  }
  if (selected >= 0) {
    stream->completed_mask &= ~(1u << selected);
  }
  taskEXIT_CRITICAL();
  return selected;
}

static void cs_process_completion(cs_impl_t *stream, uint8_t index) {
  int requeue = -1;
  int signal_ready = 0;
  int publishable;
  int settling;

  taskENTER_CRITICAL();
  publishable = stream->started && stream->completion_publishable[index] &&
                stream->completion_stop_epoch[index] == stream->stop_epoch;
  settling = stream->completion_settling[index];
  taskEXIT_CRITICAL();

  if (cs_policy(stream) == CAMERA_STREAM_POLICY_PREVIEW_LIVE) {
    uint32_t capture_started_ms;
    uint32_t capture_completed_ms;
    uint32_t capture_frame_id;
    uint32_t completion_stop_epoch;
    taskENTER_CRITICAL();
    capture_started_ms = stream->capture_started_ms[index];
    capture_completed_ms = stream->capture_completed_ms[index];
    capture_frame_id = stream->capture_frame_id[index];
    completion_stop_epoch = stream->completion_stop_epoch[index];
    taskEXIT_CRITICAL();
    /* Re-arm first: publication is intentionally only a live-buffer observer. */
    if (cs_queue_buffer_retry(stream, index) != CAMERA_STREAM_OK) {
      cs_mark_capture_fault(stream, 0);
    }
    taskENTER_CRITICAL();
    if (publishable && stream->started &&
        completion_stop_epoch == stream->stop_epoch) {
      stream->preview_capture_started_ms = capture_started_ms;
      stream->preview_capture_completed_ms = capture_completed_ms;
      stream->latest_preview_frame_id = capture_frame_id;
      stream->stats.ready_frames++;
      signal_ready = 1;
    } else {
      if (settling) {
        stream->stats.sensor_settle_discards++;
      } else {
        stream->stats.paused_discards++;
      }
    }
    taskEXIT_CRITICAL();
  } else {
    /* Exactly one capture is in flight. The next goes into a buffer nobody
     * holds, preferring a free one to the ready frame this one supersedes. With
     * two buffers and the other leased there is none: this frame is overwritten. */
    int next = -1;
    taskENTER_CRITICAL();
    for (uint8_t other = 0; other < stream->config.capture_buffer_count;
         other++) {
      if (other != index && stream->buffer_state[other] != CS_BUFFER_LEASED &&
          (next < 0 || stream->buffer_state[other] == CS_BUFFER_FREE)) {
        next = other;
      }
    }
    if (!publishable || !stream->started ||
        stream->completion_stop_epoch[index] != stream->stop_epoch) {
      if (settling) {
        stream->stats.sensor_settle_discards++;
      } else {
        stream->stats.paused_discards++;
      }
      requeue = index;
    } else if (next < 0) {
      stream->stats.dropped_while_leased++;
      requeue = index;
    } else {
      if (stream->ready_index >= 0 && stream->ready_index != index) {
        stream->stats.dropped_ready_frames++;
        stream->buffer_state[(uint8_t)stream->ready_index] = CS_BUFFER_FREE;
      }
      stream->buffer_state[index] = CS_BUFFER_READY;
      stream->ready_index = (int8_t)index;
      stream->stats.ready_frames++;
      signal_ready = 1;
      requeue = next;
    }
    taskEXIT_CRITICAL();
    if (cs_queue_buffer_retry(stream, (uint8_t)requeue) != CAMERA_STREAM_OK) {
      cs_mark_capture_fault(stream, 0);
    }
  }

  if (signal_ready) {
    xEventGroupSetBits(stream->events, CS_EVENT_FRAME_READY);
  }
}

static void cs_restart_capture(cs_impl_t *stream) {
  int next = -1;
  int queue_status;

  xSemaphoreTake(stream->lifecycle_lock, portMAX_DELAY);
  taskENTER_CRITICAL();
  if (!stream->open || stream->closing || !stream->hardware_streaming) {
    taskEXIT_CRITICAL();
    xSemaphoreGive(stream->lifecycle_lock);
    return;
  }
  taskEXIT_CRITICAL();

  pi_camera_control(&stream->camera, PI_CAMERA_CMD_STOP, 0);

  taskENTER_CRITICAL();
  for (uint8_t index = 0; index < stream->config.capture_buffer_count; index++) {
    if ((stream->completed_mask & (1u << index)) != 0) {
      stream->stats.dropped_capture_completions++;
    }
    if (stream->buffer_state[index] == CS_BUFFER_READY) {
      stream->stats.dropped_ready_frames++;
    }
    if (stream->buffer_state[index] != CS_BUFFER_LEASED) {
      stream->buffer_state[index] = CS_BUFFER_FREE;
    }
    stream->arm_generation[index] = 0;
  }
  stream->completed_mask = 0;
  stream->armed_mask = 0;
  stream->ready_index = -1;
  stream->latest_preview_frame_id = stream->last_delivered_preview_frame_id;
  stream->stats.watchdog_restarts++;

  /* One capture in flight, in the first buffer nobody holds. */
  for (uint8_t index = 0; index < stream->config.capture_buffer_count; index++) {
    if (stream->buffer_state[index] != CS_BUFFER_LEASED) {
      next = (int)index;
      break;
    }
  }
  taskEXIT_CRITICAL();

  queue_status = next < 0 ? CAMERA_STREAM_ERR_INVALID_STATE
                          : cs_queue_buffer_retry(stream, (uint8_t)next);

  if (queue_status != CAMERA_STREAM_OK) {
    cs_mark_capture_fault(stream, 1);
    xSemaphoreGive(stream->lifecycle_lock);
    return;
  }

  taskENTER_CRITICAL();
  const int restart = stream->open && !stream->closing;
  taskEXIT_CRITICAL();
  if (restart) {
    pi_camera_control(&stream->camera, PI_CAMERA_CMD_START, 0);
  }
  xSemaphoreGive(stream->lifecycle_lock);
}

static int cs_watchdog_due(cs_impl_t *stream) {
  int due = 0;
  const uint32_t now = cs_now_ms();
  taskENTER_CRITICAL();
  if (stream->open && !stream->closing && stream->hardware_streaming &&
      stream->config.watchdog_timeout_ms != 0) {
    for (uint8_t index = 0; index < stream->config.capture_buffer_count;
         index++) {
      if ((stream->armed_mask & (1u << index)) != 0 &&
          (uint32_t)(now - stream->capture_started_ms[index]) >=
              stream->config.watchdog_timeout_ms) {
        due = 1;
        break;
      }
    }
  }
  taskEXIT_CRITICAL();
  return due;
}

static TickType_t cs_watchdog_wait_ticks(cs_impl_t *stream) {
  uint32_t remaining_ms = UINT32_MAX;
  const uint32_t now = cs_now_ms();

  taskENTER_CRITICAL();
  if (stream->config.watchdog_timeout_ms != 0) {
    for (uint8_t index = 0; index < stream->config.capture_buffer_count;
         index++) {
      if ((stream->armed_mask & (1u << index)) != 0) {
        const uint32_t age = now - stream->capture_started_ms[index];
        const uint32_t remaining =
            age >= stream->config.watchdog_timeout_ms
                ? 0
                : stream->config.watchdog_timeout_ms - age;
        if (remaining < remaining_ms) {
          remaining_ms = remaining;
        }
      }
    }
  }
  taskEXIT_CRITICAL();

  return remaining_ms == UINT32_MAX
             ? portMAX_DELAY
             : cs_ms_to_ticks(remaining_ms == 0 ? 1 : remaining_ms);
}

static void cs_worker(void *argument) {
  cs_impl_t *stream = (cs_impl_t *)argument;

  while (1) {
    TickType_t wait_ticks;
    taskENTER_CRITICAL();
    const int closing = stream->closing;
    taskEXIT_CRITICAL();
    if (closing) {
      break;
    }
    wait_ticks = cs_watchdog_wait_ticks(stream);

    const EventBits_t bits = xEventGroupWaitBits(
        stream->events,
        CS_EVENT_CAPTURE_DONE | CS_EVENT_CONTROL | CS_EVENT_SHUTDOWN,
        pdTRUE, pdFALSE, wait_ticks);
    if ((bits & CS_EVENT_SHUTDOWN) != 0) {
      break;
    }

    while (1) {
      const int index = cs_oldest_completed(stream);
      if (index < 0) {
        break;
      }
      cs_process_completion(stream, (uint8_t)index);
    }

    if (cs_watchdog_due(stream)) {
      cs_restart_capture(stream);
    }
  }

  taskENTER_CRITICAL();
  stream->worker_exited = 1;
  taskEXIT_CRITICAL();
  vTaskDelete(NULL);
}

static void cs_free_buffers(cs_impl_t *stream) {
  for (int index = (int)stream->config.capture_buffer_count - 1;
       index >= 0; index--) {
    if (stream->buffers[index] != NULL) {
      pi_l2_free(stream->buffers[index], stream->frame_size);
      stream->buffers[index] = NULL;
    }
  }
}

static void cs_rollback_open(cs_impl_t *stream) {
  if (stream->lifecycle_lock != NULL) {
    vSemaphoreDelete(stream->lifecycle_lock);
    stream->lifecycle_lock = NULL;
  }
  if (stream->events != NULL) {
    vEventGroupDelete(stream->events);
    stream->events = NULL;
  }
  cs_free_buffers(stream);
  if (stream->camera_open) {
    pi_camera_close(&stream->camera);
    stream->camera_open = 0;
  }
  stream->open = 0;
  stream->closing = 0;
}

int camera_stream_config_init(camera_stream_config_t *config) {
  if (config == NULL) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  memset(config, 0, sizeof(*config));
#if CAMERA_STREAM_SPECIALIZE_SENSOR_MODE >= 0
  config->sensor_mode =
      (camera_stream_sensor_mode_t)CAMERA_STREAM_SPECIALIZE_SENSOR_MODE;
#else
  config->sensor_mode = CAMERA_STREAM_SENSOR_QVGA_60;
#endif
#if CAMERA_STREAM_SPECIALIZE_POLICY >= 0
  config->policy = (camera_stream_policy_t)CAMERA_STREAM_SPECIALIZE_POLICY;
#else
  config->policy = CAMERA_STREAM_POLICY_PREVIEW_LIVE;
#endif
  config->orientation = CAMERA_STREAM_DEFAULT_ORIENTATION;
  config->window_top = CAMERA_STREAM_WINDOW_DEFAULT;
  config->exposure_mode = (camera_stream_exposure_mode_t)
      CAMERA_STREAM_DEFAULT_EXPOSURE_MODE;
  config->ae_target_mean = CAMERA_STREAM_DEFAULT_AE_TARGET_MEAN;
  config->ae_max_exposure_us = CAMERA_STREAM_DEFAULT_AE_MAX_EXPOSURE_US;
  config->ae_max_analog_gain = CAMERA_STREAM_DEFAULT_AE_MAX_ANALOG_GAIN;
  config->ae_max_digital_gain = CAMERA_STREAM_DEFAULT_AE_MAX_DIGITAL_GAIN;
  config->manual_exposure_us = CAMERA_STREAM_DEFAULT_MANUAL_EXPOSURE_US;
  config->manual_analog_gain = CAMERA_STREAM_DEFAULT_MANUAL_ANALOG_GAIN;
  config->manual_digital_gain = CAMERA_STREAM_DEFAULT_MANUAL_DIGITAL_GAIN;
  config->flicker_mode =
      (camera_stream_flicker_mode_t)CAMERA_STREAM_DEFAULT_FLICKER_MODE;
  config->sensor_settle_ms = CAMERA_STREAM_DEFAULT_SENSOR_SETTLE_MS;
  config->watchdog_timeout_ms = 2000;
  return CAMERA_STREAM_OK;
}

int camera_stream_memory_plan(const camera_stream_config_t *config,
                              camera_stream_memory_plan_t *plan) {
  camera_stream_config_t resolved;
  cs_sensor_profile_t profile;
  cs_window_t window;
  const int status = cs_resolve_config(config, &resolved, &profile, &window);
  if (status != CAMERA_STREAM_OK || plan == NULL) {
    return status != CAMERA_STREAM_OK ? status
                                      : CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  plan->width = profile.width;
  plan->height = profile.height;
  plan->stride = profile.width;
  plan->capture_buffer_count = resolved.capture_buffer_count;
  plan->frame_size_bytes = (size_t)profile.width * profile.height;
  plan->capture_l2_bytes =
      plan->frame_size_bytes * resolved.capture_buffer_count;
  return CAMERA_STREAM_OK;
}

int camera_stream_init(camera_stream_t *storage,
                       const camera_stream_config_t *config) {
  camera_stream_config_t resolved;
  cs_sensor_profile_t profile;
  cs_window_t window;
  cs_impl_t *stream;
  int status;

  if (storage == NULL) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  if (cs_storage_looks_open(storage)) {
    return CAMERA_STREAM_ERR_BUSY;
  }
  status = cs_resolve_config(config, &resolved, &profile, &window);
  if (status != CAMERA_STREAM_OK) {
    return status;
  }

  memset(storage, 0, sizeof(*storage));
  stream = cs_impl(storage);
  stream->magic = CS_MAGIC;
  stream->self = (uintptr_t)stream;
  stream->config = resolved;
  stream->profile = profile;
  stream->window = window;
  stream->frame_size = (size_t)profile.width * profile.height;
  stream->ready_index = -1;
  stream->lease_index = -1;
  return CAMERA_STREAM_OK;
}

int camera_stream_open(camera_stream_t *storage) {
  cs_impl_t *stream;
  int status;

  if (storage == NULL) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  stream = cs_impl(storage);
  if (!cs_is_initialized(stream)) {
    return CAMERA_STREAM_ERR_INVALID_STATE;
  }
  if (stream->open || stream->camera_open) {
    return CAMERA_STREAM_ERR_INVALID_STATE;
  }

  memset(&stream->stats, 0, sizeof(stream->stats));
  stream->completed_mask = 0;
  stream->armed_mask = 0;
  stream->next_camera_frame_id = 0;
  stream->latest_preview_frame_id = 0;
  stream->last_delivered_preview_frame_id = 0;
  stream->preview_capture_started_ms = 0;
  stream->preview_capture_completed_ms = 0;
  stream->stop_epoch = 0;
  stream->settle_until_ms = 0;
  stream->ready_index = -1;
  stream->lease_index = -1;
  stream->closing = 0;
  stream->worker_exited = 0;
  stream->reader_active = 0;
  stream->lease_active = 0;
  stream->lifecycle_api_users = 0;
  stream->faulted = 0;
  stream->settling = 0;
  stream->active_callbacks = 0;
  stream->lease_token = 0;
  memset(stream->task_slot_active, 0, sizeof(stream->task_slot_active));
  memset(stream->arm_generation, 0, sizeof(stream->arm_generation));
  memset(stream->completion_settling, 0,
         sizeof(stream->completion_settling));

  status = cs_open_camera(stream);
  if (status != CAMERA_STREAM_OK) {
    cs_rollback_open(stream);
    return status;
  }

  /* Himax/CPI driver state is already secured before these bulk L2 buffers. */
  for (uint8_t index = 0; index < stream->config.capture_buffer_count; index++) {
    stream->buffers[index] = (uint8_t *)pmsis_l2_malloc(stream->frame_size);
    if (stream->buffers[index] == NULL) {
      cs_rollback_open(stream);
      return CAMERA_STREAM_ERR_NO_MEMORY;
    }
    stream->buffer_state[index] = CS_BUFFER_FREE;
  }

  stream->events = xEventGroupCreate();
  if (stream->events == NULL) {
    cs_rollback_open(stream);
    return CAMERA_STREAM_ERR_RTOS;
  }
  stream->lifecycle_lock = xSemaphoreCreateMutex();
  if (stream->lifecycle_lock == NULL) {
    cs_rollback_open(stream);
    return CAMERA_STREAM_ERR_RTOS;
  }

  stream->open = 1;
  if (xTaskCreate(cs_worker, "camera_stream", CAMERA_STREAM_TASK_STACK_WORDS,
                  stream, CAMERA_STREAM_TASK_PRIORITY, NULL) != pdPASS) {
    stream->open = 0;
    cs_rollback_open(stream);
    return CAMERA_STREAM_ERR_RTOS;
  }
  return CAMERA_STREAM_OK;
}

int camera_stream_start(camera_stream_t *storage) {
  cs_impl_t *stream;
  SemaphoreHandle_t lock;
  int first_start;
  int status;

  if (storage == NULL) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  stream = cs_impl(storage);
  if (!cs_is_initialized(stream)) {
    return CAMERA_STREAM_ERR_INVALID_STATE;
  }

  status = cs_lifecycle_enter(stream, &lock);
  if (status != CAMERA_STREAM_OK) {
    return status;
  }
  taskENTER_CRITICAL();
  if (!stream->open || stream->closing) {
    taskEXIT_CRITICAL();
    cs_lifecycle_leave(stream, lock);
    return CAMERA_STREAM_ERR_CLOSED;
  }
  if (stream->faulted) {
    taskEXIT_CRITICAL();
    cs_lifecycle_leave(stream, lock);
    return CAMERA_STREAM_ERR_CAMERA;
  }
  if (stream->started) {
    taskEXIT_CRITICAL();
    cs_lifecycle_leave(stream, lock);
    return CAMERA_STREAM_OK;
  }
  stream->started = 1;
  first_start = !stream->hardware_streaming;
  if (first_start) {
    stream->hardware_streaming = 1;
    cs_begin_sensor_settle(stream);
  }
  taskEXIT_CRITICAL();

  if (first_start) {
    /* Nothing can be leased before the first start: buffer 0 goes first. */
    if (cs_queue_buffer_retry(stream, 0) == CAMERA_STREAM_OK) {
      pi_camera_control(&stream->camera, PI_CAMERA_CMD_START, 0);
    } else {
      cs_mark_capture_fault(stream, 1);
      cs_lifecycle_leave(stream, lock);
      return CAMERA_STREAM_ERR_CAMERA;
    }
  }
  xEventGroupSetBits(stream->events, CS_EVENT_CONTROL);
  cs_lifecycle_leave(stream, lock);
  return CAMERA_STREAM_OK;
}

int camera_stream_stop(camera_stream_t *storage) {
  cs_impl_t *stream;
  SemaphoreHandle_t lock;
  int status;

  if (storage == NULL) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  stream = cs_impl(storage);
  if (!cs_is_initialized(stream)) {
    return CAMERA_STREAM_ERR_INVALID_STATE;
  }

  status = cs_lifecycle_enter(stream, &lock);
  if (status != CAMERA_STREAM_OK) {
    return status;
  }
  taskENTER_CRITICAL();
  if (stream->closing) {
    taskEXIT_CRITICAL();
    cs_lifecycle_leave(stream, lock);
    return CAMERA_STREAM_ERR_CLOSED;
  }
  if (!stream->started) {
    taskEXIT_CRITICAL();
    cs_lifecycle_leave(stream, lock);
    return CAMERA_STREAM_OK;
  }
  stream->started = 0;
  stream->stop_epoch++;
  stream->last_delivered_preview_frame_id = stream->latest_preview_frame_id;

  if (cs_policy(stream) == CAMERA_STREAM_POLICY_EXACT_FRAME &&
      stream->ready_index >= 0) {
    stream->buffer_state[(uint8_t)stream->ready_index] = CS_BUFFER_FREE;
    stream->ready_index = -1;
    stream->stats.paused_discards++;
  }
  taskEXIT_CRITICAL();

  xEventGroupSetBits(stream->events, CS_EVENT_CONTROL | CS_EVENT_READER_WAKE);
  cs_lifecycle_leave(stream, lock);
  return CAMERA_STREAM_OK;
}

static int cs_select_ready(cs_impl_t *stream) {
  int selected = -1;
  if (cs_policy(stream) == CAMERA_STREAM_POLICY_PREVIEW_LIVE) {
    if (stream->latest_preview_frame_id != 0 &&
        stream->latest_preview_frame_id !=
            stream->last_delivered_preview_frame_id) {
      selected = 0;
    }
  } else {
    selected = stream->ready_index;
  }
  return selected;
}

static void cs_fill_frame(cs_impl_t *stream, uint8_t index,
                          camera_stream_frame_t *frame) {
  frame->data = stream->buffers[index];
  frame->size_bytes = stream->frame_size;
  frame->width = stream->profile.width;
  frame->height = stream->profile.height;
  frame->stride = stream->profile.width;
  if (cs_policy(stream) == CAMERA_STREAM_POLICY_PREVIEW_LIVE) {
    frame->capture_started_ms = stream->preview_capture_started_ms;
    frame->capture_completed_ms = stream->preview_capture_completed_ms;
  } else {
    frame->capture_started_ms = stream->capture_started_ms[index];
    frame->capture_completed_ms = stream->capture_completed_ms[index];
  }
  frame->capture_duration_ms =
      frame->capture_completed_ms - frame->capture_started_ms;
  frame->camera_frame_id =
      cs_policy(stream) == CAMERA_STREAM_POLICY_PREVIEW_LIVE
          ? stream->latest_preview_frame_id
          : stream->capture_frame_id[index];
  frame->token = stream->lease_token;
}

int camera_stream_acquire(camera_stream_t *storage,
                          camera_stream_frame_t *frame,
                          uint32_t timeout_ms) {
  cs_impl_t *stream;
  TickType_t remaining_ticks = 0;
  uint32_t observed_stop_epoch;

  if (storage == NULL || frame == NULL) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  stream = cs_impl(storage);
  if (!cs_is_initialized(stream)) {
    return CAMERA_STREAM_ERR_INVALID_STATE;
  }
  memset(frame, 0, sizeof(*frame));

  taskENTER_CRITICAL();
  if (!stream->open || stream->closing) {
    taskEXIT_CRITICAL();
    return CAMERA_STREAM_ERR_CLOSED;
  }
  if (!stream->started) {
    taskEXIT_CRITICAL();
    return CAMERA_STREAM_ERR_STOPPED;
  }
  if (stream->reader_active || stream->lease_active) {
    taskEXIT_CRITICAL();
    return CAMERA_STREAM_ERR_BUSY;
  }
  stream->reader_active = 1;
  observed_stop_epoch = stream->stop_epoch;
  taskEXIT_CRITICAL();

  if (timeout_ms != CAMERA_STREAM_WAIT_FOREVER) {
    remaining_ticks = cs_ms_to_ticks(timeout_ms);
  }

  while (1) {
    int selected;
    taskENTER_CRITICAL();
    if (!stream->open || stream->closing) {
      stream->reader_active = 0;
      taskEXIT_CRITICAL();
      return CAMERA_STREAM_ERR_CLOSED;
    }
    if (!stream->started || stream->stop_epoch != observed_stop_epoch) {
      stream->reader_active = 0;
      taskEXIT_CRITICAL();
      return CAMERA_STREAM_ERR_STOPPED;
    }

    selected = cs_select_ready(stream);
    if (selected >= 0) {
      const uint8_t index = (uint8_t)selected;
      stream->lease_active = 1;
      stream->reader_active = 0;
      stream->lease_index = (int8_t)index;
      cs_global_lease_token++;
      if (cs_global_lease_token == 0) {
        cs_global_lease_token++;
      }
      stream->lease_token = cs_global_lease_token;
      if (cs_policy(stream) == CAMERA_STREAM_POLICY_PREVIEW_LIVE) {
        stream->last_delivered_preview_frame_id =
            stream->latest_preview_frame_id;
      } else {
        stream->buffer_state[index] = CS_BUFFER_LEASED;
        stream->ready_index = -1;
      }
      cs_fill_frame(stream, index, frame);
      taskEXIT_CRITICAL();
      return CAMERA_STREAM_OK;
    }
    taskEXIT_CRITICAL();

    if (timeout_ms != CAMERA_STREAM_WAIT_FOREVER && remaining_ticks == 0) {
      taskENTER_CRITICAL();
      stream->reader_active = 0;
      taskEXIT_CRITICAL();
      return CAMERA_STREAM_ERR_TIMEOUT;
    }

    const TickType_t before = xTaskGetTickCount();
    xEventGroupWaitBits(stream->events,
                        CS_EVENT_FRAME_READY | CS_EVENT_READER_WAKE,
                        pdTRUE, pdFALSE,
                        timeout_ms == CAMERA_STREAM_WAIT_FOREVER
                            ? portMAX_DELAY
                            : remaining_ticks);
    if (timeout_ms != CAMERA_STREAM_WAIT_FOREVER) {
      const TickType_t elapsed = xTaskGetTickCount() - before;
      remaining_ticks = elapsed >= remaining_ticks ? 0
                                                   : remaining_ticks - elapsed;
    }
  }
}

int camera_stream_release(camera_stream_t *storage,
                          camera_stream_frame_t *frame) {
  cs_impl_t *stream;
  SemaphoreHandle_t lock;
  int status;

  if (storage == NULL || frame == NULL) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  stream = cs_impl(storage);
  if (!cs_is_initialized(stream)) {
    return CAMERA_STREAM_ERR_INVALID_STATE;
  }

  status = cs_lifecycle_enter(stream, &lock);
  if (status != CAMERA_STREAM_OK) {
    return status;
  }

  taskENTER_CRITICAL();
  if (!stream->lease_active || stream->lease_index < 0 ||
      frame->token == 0 || frame->token != stream->lease_token ||
      frame->data != stream->buffers[(uint8_t)stream->lease_index]) {
    taskEXIT_CRITICAL();
    cs_lifecycle_leave(stream, lock);
    return CAMERA_STREAM_ERR_STALE_FRAME;
  }

  const uint8_t index = (uint8_t)stream->lease_index;
  stream->lease_active = 0;
  stream->lease_index = -1;
  stream->lease_token = 0;
  if (cs_policy(stream) == CAMERA_STREAM_POLICY_EXACT_FRAME) {
    stream->buffer_state[index] = CS_BUFFER_FREE;
  }
  taskEXIT_CRITICAL();

  memset(frame, 0, sizeof(*frame));
  if (stream->events != NULL) {
    xEventGroupSetBits(stream->events, CS_EVENT_CONTROL);
  }
  cs_lifecycle_leave(stream, lock);
  return CAMERA_STREAM_OK;
}

int camera_stream_close(camera_stream_t *storage) {
  cs_impl_t *stream;
  SemaphoreHandle_t lock;
  EventGroupHandle_t events;

  if (storage == NULL) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  stream = cs_impl(storage);
  if (!cs_is_initialized(stream)) {
    return CAMERA_STREAM_ERR_INVALID_STATE;
  }
  taskENTER_CRITICAL();
  if (!stream->open) {
    taskEXIT_CRITICAL();
    return CAMERA_STREAM_OK;
  }
  if (stream->closing) {
    taskEXIT_CRITICAL();
    return CAMERA_STREAM_ERR_BUSY;
  }
  if (stream->lease_active) {
    taskEXIT_CRITICAL();
    return CAMERA_STREAM_ERR_BUSY;
  }
  stream->closing = 1;
  lock = stream->lifecycle_lock;
  taskEXIT_CRITICAL();

  /* Mark closing before queueing on the lifecycle lock. This admits exactly
   * one closer and prevents another task from waiting on a mutex that the
   * admitted close is about to delete. */
  xSemaphoreTake(lock, portMAX_DELAY);
  taskENTER_CRITICAL();
  stream->started = 0;
  stream->stop_epoch++;
  const int stop_hardware = stream->hardware_streaming;
  stream->hardware_streaming = 0;
  events = stream->events;
  taskEXIT_CRITICAL();

  if (stop_hardware) {
    pi_camera_control(&stream->camera, PI_CAMERA_CMD_STOP, 0);
  }
  xEventGroupSetBits(events, CS_EVENT_SHUTDOWN | CS_EVENT_CONTROL |
                                CS_EVENT_READER_WAKE);
  xSemaphoreGive(lock);

  while (!cs_close_api_quiescent(stream)) {
    vTaskDelay(1);
  }

  /* Closing the device before deleting callback-visible synchronization state
   * makes late/cancelled CPI completions harmless. */
  if (stream->camera_open) {
    pi_camera_close(&stream->camera);
    stream->camera_open = 0;
  }
  while (!cs_callbacks_quiescent(stream)) {
    vTaskDelay(1);
  }
  vSemaphoreDelete(stream->lifecycle_lock);
  stream->lifecycle_lock = NULL;
  vEventGroupDelete(stream->events);
  stream->events = NULL;
  cs_free_buffers(stream);

  taskENTER_CRITICAL();
  stream->open = 0;
  stream->closing = 0;
  stream->completed_mask = 0;
  stream->armed_mask = 0;
  taskEXIT_CRITICAL();
  return CAMERA_STREAM_OK;
}

int camera_stream_get_info(const camera_stream_t *storage,
                           camera_stream_info_t *info) {
  const cs_impl_t *stream;
  if (storage == NULL || info == NULL) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  stream = cs_const_impl(storage);
  if (!cs_is_initialized(stream)) {
    return CAMERA_STREAM_ERR_INVALID_STATE;
  }

  taskENTER_CRITICAL();
  memset(info, 0, sizeof(*info));
  info->config = stream->config;
  info->width = stream->profile.width;
  info->height = stream->profile.height;
  info->stride = stream->profile.width;
  info->named_fps = stream->profile.named_fps;
  info->capture_buffer_count = stream->config.capture_buffer_count;
  info->is_open = stream->open;
  info->is_started = stream->started;
  info->is_faulted = stream->faulted;
  info->frame_size_bytes = stream->frame_size;
  for (uint8_t index = 0; index < stream->config.capture_buffer_count; index++) {
    info->capture_buffers[index] = stream->buffers[index];
  }
  taskEXIT_CRITICAL();
  return CAMERA_STREAM_OK;
}

static const char *cs_sensor_name(camera_stream_sensor_mode_t mode) {
  switch (mode) {
    case CAMERA_STREAM_SENSOR_QVGA_30:
      return "QVGA_30";
    case CAMERA_STREAM_SENSOR_QVGA_60:
      return "QVGA_60";
    case CAMERA_STREAM_SENSOR_QQVGA_BIN2_30:
      return "QQVGA_BIN2_30";
    case CAMERA_STREAM_SENSOR_QQVGA_BIN2_60:
      return "QQVGA_BIN2_60";
    default:
      return "QQVGA_BIN2_120";
  }
}

int camera_stream_describe(const camera_stream_t *stream, char *out, size_t size) {
  camera_stream_info_t info;
  const int status = camera_stream_get_info(stream, &info);
  if (status != CAMERA_STREAM_OK) {
    (void)snprintf(out, size, "camera=unavailable");
    return status;
  }
  const camera_stream_config_t *config = &info.config;
  /* Gains as the factors the build options use: 1 2 4 8. */
  (void)snprintf(
      out, size,
      "sensor=%s capture=%ux%u sensor_fps=%u policy=%s buffers=%u flip=%u window=%u "
      "exposure=%s ae_target=%u ae_max_exposure_us=%lu ae_max_analog_gain=%u "
      "ae_max_digital_gain=%u exposure_us=%lu analog_gain=%u digital_gain=%u flicker=%s "
      "settle_ms=%lu watchdog_ms=%lu",
      cs_sensor_name(config->sensor_mode), info.width, info.height, info.named_fps,
      config->policy == CAMERA_STREAM_POLICY_EXACT_FRAME ? "exact" : "preview",
      info.capture_buffer_count, config->orientation, config->window_top,
      config->exposure_mode == CAMERA_STREAM_EXPOSURE_MANUAL ? "manual" : "auto",
      config->ae_target_mean, (unsigned long)config->ae_max_exposure_us,
      1u << config->ae_max_analog_gain, config->ae_max_digital_gain,
      (unsigned long)config->manual_exposure_us, 1u << config->manual_analog_gain,
      config->manual_digital_gain,
      config->flicker_mode == CAMERA_STREAM_FLICKER_50_HZ   ? "50hz"
      : config->flicker_mode == CAMERA_STREAM_FLICKER_60_HZ ? "60hz"
                                                            : "off",
      (unsigned long)config->sensor_settle_ms, (unsigned long)config->watchdog_timeout_ms);
  return CAMERA_STREAM_OK;
}

int camera_stream_get_stats(const camera_stream_t *storage,
                            camera_stream_stats_t *stats) {
  const cs_impl_t *stream;
  if (storage == NULL || stats == NULL) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  stream = cs_const_impl(storage);
  if (!cs_is_initialized(stream)) {
    return CAMERA_STREAM_ERR_INVALID_STATE;
  }
  taskENTER_CRITICAL();
  *stats = stream->stats;
  taskEXIT_CRITICAL();
  return CAMERA_STREAM_OK;
}

int camera_stream_set_exposure(camera_stream_t *storage,
                               const camera_stream_config_t *config) {
  cs_impl_t *stream;
  SemaphoreHandle_t lock;
  uint16_t manual_rows;
  uint16_t ae_max_rows;
  int status;

  if (storage == NULL || config == NULL) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  stream = cs_impl(storage);
  if (!cs_is_initialized(stream)) {
    return CAMERA_STREAM_ERR_INVALID_STATE;
  }
  status = cs_resolve_exposure(config, &stream->profile, &manual_rows,
                               &ae_max_rows);
  if (status != CAMERA_STREAM_OK) {
    return status;
  }
  status = cs_lifecycle_enter(stream, &lock);
  if (status != CAMERA_STREAM_OK) {
    return status;
  }
  /* One parameter group, so the sensor switches at a frame boundary. */
  cs_set_reg(&stream->camera, CS_HIMAX_GRP_PARAM_HOLD, 0x01);
  cs_write_exposure(&stream->camera, config, &stream->profile, manual_rows,
                    ae_max_rows);
  cs_set_reg(&stream->camera, CS_HIMAX_GRP_PARAM_HOLD, 0x00);
  taskENTER_CRITICAL();
  stream->config.exposure_mode = config->exposure_mode;
  stream->config.ae_target_mean = config->ae_target_mean;
  stream->config.ae_max_exposure_us = config->ae_max_exposure_us;
  stream->config.ae_max_analog_gain = config->ae_max_analog_gain;
  stream->config.ae_max_digital_gain = config->ae_max_digital_gain;
  stream->config.flicker_mode = config->flicker_mode;
  stream->config.manual_exposure_us = config->manual_exposure_us;
  stream->config.manual_analog_gain = config->manual_analog_gain;
  stream->config.manual_digital_gain = config->manual_digital_gain;
  taskEXIT_CRITICAL();
  cs_lifecycle_leave(stream, lock);
  return CAMERA_STREAM_OK;
}

int camera_stream_get_exposure(const camera_stream_t *storage,
                               camera_stream_exposure_t *exposure) {
  /* Reading registers leaves the stream as it was; the cast only lets this
   * take the lifecycle lock that serializes the sensor's I2C. */
  cs_impl_t *stream;
  SemaphoreHandle_t lock;
  uint32_t rows;
  int status;

  if (storage == NULL || exposure == NULL) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  stream = cs_impl((camera_stream_t *)storage);
  if (!cs_is_initialized(stream)) {
    return CAMERA_STREAM_ERR_INVALID_STATE;
  }
  status = cs_lifecycle_enter(stream, &lock);
  if (status != CAMERA_STREAM_OK) {
    return status;
  }
  rows = ((uint32_t)cs_get_reg(&stream->camera, CS_HIMAX_INTEGRATION_H) << 8) |
         cs_get_reg(&stream->camera, CS_HIMAX_INTEGRATION_L);
  exposure->exposure_us =
      (uint32_t)(((uint64_t)rows * stream->profile.row_ns + 500u) / 1000u);
  exposure->analog_gain = (camera_stream_analog_gain_t)(
      (cs_get_reg(&stream->camera, CS_HIMAX_ANALOG_GAIN) >> 4) & 0x07u);
  exposure->digital_gain = (uint16_t)(
      ((cs_get_reg(&stream->camera, CS_HIMAX_DIGITAL_GAIN_H) & 0x03u) << 8) |
      cs_get_reg(&stream->camera, CS_HIMAX_DIGITAL_GAIN_L));
  cs_lifecycle_leave(stream, lock);
  return CAMERA_STREAM_OK;
}
