/* pmsis.h must precede libc headers so the GAP8 freestanding shims apply. */
#include "pmsis.h"

#include <stddef.h>
#include <string.h>

#include "camera_preview.h"
#include "cpx_config.h"
#include "imgstream.h"

#define CAMERA_PREVIEW_MAGIC 0x43505256u /* CPRV */
#define CAMERA_PREVIEW_INTERNAL_SSID "camera-stream"
#define CAMERA_PREVIEW_WATCHDOG_MS 2000u
#define CAMERA_PREVIEW_CLUSTER_FREQUENCY_HZ 150000000u
#define CAMERA_PREVIEW_CLIENT_RETRY_MS 20u

typedef enum {
  CAMERA_PREVIEW_STATE_UNINITIALIZED = 0,
  CAMERA_PREVIEW_STATE_INITIALIZED = 1,
  CAMERA_PREVIEW_STATE_STARTING = 2,
  CAMERA_PREVIEW_STATE_RUNNING = 3,
  CAMERA_PREVIEW_STATE_CLOSING = 4,
  CAMERA_PREVIEW_STATE_CLOSED = 5,
} camera_preview_state_t;

typedef struct {
  uint32_t magic;
  uintptr_t self;
  camera_preview_state_t state;

  camera_preview_config_t config;
  char ssid[CPX_WIFI_SSID_MAX_LENGTH + 1u];
  char key[CPX_WIFI_KEY_MAX_LENGTH + 1u];

  camera_stream_t camera;
  jpeg_pipeline_t jpeg;
  imgstream_t imgstream;
  cpx_wifi_session_t *wifi;

  uint8_t worker_created;
  uint8_t worker_exited;
  uint8_t stop_requested;
  uint8_t stream_ready;
  uint8_t camera_open;
  uint8_t camera_started;
  uint8_t jpeg_open;
  uint8_t active_claimed;

  uint32_t frame_id;
  camera_preview_stats_t stats;
} camera_preview_impl_t;

typedef struct {
  cpx_wifi_session_t session;
  char ssid[CPX_WIFI_SSID_MAX_LENGTH + 1u];
  char key[CPX_WIFI_KEY_MAX_LENGTH + 1u];
  camera_preview_transport_t transport;
  cpx_wifi_mode_t mode;
  uint32_t client_settle_ms;
  uint8_t initialized;
  uint8_t opened;
  uint8_t connect_requested;
} camera_preview_internal_wifi_t;

static camera_preview_internal_wifi_t internal_wifi;
static camera_preview_impl_t *active_preview;
static uint8_t cpx_ready;

typedef char camera_preview_object_must_fit[
    sizeof(camera_preview_impl_t) <= CAMERA_PREVIEW_OBJECT_BYTES ? 1 : -1];

static camera_preview_impl_t *preview_impl(camera_preview_t *preview) {
  return (camera_preview_impl_t *)(void *)preview->_storage;
}

static const camera_preview_impl_t *preview_const_impl(
    const camera_preview_t *preview) {
  return (const camera_preview_impl_t *)(const void *)preview->_storage;
}

static bool preview_is_initialized(const camera_preview_impl_t *preview) {
  return preview != NULL && preview->magic == CAMERA_PREVIEW_MAGIC &&
         preview->self == (uintptr_t)preview &&
         preview->state >= CAMERA_PREVIEW_STATE_INITIALIZED &&
         preview->state <= CAMERA_PREVIEW_STATE_CLOSED;
}

static bool storage_looks_active(const camera_preview_t *storage) {
  const camera_preview_impl_t *address = preview_const_impl(storage);
  const uint32_t expected_magic = CAMERA_PREVIEW_MAGIC;
  const uintptr_t expected_self = (uintptr_t)address;
  const uint8_t *bytes = storage->_storage;

  if (memcmp(bytes + offsetof(camera_preview_impl_t, magic), &expected_magic,
             sizeof(expected_magic)) != 0 ||
      memcmp(bytes + offsetof(camera_preview_impl_t, self), &expected_self,
             sizeof(expected_self)) != 0) {
    return false;
  }
  for (int value = CAMERA_PREVIEW_STATE_STARTING;
       value <= CAMERA_PREVIEW_STATE_CLOSING; value++) {
    const camera_preview_state_t expected = (camera_preview_state_t)value;
    if (memcmp(bytes + offsetof(camera_preview_impl_t, state), &expected,
               sizeof(expected)) == 0) {
      return true;
    }
  }
  return false;
}

static bool valid_transport(camera_preview_transport_t transport) {
  return transport == CAMERA_PREVIEW_TRANSPORT_TCP ||
         transport == CAMERA_PREVIEW_TRANSPORT_UDP;
}

static bool valid_mode(cpx_wifi_mode_t mode) {
  return mode == CPX_WIFI_MODE_STATION || mode == CPX_WIFI_MODE_ACCESS_POINT;
}

static bool bounded_string_length(const char *value, size_t max_length,
                                  bool allow_empty, size_t *length_out) {
  size_t length;
  if (value == NULL) {
    if (length_out != NULL) {
      *length_out = 0u;
    }
    return allow_empty;
  }
  for (length = 0u; length <= max_length; length++) {
    if (value[length] == '\0') {
      if (length_out != NULL) {
        *length_out = length;
      }
      return allow_empty || length != 0u;
    }
  }
  return false;
}

static bool preview_config_valid(const camera_preview_config_t *config) {
  size_t ssid_length;
  size_t key_length;
  camera_stream_memory_plan_t camera_plan;
  jpeg_pipeline_memory_plan_t jpeg_plan;

  if (config == NULL || !valid_transport(config->transport) ||
      !valid_mode(config->wifi_mode) || config->target_fps == 0u ||
      config->target_fps > 1000u ||
      !bounded_string_length(config->ssid, CPX_WIFI_SSID_MAX_LENGTH, false,
                             &ssid_length) ||
      !bounded_string_length(config->key, CPX_WIFI_KEY_MAX_LENGTH, true,
                             &key_length) ||
      (config->borrowed_wifi_session != NULL && config->initialize_cpx) ||
      (config->wifi_mode == CPX_WIFI_MODE_ACCESS_POINT && key_length != 0u &&
       key_length < 8u) ||
      config->camera.sensor_mode != CAMERA_STREAM_SENSOR_QVGA_60 ||
      config->camera.policy != CAMERA_STREAM_POLICY_PREVIEW_LIVE ||
      (config->camera.capture_buffer_count != 0u &&
       config->camera.capture_buffer_count != 1u) ||
      config->jpeg.width != CAMERA_PREVIEW_DEFAULT_WIDTH ||
      config->jpeg.height != CAMERA_PREVIEW_DEFAULT_HEIGHT ||
      config->jpeg.stride != CAMERA_PREVIEW_DEFAULT_WIDTH ||
      camera_stream_memory_plan(&config->camera, &camera_plan) !=
          CAMERA_STREAM_OK ||
      jpeg_pipeline_memory_plan(&config->jpeg, &jpeg_plan) !=
          JPEG_PIPELINE_OK) {
    return false;
  }
  return true;
}

static TickType_t nonzero_delay_ticks(uint32_t milliseconds) {
  TickType_t ticks = pdMS_TO_TICKS(milliseconds);
  return milliseconds != 0u && ticks == 0u ? 1u : ticks;
}

static uint16_t clamp_u16(uint32_t value) {
  return value > UINT16_MAX ? UINT16_MAX : (uint16_t)value;
}

static bool stop_requested(const camera_preview_impl_t *preview) {
  bool stop;
  taskENTER_CRITICAL();
  stop = preview->stop_requested != 0u;
  taskEXIT_CRITICAL();
  return stop;
}

static void count_acquired(camera_preview_impl_t *preview) {
  taskENTER_CRITICAL();
  preview->stats.acquired_frames++;
  taskEXIT_CRITICAL();
}

static void count_no_client(camera_preview_impl_t *preview) {
  taskENTER_CRITICAL();
  preview->stats.no_client_drops++;
  taskEXIT_CRITICAL();
}

static void count_jpeg(camera_preview_impl_t *preview, uint32_t elapsed_us,
                       bool encoded) {
  taskENTER_CRITICAL();
  preview->stats.last_jpeg_us = elapsed_us;
  if (elapsed_us > preview->stats.max_jpeg_us) {
    preview->stats.max_jpeg_us = elapsed_us;
  }
  if (!encoded) {
    preview->stats.jpeg_drops++;
  }
  taskEXIT_CRITICAL();
}

static void count_send(camera_preview_impl_t *preview, uint32_t packets) {
  taskENTER_CRITICAL();
  if (packets == 0u) {
    preview->stats.send_failures++;
  } else {
    preview->stats.sent_frames++;
    preview->stats.sent_packets += packets;
    preview->frame_id++;
  }
  taskEXIT_CRITICAL();
}

static void preview_worker(void *argument) {
  camera_preview_impl_t *preview = (camera_preview_impl_t *)argument;
  uint32_t period_remainder = 0u;
  TickType_t next_frame = xTaskGetTickCount();

  while (!stop_requested(preview)) {
    camera_stream_frame_t frame;
    jpeg_pipeline_output_t jpeg_output;
    camera_stream_frame_ext_t metadata;
    imgstream_frame_t wire_frame;
    camera_stream_stats_t camera_stats;
    uint32_t packets;

    taskENTER_CRITICAL();
    const bool ready = preview->stream_ready != 0u;
    taskEXIT_CRITICAL();
    if (!ready) {
      vTaskDelay(1u);
      continue;
    }

    if (camera_stream_acquire(&preview->camera, &frame,
                              CAMERA_STREAM_WAIT_FOREVER) !=
        CAMERA_STREAM_OK) {
      if (!stop_requested(preview)) {
        vTaskDelay(1u);
      }
      continue;
    }
    count_acquired(preview);

    if (stop_requested(preview)) {
      (void)camera_stream_release(&preview->camera, &frame);
      break;
    }
    if (!cpx_wifi_session_client_ready(preview->wifi)) {
      count_no_client(preview);
      (void)camera_stream_release(&preview->camera, &frame);
      vTaskDelay(nonzero_delay_ticks(CAMERA_PREVIEW_CLIENT_RETRY_MS));
      continue;
    }

    const int jpeg_status =
        jpeg_pipeline_encode(&preview->jpeg, frame.data, &jpeg_output);
    count_jpeg(preview, jpeg_output.encode_us,
               jpeg_status == JPEG_PIPELINE_OK);
    if (jpeg_status != JPEG_PIPELINE_OK || stop_requested(preview)) {
      (void)camera_stream_release(&preview->camera, &frame);
      continue;
    }

    memset(&camera_stats, 0, sizeof(camera_stats));
    (void)camera_stream_get_stats(&preview->camera, &camera_stats);
    metadata.frame_id = preview->frame_id;
    metadata.timestamp_ms = frame.capture_started_ms;
    metadata.capture_ms = clamp_u16(frame.capture_duration_ms);
    metadata.preprocess_ms = 0u;
    metadata.jpeg_ms = clamp_u16((jpeg_output.encode_us + 999u) / 1000u);
    metadata.encode_ms = clamp_u16((uint32_t)metadata.capture_ms +
                                   (uint32_t)metadata.jpeg_ms);
    metadata.camera_frame_id = frame.camera_frame_id;
    metadata.camera_done_ms = frame.capture_completed_ms;
    metadata.dropped_capture_completions =
        camera_stats.dropped_capture_completions;
    metadata.dropped_ready_frames = camera_stats.dropped_ready_frames;
    metadata.dropped_while_streaming = camera_stats.dropped_while_leased;

    memset(&wire_frame, 0, sizeof(wire_frame));
    wire_frame.payload = jpeg_output.data;
    wire_frame.size = (uint32_t)jpeg_output.size_bytes;
    wire_frame.width = CAMERA_PREVIEW_DEFAULT_WIDTH;
    wire_frame.height = CAMERA_PREVIEW_DEFAULT_HEIGHT;
    wire_frame.type = CF_IMG_TYPE_JPEG;
    if (preview->config.include_frame_metadata) {
      wire_frame.ext = &metadata;
      wire_frame.ext_len = sizeof(metadata);
    }
    packets = imgstream_send(&preview->imgstream, &wire_frame);
    count_send(preview, packets);
    (void)camera_stream_release(&preview->camera, &frame);

    const uint32_t base_period_ms = 1000u / preview->config.target_fps;
    uint32_t target_period_ms = base_period_ms;
    period_remainder += 1000u % preview->config.target_fps;
    if (period_remainder >= preview->config.target_fps) {
      target_period_ms++;
      period_remainder -= preview->config.target_fps;
    }
    const TickType_t target_period_ticks =
        nonzero_delay_ticks(target_period_ms);
    // The schedule covers the whole loop, the wait for a frame included. A
    // worker a period behind restarts it rather than burst.
    next_frame += target_period_ticks;
    const TickType_t now = xTaskGetTickCount();
    if ((int32_t)(next_frame - now) > 0) {
      if (!stop_requested(preview)) {
        vTaskDelay(next_frame - now);
      }
    } else if (now - next_frame > target_period_ticks) {
      next_frame = now;
    }
  }

  taskENTER_CRITICAL();
  preview->worker_exited = 1u;
  taskEXIT_CRITICAL();
  vTaskDelete(NULL);
}

static int claim_active(camera_preview_impl_t *preview) {
  int status = CAMERA_PREVIEW_OK;
  taskENTER_CRITICAL();
  if (active_preview != NULL && active_preview != preview) {
    status = CAMERA_PREVIEW_ERR_BUSY;
  } else {
    active_preview = preview;
    preview->active_claimed = 1u;
  }
  taskEXIT_CRITICAL();
  return status;
}

static void release_active(camera_preview_impl_t *preview) {
  taskENTER_CRITICAL();
  if (active_preview == preview) {
    active_preview = NULL;
  }
  preview->active_claimed = 0u;
  taskEXIT_CRITICAL();
}

static bool internal_wifi_matches(const camera_preview_impl_t *preview) {
  const char *key = preview->config.key != NULL ? preview->config.key : "";
  return internal_wifi.transport == preview->config.transport &&
         internal_wifi.mode == preview->config.wifi_mode &&
         internal_wifi.client_settle_ms == preview->config.client_settle_ms &&
         strcmp(internal_wifi.ssid, preview->config.ssid) == 0 &&
         strcmp(internal_wifi.key, key) == 0;
}

static int prepare_wifi(camera_preview_impl_t *preview) {
  cpx_wifi_session_config_t wifi_config;
  int status;

  if (preview->config.borrowed_wifi_session != NULL) {
    cpx_wifi_session_status_t wifi_status;
    if (cpx_wifi_session_get_status(preview->config.borrowed_wifi_session,
                                    &wifi_status) != CPX_WIFI_SESSION_OK ||
        (int)wifi_status.transport != (int)preview->config.transport) {
      return CAMERA_PREVIEW_ERR_WIFI;
    }
    preview->wifi = preview->config.borrowed_wifi_session;
    return CAMERA_PREVIEW_OK;
  }

  if (!internal_wifi.initialized) {
    status = cpx_wifi_session_config_init(&wifi_config);
    if (status != CPX_WIFI_SESSION_OK) {
      return CAMERA_PREVIEW_ERR_WIFI;
    }
    wifi_config.ssid = preview->config.ssid;
    wifi_config.key = preview->config.key;
    wifi_config.transport = (cpx_wifi_transport_t)preview->config.transport;
    wifi_config.mode = preview->config.wifi_mode;
    wifi_config.client_settle_ms = preview->config.client_settle_ms;
    status = cpx_wifi_session_init(&internal_wifi.session, &wifi_config);
    if (status != CPX_WIFI_SESSION_OK) {
      return CAMERA_PREVIEW_ERR_WIFI;
    }
    memcpy(internal_wifi.ssid, preview->config.ssid,
           strlen(preview->config.ssid) + 1u);
    const char *key = preview->config.key != NULL ? preview->config.key : "";
    memcpy(internal_wifi.key, key, strlen(key) + 1u);
    internal_wifi.transport = preview->config.transport;
    internal_wifi.mode = preview->config.wifi_mode;
    internal_wifi.client_settle_ms = preview->config.client_settle_ms;
    internal_wifi.initialized = 1u;
  } else if (!internal_wifi_matches(preview)) {
    return CAMERA_PREVIEW_ERR_TRANSPORT_CONFLICT;
  }

  if (!internal_wifi.opened) {
    status = cpx_wifi_session_open(&internal_wifi.session);
    if (status != CPX_WIFI_SESSION_OK) {
      return CAMERA_PREVIEW_ERR_WIFI;
    }
    internal_wifi.opened = 1u;
  }
  if (!internal_wifi.connect_requested) {
    status = cpx_wifi_session_connect(&internal_wifi.session);
    if (status != CPX_WIFI_SESSION_OK) {
      return CAMERA_PREVIEW_ERR_WIFI;
    }
    internal_wifi.connect_requested = 1u;
  }
  preview->wifi = &internal_wifi.session;
  return CAMERA_PREVIEW_OK;
}

static int prepare_cpx(const camera_preview_impl_t *preview) {
  if (!cpx_ready && preview->config.initialize_cpx) {
    cpxInit();
  }
  /* false is the application's assertion that CPX was initialized already. */
  cpx_ready = 1u;
  return CAMERA_PREVIEW_OK;
}

static void request_worker_stop(camera_preview_impl_t *preview) {
  taskENTER_CRITICAL();
  preview->stop_requested = 1u;
  preview->stream_ready = 0u;
  taskEXIT_CRITICAL();
  if (preview->camera_started) {
    (void)camera_stream_stop(&preview->camera);
    preview->camera_started = 0u;
  }
}

static void wait_for_worker(camera_preview_impl_t *preview) {
  if (!preview->worker_created) {
    return;
  }
  while (1) {
    bool exited;
    taskENTER_CRITICAL();
    exited = preview->worker_exited != 0u;
    taskEXIT_CRITICAL();
    if (exited) {
      break;
    }
    vTaskDelay(1u);
  }
  preview->worker_created = 0u;
}

static void rollback_start(camera_preview_impl_t *preview) {
  request_worker_stop(preview);
  wait_for_worker(preview);
  if (preview->jpeg_open) {
    (void)jpeg_pipeline_close(&preview->jpeg);
    preview->jpeg_open = 0u;
  }
  if (preview->camera_open) {
    (void)camera_stream_close(&preview->camera);
    preview->camera_open = 0u;
  }
  preview->wifi = NULL;
  preview->state = CAMERA_PREVIEW_STATE_INITIALIZED;
  release_active(preview);
}

int camera_preview_config_init(camera_preview_config_t *config) {
  if (config == NULL) {
    return CAMERA_PREVIEW_ERR_INVALID_ARGUMENT;
  }
  memset(config, 0, sizeof(*config));
  if (camera_stream_config_init(&config->camera) != CAMERA_STREAM_OK ||
      jpeg_pipeline_config_init(&config->jpeg) != JPEG_PIPELINE_OK) {
    return CAMERA_PREVIEW_ERR_INVALID_STATE;
  }

  config->camera.sensor_mode = CAMERA_STREAM_SENSOR_QVGA_60;
  config->camera.policy = CAMERA_STREAM_POLICY_PREVIEW_LIVE;
  config->camera.capture_buffer_count = 1u;
  config->camera.watchdog_timeout_ms = CAMERA_PREVIEW_WATCHDOG_MS;

  config->jpeg.width = CAMERA_PREVIEW_DEFAULT_WIDTH;
  config->jpeg.height = CAMERA_PREVIEW_DEFAULT_HEIGHT;
  config->jpeg.stride = CAMERA_PREVIEW_DEFAULT_WIDTH;
  config->jpeg.encoder.path = JPEG_PATH_CLUSTER;
  config->jpeg.encoder.cores = 8;
  config->jpeg.encoder.quality = 50;
  config->jpeg.encoder.restart_interval = 0;
  config->jpeg.encoder.l1_policy = JPEG_CL_L1_RELEASE_ALL;
  config->jpeg.output_budget_bytes = CAMERA_PREVIEW_DEFAULT_JPEG_BUDGET;
  config->jpeg.segment_budget_bytes = 0u;
  config->jpeg.cluster_mode = JPEG_PIPELINE_CLUSTER_OWNED;
  config->jpeg.owned_cluster_frequency_hz =
      CAMERA_PREVIEW_CLUSTER_FREQUENCY_HZ;

  config->ssid = CAMERA_PREVIEW_INTERNAL_SSID;
  config->key = NULL;
  config->transport = CAMERA_PREVIEW_TRANSPORT_TCP;
  config->wifi_mode = CPX_WIFI_MODE_ACCESS_POINT;
  config->client_settle_ms = CPX_WIFI_SESSION_DEFAULT_SETTLE_MS;
  config->initialize_cpx = true;
  config->target_fps = CAMERA_PREVIEW_DEFAULT_TARGET_FPS;
  config->include_frame_metadata = true;
  return CAMERA_PREVIEW_OK;
}

int camera_preview_config_use_fc_jpeg(camera_preview_config_t *config) {
  if (config == NULL) {
    return CAMERA_PREVIEW_ERR_INVALID_ARGUMENT;
  }
  config->jpeg.encoder.path = JPEG_PATH_FC;
  config->jpeg.cluster_mode = JPEG_PIPELINE_CLUSTER_NONE;
  config->jpeg.borrowed_cluster = NULL;
  config->jpeg.owned_cluster_frequency_hz = 0u;
  return CAMERA_PREVIEW_OK;
}

int camera_preview_config_use_existing_cpx(camera_preview_config_t *config) {
  if (config == NULL) {
    return CAMERA_PREVIEW_ERR_INVALID_ARGUMENT;
  }
  config->initialize_cpx = false;
  return CAMERA_PREVIEW_OK;
}

int camera_preview_memory_plan(const camera_preview_config_t *config,
                               camera_preview_memory_plan_t *plan) {
  if (config == NULL || plan == NULL || !preview_config_valid(config)) {
    return CAMERA_PREVIEW_ERR_INVALID_ARGUMENT;
  }
  memset(plan, 0, sizeof(*plan));
  if (camera_stream_memory_plan(&config->camera, &plan->camera) !=
          CAMERA_STREAM_OK ||
      jpeg_pipeline_memory_plan(&config->jpeg, &plan->jpeg) !=
          JPEG_PIPELINE_OK) {
    return CAMERA_PREVIEW_ERR_INVALID_ARGUMENT;
  }
  plan->bulk_l2_bytes =
      plan->camera.capture_l2_bytes + plan->jpeg.total_l2_bytes;
  plan->object_bytes = CAMERA_PREVIEW_OBJECT_BYTES;
  /* The process-wide fallback is compiled into the facade even when this
   * particular configuration borrows a session, so always report its BSS. */
  plan->internal_wifi_object_bytes = sizeof(camera_preview_internal_wifi_t);
  plan->camera_task_stack_words = CAMERA_STREAM_TASK_STACK_WORDS;
  if (config->initialize_cpx) {
    plan->cpx_queue_payload_bytes =
        (TXQ_SIZE + TXQ_URGENT_SIZE + RXQ_SIZE) * sizeof(packet_t);
    plan->cpx_task_stack_words =
        CPX_COM_TASK_STACK_WORDS + CPX_RX_TASK_STACK_WORDS;
  }
  if (config->borrowed_wifi_session == NULL) {
    plan->wifi_queue_payload_bytes =
        CPX_FUNCTION_QUEUE_LENGTH * sizeof(CPXPacket_t);
    plan->wifi_task_stack_words = CPX_WIFI_SESSION_RECEIVER_STACK_WORDS;
  }
  plan->worker_task_stack_words =
      config->worker_task_stack_words != 0u
          ? config->worker_task_stack_words
          : (uint32_t)(configMINIMAL_STACK_SIZE * 4u);
  plan->total_task_stack_words =
      plan->camera_task_stack_words + plan->cpx_task_stack_words +
      plan->wifi_task_stack_words + plan->worker_task_stack_words;
  plan->known_l2_bytes =
      plan->bulk_l2_bytes + plan->object_bytes +
      plan->internal_wifi_object_bytes + plan->cpx_queue_payload_bytes +
      plan->wifi_queue_payload_bytes +
      plan->total_task_stack_words * sizeof(StackType_t);
  return CAMERA_PREVIEW_OK;
}

int camera_preview_init(camera_preview_t *storage,
                        const camera_preview_config_t *config) {
  camera_preview_impl_t *preview;
  camera_preview_memory_plan_t plan;
  size_t ssid_length;
  size_t key_length;

  if (storage == NULL || config == NULL || storage_looks_active(storage) ||
      camera_preview_memory_plan(config, &plan) != CAMERA_PREVIEW_OK ||
      !bounded_string_length(config->ssid, CPX_WIFI_SSID_MAX_LENGTH, false,
                             &ssid_length) ||
      !bounded_string_length(config->key, CPX_WIFI_KEY_MAX_LENGTH, true,
                             &key_length)) {
    return storage != NULL && storage_looks_active(storage)
               ? CAMERA_PREVIEW_ERR_BUSY
               : CAMERA_PREVIEW_ERR_INVALID_ARGUMENT;
  }

  memset(storage, 0, sizeof(*storage));
  preview = preview_impl(storage);
  preview->magic = CAMERA_PREVIEW_MAGIC;
  preview->self = (uintptr_t)preview;
  preview->state = CAMERA_PREVIEW_STATE_INITIALIZED;
  preview->config = *config;
  memcpy(preview->ssid, config->ssid, ssid_length + 1u);
  if (config->key != NULL) {
    memcpy(preview->key, config->key, key_length + 1u);
  }
  preview->config.ssid = preview->ssid;
  preview->config.key = preview->key[0] != '\0' ? preview->key : NULL;

  if (camera_stream_init(&preview->camera, &preview->config.camera) !=
      CAMERA_STREAM_OK) {
    memset(storage, 0, sizeof(*storage));
    return CAMERA_PREVIEW_ERR_CAMERA;
  }
  if (jpeg_pipeline_init(&preview->jpeg, &preview->config.jpeg) !=
      JPEG_PIPELINE_OK) {
    memset(storage, 0, sizeof(*storage));
    return CAMERA_PREVIEW_ERR_JPEG;
  }
  return CAMERA_PREVIEW_OK;
}

int camera_preview_start(camera_preview_t *storage) {
  camera_preview_impl_t *preview;
  uint32_t stack_words;
  uint32_t priority;
  int status;

  if (storage == NULL) {
    return CAMERA_PREVIEW_ERR_INVALID_ARGUMENT;
  }
  preview = preview_impl(storage);
  if (!preview_is_initialized(preview) ||
      (preview->state != CAMERA_PREVIEW_STATE_INITIALIZED &&
       preview->state != CAMERA_PREVIEW_STATE_CLOSED)) {
    return preview_is_initialized(preview) &&
                   preview->state == CAMERA_PREVIEW_STATE_RUNNING
               ? CAMERA_PREVIEW_ERR_BUSY
               : CAMERA_PREVIEW_ERR_INVALID_STATE;
  }
  status = claim_active(preview);
  if (status != CAMERA_PREVIEW_OK) {
    return status;
  }
  preview->state = CAMERA_PREVIEW_STATE_STARTING;
  preview->stop_requested = 0u;
  preview->stream_ready = 0u;
  preview->worker_exited = 0u;

  if (camera_stream_open(&preview->camera) != CAMERA_STREAM_OK) {
    rollback_start(preview);
    return CAMERA_PREVIEW_ERR_CAMERA;
  }
  preview->camera_open = 1u;

  status = prepare_cpx(preview);
  if (status != CAMERA_PREVIEW_OK) {
    rollback_start(preview);
    return status;
  }
  status = prepare_wifi(preview);
  if (status != CAMERA_PREVIEW_OK) {
    rollback_start(preview);
    return status;
  }
  imgstream_init(&preview->imgstream, CPX_T_WIFI_HOST);

  if (jpeg_pipeline_open(&preview->jpeg) != JPEG_PIPELINE_OK) {
    rollback_start(preview);
    return CAMERA_PREVIEW_ERR_JPEG;
  }
  preview->jpeg_open = 1u;

  stack_words = preview->config.worker_task_stack_words != 0u
                    ? preview->config.worker_task_stack_words
                    : (uint32_t)(configMINIMAL_STACK_SIZE * 4u);
  priority = preview->config.worker_task_priority != 0u
                 ? preview->config.worker_task_priority
                 : (uint32_t)(tskIDLE_PRIORITY + 1u);
  preview->worker_created = 1u;
  if (xTaskCreate(preview_worker, "cam_preview", stack_words, preview, priority,
                  NULL) != pdPASS) {
    preview->worker_created = 0u;
    rollback_start(preview);
    return CAMERA_PREVIEW_ERR_RTOS;
  }

  if (camera_stream_start(&preview->camera) != CAMERA_STREAM_OK) {
    rollback_start(preview);
    return CAMERA_PREVIEW_ERR_CAMERA;
  }
  preview->camera_started = 1u;
  taskENTER_CRITICAL();
  preview->stream_ready = 1u;
  preview->state = CAMERA_PREVIEW_STATE_RUNNING;
  taskEXIT_CRITICAL();
  return CAMERA_PREVIEW_OK;
}

int camera_preview_start_default(camera_preview_t *preview,
                                 camera_preview_transport_t transport) {
  camera_preview_config_t config;
  int status;

  if (preview == NULL || !valid_transport(transport)) {
    return CAMERA_PREVIEW_ERR_INVALID_ARGUMENT;
  }
  status = camera_preview_config_init(&config);
  if (status != CAMERA_PREVIEW_OK) {
    return status;
  }
  config.transport = transport;
  status = camera_preview_init(preview, &config);
  if (status != CAMERA_PREVIEW_OK) {
    return status;
  }
  return camera_preview_start(preview);
}

int camera_preview_close(camera_preview_t *storage) {
  camera_preview_impl_t *preview;
  int jpeg_status = JPEG_PIPELINE_OK;
  int camera_status = CAMERA_STREAM_OK;

  if (storage == NULL) {
    return CAMERA_PREVIEW_ERR_INVALID_ARGUMENT;
  }
  preview = preview_impl(storage);
  if (!preview_is_initialized(preview)) {
    return CAMERA_PREVIEW_ERR_INVALID_STATE;
  }
  if (preview->state == CAMERA_PREVIEW_STATE_INITIALIZED ||
      preview->state == CAMERA_PREVIEW_STATE_CLOSED) {
    return CAMERA_PREVIEW_OK;
  }
  if (preview->state != CAMERA_PREVIEW_STATE_RUNNING &&
      preview->state != CAMERA_PREVIEW_STATE_CLOSING) {
    return CAMERA_PREVIEW_ERR_BUSY;
  }

  if (preview->state == CAMERA_PREVIEW_STATE_RUNNING) {
    taskENTER_CRITICAL();
    preview->state = CAMERA_PREVIEW_STATE_CLOSING;
    taskEXIT_CRITICAL();
    request_worker_stop(preview);
    wait_for_worker(preview);
  }

  if (preview->jpeg_open) {
    jpeg_status = jpeg_pipeline_close(&preview->jpeg);
    if (jpeg_status == JPEG_PIPELINE_OK) {
      preview->jpeg_open = 0u;
    }
  }
  if (preview->camera_open) {
    camera_status = camera_stream_close(&preview->camera);
    if (camera_status == CAMERA_STREAM_OK) {
      preview->camera_open = 0u;
    }
  }
  preview->wifi = NULL;

  if (jpeg_status != JPEG_PIPELINE_OK) {
    return CAMERA_PREVIEW_ERR_JPEG;
  }
  if (camera_status != CAMERA_STREAM_OK) {
    return CAMERA_PREVIEW_ERR_CAMERA;
  }
  preview->state = CAMERA_PREVIEW_STATE_CLOSED;
  release_active(preview);
  return CAMERA_PREVIEW_OK;
}

int camera_preview_get_stats(const camera_preview_t *storage,
                             camera_preview_stats_t *stats) {
  const camera_preview_impl_t *preview;
  if (storage == NULL || stats == NULL) {
    return CAMERA_PREVIEW_ERR_INVALID_ARGUMENT;
  }
  preview = preview_const_impl(storage);
  if (!preview_is_initialized(preview)) {
    return CAMERA_PREVIEW_ERR_INVALID_STATE;
  }
  taskENTER_CRITICAL();
  *stats = preview->stats;
  taskEXIT_CRITICAL();
  return CAMERA_PREVIEW_OK;
}

bool camera_preview_is_running(const camera_preview_t *storage) {
  const camera_preview_impl_t *preview;
  bool running;
  if (storage == NULL) {
    return false;
  }
  preview = preview_const_impl(storage);
  if (!preview_is_initialized(preview)) {
    return false;
  }
  taskENTER_CRITICAL();
  running = preview->state == CAMERA_PREVIEW_STATE_RUNNING;
  taskEXIT_CRITICAL();
  return running;
}
