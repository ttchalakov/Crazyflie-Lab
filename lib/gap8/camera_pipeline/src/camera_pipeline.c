#include "camera_pipeline.h"

#include <stddef.h>
#include <string.h>

#include "pmsis.h"

#define CP_MAGIC 0x43504950u /* CPIP */
#define CP_CAMERA_WAIT_SLICE_MS 50u

/* Process-wide monotonic lease identity, matching camera_stream itself.  Keeping
 * the epoch outside caller storage prevents a stale frame from becoming valid
 * again after close() followed by init() at the same address. */
static volatile uintptr_t cp_global_lease_token;

typedef struct {
  uint32_t magic;
  uintptr_t self;
  camera_pipeline_config_t config;
  camera_stream_info_t source_info;
  camera_pipeline_memory_plan_t plan;

  uint8_t *output;
  uintptr_t active_token;
  uint8_t initialized;
  uint8_t open;
  uint8_t started;
  uint8_t transitioning;
  uint8_t closing;
  uint8_t reader_active;
  uint8_t lease_active;
} cp_impl_t;

_Static_assert(sizeof(cp_impl_t) <= CAMERA_PIPELINE_OBJECT_BYTES,
               "CAMERA_PIPELINE_OBJECT_BYTES is too small");

static cp_impl_t *cp_impl(camera_pipeline_t *pipeline) {
  return (cp_impl_t *)(void *)pipeline->_storage;
}

static int cp_is_initialized(const cp_impl_t *pipeline) {
  return pipeline != NULL && pipeline->magic == CP_MAGIC &&
         pipeline->self == (uintptr_t)pipeline && pipeline->initialized;
}

static int cp_storage_looks_active(const camera_pipeline_t *storage) {
  const cp_impl_t *address =
      (const cp_impl_t *)(const void *)storage->_storage;
  const uint32_t expected_magic = CP_MAGIC;
  const uintptr_t expected_self = (uintptr_t)address;
  uint8_t open;
  uint8_t transitioning;
  uint8_t closing;
  const uint8_t *bytes = storage->_storage;

  if (memcmp(bytes + offsetof(cp_impl_t, magic), &expected_magic,
             sizeof(expected_magic)) != 0 ||
      memcmp(bytes + offsetof(cp_impl_t, self), &expected_self,
             sizeof(expected_self)) != 0) {
    return 0;
  }
  memcpy(&open, bytes + offsetof(cp_impl_t, open), sizeof(open));
  memcpy(&transitioning, bytes + offsetof(cp_impl_t, transitioning),
         sizeof(transitioning));
  memcpy(&closing, bytes + offsetof(cp_impl_t, closing), sizeof(closing));
  return open || transitioning || closing;
}

static int cp_map_camera_status(int status) {
  switch (status) {
    case CAMERA_STREAM_OK:
      return CAMERA_PIPELINE_OK;
    case CAMERA_STREAM_ERR_INVALID_ARGUMENT:
      return CAMERA_PIPELINE_ERR_INVALID_ARGUMENT;
    case CAMERA_STREAM_ERR_NO_MEMORY:
      return CAMERA_PIPELINE_ERR_NO_MEMORY;
    case CAMERA_STREAM_ERR_RTOS:
      return CAMERA_PIPELINE_ERR_RTOS;
    case CAMERA_STREAM_ERR_TIMEOUT:
      return CAMERA_PIPELINE_ERR_TIMEOUT;
    case CAMERA_STREAM_ERR_STOPPED:
      return CAMERA_PIPELINE_ERR_STOPPED;
    case CAMERA_STREAM_ERR_BUSY:
      return CAMERA_PIPELINE_ERR_BUSY;
    case CAMERA_STREAM_ERR_STALE_FRAME:
      return CAMERA_PIPELINE_ERR_STALE_FRAME;
    case CAMERA_STREAM_ERR_CAMERA:
      return CAMERA_PIPELINE_ERR_CAMERA;
    case CAMERA_STREAM_ERR_CLOSED:
    case CAMERA_STREAM_ERR_INVALID_STATE:
    default:
      return CAMERA_PIPELINE_ERR_INVALID_STATE;
  }
}

static uint32_t cp_now_ms(void) {
  return (uint32_t)xTaskGetTickCount() * (uint32_t)portTICK_PERIOD_MS;
}

/*
 * camera_stream and camera_pipeline deliberately have independent soft-stop
 * lifecycles.  Do not pass an infinite wait through to camera_stream here:
 * doing so would leave camera_pipeline_stop() unable to wake a reader when the
 * sensor has stopped completing captures.  The event-backed camera wait normally
 * returns immediately; the bounded slice is only observable when the camera is
 * idle or faulted.
 */
static int cp_acquire_source(cp_impl_t *pipeline,
                             camera_stream_frame_t *frame,
                             uint32_t timeout_ms) {
  const uint32_t started_ms = cp_now_ms();
  while (1) {
    taskENTER_CRITICAL();
    const int publication_active =
        pipeline->open && pipeline->started && !pipeline->closing &&
        !pipeline->transitioning;
    taskEXIT_CRITICAL();
    if (!publication_active) {
      return CAMERA_PIPELINE_ERR_STOPPED;
    }

    uint32_t wait_ms = CP_CAMERA_WAIT_SLICE_MS;
    if (timeout_ms != CAMERA_STREAM_WAIT_FOREVER) {
      const uint32_t elapsed_ms = cp_now_ms() - started_ms;
      if (timeout_ms == 0) {
        wait_ms = 0;
      } else if (elapsed_ms >= timeout_ms) {
        return CAMERA_PIPELINE_ERR_TIMEOUT;
      } else {
        const uint32_t remaining_ms = timeout_ms - elapsed_ms;
        if (remaining_ms < wait_ms) {
          wait_ms = remaining_ms;
        }
      }
    }

    const int camera_status =
        camera_stream_acquire(pipeline->config.camera, frame, wait_ms);
    if (camera_status == CAMERA_STREAM_OK) {
      return CAMERA_PIPELINE_OK;
    }
    if (camera_status != CAMERA_STREAM_ERR_TIMEOUT) {
      return cp_map_camera_status(camera_status);
    }
    if (timeout_ms == 0) {
      return CAMERA_PIPELINE_ERR_TIMEOUT;
    }
    if (timeout_ms != CAMERA_STREAM_WAIT_FOREVER &&
        cp_now_ms() - started_ms >= timeout_ms) {
      return CAMERA_PIPELINE_ERR_TIMEOUT;
    }
  }
}

static int cp_plan(const camera_pipeline_config_t *config,
                   camera_stream_info_t *source,
                   camera_pipeline_memory_plan_t *plan) {
  if (config == NULL || config->camera == NULL || source == NULL ||
      plan == NULL) {
    return CAMERA_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  if ((config->output_width == 0) != (config->output_height == 0) ||
      (config->fit != DS_SQUASH && config->fit != DS_CROP)) {
    return CAMERA_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  if (camera_stream_get_info(config->camera, source) != CAMERA_STREAM_OK) {
    return CAMERA_PIPELINE_ERR_INVALID_STATE;
  }

  memset(plan, 0, sizeof(*plan));
  plan->output_width =
      config->output_width != 0 ? config->output_width : source->width;
  plan->output_height =
      config->output_height != 0 ? config->output_height : source->height;
  if (plan->output_width == source->width &&
      plan->output_height == source->height) {
    plan->is_native = 1;
    plan->output_stride = source->stride;
    plan->output_frame_bytes = source->frame_size_bytes;
    return CAMERA_PIPELINE_OK;
  }
  plan->output_stride = plan->output_width;
  plan->output_frame_bytes =
      (size_t)plan->output_width * (size_t)plan->output_height;
  plan->output_l2_bytes = plan->output_frame_bytes;
  return CAMERA_PIPELINE_OK;
}

static int cp_same_source_contract(const camera_stream_info_t *left,
                                   const camera_stream_info_t *right) {
  const camera_stream_config_t *a = &left->config;
  const camera_stream_config_t *b = &right->config;
  return a->sensor_mode == b->sensor_mode && a->policy == b->policy &&
         a->capture_buffer_count == b->capture_buffer_count &&
         a->orientation == b->orientation &&
         a->window_top == b->window_top &&
         a->sensor_settle_ms == b->sensor_settle_ms &&
         a->watchdog_timeout_ms == b->watchdog_timeout_ms &&
         left->width == right->width && left->height == right->height &&
         left->stride == right->stride &&
         left->named_fps == right->named_fps &&
         left->capture_buffer_count == right->capture_buffer_count &&
         left->frame_size_bytes == right->frame_size_bytes;
}

static int cp_same_plan(const camera_pipeline_memory_plan_t *left,
                        const camera_pipeline_memory_plan_t *right) {
  return left->is_native == right->is_native &&
         left->output_width == right->output_width &&
         left->output_height == right->output_height &&
         left->output_stride == right->output_stride &&
         left->output_frame_bytes == right->output_frame_bytes &&
         left->output_l2_bytes == right->output_l2_bytes;
}

void camera_pipeline_config_init(camera_pipeline_config_t *config) {
  if (config != NULL) {
    memset(config, 0, sizeof(*config));
    config->fit = DS_SQUASH;
  }
}

int camera_pipeline_memory_plan(const camera_pipeline_config_t *config,
                                camera_pipeline_memory_plan_t *plan) {
  camera_stream_info_t source;
  return cp_plan(config, &source, plan);
}

int camera_pipeline_init(camera_pipeline_t *storage,
                         const camera_pipeline_config_t *config) {
  camera_stream_info_t source;
  camera_pipeline_memory_plan_t plan;
  if (storage == NULL || config == NULL) {
    return CAMERA_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  if (cp_storage_looks_active(storage)) {
    return CAMERA_PIPELINE_ERR_BUSY;
  }
  const int status = cp_plan(config, &source, &plan);
  if (status != CAMERA_PIPELINE_OK) {
    return status;
  }

  memset(storage->_storage, 0, sizeof(storage->_storage));
  cp_impl_t *pipeline = cp_impl(storage);
  pipeline->magic = CP_MAGIC;
  pipeline->self = (uintptr_t)pipeline;
  pipeline->config = *config;
  pipeline->source_info = source;
  pipeline->plan = plan;
  pipeline->initialized = 1;
  return CAMERA_PIPELINE_OK;
}

static void cp_free_output(cp_impl_t *pipeline) {
  if (pipeline->output != NULL) {
    pi_l2_free(pipeline->output, pipeline->plan.output_l2_bytes);
    pipeline->output = NULL;
  }
}

int camera_pipeline_open(camera_pipeline_t *storage) {
  if (storage == NULL) {
    return CAMERA_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  cp_impl_t *pipeline = cp_impl(storage);
  if (!cp_is_initialized(pipeline)) {
    return CAMERA_PIPELINE_ERR_INVALID_STATE;
  }
  taskENTER_CRITICAL();
  if (pipeline->open || pipeline->transitioning || pipeline->closing) {
    taskEXIT_CRITICAL();
    return CAMERA_PIPELINE_ERR_INVALID_STATE;
  }
  pipeline->transitioning = 1;
  taskEXIT_CRITICAL();
  camera_stream_info_t current_source;
  camera_pipeline_memory_plan_t current_plan;
  int status = cp_plan(&pipeline->config, &current_source, &current_plan);
  if (status == CAMERA_PIPELINE_OK &&
      (!current_source.is_open ||
       !cp_same_source_contract(&pipeline->source_info, &current_source) ||
       !cp_same_plan(&pipeline->plan, &current_plan))) {
    status = CAMERA_PIPELINE_ERR_INVALID_STATE;
  }
  if (status == CAMERA_PIPELINE_OK) {
    pipeline->source_info = current_source;
    if (pipeline->plan.output_l2_bytes != 0) {
      pipeline->output =
          (uint8_t *)pmsis_l2_malloc(pipeline->plan.output_l2_bytes);
      if (pipeline->output == NULL) {
        status = CAMERA_PIPELINE_ERR_NO_MEMORY;
      }
    }
  }

  taskENTER_CRITICAL();
  pipeline->open = status == CAMERA_PIPELINE_OK;
  pipeline->transitioning = 0;
  taskEXIT_CRITICAL();
  return status;
}

int camera_pipeline_start(camera_pipeline_t *storage) {
  if (storage == NULL) {
    return CAMERA_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  cp_impl_t *pipeline = cp_impl(storage);
  if (!cp_is_initialized(pipeline)) {
    return CAMERA_PIPELINE_ERR_INVALID_STATE;
  }
  taskENTER_CRITICAL();
  if (!pipeline->open || pipeline->started || pipeline->transitioning ||
      pipeline->closing) {
    taskEXIT_CRITICAL();
    return CAMERA_PIPELINE_ERR_INVALID_STATE;
  }
  if (pipeline->lease_active || pipeline->reader_active) {
    taskEXIT_CRITICAL();
    return CAMERA_PIPELINE_ERR_BUSY;
  }
  pipeline->started = 1;
  taskEXIT_CRITICAL();
  return CAMERA_PIPELINE_OK;
}

static uintptr_t cp_new_token(void) {
  cp_global_lease_token++;
  if (cp_global_lease_token == 0) {
    cp_global_lease_token++;
  }
  return cp_global_lease_token;
}

/*
 * Fill frame from its camera lease. A resized frame is resized here, in the
 * caller's task, and the camera gets its buffer back before the caller
 * encodes or sends: only frames that are taken cost a resize.
 */
static int cp_take_source(cp_impl_t *pipeline,
                          camera_pipeline_frame_t *frame) {
  const camera_stream_frame_t *source = &frame->_source_frame;
  frame->capture_started_ms = source->capture_started_ms;
  frame->capture_completed_ms = source->capture_completed_ms;
  frame->capture_duration_ms = source->capture_duration_ms;
  frame->camera_frame_id = source->camera_frame_id;
  if (pipeline->plan.is_native) {
    frame->data = source->data;
    frame->size_bytes = source->size_bytes;
    frame->width = source->width;
    frame->height = source->height;
    frame->stride = source->stride;
    frame->_source_lease = 1;
    return CAMERA_PIPELINE_OK;
  }

  const uint32_t resize_started_ms = cp_now_ms();
  ds_resize_gray(source->data, source->width, source->height, source->stride,
                 pipeline->output, pipeline->plan.output_width,
                 pipeline->plan.output_height, pipeline->plan.output_stride,
                 pipeline->config.fit);
  frame->resize_duration_ms = cp_now_ms() - resize_started_ms;
  frame->data = pipeline->output;
  frame->size_bytes = pipeline->plan.output_frame_bytes;
  frame->width = pipeline->plan.output_width;
  frame->height = pipeline->plan.output_height;
  frame->stride = pipeline->plan.output_stride;
  return cp_map_camera_status(
      camera_stream_release(pipeline->config.camera, &frame->_source_frame));
}

int camera_pipeline_acquire(camera_pipeline_t *storage,
                            camera_pipeline_frame_t *frame,
                            uint32_t timeout_ms) {
  if (storage == NULL || frame == NULL) {
    return CAMERA_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  cp_impl_t *pipeline = cp_impl(storage);
  if (!cp_is_initialized(pipeline)) {
    return CAMERA_PIPELINE_ERR_INVALID_STATE;
  }
  taskENTER_CRITICAL();
  if (!pipeline->open || pipeline->closing || pipeline->transitioning) {
    taskEXIT_CRITICAL();
    return CAMERA_PIPELINE_ERR_INVALID_STATE;
  }
  if (!pipeline->started) {
    taskEXIT_CRITICAL();
    return CAMERA_PIPELINE_ERR_STOPPED;
  }
  if (pipeline->reader_active || pipeline->lease_active) {
    taskEXIT_CRITICAL();
    return CAMERA_PIPELINE_ERR_BUSY;
  }
  pipeline->reader_active = 1;
  taskEXIT_CRITICAL();

  memset(frame, 0, sizeof(*frame));
  int status = cp_acquire_source(pipeline, &frame->_source_frame, timeout_ms);
  if (status == CAMERA_PIPELINE_OK) {
    status = cp_take_source(pipeline, frame);
  }

  taskENTER_CRITICAL();
  const int publication_active =
      pipeline->open && pipeline->started && !pipeline->closing &&
      !pipeline->transitioning;
  if (status == CAMERA_PIPELINE_OK && publication_active) {
    pipeline->lease_active = 1;
    pipeline->active_token = cp_new_token();
    frame->token = pipeline->active_token;
    pipeline->reader_active = 0;
    taskEXIT_CRITICAL();
    return CAMERA_PIPELINE_OK;
  }
  taskEXIT_CRITICAL();

  int cleanup_status = CAMERA_PIPELINE_OK;
  if (frame->_source_lease) {
    cleanup_status = cp_map_camera_status(camera_stream_release(
        pipeline->config.camera, &frame->_source_frame));
    frame->_source_lease = 0;
  }
  taskENTER_CRITICAL();
  pipeline->reader_active = 0;
  taskEXIT_CRITICAL();
  if (status == CAMERA_PIPELINE_OK && !publication_active) {
    status = CAMERA_PIPELINE_ERR_STOPPED;
  }
  return cleanup_status != CAMERA_PIPELINE_OK ? cleanup_status : status;
}

int camera_pipeline_release(camera_pipeline_t *storage,
                            camera_pipeline_frame_t *frame) {
  if (storage == NULL || frame == NULL) {
    return CAMERA_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  cp_impl_t *pipeline = cp_impl(storage);
  if (!cp_is_initialized(pipeline)) {
    return CAMERA_PIPELINE_ERR_INVALID_STATE;
  }
  taskENTER_CRITICAL();
  if (!pipeline->open || !pipeline->lease_active || frame->token == 0 ||
      frame->token != pipeline->active_token) {
    taskEXIT_CRITICAL();
    return CAMERA_PIPELINE_ERR_STALE_FRAME;
  }
  taskEXIT_CRITICAL();

  int result = CAMERA_PIPELINE_OK;
  if (frame->_source_lease) {
    const int camera_status = camera_stream_release(
        pipeline->config.camera, &frame->_source_frame);
    if (frame->_source_frame.token != 0) {
      return cp_map_camera_status(camera_status);
    }
    frame->_source_lease = 0;
    result = cp_map_camera_status(camera_status);
  }

  taskENTER_CRITICAL();
  pipeline->lease_active = 0;
  pipeline->active_token = 0;
  taskEXIT_CRITICAL();
  frame->token = 0;
  return result;
}

/* A blocked reader sees the change within one camera wait slice. */
static void cp_wait_for_reader(cp_impl_t *pipeline) {
  while (1) {
    int active;
    taskENTER_CRITICAL();
    active = pipeline->reader_active;
    taskEXIT_CRITICAL();
    if (!active) {
      return;
    }
    vTaskDelay(1);
  }
}

int camera_pipeline_stop(camera_pipeline_t *storage) {
  if (storage == NULL) {
    return CAMERA_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  cp_impl_t *pipeline = cp_impl(storage);
  if (!cp_is_initialized(pipeline)) {
    return CAMERA_PIPELINE_ERR_INVALID_STATE;
  }
  taskENTER_CRITICAL();
  if (!pipeline->open || pipeline->closing || pipeline->transitioning) {
    taskEXIT_CRITICAL();
    return CAMERA_PIPELINE_ERR_INVALID_STATE;
  }
  if (!pipeline->started) {
    taskEXIT_CRITICAL();
    return CAMERA_PIPELINE_OK;
  }
  pipeline->transitioning = 1;
  pipeline->started = 0;
  taskEXIT_CRITICAL();

  cp_wait_for_reader(pipeline);
  taskENTER_CRITICAL();
  pipeline->transitioning = 0;
  taskEXIT_CRITICAL();
  return CAMERA_PIPELINE_OK;
}

int camera_pipeline_close(camera_pipeline_t *storage) {
  if (storage == NULL) {
    return CAMERA_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  cp_impl_t *pipeline = cp_impl(storage);
  if (!cp_is_initialized(pipeline)) {
    return CAMERA_PIPELINE_ERR_INVALID_STATE;
  }
  taskENTER_CRITICAL();
  if (pipeline->closing || pipeline->transitioning) {
    taskEXIT_CRITICAL();
    return CAMERA_PIPELINE_ERR_BUSY;
  }
  if (!pipeline->open) {
    taskEXIT_CRITICAL();
    return CAMERA_PIPELINE_OK;
  }
  if (pipeline->lease_active) {
    taskEXIT_CRITICAL();
    return CAMERA_PIPELINE_ERR_BUSY;
  }
  pipeline->closing = 1;
  pipeline->started = 0;
  taskEXIT_CRITICAL();

  cp_wait_for_reader(pipeline);
  cp_free_output(pipeline);
  taskENTER_CRITICAL();
  pipeline->open = 0;
  pipeline->closing = 0;
  taskEXIT_CRITICAL();
  return CAMERA_PIPELINE_OK;
}
