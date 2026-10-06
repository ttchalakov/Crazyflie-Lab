#include "jpeg_pipeline.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

#define JPEG_PIPELINE_MAGIC 0x4A504950u /* "JPIP" */
#define JPEG_PIPELINE_DEFAULT_OUTPUT_BUDGET (64u * 1024u)

typedef struct {
  uint32_t magic;
  uintptr_t self;
  jpeg_pipeline_config_t config;
  jpeg_pipeline_memory_plan_t plan;
  jpeg_pipeline_stats_t stats;

  jpeg_enc_t encoder;
  struct pi_device owned_cluster;
  struct pi_device *cluster;
  uint8_t *output_buffer;
  uint8_t *segment_pool;

  volatile uint8_t is_open;
  volatile uint8_t encoder_initialized;
  volatile uint8_t owned_cluster_open;
  volatile uint8_t encoding;
  volatile uint8_t lifecycle_busy;
  uint32_t slice_busy_us; /* the sliced encode's cluster time so far */
} jpeg_pipeline_impl_t;

typedef char jpeg_pipeline_object_size_check[
    (sizeof(jpeg_pipeline_impl_t) <= JPEG_PIPELINE_OBJECT_BYTES) ? 1 : -1];

static jpeg_pipeline_impl_t *jpeg_pipeline_impl(jpeg_pipeline_t *pipeline) {
  return (jpeg_pipeline_impl_t *)(void *)pipeline->_storage;
}

static const jpeg_pipeline_impl_t *jpeg_pipeline_const_impl(
    const jpeg_pipeline_t *pipeline) {
  return (const jpeg_pipeline_impl_t *)(const void *)pipeline->_storage;
}

static int jpeg_pipeline_is_initialized(const jpeg_pipeline_impl_t *pipeline) {
  return pipeline != NULL && pipeline->magic == JPEG_PIPELINE_MAGIC &&
         pipeline->self == (uintptr_t)pipeline;
}

static int jpeg_pipeline_storage_looks_active(
    const jpeg_pipeline_t *storage) {
  const jpeg_pipeline_impl_t *address = jpeg_pipeline_const_impl(storage);
  const uint32_t expected_magic = JPEG_PIPELINE_MAGIC;
  const uintptr_t expected_self = (uintptr_t)address;
  uint8_t is_open;
  uint8_t lifecycle_busy;
  uint8_t encoding;
  const uint8_t *bytes = storage->_storage;

  /* init() accepts never-initialized caller storage, so inspect only its byte
   * representation until the magic/self pair establishes the private type. */
  if (memcmp(bytes + offsetof(jpeg_pipeline_impl_t, magic), &expected_magic,
             sizeof(expected_magic)) != 0 ||
      memcmp(bytes + offsetof(jpeg_pipeline_impl_t, self), &expected_self,
             sizeof(expected_self)) != 0) {
    return 0;
  }
  memcpy(&is_open, bytes + offsetof(jpeg_pipeline_impl_t, is_open),
         sizeof(is_open));
  memcpy(&lifecycle_busy,
         bytes + offsetof(jpeg_pipeline_impl_t, lifecycle_busy),
         sizeof(lifecycle_busy));
  memcpy(&encoding, bytes + offsetof(jpeg_pipeline_impl_t, encoding),
         sizeof(encoding));
  return is_open || lifecycle_busy || encoding;
}

static size_t jpeg_pipeline_min_size(size_t left, size_t right) {
  return left < right ? left : right;
}

static int jpeg_pipeline_validate(const jpeg_pipeline_config_t *config) {
  if (config == NULL || config->width == 0 || config->height == 0 ||
      config->stride < config->width || config->encoder.cores < 1 ||
      config->encoder.cores > JPEG_CL_MAX_CORES ||
      config->encoder.quality < 1 || config->encoder.quality > 100 ||
      config->encoder.restart_interval < 0 || config->encoder.slices < 0 ||
      config->encoder.slices > JPEG_ENC_MAX_SLICES ||
      config->encoder.l1_policy < JPEG_CL_L1_KEEP_ALL ||
      config->encoder.l1_policy > JPEG_CL_L1_RELEASE_ALL) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }

  if (config->cluster_mode < JPEG_PIPELINE_CLUSTER_NONE ||
      config->cluster_mode > JPEG_PIPELINE_CLUSTER_BORROWED) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }

  if (config->encoder.path == JPEG_PATH_FC) {
    if (config->cluster_mode != JPEG_PIPELINE_CLUSTER_NONE ||
        config->borrowed_cluster != NULL) {
      return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
    }
  } else if (config->encoder.path == JPEG_PATH_CLUSTER) {
    if (config->cluster_mode == JPEG_PIPELINE_CLUSTER_NONE ||
        (config->cluster_mode == JPEG_PIPELINE_CLUSTER_BORROWED &&
         config->borrowed_cluster == NULL) ||
        (config->cluster_mode == JPEG_PIPELINE_CLUSTER_OWNED &&
         config->borrowed_cluster != NULL)) {
      return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
    }
  } else {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  return JPEG_PIPELINE_OK;
}

int jpeg_pipeline_config_init(jpeg_pipeline_config_t *config) {
  if (config == NULL) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  memset(config, 0, sizeof(*config));
  config->width = 320;
  config->height = 240;
  config->stride = 320;
  config->encoder.path = JPEG_PATH_FC;
  config->encoder.cores = 1;
  config->encoder.quality = 50;
  config->encoder.restart_interval = 0;
  config->encoder.l1_policy = JPEG_CL_L1_RELEASE_ALL;
  config->output_budget_bytes = JPEG_PIPELINE_DEFAULT_OUTPUT_BUDGET;
  config->cluster_mode = JPEG_PIPELINE_CLUSTER_NONE;
  return JPEG_PIPELINE_OK;
}

int jpeg_pipeline_memory_plan(const jpeg_pipeline_config_t *config,
                              jpeg_pipeline_memory_plan_t *plan) {
  jpeg_pipeline_memory_plan_t resolved;
  size_t fair_segment_budget;
  int status;

  if (plan == NULL) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  status = jpeg_pipeline_validate(config);
  if (status != JPEG_PIPELINE_OK) {
    return status;
  }

  memset(&resolved, 0, sizeof(resolved));
  /* A sliced frame has cores x slices restart segments, on either path. */
  const int segments = jpeg_enc_segments(&config->encoder);
  resolved.output_ceiling_bytes = jpeg_enc_out_cap(
      config->width, config->height, config->encoder.quality, segments);
  resolved.segment_ceiling_bytes = jpeg_enc_seg_cap(
      config->width, config->height, config->encoder.quality, segments);
  if (resolved.output_ceiling_bytes == 0 ||
      resolved.segment_ceiling_bytes == 0) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }

  resolved.output_buffer_bytes =
      config->output_budget_bytes == 0
          ? resolved.output_ceiling_bytes
          : jpeg_pipeline_min_size(config->output_budget_bytes,
                                   resolved.output_ceiling_bytes);
  if (resolved.output_buffer_bytes == 0) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  resolved.output_may_overflow =
      resolved.output_buffer_bytes < resolved.output_ceiling_bytes;

  if (config->encoder.path == JPEG_PATH_CLUSTER) {
    resolved.segment_count = (uint8_t)segments;
    fair_segment_budget = config->segment_budget_bytes != 0
                              ? config->segment_budget_bytes
                              : resolved.output_buffer_bytes /
                                    (size_t)resolved.segment_count;
    resolved.segment_buffer_bytes = jpeg_pipeline_min_size(
        fair_segment_budget, resolved.segment_ceiling_bytes);
    if (resolved.segment_buffer_bytes == 0 ||
        resolved.segment_buffer_bytes > SIZE_MAX / resolved.segment_count) {
      return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
    }
    resolved.segment_pool_bytes =
        resolved.segment_buffer_bytes * resolved.segment_count;
    resolved.segments_may_overflow =
        resolved.segment_buffer_bytes < resolved.segment_ceiling_bytes;
  }

  if (resolved.output_buffer_bytes > SIZE_MAX - resolved.segment_pool_bytes) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  resolved.total_l2_bytes =
      resolved.output_buffer_bytes + resolved.segment_pool_bytes;
  *plan = resolved;
  return JPEG_PIPELINE_OK;
}

int jpeg_pipeline_init(jpeg_pipeline_t *storage,
                       const jpeg_pipeline_config_t *config) {
  jpeg_pipeline_memory_plan_t plan;
  jpeg_pipeline_impl_t *pipeline;
  int status;

  if (storage == NULL) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  if (jpeg_pipeline_storage_looks_active(storage)) {
    return JPEG_PIPELINE_ERR_BUSY;
  }
  status = jpeg_pipeline_memory_plan(config, &plan);
  if (status != JPEG_PIPELINE_OK) {
    return status;
  }

  memset(storage, 0, sizeof(*storage));
  pipeline = jpeg_pipeline_impl(storage);
  pipeline->magic = JPEG_PIPELINE_MAGIC;
  pipeline->self = (uintptr_t)pipeline;
  pipeline->config = *config;
  pipeline->plan = plan;
  return JPEG_PIPELINE_OK;
}

static void jpeg_pipeline_rollback_open(jpeg_pipeline_impl_t *pipeline) {
  if (pipeline->encoder_initialized) {
    jpeg_enc_deinit(&pipeline->encoder);
    pipeline->encoder_initialized = 0;
  }
  if (pipeline->segment_pool != NULL) {
    pi_l2_free(pipeline->segment_pool, pipeline->plan.segment_pool_bytes);
    pipeline->segment_pool = NULL;
  }
  if (pipeline->output_buffer != NULL) {
    pi_l2_free(pipeline->output_buffer, pipeline->plan.output_buffer_bytes);
    pipeline->output_buffer = NULL;
  }
  if (pipeline->owned_cluster_open) {
    pi_cluster_close(&pipeline->owned_cluster);
    pipeline->owned_cluster_open = 0;
  }
  pipeline->cluster = NULL;
  taskENTER_CRITICAL();
  pipeline->is_open = 0;
  pipeline->lifecycle_busy = 0;
  taskEXIT_CRITICAL();
}

int jpeg_pipeline_open(jpeg_pipeline_t *storage) {
  jpeg_pipeline_impl_t *pipeline;
  struct pi_cluster_conf cluster_config;

  if (storage == NULL) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  pipeline = jpeg_pipeline_impl(storage);
  if (!jpeg_pipeline_is_initialized(pipeline)) {
    return JPEG_PIPELINE_ERR_INVALID_STATE;
  }
  taskENTER_CRITICAL();
  if (pipeline->is_open) {
    taskEXIT_CRITICAL();
    return JPEG_PIPELINE_OK;
  }
  if (pipeline->lifecycle_busy || pipeline->encoding) {
    taskEXIT_CRITICAL();
    return JPEG_PIPELINE_ERR_BUSY;
  }
  pipeline->lifecycle_busy = 1;
  taskEXIT_CRITICAL();

  pipeline->output_buffer =
      (uint8_t *)pmsis_l2_malloc(pipeline->plan.output_buffer_bytes);
  if (pipeline->output_buffer == NULL) {
    jpeg_pipeline_rollback_open(pipeline);
    return JPEG_PIPELINE_ERR_NO_MEMORY;
  }
  if (pipeline->plan.segment_pool_bytes != 0) {
    pipeline->segment_pool =
        (uint8_t *)pmsis_l2_malloc(pipeline->plan.segment_pool_bytes);
    if (pipeline->segment_pool == NULL) {
      jpeg_pipeline_rollback_open(pipeline);
      return JPEG_PIPELINE_ERR_NO_MEMORY;
    }
  }

  if (pipeline->config.cluster_mode == JPEG_PIPELINE_CLUSTER_BORROWED) {
    pipeline->cluster = pipeline->config.borrowed_cluster;
  } else if (pipeline->config.cluster_mode == JPEG_PIPELINE_CLUSTER_OWNED) {
    pi_cluster_conf_init(&cluster_config);
    pi_open_from_conf(&pipeline->owned_cluster, &cluster_config);
    if (pi_cluster_open(&pipeline->owned_cluster) != 0) {
      jpeg_pipeline_rollback_open(pipeline);
      return JPEG_PIPELINE_ERR_CLUSTER;
    }
    pipeline->owned_cluster_open = 1;
    pipeline->cluster = &pipeline->owned_cluster;
    if (pipeline->config.owned_cluster_frequency_hz != 0) {
      (void)pi_freq_set(PI_FREQ_DOMAIN_CL,
                        pipeline->config.owned_cluster_frequency_hz);
    }
  }

  if (jpeg_enc_init(&pipeline->encoder, &pipeline->config.encoder,
                    pipeline->config.width, pipeline->config.height,
                    pipeline->config.stride, pipeline->cluster,
                    pipeline->segment_pool,
                    pipeline->plan.segment_buffer_bytes) != 0) {
    jpeg_pipeline_rollback_open(pipeline);
    return JPEG_PIPELINE_ERR_ENCODER;
  }
  pipeline->encoder_initialized = 1;
  taskENTER_CRITICAL();
  pipeline->is_open = 1;
  pipeline->lifecycle_busy = 0;
  taskEXIT_CRITICAL();
  return JPEG_PIPELINE_OK;
}

/* Takes the encoder for one frame; both encode APIs start here. */
static int jpeg_pipeline_claim(jpeg_pipeline_impl_t *pipeline) {
  if (!jpeg_pipeline_is_initialized(pipeline)) {
    return JPEG_PIPELINE_ERR_INVALID_STATE;
  }
  taskENTER_CRITICAL();
  if (!pipeline->is_open) {
    taskEXIT_CRITICAL();
    return JPEG_PIPELINE_ERR_INVALID_STATE;
  }
  if (pipeline->lifecycle_busy || pipeline->encoding) {
    taskEXIT_CRITICAL();
    return JPEG_PIPELINE_ERR_BUSY;
  }
  pipeline->encoding = 1;
  taskEXIT_CRITICAL();
  return JPEG_PIPELINE_OK;
}

/* Counts the finished frame, hands the encoder back and fills `output`; both
 * encode APIs end here. */
static int jpeg_pipeline_finish(jpeg_pipeline_impl_t *pipeline,
                                size_t encoded_size, uint32_t elapsed_us,
                                jpeg_pipeline_output_t *output) {
  taskENTER_CRITICAL();
  pipeline->stats.encode_attempts++;
  pipeline->stats.last_encode_us = elapsed_us;
  pipeline->stats.total_encode_us += elapsed_us;
  if (elapsed_us > pipeline->stats.max_encode_us) {
    pipeline->stats.max_encode_us = elapsed_us;
  }
  if (encoded_size == 0) {
    pipeline->stats.oversize_drops++;
  } else {
    pipeline->stats.encoded_frames++;
    if (encoded_size > pipeline->stats.peak_encoded_bytes) {
      pipeline->stats.peak_encoded_bytes = (uint32_t)encoded_size;
    }
  }
  pipeline->encoding = 0;
  taskEXIT_CRITICAL();

  output->encode_us = elapsed_us;
  output->capacity_bytes = pipeline->plan.output_buffer_bytes;
  if (encoded_size == 0) {
    return JPEG_PIPELINE_ERR_OVERFLOW;
  }
  output->data = pipeline->output_buffer;
  output->size_bytes = encoded_size;
  return JPEG_PIPELINE_OK;
}

/* begin + steps + end back to back (jpeg_enc_frame()), timed as one interval. */
int jpeg_pipeline_encode(jpeg_pipeline_t *storage, const uint8_t *gray,
                         jpeg_pipeline_output_t *output) {
  jpeg_pipeline_impl_t *pipeline;
  uint32_t started_us;
  uint32_t elapsed_us;
  size_t encoded_size;
  int status;

  if (storage == NULL || gray == NULL || output == NULL) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  memset(output, 0, sizeof(*output));
  pipeline = jpeg_pipeline_impl(storage);
  status = jpeg_pipeline_claim(pipeline);
  if (status != JPEG_PIPELINE_OK) {
    return status;
  }

  started_us = pi_time_get_us();
  encoded_size = jpeg_enc_frame(&pipeline->encoder, gray,
                                pipeline->output_buffer,
                                pipeline->plan.output_buffer_bytes);
  elapsed_us = pi_time_get_us() - started_us;
  return jpeg_pipeline_finish(pipeline, encoded_size, elapsed_us, output);
}

/* The sliced encode: jpeg_pipeline_encode() split at its cluster jobs. */
int jpeg_pipeline_encode_begin(jpeg_pipeline_t *storage, const uint8_t *gray) {
  jpeg_pipeline_impl_t *pipeline;
  int status;
  if (storage == NULL || gray == NULL) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  pipeline = jpeg_pipeline_impl(storage);
  status = jpeg_pipeline_claim(pipeline);
  if (status != JPEG_PIPELINE_OK) {
    return status;
  }
  pipeline->slice_busy_us = 0;
  jpeg_enc_begin(&pipeline->encoder, gray);
  return JPEG_PIPELINE_OK;
}

int jpeg_pipeline_encode_step(jpeg_pipeline_t *storage) {
  if (storage == NULL) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  jpeg_pipeline_impl_t *pipeline = jpeg_pipeline_impl(storage);
  if (!jpeg_pipeline_is_initialized(pipeline) || !pipeline->encoding) {
    return JPEG_PIPELINE_ERR_INVALID_STATE;
  }
  const uint32_t started_us = pi_time_get_us();
  const int left = jpeg_enc_step(&pipeline->encoder);
  pipeline->slice_busy_us += pi_time_get_us() - started_us;
  return left < 0 ? JPEG_PIPELINE_ERR_CLUSTER : left;
}

void jpeg_pipeline_encode_yield(jpeg_pipeline_t *storage) {
  if (storage == NULL) {
    return;
  }
  jpeg_pipeline_impl_t *pipeline = jpeg_pipeline_impl(storage);
  if (jpeg_pipeline_is_initialized(pipeline) && pipeline->encoding) {
    jpeg_enc_yield(&pipeline->encoder);
  }
}

int jpeg_pipeline_encode_end(jpeg_pipeline_t *storage, jpeg_pipeline_output_t *output) {
  if (storage == NULL || output == NULL) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  jpeg_pipeline_impl_t *pipeline = jpeg_pipeline_impl(storage);
  memset(output, 0, sizeof(*output));
  if (!jpeg_pipeline_is_initialized(pipeline) || !pipeline->encoding) {
    return JPEG_PIPELINE_ERR_INVALID_STATE;
  }
  const uint32_t started_us = pi_time_get_us();
  const size_t encoded_size = jpeg_enc_end(&pipeline->encoder, pipeline->output_buffer,
                                           pipeline->plan.output_buffer_bytes);
  /* The encoder's own time: the slices and the merge, not what ran between. */
  const uint32_t elapsed_us = pipeline->slice_busy_us + (pi_time_get_us() - started_us);
  return jpeg_pipeline_finish(pipeline, encoded_size, elapsed_us, output);
}

int jpeg_pipeline_close(jpeg_pipeline_t *storage) {
  jpeg_pipeline_impl_t *pipeline;

  if (storage == NULL) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  pipeline = jpeg_pipeline_impl(storage);
  if (!jpeg_pipeline_is_initialized(pipeline)) {
    return JPEG_PIPELINE_ERR_INVALID_STATE;
  }
  taskENTER_CRITICAL();
  if (pipeline->encoding || pipeline->lifecycle_busy) {
    taskEXIT_CRITICAL();
    return JPEG_PIPELINE_ERR_BUSY;
  }
  if (!pipeline->is_open) {
    taskEXIT_CRITICAL();
    return JPEG_PIPELINE_OK;
  }
  pipeline->lifecycle_busy = 1;
  pipeline->is_open = 0;
  taskEXIT_CRITICAL();

  jpeg_pipeline_rollback_open(pipeline);
  return JPEG_PIPELINE_OK;
}

int jpeg_pipeline_get_plan(const jpeg_pipeline_t *storage,
                           jpeg_pipeline_memory_plan_t *plan) {
  const jpeg_pipeline_impl_t *pipeline;
  if (storage == NULL || plan == NULL) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  pipeline = jpeg_pipeline_const_impl(storage);
  if (!jpeg_pipeline_is_initialized(pipeline)) {
    return JPEG_PIPELINE_ERR_INVALID_STATE;
  }
  *plan = pipeline->plan;
  return JPEG_PIPELINE_OK;
}

int jpeg_pipeline_get_stats(const jpeg_pipeline_t *storage,
                            jpeg_pipeline_stats_t *stats) {
  const jpeg_pipeline_impl_t *pipeline;
  if (storage == NULL || stats == NULL) {
    return JPEG_PIPELINE_ERR_INVALID_ARGUMENT;
  }
  pipeline = jpeg_pipeline_const_impl(storage);
  if (!jpeg_pipeline_is_initialized(pipeline)) {
    return JPEG_PIPELINE_ERR_INVALID_STATE;
  }
  taskENTER_CRITICAL();
  *stats = pipeline->stats;
  taskEXIT_CRITICAL();
  return JPEG_PIPELINE_OK;
}
