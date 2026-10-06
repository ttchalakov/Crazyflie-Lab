#ifndef CF_GAP8_JPEG_PIPELINE_H
#define CF_GAP8_JPEG_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#include "jpeg_enc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  JPEG_PIPELINE_OK = 0,
  JPEG_PIPELINE_ERR_INVALID_ARGUMENT = -1,
  JPEG_PIPELINE_ERR_INVALID_STATE = -2,
  JPEG_PIPELINE_ERR_NO_MEMORY = -3,
  JPEG_PIPELINE_ERR_CLUSTER = -4,
  JPEG_PIPELINE_ERR_ENCODER = -5,
  JPEG_PIPELINE_ERR_OVERFLOW = -6,
  JPEG_PIPELINE_ERR_BUSY = -7,
} jpeg_pipeline_status_t;

typedef enum {
  /* FC encoding uses no cluster and must select this mode. */
  JPEG_PIPELINE_CLUSTER_NONE = 0,
  /* open() opens, optionally clocks, and close() closes an internal device. */
  JPEG_PIPELINE_CLUSTER_OWNED = 1,
  /* The caller keeps the supplied device open for the pipeline lifetime. */
  JPEG_PIPELINE_CLUSTER_BORROWED = 2,
} jpeg_pipeline_cluster_mode_t;

typedef struct {
  uint16_t width;
  uint16_t height;
  uint16_t stride;
  jpeg_cfg_t encoder;

  /*
   * Zero uses the rigorous whole-frame ceiling. A nonzero value is a budget;
   * memory_plan() caps it at the ceiling and reports that overflow is possible.
   */
  size_t output_budget_bytes;

  /*
   * Cluster path only, per core. Zero gives each core an equal share of the
   * resolved output budget, capped at the rigorous segment ceiling. A nonzero
   * value is an explicit per-core budget, also capped at the ceiling.
   */
  size_t segment_budget_bytes;

  jpeg_pipeline_cluster_mode_t cluster_mode;
  struct pi_device *borrowed_cluster;

  /* Owned cluster only. Zero leaves the SDK's default frequency unchanged. */
  uint32_t owned_cluster_frequency_hz;
} jpeg_pipeline_config_t;

typedef struct {
  size_t output_ceiling_bytes;
  size_t segment_ceiling_bytes;
  size_t output_buffer_bytes;
  size_t segment_buffer_bytes;
  size_t segment_pool_bytes;
  size_t total_l2_bytes;
  uint8_t segment_count;
  uint8_t output_may_overflow;
  uint8_t segments_may_overflow;
} jpeg_pipeline_memory_plan_t;

typedef struct {
  uint32_t encode_attempts;
  uint32_t encoded_frames;
  uint32_t oversize_drops;
  uint32_t last_encode_us;
  uint32_t max_encode_us;
  uint32_t total_encode_us;
  uint32_t peak_encoded_bytes;
} jpeg_pipeline_stats_t;

typedef struct {
  /* Valid until the next encode() on this object or close(). */
  const uint8_t *data;
  size_t size_bytes;
  size_t capacity_bytes;
  uint32_t encode_us;
} jpeg_pipeline_output_t;

/*
 * Opaque caller-owned storage. The expression follows jpeg_enc_t's target ABI
 * instead of reserving a worst-case fixed block for every architecture.
 */
#define JPEG_PIPELINE_OBJECT_BYTES \
  (sizeof(jpeg_enc_t) + sizeof(struct pi_device) + 256u)

typedef union {
  uint64_t _alignment;
  uint8_t _storage[JPEG_PIPELINE_OBJECT_BYTES];
} jpeg_pipeline_t;

/*
 * The object must not be copied after init(). Lifecycle calls must be serialized
 * by the application; encode() itself rejects overlapping callers, and close()
 * returns BUSY while an encode is in flight.
 */

/* A valid QVGA, quality-50, 64 KiB-budget FC configuration. */
int jpeg_pipeline_config_init(jpeg_pipeline_config_t *config);

/* Pure planning/validation: no hardware access and no allocation. */
int jpeg_pipeline_memory_plan(const jpeg_pipeline_config_t *config,
                              jpeg_pipeline_memory_plan_t *plan);

/* validate/copy only; safe before the scheduler, allocator, or cluster is up. */
int jpeg_pipeline_init(jpeg_pipeline_t *pipeline,
                       const jpeg_pipeline_config_t *config);

/* Allocates all L2 once and initializes jpeg_enc. Retryable after failure. */
int jpeg_pipeline_open(jpeg_pipeline_t *pipeline);

/*
 * Synchronous with no per-frame L2 allocation. Only one caller may encode at a
 * time. The cluster path acquires its policy-controlled L1 working set during
 * encode and may release some or all of it afterward; the FC path uses no L1.
 * JPEG_PIPELINE_ERR_OVERFLOW leaves output->data NULL and counts a dropped
 * frame. jpeg_enc reports both capacity and cluster-L1 acquisition failures as
 * a zero-sized encode, so oversize_drops is the conservative aggregate.
 */
int jpeg_pipeline_encode(jpeg_pipeline_t *pipeline, const uint8_t *gray,
                         jpeg_pipeline_output_t *output);

/*
 * The same encode in slices (config.encoder.slices), so the caller can use the
 * cluster between them, for example to serve an urgent inference:
 *
 *     jpeg_pipeline_encode_begin(p, gray);
 *     while ((left = jpeg_pipeline_encode_step(p)) > 0) {
 *       jpeg_pipeline_encode_yield(p);  // before anything that needs L1
 *       ...
 *     }
 *     status = jpeg_pipeline_encode_end(p, &output);  // also after an error
 *
 * step() returns the slices left, or a negative status. `gray` must stay
 * unchanged until end(). output->encode_us counts the slices and the merge only.
 */
int jpeg_pipeline_encode_begin(jpeg_pipeline_t *pipeline, const uint8_t *gray);
int jpeg_pipeline_encode_step(jpeg_pipeline_t *pipeline);
void jpeg_pipeline_encode_yield(jpeg_pipeline_t *pipeline);
int jpeg_pipeline_encode_end(jpeg_pipeline_t *pipeline, jpeg_pipeline_output_t *output);

/* Idempotent after init; returns BUSY if an encode is currently executing. */
int jpeg_pipeline_close(jpeg_pipeline_t *pipeline);

int jpeg_pipeline_get_plan(const jpeg_pipeline_t *pipeline,
                           jpeg_pipeline_memory_plan_t *plan);
int jpeg_pipeline_get_stats(const jpeg_pipeline_t *pipeline,
                            jpeg_pipeline_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_JPEG_PIPELINE_H */
