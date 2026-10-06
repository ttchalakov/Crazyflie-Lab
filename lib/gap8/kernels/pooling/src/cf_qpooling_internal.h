#ifndef CF_QPOOLING_INTERNAL_H
#define CF_QPOOLING_INTERNAL_H

#include <stdint.h>

#include "cf_qpooling.h"

static inline uint8_t cf_qpooling_clip_u8(int64_t value) {
  if (value < 0) return 0u;
  if (value > 255) return 255u;
  return (uint8_t)value;
}

static inline uint8_t cf_qpooling_average_output(
    const cf_qpooling_job_t *job, uint32_t accumulator, uint32_t divisor) {
  int64_t value;
  if (divisor == 0u) return 0u;
  if (!job->requantize) return (uint8_t)(accumulator / divisor);
  /* DORY's ordinary uint8 pooling ranges fit a signed 32-bit product. Keep
   * that hot path on GAP8 so GCC emits the much cheaper 32-bit divide, while
   * retaining a wide fallback for larger custom multipliers/windows. */
  if (divisor <= INT32_MAX) {
    const int64_t multiplier = job->output_multiplier;
    const uint64_t magnitude =
        multiplier < 0 ? (uint64_t)(-multiplier) : (uint64_t)multiplier;
    if (magnitude == 0u ||
        accumulator <= (uint32_t)((uint64_t)INT32_MAX / magnitude)) {
      const int32_t quotient =
          magnitude == 0u
              ? 0
              : ((int32_t)accumulator * job->output_multiplier) /
                    (int32_t)divisor;
      value = (int64_t)quotient + job->output_add;
      value >>= job->output_shift;
      return cf_qpooling_clip_u8(value);
    }
  }
  value = ((int64_t)accumulator * job->output_multiplier) / divisor;
  value = (value + job->output_add) >> job->output_shift;
  return cf_qpooling_clip_u8(value);
}

static inline int32_t cf_qpooling_max_i32(int32_t a, int32_t b) {
  return a > b ? a : b;
}

static inline int32_t cf_qpooling_min_i32(int32_t a, int32_t b) {
  return a < b ? a : b;
}

static inline void cf_qpooling_fixed_window(
    const cf_qpooling_job_t *job, uint16_t output_y, uint16_t output_x,
    int32_t *start_y, int32_t *end_y, int32_t *start_x, int32_t *end_x,
    uint32_t *divisor) {
  const int32_t nominal_y =
      (int32_t)output_y * job->stride_height - job->padding_top;
  const int32_t nominal_x =
      (int32_t)output_x * job->stride_width - job->padding_left;
  *start_y = cf_qpooling_max_i32(0, nominal_y);
  *end_y = cf_qpooling_min_i32(job->input_height,
                               nominal_y + job->kernel_height);
  *start_x = cf_qpooling_max_i32(0, nominal_x);
  *end_x = cf_qpooling_min_i32(job->input_width,
                               nominal_x + job->kernel_width);
  if (job->count_include_pad) {
    *divisor = (uint32_t)job->kernel_height * job->kernel_width;
  } else {
    *divisor = (uint32_t)(*end_y - *start_y) * (*end_x - *start_x);
  }
}

static inline void cf_qpooling_adaptive_window(
    const cf_qpooling_job_t *job, uint16_t output_y, uint16_t output_x,
    int32_t *start_y, int32_t *end_y, int32_t *start_x, int32_t *end_x,
    uint32_t *divisor) {
  *start_y = ((uint32_t)output_y * job->input_height) / job->output_height;
  *end_y = ((uint32_t)(output_y + 1u) * job->input_height +
            job->output_height - 1u) /
           job->output_height;
  *start_x = ((uint32_t)output_x * job->input_width) / job->output_width;
  *end_x = ((uint32_t)(output_x + 1u) * job->input_width +
            job->output_width - 1u) /
           job->output_width;
  *divisor = (uint32_t)(*end_y - *start_y) * (*end_x - *start_x);
}

static inline void cf_qpooling_window(
    const cf_qpooling_job_t *job, uint16_t output_y, uint16_t output_x,
    int32_t *start_y, int32_t *end_y, int32_t *start_x, int32_t *end_x,
    uint32_t *divisor) {
  if (job->kind == CF_QPOOLING_ADAPTIVE_AVERAGE) {
    cf_qpooling_adaptive_window(job, output_y, output_x, start_y, end_y,
                                start_x, end_x, divisor);
  } else {
    cf_qpooling_fixed_window(job, output_y, output_x, start_y, end_y, start_x,
                             end_x, divisor);
  }
}

#endif /* CF_QPOOLING_INTERNAL_H */
