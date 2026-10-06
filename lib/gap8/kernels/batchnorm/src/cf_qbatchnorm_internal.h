#ifndef CF_QBATCHNORM_INTERNAL_H
#define CF_QBATCHNORM_INTERNAL_H

#include <limits.h>
#include <stdint.h>

#include "cf_qbatchnorm.h"

static inline uint8_t cf_qbatchnorm_clip_u8(int64_t value) {
  if (value < 0) return 0u;
  if (value > 255) return 255u;
  return (uint8_t)value;
}

static inline uint64_t cf_qbatchnorm_isqrt_u64(uint64_t value) {
  uint64_t result = 0u;
  uint64_t bit = (uint64_t)1u << 62;
  while (bit > value) bit >>= 2;
  while (bit != 0u) {
    if (value >= result + bit) {
      value -= result + bit;
      result = (result >> 1) + bit;
    } else {
      result >>= 1;
    }
    bit >>= 2;
  }
  return result;
}

static inline int64_t cf_qbatchnorm_round_div_s64(int64_t numerator,
                                                   int64_t denominator) {
  if (denominator <= 0) return 0;
  if (numerator >= 0) {
    return (numerator + denominator / 2) / denominator;
  }
  return -((-numerator + denominator / 2) / denominator);
}

static inline int64_t cf_qbatchnorm_round_shift_s64(int64_t value,
                                                     uint8_t shift) {
  if (shift == 0u) return value;
  if (value >= 0) return (value + ((int64_t)1 << (shift - 1u))) >> shift;
  return -((-value + ((int64_t)1 << (shift - 1u))) >> shift);
}

static inline void cf_qbatchnorm_statistics(
    const cf_qbatchnorm_update_job_t *job, uint16_t channel,
    int32_t *mean_q16, uint32_t *biased_variance_q16,
    uint32_t *unbiased_variance_q16) {
  const uint64_t samples = job->spatial_samples;
  uint64_t sum = 0u;
  uint64_t square_sum = 0u;
  uint32_t sample;
  for (sample = 0u; sample < job->spatial_samples; ++sample) {
    const uint32_t value =
        job->input[(uint32_t)sample * job->channels + channel];
    sum += value;
    square_sum += value * value;
  }
  {
    const uint64_t variance_numerator =
        samples * square_sum - sum * sum;
    *mean_q16 = (int32_t)((sum * CF_QBATCHNORM_Q16_ONE + samples / 2u) /
                          samples);
    *biased_variance_q16 = (uint32_t)(
        (variance_numerator * CF_QBATCHNORM_Q16_ONE + samples * samples / 2u) /
        (samples * samples));
    *unbiased_variance_q16 =
        samples > 1u
            ? (uint32_t)((variance_numerator * CF_QBATCHNORM_Q16_ONE +
                          samples * (samples - 1u) / 2u) /
                         (samples * (samples - 1u)))
            : 0u;
  }
}

static inline int cf_qbatchnorm_affine_from_stats(
    int32_t mean_q16, uint32_t variance_q16, int32_t gamma_q16,
    int32_t beta_q16, uint32_t epsilon_q16, uint8_t output_zero_point,
    uint8_t shift, int32_t *multiplier, int32_t *bias) {
  const uint64_t radicand =
      ((uint64_t)variance_q16 + epsilon_q16) << 16;
  const uint64_t standard_deviation_q16 = cf_qbatchnorm_isqrt_u64(radicand);
  int64_t scale;
  int64_t intercept;
  if (standard_deviation_q16 == 0u || shift > 30u || multiplier == NULL ||
      bias == NULL) {
    return -1;
  }
  scale = cf_qbatchnorm_round_div_s64(
      (int64_t)gamma_q16 * ((int64_t)1 << shift),
      (int64_t)standard_deviation_q16);
  intercept = ((int64_t)output_zero_point << shift) +
              cf_qbatchnorm_round_shift_s64(
                  (int64_t)beta_q16 * ((int64_t)1 << shift), 16u) -
              cf_qbatchnorm_round_shift_s64((int64_t)mean_q16 * scale, 16u);
  /* Bake round-to-nearest into the affine bias. */
  if (shift != 0u) intercept += (int64_t)1 << (shift - 1u);
  if (scale < INT32_MIN || scale > INT32_MAX || intercept < INT32_MIN ||
      intercept > INT32_MAX) {
    return -1;
  }
  *multiplier = (int32_t)scale;
  *bias = (int32_t)intercept;
  return 0;
}

static inline uint8_t cf_qbatchnorm_apply_affine(uint8_t input,
                                                 int32_t multiplier,
                                                 int32_t bias,
                                                 uint8_t shift) {
  const int64_t accumulator = (int64_t)input * multiplier + bias;
  return cf_qbatchnorm_clip_u8(accumulator >> shift);
}

static inline int32_t cf_qbatchnorm_ema_s32(int32_t old_value,
                                            int32_t new_value,
                                            uint16_t momentum_q16) {
  const int64_t keep = CF_QBATCHNORM_Q16_ONE - momentum_q16;
  return (int32_t)((keep * old_value +
                    (int64_t)momentum_q16 * new_value + 32768) >>
                   16);
}

static inline uint32_t cf_qbatchnorm_ema_u32(uint32_t old_value,
                                             uint32_t new_value,
                                             uint16_t momentum_q16) {
  const uint64_t keep = CF_QBATCHNORM_Q16_ONE - momentum_q16;
  return (uint32_t)((keep * old_value +
                     (uint64_t)momentum_q16 * new_value + 32768u) >>
                    16);
}

#endif /* CF_QBATCHNORM_INTERNAL_H */
