#include <stddef.h>
#include <stdint.h>

#include "cf_qbatchnorm.h"
#include "cf_qbatchnorm_internal.h"

void cf_qbatchnorm_fixed_reference(const cf_qbatchnorm_fixed_job_t *job) {
  uint32_t index;
  if (job == NULL || job->input == NULL || job->output == NULL ||
      job->multiplier == NULL || job->bias == NULL || job->channels == 0u ||
      job->shift > 30u) {
    return;
  }
  for (index = 0u; index < job->elements; ++index) {
    const uint16_t channel = (uint16_t)(index % job->channels);
    job->output[index] = cf_qbatchnorm_apply_affine(
        job->input[index], job->multiplier[channel], job->bias[channel],
        job->shift);
  }
}

void cf_qbatchnorm_update_reference(cf_qbatchnorm_update_job_t *job) {
  uint16_t channel;
  if (job == NULL || job->input == NULL || job->gamma_q16 == NULL ||
      job->beta_q16 == NULL || job->running_mean_q16 == NULL ||
      job->running_variance_q16 == NULL || job->spatial_samples < 2u ||
      job->channels == 0u ||
      (job->mode != CF_QBATCHNORM_CALIBRATE_ONLY && job->output == NULL)) {
    return;
  }
  for (channel = 0u; channel < job->channels; ++channel) {
    int32_t mean_q16;
    uint32_t biased_variance_q16;
    uint32_t unbiased_variance_q16;
    int32_t multiplier;
    int32_t bias;
    cf_qbatchnorm_statistics(job, channel, &mean_q16, &biased_variance_q16,
                             &unbiased_variance_q16);
    if (job->mode != CF_QBATCHNORM_CALIBRATE_ONLY &&
        cf_qbatchnorm_affine_from_stats(
            mean_q16, biased_variance_q16, job->gamma_q16[channel],
            job->beta_q16[channel], job->epsilon_q16,
            job->output_zero_point, 16u, &multiplier, &bias) == 0) {
      uint32_t sample;
      for (sample = 0u; sample < job->spatial_samples; ++sample) {
        const uint32_t index = sample * job->channels + channel;
        job->output[index] = cf_qbatchnorm_apply_affine(
            job->input[index], multiplier, bias, 16u);
      }
    }
    if (job->mode == CF_QBATCHNORM_UPDATE_RUNNING ||
        job->mode == CF_QBATCHNORM_CALIBRATE_ONLY) {
      job->running_mean_q16[channel] = cf_qbatchnorm_ema_s32(
          job->running_mean_q16[channel], mean_q16, job->momentum_q16);
      if (job->spatial_samples > 1u) {
        job->running_variance_q16[channel] = cf_qbatchnorm_ema_u32(
            job->running_variance_q16[channel], unbiased_variance_q16,
            job->momentum_q16);
      }
    }
  }
}

int cf_qbatchnorm_prepare_fixed(const cf_qbatchnorm_update_job_t *job,
                                int32_t *multiplier, int32_t *bias,
                                uint8_t shift) {
  uint16_t channel;
  if (job == NULL || multiplier == NULL || bias == NULL ||
      job->gamma_q16 == NULL || job->beta_q16 == NULL ||
      job->running_mean_q16 == NULL || job->running_variance_q16 == NULL ||
      job->channels == 0u || shift > 30u) {
    return -1;
  }
  for (channel = 0u; channel < job->channels; ++channel) {
    if (cf_qbatchnorm_affine_from_stats(
            job->running_mean_q16[channel],
            job->running_variance_q16[channel], job->gamma_q16[channel],
            job->beta_q16[channel], job->epsilon_q16,
            job->output_zero_point, shift, &multiplier[channel],
            &bias[channel]) != 0) {
      return -1;
    }
  }
  return 0;
}
