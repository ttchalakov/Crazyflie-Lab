#include <stdint.h>

#include "pmsis.h"

#include "cf_qbatchnorm.h"
#include "cf_qbatchnorm_internal.h"

void cf_qbatchnorm_update_channel_team(void *argument) {
  cf_qbatchnorm_update_job_t *job = (cf_qbatchnorm_update_job_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();
  uint32_t channel_start;
  uint32_t channel_end;
  if (job == NULL || job->input == NULL || job->gamma_q16 == NULL ||
      job->beta_q16 == NULL || job->running_mean_q16 == NULL ||
      job->running_variance_q16 == NULL || job->spatial_samples < 2u ||
      job->channels == 0u || job->cores == 0u ||
      job->cores > CF_QBATCHNORM_MAX_CORES || core >= job->cores ||
      (job->mode != CF_QBATCHNORM_CALIBRATE_ONLY && job->output == NULL)) {
    return;
  }
  channel_start =
      (uint32_t)(((uint64_t)job->channels * core) / job->cores);
  channel_end =
      (uint32_t)(((uint64_t)job->channels * (core + 1u)) / job->cores);
  {
    uint32_t channel;
    for (channel = channel_start; channel < channel_end; ++channel) {
      int32_t mean_q16;
      uint32_t biased_variance_q16;
      uint32_t unbiased_variance_q16;
      int32_t multiplier;
      int32_t bias;
      cf_qbatchnorm_statistics(job, (uint16_t)channel, &mean_q16,
                               &biased_variance_q16,
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
}
