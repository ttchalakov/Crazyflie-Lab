#include <stdint.h>

#include "pmsis.h"

#include "cf_qbatchnorm.h"
#include "cf_qbatchnorm_internal.h"

static void cf_qbatchnorm_partition(uint32_t work, uint32_t core,
                                    uint32_t cores, uint32_t *start,
                                    uint32_t *end) {
  *start = (uint32_t)(((uint64_t)work * core) / cores);
  *end = (uint32_t)(((uint64_t)work * (core + 1u)) / cores);
}

void cf_qbatchnorm_fixed_scalar_team(void *argument) {
  cf_qbatchnorm_fixed_job_t *job = (cf_qbatchnorm_fixed_job_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();
  uint32_t start;
  uint32_t end;
  if (job == NULL || job->input == NULL || job->output == NULL ||
      job->multiplier == NULL || job->bias == NULL || job->channels == 0u ||
      job->cores == 0u || job->cores > CF_QBATCHNORM_MAX_CORES ||
      core >= job->cores || job->shift > 30u) {
    return;
  }
  cf_qbatchnorm_partition(job->elements, core, job->cores, &start, &end);
  {
    uint32_t index;
    uint16_t channel = (uint16_t)(start % job->channels);
    for (index = start; index < end; ++index) {
      job->output[index] = cf_qbatchnorm_apply_affine(
          job->input[index], job->multiplier[channel], job->bias[channel],
          job->shift);
      if (++channel == job->channels) channel = 0u;
    }
  }
}

void cf_qbatchnorm_fixed_channel4_team(void *argument) {
  cf_qbatchnorm_fixed_job_t *job = (cf_qbatchnorm_fixed_job_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();
  uint32_t pixel_start;
  uint32_t pixel_end;
  uint32_t pixels;
  if (job == NULL || job->input == NULL || job->output == NULL ||
      job->multiplier == NULL || job->bias == NULL || job->channels == 0u ||
      job->elements % job->channels != 0u || job->cores == 0u ||
      job->cores > CF_QBATCHNORM_MAX_CORES || core >= job->cores ||
      job->shift > 30u) {
    return;
  }
  if ((job->channels & 3u) != 0u ||
      (((uintptr_t)job->input | (uintptr_t)job->output) & 3u) != 0u) {
    cf_qbatchnorm_fixed_scalar_team(argument);
    return;
  }
  pixels = job->elements / job->channels;
  cf_qbatchnorm_partition(pixels, core, job->cores, &pixel_start, &pixel_end);
  {
    uint32_t pixel;
    for (pixel = pixel_start; pixel < pixel_end; ++pixel) {
      const uint32_t base = pixel * job->channels;
      uint16_t channel;
      for (channel = 0u; channel < job->channels; channel += 4u) {
        const v4u input = *(const v4u *)(job->input + base + channel);
        *(v4u *)(job->output + base + channel) = __builtin_pulp_pack4(
            cf_qbatchnorm_apply_affine(input[0], job->multiplier[channel],
                                       job->bias[channel], job->shift),
            cf_qbatchnorm_apply_affine(input[1], job->multiplier[channel + 1u],
                                       job->bias[channel + 1u], job->shift),
            cf_qbatchnorm_apply_affine(input[2], job->multiplier[channel + 2u],
                                       job->bias[channel + 2u], job->shift),
            cf_qbatchnorm_apply_affine(input[3], job->multiplier[channel + 3u],
                                       job->bias[channel + 3u], job->shift));
      }
    }
  }
}
