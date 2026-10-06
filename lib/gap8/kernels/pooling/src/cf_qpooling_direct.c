#include <stdint.h>

#include "pmsis.h"

#include "cf_qpooling.h"
#include "cf_qpooling_internal.h"

static void cf_qpooling_partition(uint32_t work, uint32_t core,
                                  uint32_t cores, uint32_t *start,
                                  uint32_t *end) {
  *start = (uint32_t)(((uint64_t)work * core) / cores);
  *end = (uint32_t)(((uint64_t)work * (core + 1u)) / cores);
}

static int cf_qpooling_valid_job(const cf_qpooling_job_t *job,
                                 uint32_t core) {
  return job != NULL && job->input != NULL && job->output != NULL &&
         job->input_width != 0u && job->input_height != 0u &&
         job->channels != 0u && job->output_width != 0u &&
         job->output_height != 0u && job->cores != 0u &&
         job->cores <= CF_QPOOLING_MAX_CORES && core < job->cores;
}

void cf_qpooling_spatial_team(void *argument) {
  cf_qpooling_job_t *job = (cf_qpooling_job_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t output_pixels =
      job != NULL ? (uint32_t)job->output_height * job->output_width : 0u;
  uint32_t start;
  uint32_t end;
  if (!cf_qpooling_valid_job(job, core)) return;
  cf_qpooling_partition(output_pixels, core, job->cores, &start, &end);
  {
    uint32_t output_pixel;
    for (output_pixel = start; output_pixel < end; ++output_pixel) {
      const uint16_t output_y = output_pixel / job->output_width;
      const uint16_t output_x = output_pixel % job->output_width;
      int32_t start_y;
      int32_t end_y;
      int32_t start_x;
      int32_t end_x;
      uint32_t divisor;
      uint16_t channel;
      cf_qpooling_window(job, output_y, output_x, &start_y, &end_y, &start_x,
                         &end_x, &divisor);
      for (channel = 0u; channel < job->channels; ++channel) {
        uint32_t accumulator = 0u;
        int32_t y;
        for (y = start_y; y < end_y; ++y) {
          int32_t x;
          for (x = start_x; x < end_x; ++x) {
            const uint8_t value =
                job->input[((uint32_t)y * job->input_width + x) *
                               job->channels +
                           channel];
            if (job->kind == CF_QPOOLING_MAX) {
              if (value > accumulator) accumulator = value;
            } else {
              accumulator += value;
            }
          }
        }
        job->output[output_pixel * job->channels + channel] =
            job->kind == CF_QPOOLING_MAX
                ? (uint8_t)accumulator
                : cf_qpooling_average_output(job, accumulator, divisor);
      }
    }
  }
  pi_cl_team_barrier(0);
}

void cf_qpooling_channel4_team(void *argument) {
  cf_qpooling_job_t *job = (cf_qpooling_job_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t output_pixels =
      job != NULL ? (uint32_t)job->output_height * job->output_width : 0u;
  uint32_t start;
  uint32_t end;
  if (!cf_qpooling_valid_job(job, core)) return;
  if ((job->channels & 3u) != 0u ||
      (((uintptr_t)job->input | (uintptr_t)job->output) & 3u) != 0u) {
    cf_qpooling_spatial_team(argument);
    return;
  }
  cf_qpooling_partition(output_pixels, core, job->cores, &start, &end);
  {
    uint32_t output_pixel;
    for (output_pixel = start; output_pixel < end; ++output_pixel) {
      const uint16_t output_y = output_pixel / job->output_width;
      const uint16_t output_x = output_pixel % job->output_width;
      int32_t start_y;
      int32_t end_y;
      int32_t start_x;
      int32_t end_x;
      uint32_t divisor;
      uint16_t channel;
      cf_qpooling_window(job, output_y, output_x, &start_y, &end_y, &start_x,
                         &end_x, &divisor);
      for (channel = 0u; channel < job->channels; channel += 4u) {
        if (job->kind == CF_QPOOLING_MAX) {
          v4u maximum = (v4u){0u, 0u, 0u, 0u};
          int32_t y;
          for (y = start_y; y < end_y; ++y) {
            int32_t x;
            for (x = start_x; x < end_x; ++x) {
              const v4u value = *(const v4u *)(
                  job->input +
                  ((uint32_t)y * job->input_width + x) * job->channels +
                  channel);
              maximum = __builtin_pulp_maxu4(maximum, value);
            }
          }
          *(v4u *)(job->output + output_pixel * job->channels + channel) =
              maximum;
          continue;
        }
        uint32_t accumulator0 = 0u;
        uint32_t accumulator1 = 0u;
        uint32_t accumulator2 = 0u;
        uint32_t accumulator3 = 0u;
        int32_t y;
        for (y = start_y; y < end_y; ++y) {
          int32_t x;
          for (x = start_x; x < end_x; ++x) {
            const v4u value = *(const v4u *)(
                job->input +
                ((uint32_t)y * job->input_width + x) * job->channels +
                channel);
            accumulator0 += value[0];
            accumulator1 += value[1];
            accumulator2 += value[2];
            accumulator3 += value[3];
          }
        }
        *(v4u *)(job->output + output_pixel * job->channels + channel) =
            __builtin_pulp_pack4(
                cf_qpooling_average_output(job, accumulator0, divisor),
                cf_qpooling_average_output(job, accumulator1, divisor),
                cf_qpooling_average_output(job, accumulator2, divisor),
                cf_qpooling_average_output(job, accumulator3, divisor));
      }
    }
  }
  pi_cl_team_barrier(0);
}

void cf_qglobal_avgpool_channel_team(void *argument) {
  cf_qpooling_job_t *job = (cf_qpooling_job_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t samples =
      job != NULL ? (uint32_t)job->input_height * job->input_width : 0u;
  if (!cf_qpooling_valid_job(job, core) ||
      job->kind != CF_QPOOLING_AVERAGE || job->output_width != 1u ||
      job->output_height != 1u || samples == 0u) {
    return;
  }
  if ((job->channels & 3u) == 0u &&
      (((uintptr_t)job->input | (uintptr_t)job->output) & 3u) == 0u) {
    const uint32_t channel_words = job->channels >> 2;
    uint32_t word_start;
    uint32_t word_end;
    cf_qpooling_partition(channel_words, core, job->cores, &word_start,
                           &word_end);
    {
      uint32_t word;
      for (word = word_start; word < word_end; ++word) {
        uint32_t sum0 = 0u;
        uint32_t sum1 = 0u;
        uint32_t sum2 = 0u;
        uint32_t sum3 = 0u;
        uint32_t sample;
        for (sample = 0u; sample < samples; ++sample) {
          const v4u value = *(const v4u *)(job->input +
                                           sample * job->channels +
                                           (word << 2));
          sum0 += value[0];
          sum1 += value[1];
          sum2 += value[2];
          sum3 += value[3];
        }
        *(v4u *)(job->output + (word << 2)) = __builtin_pulp_pack4(
            cf_qpooling_average_output(job, sum0, samples),
            cf_qpooling_average_output(job, sum1, samples),
            cf_qpooling_average_output(job, sum2, samples),
            cf_qpooling_average_output(job, sum3, samples));
      }
    }
  } else {
    uint32_t channel_start;
    uint32_t channel_end;
    cf_qpooling_partition(job->channels, core, job->cores, &channel_start,
                           &channel_end);
    {
      uint32_t channel;
      for (channel = channel_start; channel < channel_end; ++channel) {
        uint32_t sum = 0u;
        uint32_t sample;
        for (sample = 0u; sample < samples; ++sample) {
          sum += job->input[sample * job->channels + channel];
        }
        job->output[channel] = cf_qpooling_average_output(job, sum, samples);
      }
    }
  }
  pi_cl_team_barrier(0);
}
