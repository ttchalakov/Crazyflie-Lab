#include <stddef.h>
#include <stdint.h>

#include "cf_qpooling.h"
#include "cf_qpooling_internal.h"

void cf_qpooling_reference(const cf_qpooling_job_t *job) {
  uint16_t output_y;
  if (job == NULL || job->input == NULL || job->output == NULL ||
      job->input_width == 0u || job->input_height == 0u ||
      job->channels == 0u || job->output_width == 0u ||
      job->output_height == 0u) {
    return;
  }
  for (output_y = 0u; output_y < job->output_height; ++output_y) {
    uint16_t output_x;
    for (output_x = 0u; output_x < job->output_width; ++output_x) {
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
        job->output[((uint32_t)output_y * job->output_width + output_x) *
                        job->channels +
                    channel] =
            job->kind == CF_QPOOLING_MAX
                ? (uint8_t)accumulator
                : cf_qpooling_average_output(job, accumulator, divisor);
      }
    }
  }
}
