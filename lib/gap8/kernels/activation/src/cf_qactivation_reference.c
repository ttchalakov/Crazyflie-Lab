#include <stdint.h>

#include "cf_qactivation.h"

void cf_qactivation_lut_reference(const cf_qactivation_lut_job_t *job) {
  uint32_t index;
  if (job == NULL || job->input == NULL || job->output == NULL ||
      job->lut == NULL) {
    return;
  }
  for (index = 0u; index < job->elements; ++index) {
    job->output[index] = job->lut[job->input[index]];
  }
}

static int32_t cf_qsoftmax_value(const cf_qsoftmax_job_t *job,
                                 uint32_t index) {
  return job->input_signed ? (int32_t)(int8_t)job->input[index]
                           : (int32_t)job->input[index];
}

void cf_qsoftmax_reference(const cf_qsoftmax_job_t *job) {
  uint16_t row;
  if (job == NULL || job->input == NULL || job->output == NULL ||
      job->exp_lut == NULL || job->rows == 0u || job->columns == 0u ||
      job->columns > CF_QSOFTMAX_MAX_COLUMNS) {
    return;
  }

  for (row = 0u; row < job->rows; ++row) {
    const uint32_t base = (uint32_t)row * job->columns;
    int32_t maximum = cf_qsoftmax_value(job, base);
    uint16_t maximum_index = 0u;
    uint32_t exponential_sum = 0u;
    uint32_t output_sum = 0u;
    uint16_t column;

    for (column = 1u; column < job->columns; ++column) {
      const int32_t value = cf_qsoftmax_value(job, base + column);
      if (value > maximum) {
        maximum = value;
        maximum_index = column;
      }
    }
    for (column = 0u; column < job->columns; ++column) {
      const uint32_t difference =
          (uint32_t)(maximum - cf_qsoftmax_value(job, base + column));
      exponential_sum += job->exp_lut[difference];
    }
    if (exponential_sum == 0u) return;
    for (column = 0u; column < job->columns; ++column) {
      const uint32_t difference =
          (uint32_t)(maximum - cf_qsoftmax_value(job, base + column));
      const uint8_t quantized = (uint8_t)(
          ((uint32_t)job->exp_lut[difference] * 255u) / exponential_sum);
      job->output[base + column] = quantized;
      output_sum += quantized;
    }
    job->output[base + maximum_index] = (uint8_t)(
        job->output[base + maximum_index] + (255u - output_sum));
  }
}
