#include "cf_qlinear.h"

void cf_qlinear_u8s8_i32_reference(const cf_qlinear_u8s8_i32_job_t *job) {
  uint16_t output_index;

  if (job == 0 || job->input == 0 || job->weights == 0 ||
      job->output == 0 || job->input_dim == 0u || job->output_dim == 0u) {
    return;
  }
  if (job->weight_row_stride_bytes < job->input_dim) {
    return;
  }

  for (output_index = 0; output_index < job->output_dim; ++output_index) {
    const int8_t *weight = job->weights +
                           ((uint32_t)output_index *
                            job->weight_row_stride_bytes);
    int32_t sum = job->bias != 0 ? job->bias[output_index] : 0;
    uint16_t input_index;

    for (input_index = 0; input_index < job->input_dim; ++input_index) {
      sum += (int32_t)job->input[input_index] * (int32_t)weight[input_index];
    }
    job->output[output_index] = sum;
  }
}
