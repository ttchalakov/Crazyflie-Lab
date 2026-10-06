#include <limits.h>
#include <stdint.h>

#include "cf_qlinear.h"

static uint8_t cf_qlinear_clip_u8(int32_t value) {
  if (value < 0) {
    return 0u;
  }
  if (value > UINT8_MAX) {
    return UINT8_MAX;
  }
  return (uint8_t)value;
}

static uint8_t cf_qlinear_fused_output(
    const cf_qlinear_u8s8_u8_job_t *job, uint16_t output_index,
    int32_t accumulator) {
  switch (job->epilogue) {
    case CF_QLINEAR_U8_SHIFT_CLIP:
      return cf_qlinear_clip_u8(accumulator >> job->output_shift);
    case CF_QLINEAR_U8_REQUANT_RELU: {
      const int32_t multiplier = (int16_t)job->output_multiplier;
      const int16_t requantized =
          (int16_t)((multiplier * accumulator) >> job->output_shift);
      return cf_qlinear_clip_u8(requantized);
    }
    case CF_QLINEAR_U8_BATCHNORM_RELU: {
      const int32_t normalized =
          job->batch_norm_multiplier[output_index] * accumulator +
          job->batch_norm_bias[output_index];
      return cf_qlinear_clip_u8(normalized >> job->output_shift);
    }
    default:
      return 0u;
  }
}

void cf_qlinear_u8s8_u8_reference(const cf_qlinear_u8s8_u8_job_t *job) {
  uint16_t output_index;

  if (job == 0 || job->input == 0 || job->weights == 0 ||
      job->output == 0 || job->input_dim == 0u || job->output_dim == 0u ||
      job->weight_row_stride_bytes < job->input_dim ||
      job->output_shift < 0 || job->output_shift > 31 ||
      job->epilogue > CF_QLINEAR_U8_BATCHNORM_RELU ||
      (job->epilogue == CF_QLINEAR_U8_BATCHNORM_RELU &&
       (job->batch_norm_multiplier == 0 || job->batch_norm_bias == 0))) {
    return;
  }

  for (output_index = 0u; output_index < job->output_dim; ++output_index) {
    const int8_t *weight =
        job->weights +
        (uint32_t)output_index * job->weight_row_stride_bytes;
    int32_t sum = job->bias != 0 ? job->bias[output_index] : 0;
    uint16_t input_index;

    for (input_index = 0u; input_index < job->input_dim; ++input_index) {
      sum += (int32_t)job->input[input_index] * (int32_t)weight[input_index];
    }
    job->output[output_index] =
        cf_qlinear_fused_output(job, output_index, sum);
  }
}
