#include <stdint.h>

#include "pmsis.h"

#include "cf_qconv.h"

extern void pulp_nn_conv_Ho_parallel(
    const uint8_t *input, uint8_t *im2col, const int8_t *bias,
    uint8_t *output, const int8_t *weights, int32_t *batch_norm_multiplier,
    int32_t *batch_norm_bias, uint16_t output_multiplier,
    uint16_t output_shift, uint16_t input_width, uint16_t input_height,
    uint16_t input_channels, uint16_t output_width, uint16_t output_height,
    uint16_t output_channels, uint16_t kernel_width, uint16_t kernel_height,
    uint16_t padding_top, uint16_t padding_bottom, uint16_t padding_left,
    uint16_t padding_right, uint16_t stride_width, uint16_t stride_height,
    int flag_relu, int flag_batch_norm);

/* Single-invocation team entry: called by every core of an already-forked
 * team so the benchmark adapter can place performance counters around the
 * arithmetic without measuring the fork. */
void cf_qconv_u8s8_u8_pulp_nn_team(void *argument) {
  cf_qconv_u8s8_u8_job_t *job = (cf_qconv_u8s8_u8_job_t *)argument;
  const int flag_relu = job->epilogue != CF_QCONV_U8_SHIFT_CLIP;
  const int flag_batch_norm =
      job->epilogue == CF_QCONV_U8_BATCHNORM_RELU;

  pulp_nn_conv_Ho_parallel(
      job->input, job->im2col, job->bias, job->output, job->weights,
      (int32_t *)job->batch_norm_multiplier,
      (int32_t *)job->batch_norm_bias, job->output_multiplier,
      job->output_shift, job->input_width, job->input_height,
      job->input_channels, job->output_width, job->output_height,
      job->output_channels, job->kernel_width, job->kernel_height,
      job->padding_top, job->padding_bottom, job->padding_left,
      job->padding_right, job->stride_width, job->stride_height, flag_relu,
      flag_batch_norm);
}
