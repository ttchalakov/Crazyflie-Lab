#include <stdint.h>

#include "pmsis.h"

#include "cf_qdepthwise.h"

extern void pulp_nn_depthwise_generic(
    const uint8_t *input, uint8_t *scratch, const int8_t *bias,
    uint8_t *output, const int8_t *weights, uint8_t *weight_buffer,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    uint16_t output_multiplier, uint16_t output_shift, uint16_t input_width,
    uint16_t input_height, uint16_t input_channels, uint16_t output_width,
    uint16_t output_height, uint16_t output_channels, uint16_t kernel_width,
    uint16_t kernel_height, uint16_t padding_top, uint16_t padding_bottom,
    uint16_t padding_left, uint16_t padding_right, uint16_t stride_width,
    uint16_t stride_height, int8_t flag_relu, int8_t flag_batch_norm);

uint32_t cf_qdepthwise_pulp_nn_scratch_bytes(
    uint16_t input_height, uint16_t kernel_width, uint16_t padding_top,
    uint16_t padding_bottom, uint8_t cores) {
  return (uint32_t)cores * kernel_width *
         ((uint32_t)input_height + padding_top + padding_bottom + 1u);
}

/* Single-invocation team entry: called by every core of an already-forked
 * team so the benchmark adapter can place performance counters around the
 * arithmetic without measuring the fork. */
void cf_qdepthwise_u8s8_u8_pulp_nn_team(void *argument) {
  cf_qdepthwise_u8s8_u8_job_t *job =
      (cf_qdepthwise_u8s8_u8_job_t *)argument;
  const int8_t flag_relu =
      job->epilogue != CF_QDEPTHWISE_U8_SHIFT_CLIP;
  const int8_t flag_batch_norm =
      job->epilogue == CF_QDEPTHWISE_U8_BATCHNORM_RELU;

  pulp_nn_depthwise_generic(
      job->input, job->scratch, job->bias, job->output, job->weights,
      job->weight_buffer, (int32_t *)job->batch_norm_multiplier,
      (int32_t *)job->batch_norm_bias, job->output_multiplier,
      job->output_shift, job->input_width, job->input_height,
      job->input_channels, job->output_width, job->output_height,
      job->output_channels, job->kernel_width, job->kernel_height,
      job->padding_top, job->padding_bottom, job->padding_left,
      job->padding_right, job->stride_width, job->stride_height, flag_relu,
      flag_batch_norm);
}
