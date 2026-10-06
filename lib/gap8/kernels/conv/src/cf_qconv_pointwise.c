#include <stdint.h>

#include "pmsis.h"

#include "cf_dory_profile.h"
#include "cf_qconv.h"
#include "cf_qconv_internal.h"

extern void pulp_nn_pointwise_HoWo_parallel(
    const uint8_t *input, uint8_t *im2col, const int8_t *bias,
    uint8_t *output, const int8_t *weights, int32_t *batch_norm_multiplier,
    int32_t *batch_norm_bias, uint16_t output_multiplier,
    uint16_t output_shift, uint16_t input_width, uint16_t input_height,
    uint16_t input_channels, uint16_t output_width, uint16_t output_height,
    uint16_t output_channels, uint16_t kernel_width, uint16_t kernel_height,
    uint16_t padding_top, uint16_t padding_bottom, uint16_t padding_left,
    uint16_t padding_right, uint16_t stride_width, uint16_t stride_height,
    int flag_relu, int flag_batch_norm);

static cf_qconv_u8_epilogue_t cf_qpointwise_mode(int flag_relu,
                                                  int flag_batch_norm) {
  if (flag_relu && flag_batch_norm) return CF_QCONV_U8_BATCHNORM_RELU;
  if (flag_relu) return CF_QCONV_U8_REQUANT_RELU;
  return CF_QCONV_U8_SHIFT_CLIP;
}


static void cf_qpointwise_flattened(
    const uint8_t *input, uint8_t *im2col, const int8_t *bias,
    uint8_t *output, const int8_t *weights, int32_t *batch_norm_multiplier,
    int32_t *batch_norm_bias, uint16_t output_multiplier,
    uint16_t output_shift, uint16_t input_width, uint16_t input_height,
    uint16_t input_channels, uint16_t output_width, uint16_t output_height,
    uint16_t output_channels, uint16_t kernel_width, uint16_t kernel_height,
    uint16_t padding_top, uint16_t padding_bottom, uint16_t padding_left,
    uint16_t padding_right, uint16_t stride_width, uint16_t stride_height,
    int flag_relu, int flag_batch_norm) {
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t cores = (uint32_t)pi_cl_team_nb_cores();
  const uint32_t pixels = (uint32_t)output_width * output_height;
  const uint32_t pairs = (pixels + 1u) >> 1;
  const uint32_t chunk = (pairs + cores - 1u) / cores;
  const uint32_t first = cf_qconv_min_u32(core * chunk, pairs);
  const uint32_t last = cf_qconv_min_u32(first + chunk, pairs);
  const cf_qconv_u8_epilogue_t mode =
      cf_qpointwise_mode(flag_relu, flag_batch_norm);
  uint32_t pair;

  (void)im2col;
  (void)input_width;
  (void)input_height;
  (void)kernel_width;
  (void)kernel_height;
  (void)padding_top;
  (void)padding_bottom;
  (void)padding_left;
  (void)padding_right;
  (void)stride_width;
  (void)stride_height;

  /*
   * cf_qconv_matmul4x2() already handles a channel count that is not a
   * multiple of four: it consumes whole words and finishes with a scalar lane
   * loop, which is how the generic convolution path runs K=63 tiles. Routing
   * unaligned channels to a per-channel scalar kernel instead cost ~3.7x at
   * team widths below eight and, because the escape hatch was compiled out
   * under CF_QCONV_DORY_INT32_BIAS, applied at every width in the overlay
   * build. There is no alignment cliff and no ABI-conditional path here now.
   */
  for (pair = first; pair < last; ++pair) {
    const uint32_t pixel0 = pair << 1;
    const uint32_t pixel1 = pixel0 + 1u;
    const uint8_t *const input0 = input + pixel0 * input_channels;
    uint8_t *const output0 = output + pixel0 * output_channels;

    if (pixel1 < pixels) {
      (void)cf_qconv_matmul4x2(
          weights, (uint8_t *)input0, output_channels, input_channels,
          output_shift, output_multiplier, batch_norm_multiplier,
          batch_norm_bias, bias, output0,
          output + pixel1 * output_channels, flag_relu, flag_batch_norm);
    } else {
      (void)cf_qconv_matmul4x1_internal(
          weights, input0, output_channels, input_channels, output_shift,
          output_multiplier, batch_norm_multiplier, batch_norm_bias, bias,
          output0, mode);
    }
  }
  pi_cl_team_barrier(0);
}

static void cf_qpointwise_call_pulp(cf_qconv_u8s8_u8_job_t *job) {
  const int flag_relu = job->epilogue != CF_QCONV_U8_SHIFT_CLIP;
  const int flag_batch_norm =
      job->epilogue == CF_QCONV_U8_BATCHNORM_RELU;
  pulp_nn_pointwise_HoWo_parallel(
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

static void cf_qpointwise_call_flattened(cf_qconv_u8s8_u8_job_t *job) {
  const int flag_relu = job->epilogue != CF_QCONV_U8_SHIFT_CLIP;
  const int flag_batch_norm =
      job->epilogue == CF_QCONV_U8_BATCHNORM_RELU;
  cf_qpointwise_flattened(
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

void cf_qconv_u8s8_u8_pointwise_flattened_team(void *argument) {
  cf_qpointwise_call_flattened((cf_qconv_u8s8_u8_job_t *)argument);
}

/* Single-invocation team entry: called by every core of an already-forked
 * team so the benchmark adapter can place performance counters around the
 * arithmetic without measuring the fork. */
void cf_qconv_u8s8_u8_pointwise_pulp_nn_team(void *argument) {
  cf_qpointwise_call_pulp((cf_qconv_u8s8_u8_job_t *)argument);
}

#ifdef CF_QCONV_REPLACE_PULP_NN
void pulp_nn_pointwise_HoWo_parallel(
    const uint8_t *input, uint8_t *im2col, const int8_t *bias,
    uint8_t *output, const int8_t *weights, int32_t *batch_norm_multiplier,
    int32_t *batch_norm_bias, uint16_t output_multiplier,
    uint16_t output_shift, uint16_t input_width, uint16_t input_height,
    uint16_t input_channels, uint16_t output_width, uint16_t output_height,
    uint16_t output_channels, uint16_t kernel_width, uint16_t kernel_height,
    uint16_t padding_top, uint16_t padding_bottom, uint16_t padding_left,
    uint16_t padding_right, uint16_t stride_width, uint16_t stride_height,
    int flag_relu, int flag_batch_norm) {
  /* Instrumented so a pointwise-only graph reports a compute fraction rather
   * than zero. Without this, `pointwise_repeat_c32_d8` -- which is *entirely*
   * this kernel -- would have read as 100% orchestration, which is the most
   * confidently wrong number the profiler could produce. */
  CF_DORY_PROF_BEGIN();
  cf_qpointwise_flattened(
      input, im2col, bias, output, weights, batch_norm_multiplier,
      batch_norm_bias, output_multiplier, output_shift, input_width,
      input_height, input_channels, output_width, output_height,
      output_channels, kernel_width, kernel_height, padding_top, padding_bottom,
      padding_left, padding_right, stride_width, stride_height, flag_relu,
      flag_batch_norm);
  CF_DORY_PROF_END();
}
#endif
