#include <stdint.h>

#include "pmsis.h"

#include "cf_qconv.h"
#include "cf_dory_profile.h"
#include "cf_qconv_internal.h"

extern void cf_qconv_conv_Ho_parallel_matmul4x2(
    const uint8_t *input, uint8_t *im2col, const int8_t *bias,
    uint8_t *output, const int8_t *weights, int32_t *batch_norm_multiplier,
    int32_t *batch_norm_bias, uint16_t output_multiplier,
    uint16_t output_shift, uint16_t input_width, uint16_t input_height,
    uint16_t input_channels, uint16_t output_width, uint16_t output_height,
    uint16_t output_channels, uint16_t kernel_width, uint16_t kernel_height,
    uint16_t padding_top, uint16_t padding_bottom, uint16_t padding_left,
    uint16_t padding_right, uint16_t stride_width, uint16_t stride_height,
    int flag_relu, int flag_batch_norm);

uint8_t *cf_qconv_matmul4x1_internal(
    const int8_t *weights, const uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    const int32_t *batch_norm_multiplier, const int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, cf_qconv_u8_epilogue_t mode) {
  const uint32_t full_channels = (uint32_t)output_channels & ~3u;
  const uint32_t words = (uint32_t)columns >> 2;
  const uint32_t tail = (uint32_t)columns & 3u;
  const uint8_t packed_store = (((uintptr_t)output & 3u) == 0u);
  uint32_t channel;

  for (channel = 0u; channel < full_channels; channel += 4u) {
    const int8_t *weight0 = weights + channel * columns;
    const int8_t *weight1 = weight0 + columns;
    const int8_t *weight2 = weight1 + columns;
    const int8_t *weight3 = weight2 + columns;
    const uint8_t *activation = input;
    int32_t sum0 = cf_qconv_bias_at(bias, channel);
    int32_t sum1 = cf_qconv_bias_at(bias, channel + 1u);
    int32_t sum2 = cf_qconv_bias_at(bias, channel + 2u);
    int32_t sum3 = cf_qconv_bias_at(bias, channel + 3u);
    uint32_t word;
    uint32_t lane;
    uint8_t out0;
    uint8_t out1;
    uint8_t out2;
    uint8_t out3;

    for (word = 0u; word < words; ++word) {
      const v4u x = *((const v4u *)activation);
      sum0 = CF_QCONV_DOTP4(x, *((const v4s *)weight0), sum0);
      sum1 = CF_QCONV_DOTP4(x, *((const v4s *)weight1), sum1);
      sum2 = CF_QCONV_DOTP4(x, *((const v4s *)weight2), sum2);
      sum3 = CF_QCONV_DOTP4(x, *((const v4s *)weight3), sum3);
      activation += 4;
      weight0 += 4;
      weight1 += 4;
      weight2 += 4;
      weight3 += 4;
    }
    for (lane = 0u; lane < tail; ++lane) {
      const int32_t x = *activation++;
      sum0 += x * *weight0++;
      sum1 += x * *weight1++;
      sum2 += x * *weight2++;
      sum3 += x * *weight3++;
    }
    out0 = cf_qconv_quantize(mode, sum0, output_multiplier, output_shift,
                             batch_norm_multiplier, batch_norm_bias, channel);
    out1 = cf_qconv_quantize(mode, sum1, output_multiplier, output_shift,
                             batch_norm_multiplier, batch_norm_bias,
                             channel + 1u);
    out2 = cf_qconv_quantize(mode, sum2, output_multiplier, output_shift,
                             batch_norm_multiplier, batch_norm_bias,
                             channel + 2u);
    out3 = cf_qconv_quantize(mode, sum3, output_multiplier, output_shift,
                             batch_norm_multiplier, batch_norm_bias,
                             channel + 3u);
    if (packed_store) {
      *((v4u *)(output + channel)) =
          __builtin_pulp_pack4(out0, out1, out2, out3);
    } else {
      output[channel] = out0;
      output[channel + 1u] = out1;
      output[channel + 2u] = out2;
      output[channel + 3u] = out3;
    }
  }
  for (; channel < output_channels; ++channel) {
    const int8_t *weight = weights + channel * columns;
    const uint8_t *activation = input;
    int32_t sum = cf_qconv_bias_at(bias, channel);
    uint32_t word;
    uint32_t lane;
    for (word = 0u; word < words; ++word) {
      sum = CF_QCONV_DOTP4(*((const v4u *)activation),
                            *((const v4s *)weight), sum);
      activation += 4;
      weight += 4;
    }
    for (lane = 0u; lane < tail; ++lane) {
      sum += (int32_t)*activation++ * *weight++;
    }
    output[channel] = cf_qconv_quantize(
        mode, sum, output_multiplier, output_shift, batch_norm_multiplier,
        batch_norm_bias, channel);
  }
  return output + output_channels;
}

static cf_qconv_u8_epilogue_t cf_qconv_epilogue_from_flags(
    int flag_relu, int flag_batch_norm) {
  if (flag_relu && flag_batch_norm) return CF_QCONV_U8_BATCHNORM_RELU;
  if (flag_relu) return CF_QCONV_U8_REQUANT_RELU;
  return CF_QCONV_U8_SHIFT_CLIP;
}

static void cf_qconv_Ho_parallel_flattened(
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
  const uint32_t columns =
      (uint32_t)kernel_width * kernel_height * input_channels;
  const uint32_t pixels = (uint32_t)output_width * output_height;
  const uint32_t pairs = (pixels + 1u) >> 1;
  const uint32_t chunk = (pairs + cores - 1u) / cores;
  const uint32_t first = cf_qconv_min_u32(core * chunk, pairs);
  const uint32_t last = cf_qconv_min_u32(first + chunk, pairs);
  uint8_t *const columns_buffer = im2col + 2u * core * columns;
  const cf_qconv_u8_epilogue_t mode =
      cf_qconv_epilogue_from_flags(flag_relu, flag_batch_norm);
  uint32_t pair;

  (void)padding_bottom;
  (void)padding_right;
  for (pair = first; pair < last; ++pair) {
    const uint32_t pixel0 = pair << 1;
    const uint32_t pixel1 = pixel0 + 1u;
    uint8_t *const output0 = output + pixel0 * output_channels;

    cf_qconv_im2col_patch(
        input, columns_buffer, pixel0, input_width, input_height,
        input_channels, output_width, kernel_width, kernel_height, padding_top,
        padding_left, stride_width, stride_height);
    if (pixel1 < pixels) {
      uint8_t *const output1 = output + pixel1 * output_channels;
      cf_qconv_im2col_patch(
          input, columns_buffer + columns, pixel1, input_width, input_height,
          input_channels, output_width, kernel_width, kernel_height,
          padding_top, padding_left, stride_width, stride_height);
      (void)cf_qconv_matmul4x2(
          weights, columns_buffer, output_channels, (uint16_t)columns,
          output_shift, output_multiplier, batch_norm_multiplier,
          batch_norm_bias, bias, output0, output1, flag_relu,
          flag_batch_norm);
    } else {
      (void)cf_qconv_matmul4x1_internal(
          weights, columns_buffer, output_channels, (uint16_t)columns,
          output_shift, output_multiplier, batch_norm_multiplier,
          batch_norm_bias, bias, output0, mode);
    }
  }
  pi_cl_team_barrier(0);
}

/*
 * Same traversal, three output pixels per block instead of two.
 *
 * The microkernel measurement said 4x3 beats 4x2 by 1.10-1.13x on pre-built
 * im2col columns. This is what tests whether that survives a real layer, where
 * the driver also has to *build* the columns -- a third patch per block is a
 * third more im2col work, and the matmul is only part of the kernel.
 *
 * The reason this is not simply switched on for the DORY overlay is the buffer.
 * PULP-NN's contract is `pIm2ColBase = im2col + 2 * core_id * columns` and
 * DORY sizes the allocation to match, in `im2col_dim` in
 * `dory/Utils/Templates_writer/Layer2D_template_writer.py`. Three columns per
 * core needs `cores * columns` more bytes than DORY laid out, and that figure
 * also feeds the tiler's L1 budget -- so it would change the tiling of every
 * convolution layer, not just this kernel's scratch. Reachable here because
 * kernel_bench and the fused executor own their own allocation; not reachable
 * from the overlay without a DORY change.
 */
static void cf_qconv_Ho_parallel_flattened3(
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
  const uint32_t columns =
      (uint32_t)kernel_width * kernel_height * input_channels;
  const uint32_t pixels = (uint32_t)output_width * output_height;
  const uint32_t triples = (pixels + 2u) / 3u;
  const uint32_t chunk = (triples + cores - 1u) / cores;
  const uint32_t first = cf_qconv_min_u32(core * chunk, triples);
  const uint32_t last = cf_qconv_min_u32(first + chunk, triples);
  /* Three columns per core. The caller owns this; see the note above. */
  uint8_t *const columns_buffer = im2col + 3u * core * columns;
  const cf_qconv_u8_epilogue_t mode =
      cf_qconv_epilogue_from_flags(flag_relu, flag_batch_norm);
  uint32_t triple;

  (void)padding_bottom;
  (void)padding_right;
  for (triple = first; triple < last; ++triple) {
    const uint32_t pixel0 = triple * 3u;
    const uint32_t available = pixels - pixel0;
    const uint32_t take = available < 3u ? available : 3u;
    uint8_t *const output0 = output + pixel0 * output_channels;
    uint32_t i;

    for (i = 0u; i < take; ++i) {
      cf_qconv_im2col_patch(
          input, columns_buffer + i * columns, pixel0 + i, input_width,
          input_height, input_channels, output_width, kernel_width,
          kernel_height, padding_top, padding_left, stride_width,
          stride_height);
    }
    /* The tail is one or two pixels and runs at most once per core, so it
     * reuses the existing kernels rather than justifying a 4x2/4x1 variant of
     * the three-pixel one. */
    if (take == 3u) {
      (void)cf_qconv_matmul4x3(
          weights, columns_buffer, output_channels, (uint16_t)columns,
          output_shift, output_multiplier, batch_norm_multiplier,
          batch_norm_bias, bias, output0, output_channels, flag_relu,
          flag_batch_norm);
    } else if (take == 2u) {
      (void)cf_qconv_matmul4x2(
          weights, columns_buffer, output_channels, (uint16_t)columns,
          output_shift, output_multiplier, batch_norm_multiplier,
          batch_norm_bias, bias, output0, output0 + output_channels,
          flag_relu, flag_batch_norm);
    } else {
      (void)cf_qconv_matmul4x1_internal(
          weights, columns_buffer, output_channels, (uint16_t)columns,
          output_shift, output_multiplier, batch_norm_multiplier,
          batch_norm_bias, bias, output0, mode);
    }
  }
  pi_cl_team_barrier(0);
}

static void cf_qconv_call_flattened3(cf_qconv_u8s8_u8_job_t *job) {
  const int flag_relu = job->epilogue != CF_QCONV_U8_SHIFT_CLIP;
  const int flag_batch_norm =
      job->epilogue == CF_QCONV_U8_BATCHNORM_RELU;
  cf_qconv_Ho_parallel_flattened3(
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

static void cf_qconv_call_matmul_variant(cf_qconv_u8s8_u8_job_t *job) {
  const int flag_relu = job->epilogue != CF_QCONV_U8_SHIFT_CLIP;
  const int flag_batch_norm =
      job->epilogue == CF_QCONV_U8_BATCHNORM_RELU;
  cf_qconv_conv_Ho_parallel_matmul4x2(
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

static void cf_qconv_call_flattened(cf_qconv_u8s8_u8_job_t *job) {
  const int flag_relu = job->epilogue != CF_QCONV_U8_SHIFT_CLIP;
  const int flag_batch_norm =
      job->epilogue == CF_QCONV_U8_BATCHNORM_RELU;
  cf_qconv_Ho_parallel_flattened(
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

void cf_qconv_u8s8_u8_matmul4x2_team(void *argument) {
  cf_qconv_call_matmul_variant((cf_qconv_u8s8_u8_job_t *)argument);
}

void cf_qconv_u8s8_u8_flattened_team(void *argument) {
  cf_qconv_call_flattened((cf_qconv_u8s8_u8_job_t *)argument);
}

void cf_qconv_u8s8_u8_flattened3_team(void *argument) {
  cf_qconv_call_flattened3((cf_qconv_u8s8_u8_job_t *)argument);
}

#ifdef CF_QCONV_REPLACE_PULP_NN
void pulp_nn_conv_Ho_parallel(
    const uint8_t *input, uint8_t *im2col, const int8_t *bias,
    uint8_t *output, const int8_t *weights, int32_t *batch_norm_multiplier,
    int32_t *batch_norm_bias, uint16_t output_multiplier,
    uint16_t output_shift, uint16_t input_width, uint16_t input_height,
    uint16_t input_channels, uint16_t output_width, uint16_t output_height,
    uint16_t output_channels, uint16_t kernel_width, uint16_t kernel_height,
    uint16_t padding_top, uint16_t padding_bottom, uint16_t padding_left,
    uint16_t padding_right, uint16_t stride_width, uint16_t stride_height,
    int flag_relu, int flag_batch_norm) {
  CF_DORY_PROF_BEGIN();
  cf_qconv_Ho_parallel_flattened(
      input, im2col, bias, output, weights, batch_norm_multiplier,
      batch_norm_bias, output_multiplier, output_shift, input_width,
      input_height, input_channels, output_width, output_height,
      output_channels, kernel_width, kernel_height, padding_top, padding_bottom,
      padding_left, padding_right, stride_width, stride_height, flag_relu,
      flag_batch_norm);
  CF_DORY_PROF_END();
}
#endif
