#include <stdint.h>

#include "pmsis.h"

#include "cf_qdepthwise.h"

#include "cf_qdepthwise_internal.h"

#include "cf_dory_profile.h"

#define CF_QDW_ALWAYS_INLINE static inline __attribute__((always_inline))
#define CF_QDW_DOTP4(a, b, c) __builtin_pulp_sdotusp4((a), (b), (c))
#define CF_QDW_CLIP_U8(value) __builtin_pulp_clipu_r((value), 255)

extern void cf_qdepthwise_pulp_nn_generic(
    const uint8_t *input, uint8_t *scratch, const int8_t *bias,
    uint8_t *output, const int8_t *weights, uint8_t *weight_buffer,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    uint16_t output_multiplier, uint16_t output_shift, uint16_t input_width,
    uint16_t input_height, uint16_t input_channels, uint16_t output_width,
    uint16_t output_height, uint16_t output_channels, uint16_t kernel_width,
    uint16_t kernel_height, uint16_t padding_top, uint16_t padding_bottom,
    uint16_t padding_left, uint16_t padding_right, uint16_t stride_width,
    uint16_t stride_height, int8_t flag_relu, int8_t flag_batch_norm);

CF_QDW_ALWAYS_INLINE uint32_t cf_qdw_min_u32(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}

CF_QDW_ALWAYS_INLINE uint8_t cf_qdw_quantize(
    cf_qdepthwise_u8_epilogue_t mode, int32_t accumulator,
    uint16_t output_multiplier, uint16_t output_shift,
    const int32_t *batch_norm_multiplier, const int32_t *batch_norm_bias,
    uint32_t channel) {
  switch (mode) {
    case CF_QDEPTHWISE_U8_SHIFT_CLIP:
      return CF_QDW_CLIP_U8(accumulator >> output_shift);
    case CF_QDEPTHWISE_U8_REQUANT_RELU: {
      const int16_t result =
          (int16_t)(((int16_t)output_multiplier * accumulator) >>
                    output_shift);
      return CF_QDW_CLIP_U8(result);
    }
    case CF_QDEPTHWISE_U8_BATCHNORM_RELU:
      return CF_QDW_CLIP_U8(
          (batch_norm_multiplier[channel] * accumulator +
           batch_norm_bias[channel]) >>
          output_shift);
    default:
      return 0u;
  }
}

CF_QDW_ALWAYS_INLINE int32_t cf_qdw_scalar_3x3(
    const uint8_t *input_plane, const int8_t *weight, uint16_t input_width,
    uint16_t input_height, int32_t origin_y, int32_t origin_x,
    int32_t accumulator) {
  uint32_t kernel_y;
  for (kernel_y = 0u; kernel_y < 3u; ++kernel_y) {
    const int32_t input_y = origin_y + (int32_t)kernel_y;
    uint32_t kernel_x;
    if (input_y < 0 || input_y >= input_height) continue;
    for (kernel_x = 0u; kernel_x < 3u; ++kernel_x) {
      const int32_t input_x = origin_x + (int32_t)kernel_x;
      if (input_x >= 0 && input_x < input_width) {
        accumulator +=
            (int32_t)input_plane[(uint32_t)input_y * input_width +
                                 (uint32_t)input_x] *
            weight[kernel_y * 3u + kernel_x];
      }
    }
  }
  return accumulator;
}

CF_QDW_ALWAYS_INLINE int32_t cf_qdw_vector_3x3(
    const uint8_t *input_plane, uint16_t input_width, uint32_t input_y,
    uint32_t input_x, v4s weight0, v4s weight1, v4s weight2,
    int32_t accumulator) {
  const uint8_t *row = input_plane + input_y * input_width + input_x;
  accumulator = CF_QDW_DOTP4(*((const v4u *)row), weight0, accumulator);
  row += input_width;
  accumulator = CF_QDW_DOTP4(*((const v4u *)row), weight1, accumulator);
  row += input_width;
  return CF_QDW_DOTP4(*((const v4u *)row), weight2, accumulator);
}

CF_QDW_ALWAYS_INLINE void cf_qdw_store_scalar(
    const uint8_t *input_plane, uint8_t *output, const int8_t *weight,
    uint32_t output_offset, uint16_t input_width, uint16_t input_height,
    int32_t origin_y, int32_t origin_x, int32_t bias,
    cf_qdepthwise_u8_epilogue_t mode, uint16_t output_multiplier,
    uint16_t output_shift, const int32_t *batch_norm_multiplier,
    const int32_t *batch_norm_bias, uint32_t channel) {
  const int32_t sum = cf_qdw_scalar_3x3(
      input_plane, weight, input_width, input_height, origin_y, origin_x,
      bias);
  output[output_offset] = cf_qdw_quantize(
      mode, sum, output_multiplier, output_shift, batch_norm_multiplier,
      batch_norm_bias, channel);
}

/*
 * Work is partitioned by (channel, output row), not channel alone. That keeps
 * short DORY height tiles balanced while each task still streams one CHW row.
 * The stride-one interior computes two adjacent output pixels from three v4u
 * loads and six dot products. Border handling is outside that branch-free loop.
 */
CF_QDW_ALWAYS_INLINE void cf_qdepthwise_direct3x3_impl(
    const uint8_t *input, const int8_t *bias, uint8_t *output,
    const int8_t *weights, const int32_t *batch_norm_multiplier,
    const int32_t *batch_norm_bias, uint16_t output_multiplier,
    uint16_t output_shift, uint16_t input_width, uint16_t input_height,
    uint16_t output_width, uint16_t output_height, uint16_t output_channels,
    uint16_t padding_top, uint16_t padding_left, uint16_t stride_width,
    uint16_t stride_height, cf_qdepthwise_u8_epilogue_t mode) {
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t cores = (uint32_t)pi_cl_team_nb_cores();
  const uint32_t rows = (uint32_t)output_channels * output_height;
  const uint32_t chunk = (rows + cores - 1u) / cores;
  const uint32_t first = cf_qdw_min_u32(core * chunk, rows);
  const uint32_t last = cf_qdw_min_u32(first + chunk, rows);
  uint32_t task;

  for (task = first; task < last; ++task) {
    const uint32_t channel = task / output_height;
    const uint32_t output_y = task - channel * output_height;
    const uint8_t *const input_plane =
        input + channel * input_height * input_width;
    const int8_t *const weight = weights + channel * 9u;
    const int32_t initial = cf_qdepthwise_bias_at(bias, channel);
    const int32_t origin_y =
        (int32_t)output_y * stride_height - padding_top;
    const v4s weight0 = (v4s){weight[0], weight[1], weight[2], 0};
    const v4s weight1 = (v4s){weight[3], weight[4], weight[5], 0};
    const v4s weight2 = (v4s){weight[6], weight[7], weight[8], 0};
    const v4s shifted0 = (v4s){0, weight[0], weight[1], weight[2]};
    const v4s shifted1 = (v4s){0, weight[3], weight[4], weight[5]};
    const v4s shifted2 = (v4s){0, weight[6], weight[7], weight[8]};
    uint32_t interior_first = 0u;
    uint32_t interior_last = 0u;
    uint32_t output_x = 0u;

    if (origin_y >= 0 && origin_y + 2 < input_height) {
      const uint32_t numerator = (uint32_t)input_width + padding_left;
      interior_first =
          ((uint32_t)padding_left + stride_width - 1u) / stride_width;
      if (numerator >= 3u) {
        interior_last = (numerator - 3u) / stride_width + 1u;
      }
      interior_first = cf_qdw_min_u32(interior_first, output_width);
      interior_last = cf_qdw_min_u32(interior_last, output_width);
      if (interior_last < interior_first) interior_last = interior_first;
    }

    for (; output_x < interior_first; ++output_x) {
      const int32_t origin_x =
          (int32_t)output_x * stride_width - padding_left;
      const uint32_t output_offset =
          (output_y * output_width + output_x) * output_channels + channel;
      cf_qdw_store_scalar(
          input_plane, output, weight, output_offset, input_width,
          input_height, origin_y, origin_x, initial, mode, output_multiplier,
          output_shift, batch_norm_multiplier, batch_norm_bias, channel);
    }

    if (stride_width == 1u) {
      for (; output_x + 1u < interior_last; output_x += 2u) {
        const uint32_t input_x = output_x - padding_left;
        const uint8_t *row =
            input_plane + (uint32_t)origin_y * input_width + input_x;
        int32_t sum0 = initial;
        int32_t sum1 = initial;
        uint32_t output_offset;

        sum0 = CF_QDW_DOTP4(*((const v4u *)row), weight0, sum0);
        sum1 = CF_QDW_DOTP4(*((const v4u *)row), shifted0, sum1);
        row += input_width;
        sum0 = CF_QDW_DOTP4(*((const v4u *)row), weight1, sum0);
        sum1 = CF_QDW_DOTP4(*((const v4u *)row), shifted1, sum1);
        row += input_width;
        sum0 = CF_QDW_DOTP4(*((const v4u *)row), weight2, sum0);
        sum1 = CF_QDW_DOTP4(*((const v4u *)row), shifted2, sum1);

        output_offset =
            (output_y * output_width + output_x) * output_channels + channel;
        output[output_offset] = cf_qdw_quantize(
            mode, sum0, output_multiplier, output_shift,
            batch_norm_multiplier, batch_norm_bias, channel);
        output[output_offset + output_channels] = cf_qdw_quantize(
            mode, sum1, output_multiplier, output_shift,
            batch_norm_multiplier, batch_norm_bias, channel);
      }
    } else {
      for (; output_x < interior_last; ++output_x) {
        const uint32_t input_x = output_x * stride_width - padding_left;
        const uint32_t output_offset =
            (output_y * output_width + output_x) * output_channels + channel;
        int32_t sum;
        if (input_x + 3u < input_width) {
          sum = cf_qdw_vector_3x3(
              input_plane, input_width, (uint32_t)origin_y, input_x, weight0,
              weight1, weight2, initial);
        } else {
          sum = cf_qdw_scalar_3x3(
              input_plane, weight, input_width, input_height, origin_y,
              (int32_t)input_x, initial);
        }
        output[output_offset] = cf_qdw_quantize(
            mode, sum, output_multiplier, output_shift,
            batch_norm_multiplier, batch_norm_bias, channel);
      }
    }

    if (output_x < interior_last) {
      const uint32_t input_x = output_x * stride_width - padding_left;
      const uint32_t output_offset =
          (output_y * output_width + output_x) * output_channels + channel;
      int32_t sum;
      if (input_x + 3u < input_width) {
        sum = cf_qdw_vector_3x3(
            input_plane, input_width, (uint32_t)origin_y, input_x, weight0,
            weight1, weight2, initial);
      } else {
        sum = cf_qdw_scalar_3x3(input_plane, weight, input_width, input_height,
                                origin_y, (int32_t)input_x, initial);
      }
      output[output_offset] = cf_qdw_quantize(
          mode, sum, output_multiplier, output_shift, batch_norm_multiplier,
          batch_norm_bias, channel);
      ++output_x;
    }

    for (; output_x < output_width; ++output_x) {
      const int32_t origin_x =
          (int32_t)output_x * stride_width - padding_left;
      const uint32_t output_offset =
          (output_y * output_width + output_x) * output_channels + channel;
      cf_qdw_store_scalar(
          input_plane, output, weight, output_offset, input_width,
          input_height, origin_y, origin_x, initial, mode, output_multiplier,
          output_shift, batch_norm_multiplier, batch_norm_bias, channel);
    }
  }
  pi_cl_team_barrier(0);
}

void cf_qdepthwise_generic_direct(
    const uint8_t *input, uint8_t *scratch, const int8_t *bias,
    uint8_t *output, const int8_t *weights, uint8_t *weight_buffer,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    uint16_t output_multiplier, uint16_t output_shift, uint16_t input_width,
    uint16_t input_height, uint16_t input_channels, uint16_t output_width,
    uint16_t output_height, uint16_t output_channels, uint16_t kernel_width,
    uint16_t kernel_height, uint16_t padding_top, uint16_t padding_bottom,
    uint16_t padding_left, uint16_t padding_right, uint16_t stride_width,
    uint16_t stride_height, int8_t flag_relu, int8_t flag_batch_norm) {
  (void)padding_bottom;
  (void)padding_right;
  if (kernel_width != 3u || kernel_height != 3u || stride_width == 0u ||
      stride_height == 0u || input_channels != output_channels) {
    cf_qdepthwise_pulp_nn_generic(
        input, scratch, bias, output, weights, weight_buffer,
        batch_norm_multiplier, batch_norm_bias, output_multiplier,
        output_shift, input_width, input_height, input_channels, output_width,
        output_height, output_channels, kernel_width, kernel_height,
        padding_top, padding_bottom, padding_left, padding_right, stride_width,
        stride_height, flag_relu, flag_batch_norm);
    return;
  }

  if (flag_batch_norm && flag_relu) {
    cf_qdepthwise_direct3x3_impl(
        input, bias, output, weights, batch_norm_multiplier, batch_norm_bias,
        output_multiplier, output_shift, input_width, input_height,
        output_width, output_height, output_channels, padding_top,
        padding_left, stride_width, stride_height,
        CF_QDEPTHWISE_U8_BATCHNORM_RELU);
  } else if (flag_relu) {
    cf_qdepthwise_direct3x3_impl(
        input, bias, output, weights, batch_norm_multiplier, batch_norm_bias,
        output_multiplier, output_shift, input_width, input_height,
        output_width, output_height, output_channels, padding_top,
        padding_left, stride_width, stride_height,
        CF_QDEPTHWISE_U8_REQUANT_RELU);
  } else {
    cf_qdepthwise_direct3x3_impl(
        input, bias, output, weights, batch_norm_multiplier, batch_norm_bias,
        output_multiplier, output_shift, input_width, input_height,
        output_width, output_height, output_channels, padding_top,
        padding_left, stride_width, stride_height,
        CF_QDEPTHWISE_U8_SHIFT_CLIP);
  }
}

static void cf_qdepthwise_call_direct(cf_qdepthwise_u8s8_u8_job_t *job) {
  const int8_t flag_relu =
      job->epilogue != CF_QDEPTHWISE_U8_SHIFT_CLIP;
  const int8_t flag_batch_norm =
      job->epilogue == CF_QDEPTHWISE_U8_BATCHNORM_RELU;
  cf_qdepthwise_generic_direct(
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

void cf_qdepthwise_u8s8_u8_direct3x3_team(void *argument) {
  cf_qdepthwise_call_direct((cf_qdepthwise_u8s8_u8_job_t *)argument);
}

#ifdef CF_QDEPTHWISE_REPLACE_PULP_NN
void pulp_nn_depthwise_generic(
    const uint8_t *input, uint8_t *scratch, const int8_t *bias,
    uint8_t *output, const int8_t *weights, uint8_t *weight_buffer,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    uint16_t output_multiplier, uint16_t output_shift, uint16_t input_width,
    uint16_t input_height, uint16_t input_channels, uint16_t output_width,
    uint16_t output_height, uint16_t output_channels, uint16_t kernel_width,
    uint16_t kernel_height, uint16_t padding_top, uint16_t padding_bottom,
    uint16_t padding_left, uint16_t padding_right, uint16_t stride_width,
    uint16_t stride_height, int8_t flag_relu, int8_t flag_batch_norm) {
  /* Instrumented for the same reason the pointwise entry point is: an
   * uninstrumented kernel contributes zero to the compute side of the split,
   * and zero is a claim. Six of the eleven layers of nn_bench's
   * dwpw-24x24c16-c32mp-c64mp-c96mp-out4 are depthwise or pointwise, so without
   * this the graph's arithmetic share would read roughly half of what it is.
   * The bracket sits here rather than in
   * cf_qdepthwise_generic_direct() because kernel_bench calls that one in a
   * repetition loop and would multiply-count a single DORY layer. */
  CF_DORY_PROF_BEGIN();
  cf_qdepthwise_generic_direct(
      input, scratch, bias, output, weights, weight_buffer,
      batch_norm_multiplier, batch_norm_bias, output_multiplier, output_shift,
      input_width, input_height, input_channels, output_width, output_height,
      output_channels, kernel_width, kernel_height, padding_top,
      padding_bottom, padding_left, padding_right, stride_width, stride_height,
      flag_relu, flag_batch_norm);
  CF_DORY_PROF_END();
}
#endif
