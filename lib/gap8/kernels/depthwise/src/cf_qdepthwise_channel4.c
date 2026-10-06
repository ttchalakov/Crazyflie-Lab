/*
 * Channel-blocked 3x3 depthwise convolution.
 *
 * Motivation, from measurement rather than principle. The existing
 * `direct3x3_2x1` kernel reaches 1.20 MAC/cycle with L1-resident operands --
 * under 4% of GAP8's 32 MAC/cycle peak -- and staging operands into L1 buys it
 * only 2.4x where convolution gets 7.6x. It is not latency-bound, so moving the
 * data closer does not help.
 *
 * The layout is why. Input is CHW, so one channel's plane is contiguous, but
 * output is HWC: for a fixed channel, consecutive output pixels land
 * `output_channels` bytes apart. Every store in the per-channel kernel is
 * therefore a single scattered byte.
 *
 * Blocking four channels inverts that. For one output pixel the four channels
 * are *adjacent* in HWC, so four scattered byte stores collapse into one
 * aligned 32-bit store. The block also gives four independent accumulator
 * chains, which hides `pv.sdotusp.b` result latency that a single chain
 * exposes, and keeps four channels' 3x3 weights live in registers across a
 * whole output row.
 *
 * Arithmetic is identical to cf_qdepthwise_direct3x3.c and to the reference:
 * same accumulation order, same epilogue. This kernel is a scheduling change.
 */
#include <stdint.h>

#include "pmsis.h"

#include "cf_qdepthwise.h"
#include "cf_qdepthwise_internal.h"

#define CF_QDW4_ALWAYS_INLINE static inline __attribute__((always_inline))
#define CF_QDW4_DOTP4(a, b, c) __builtin_pulp_sdotusp4((a), (b), (c))
#define CF_QDW4_CLIP_U8(value) __builtin_pulp_clipu_r((value), 255)

/* Channels per block. Four is not tunable: it is what makes the HWC store one
 * aligned word. */
#define CF_QDW4_BLOCK 4u

CF_QDW4_ALWAYS_INLINE uint32_t cf_qdw4_min_u32(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}

CF_QDW4_ALWAYS_INLINE uint8_t cf_qdw4_quantize(
    cf_qdepthwise_u8_epilogue_t mode, int32_t accumulator,
    uint16_t output_multiplier, uint16_t output_shift,
    const int32_t *batch_norm_multiplier, const int32_t *batch_norm_bias,
    uint32_t channel) {
  switch (mode) {
    case CF_QDEPTHWISE_U8_SHIFT_CLIP:
      return CF_QDW4_CLIP_U8(accumulator >> output_shift);
    case CF_QDEPTHWISE_U8_REQUANT_RELU: {
      const int32_t result =
          (int32_t)(((int64_t)accumulator * output_multiplier) >> output_shift);
      return CF_QDW4_CLIP_U8(result);
    }
    default:
      return CF_QDW4_CLIP_U8(
          (batch_norm_multiplier[channel] * accumulator +
           batch_norm_bias[channel]) >>
          output_shift);
  }
}

/* Scalar 3x3 with border clamping, for the padded edges of a row. */
CF_QDW4_ALWAYS_INLINE int32_t cf_qdw4_scalar_3x3(
    const uint8_t *plane, uint16_t input_width, uint16_t input_height,
    int32_t origin_y, int32_t origin_x, const int8_t *weight,
    int32_t accumulator) {
  int32_t kernel_y;
  for (kernel_y = 0; kernel_y < 3; ++kernel_y) {
    const int32_t input_y = origin_y + kernel_y;
    int32_t kernel_x;
    if (input_y < 0 || input_y >= (int32_t)input_height) continue;
    for (kernel_x = 0; kernel_x < 3; ++kernel_x) {
      const int32_t input_x = origin_x + kernel_x;
      if (input_x < 0 || input_x >= (int32_t)input_width) continue;
      accumulator += (int32_t)plane[(uint32_t)input_y * input_width +
                                    (uint32_t)input_x] *
                     weight[kernel_y * 3 + kernel_x];
    }
  }
  return accumulator;
}

/*
 * One task is (channel block, output row). Partitioning by row as well as by
 * block keeps short DORY height tiles balanced across eight cores -- the same
 * reason cf_qdepthwise_direct3x3.c does it.
 */
CF_QDW4_ALWAYS_INLINE void cf_qdepthwise_channel4_impl(
    const uint8_t *input, const int8_t *bias, uint8_t *output,
    const int8_t *weights, const int32_t *batch_norm_multiplier,
    const int32_t *batch_norm_bias, uint16_t output_multiplier,
    uint16_t output_shift, uint16_t input_width, uint16_t input_height,
    uint16_t output_width, uint16_t output_height, uint16_t output_channels,
    uint16_t padding_top, uint16_t padding_left, uint16_t stride_width,
    uint16_t stride_height, cf_qdepthwise_u8_epilogue_t mode) {
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t cores = (uint32_t)pi_cl_team_nb_cores();
  const uint32_t blocks = (uint32_t)output_channels / CF_QDW4_BLOCK;
  const uint32_t plane_stride = (uint32_t)input_height * input_width;
  const uint32_t tasks = blocks * output_height;
  const uint32_t chunk = (tasks + cores - 1u) / cores;
  const uint32_t first = cf_qdw4_min_u32(core * chunk, tasks);
  const uint32_t last = cf_qdw4_min_u32(first + chunk, tasks);
  uint32_t task;

  for (task = first; task < last; ++task) {
    const uint32_t block = task / output_height;
    const uint32_t output_y = task - block * output_height;
    const uint32_t channel = block * CF_QDW4_BLOCK;
    const int32_t origin_y = (int32_t)output_y * stride_height - padding_top;
    const uint8_t *const plane0 = input + (channel + 0u) * plane_stride;
    const uint8_t *const plane1 = input + (channel + 1u) * plane_stride;
    const uint8_t *const plane2 = input + (channel + 2u) * plane_stride;
    const uint8_t *const plane3 = input + (channel + 3u) * plane_stride;
    const int8_t *const w0 = weights + (channel + 0u) * 9u;
    const int8_t *const w1 = weights + (channel + 1u) * 9u;
    const int8_t *const w2 = weights + (channel + 2u) * 9u;
    const int8_t *const w3 = weights + (channel + 3u) * 9u;
    /* Four channels' worth of 3x3 taps, live across the whole row. */
    const v4s v0r0 = (v4s){w0[0], w0[1], w0[2], 0};
    const v4s v0r1 = (v4s){w0[3], w0[4], w0[5], 0};
    const v4s v0r2 = (v4s){w0[6], w0[7], w0[8], 0};
    const v4s v1r0 = (v4s){w1[0], w1[1], w1[2], 0};
    const v4s v1r1 = (v4s){w1[3], w1[4], w1[5], 0};
    const v4s v1r2 = (v4s){w1[6], w1[7], w1[8], 0};
    const v4s v2r0 = (v4s){w2[0], w2[1], w2[2], 0};
    const v4s v2r1 = (v4s){w2[3], w2[4], w2[5], 0};
    const v4s v2r2 = (v4s){w2[6], w2[7], w2[8], 0};
    const v4s v3r0 = (v4s){w3[0], w3[1], w3[2], 0};
    const v4s v3r1 = (v4s){w3[3], w3[4], w3[5], 0};
    const v4s v3r2 = (v4s){w3[6], w3[7], w3[8], 0};
    const int32_t bias0 = cf_qdepthwise_bias_at(bias, channel + 0u);
    const int32_t bias1 = cf_qdepthwise_bias_at(bias, channel + 1u);
    const int32_t bias2 = cf_qdepthwise_bias_at(bias, channel + 2u);
    const int32_t bias3 = cf_qdepthwise_bias_at(bias, channel + 3u);
    uint32_t interior_first = 0u;
    uint32_t interior_last = 0u;
    uint32_t output_x = 0u;

    /* A row whose 3x3 window is fully inside the plane vertically can use the
     * unclamped vector path between these bounds. */
    if (origin_y >= 0 && origin_y + 2 < (int32_t)input_height) {
      const uint32_t numerator = (uint32_t)input_width + padding_left;
      interior_first =
          ((uint32_t)padding_left + stride_width - 1u) / stride_width;
      if (numerator >= 3u) {
        interior_last = (numerator - 3u) / stride_width + 1u;
      }
      interior_first = cf_qdw4_min_u32(interior_first, output_width);
      interior_last = cf_qdw4_min_u32(interior_last, output_width);
      if (interior_last < interior_first) interior_last = interior_first;
    }

    for (; output_x < interior_first; ++output_x) {
      const int32_t origin_x = (int32_t)output_x * stride_width - padding_left;
      const uint32_t offset =
          (output_y * output_width + output_x) * output_channels + channel;
      output[offset + 0u] = cf_qdw4_quantize(
          mode,
          cf_qdw4_scalar_3x3(plane0, input_width, input_height, origin_y,
                             origin_x, w0, bias0),
          output_multiplier, output_shift, batch_norm_multiplier,
          batch_norm_bias, channel + 0u);
      output[offset + 1u] = cf_qdw4_quantize(
          mode,
          cf_qdw4_scalar_3x3(plane1, input_width, input_height, origin_y,
                             origin_x, w1, bias1),
          output_multiplier, output_shift, batch_norm_multiplier,
          batch_norm_bias, channel + 1u);
      output[offset + 2u] = cf_qdw4_quantize(
          mode,
          cf_qdw4_scalar_3x3(plane2, input_width, input_height, origin_y,
                             origin_x, w2, bias2),
          output_multiplier, output_shift, batch_norm_multiplier,
          batch_norm_bias, channel + 2u);
      output[offset + 3u] = cf_qdw4_quantize(
          mode,
          cf_qdw4_scalar_3x3(plane3, input_width, input_height, origin_y,
                             origin_x, w3, bias3),
          output_multiplier, output_shift, batch_norm_multiplier,
          batch_norm_bias, channel + 3u);
    }

    /*
     * Interior. Four independent accumulator chains, twelve loads, twelve dot
     * products, and one aligned 32-bit store per output pixel -- against four
     * scattered byte stores for the per-channel kernel.
     */
    for (; output_x < interior_last; ++output_x) {
      const uint32_t input_x = output_x * stride_width - padding_left;
      const uint32_t base = (uint32_t)origin_y * input_width + input_x;
      const uint8_t *row0 = plane0 + base;
      const uint8_t *row1 = plane1 + base;
      const uint8_t *row2 = plane2 + base;
      const uint8_t *row3 = plane3 + base;
      int32_t sum0 = bias0;
      int32_t sum1 = bias1;
      int32_t sum2 = bias2;
      int32_t sum3 = bias3;
      uint32_t offset;

      sum0 = CF_QDW4_DOTP4(*((const v4u *)row0), v0r0, sum0);
      sum1 = CF_QDW4_DOTP4(*((const v4u *)row1), v1r0, sum1);
      sum2 = CF_QDW4_DOTP4(*((const v4u *)row2), v2r0, sum2);
      sum3 = CF_QDW4_DOTP4(*((const v4u *)row3), v3r0, sum3);
      row0 += input_width;
      row1 += input_width;
      row2 += input_width;
      row3 += input_width;
      sum0 = CF_QDW4_DOTP4(*((const v4u *)row0), v0r1, sum0);
      sum1 = CF_QDW4_DOTP4(*((const v4u *)row1), v1r1, sum1);
      sum2 = CF_QDW4_DOTP4(*((const v4u *)row2), v2r1, sum2);
      sum3 = CF_QDW4_DOTP4(*((const v4u *)row3), v3r1, sum3);
      row0 += input_width;
      row1 += input_width;
      row2 += input_width;
      row3 += input_width;
      sum0 = CF_QDW4_DOTP4(*((const v4u *)row0), v0r2, sum0);
      sum1 = CF_QDW4_DOTP4(*((const v4u *)row1), v1r2, sum1);
      sum2 = CF_QDW4_DOTP4(*((const v4u *)row2), v2r2, sum2);
      sum3 = CF_QDW4_DOTP4(*((const v4u *)row3), v3r2, sum3);

      offset = (output_y * output_width + output_x) * output_channels + channel;
      /* output_channels % 4 == 0 and channel % 4 == 0, so this is word
       * aligned; the validate path guarantees both. */
      *((uint32_t *)(void *)(output + offset)) =
          (uint32_t)cf_qdw4_quantize(mode, sum0, output_multiplier,
                                     output_shift, batch_norm_multiplier,
                                     batch_norm_bias, channel + 0u) |
          ((uint32_t)cf_qdw4_quantize(mode, sum1, output_multiplier,
                                      output_shift, batch_norm_multiplier,
                                      batch_norm_bias, channel + 1u)
           << 8) |
          ((uint32_t)cf_qdw4_quantize(mode, sum2, output_multiplier,
                                      output_shift, batch_norm_multiplier,
                                      batch_norm_bias, channel + 2u)
           << 16) |
          ((uint32_t)cf_qdw4_quantize(mode, sum3, output_multiplier,
                                      output_shift, batch_norm_multiplier,
                                      batch_norm_bias, channel + 3u)
           << 24);
    }

    for (; output_x < output_width; ++output_x) {
      const int32_t origin_x = (int32_t)output_x * stride_width - padding_left;
      const uint32_t offset =
          (output_y * output_width + output_x) * output_channels + channel;
      output[offset + 0u] = cf_qdw4_quantize(
          mode,
          cf_qdw4_scalar_3x3(plane0, input_width, input_height, origin_y,
                             origin_x, w0, bias0),
          output_multiplier, output_shift, batch_norm_multiplier,
          batch_norm_bias, channel + 0u);
      output[offset + 1u] = cf_qdw4_quantize(
          mode,
          cf_qdw4_scalar_3x3(plane1, input_width, input_height, origin_y,
                             origin_x, w1, bias1),
          output_multiplier, output_shift, batch_norm_multiplier,
          batch_norm_bias, channel + 1u);
      output[offset + 2u] = cf_qdw4_quantize(
          mode,
          cf_qdw4_scalar_3x3(plane2, input_width, input_height, origin_y,
                             origin_x, w2, bias2),
          output_multiplier, output_shift, batch_norm_multiplier,
          batch_norm_bias, channel + 2u);
      output[offset + 3u] = cf_qdw4_quantize(
          mode,
          cf_qdw4_scalar_3x3(plane3, input_width, input_height, origin_y,
                             origin_x, w3, bias3),
          output_multiplier, output_shift, batch_norm_multiplier,
          batch_norm_bias, channel + 3u);
    }
  }
}

void cf_qdepthwise_u8s8_u8_channel4_team(void *argument) {
  const cf_qdepthwise_u8s8_u8_job_t *const job =
      (const cf_qdepthwise_u8s8_u8_job_t *)argument;
  if (job == 0) return;
  cf_qdepthwise_channel4_impl(
      job->input, job->bias, job->output, job->weights,
      job->batch_norm_multiplier, job->batch_norm_bias, job->output_multiplier,
      job->output_shift, job->input_width, job->input_height, job->output_width,
      job->output_height, job->output_channels, job->padding_top,
      job->padding_left, job->stride_width, job->stride_height, job->epilogue);
}
