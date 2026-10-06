/*
 * 4x4-blocked convolution matmul: four output pixels by four output channels.
 *
 * Why four rather than two. RI5CY issues roughly one operation per cycle, so
 * the inner loop's ceiling is set by how many loads it takes to feed a dot
 * product. `cf_qconv_matmul4x2` loads four weight words and two input words to
 * issue eight `pv.sdotusp.b`:
 *
 *     4x2   6 loads + 8 dotp  = 14 slots for 32 MACs  -> 2.29 MAC/cycle/core
 *     4x4   8 loads + 16 dotp = 24 slots for 64 MACs  -> 2.67 MAC/cycle/core
 *
 * about 17% more arithmetic per cycle, and 4x2 runs at about 2.02, roughly 88%
 * of its own ceiling, so there is no slack left to find inside it. The extra
 * pixel pair is where the headroom is.
 *
 * The thing that can defeat this is register pressure, not arithmetic. Sixteen
 * live int32 accumulators is half the register file, and a spill in the inner
 * loop would cost more than the loads it saves. Two decisions follow from that,
 * and they are the whole design:
 *
 *   1. Two running pointers, not eight. The four weight rows and the four
 *      im2col columns are each `columns` apart, so the loop carries `weight`
 *      and `input` plus the three stride constants and reaches the rest by
 *      displacement. Eight independently incremented pointers is eight live
 *      registers to keep sixteen accumulators company, and that is the version
 *      that spills.
 *   2. The four input words are loaded once and reused against each weight row
 *      in turn, so only one weight word is live at a time. Peak is 16
 *      accumulators + 4 inputs + 1 weight + 5 address registers.
 *
 * kernel_bench measures it against 4x2 (matmul4x4_shapes) rather than swapping
 * it in for 4x2.
 */
#include <stdint.h>

#include "pmsis.h"

#include "cf_qconv.h"
#include "cf_qconv_internal.h"

CF_QCONV_ALWAYS_INLINE uint8_t *cf_qconv_matmul4x4_impl(
    const int8_t *weights, const uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    const int32_t *batch_norm_multiplier, const int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, uint16_t output_stride,
    cf_qconv_u8_epilogue_t mode) {
  const uint8_t *const output_start = output;
  const uint32_t full_channels = (uint32_t)output_channels & ~3u;
  const uint32_t words = (uint32_t)columns >> 2;
  const uint32_t tail = (uint32_t)columns & 3u;
  const uint32_t cs = (uint32_t)columns;
  uint8_t *const out0 = output;
  uint8_t *const out1 = output + output_stride;
  uint8_t *const out2 = output + 2u * (uint32_t)output_stride;
  uint8_t *const out3 = output + 3u * (uint32_t)output_stride;
  const uint8_t packed_store =
      ((((uintptr_t)out0 | (uintptr_t)out1 | (uintptr_t)out2 |
         (uintptr_t)out3) &
        3u) == 0u);
  uint32_t channel;

  for (channel = 0u; channel < full_channels; channel += 4u) {
    const int8_t *weight = weights + (size_t)channel * cs;
    const uint8_t *in = input;
    int32_t s00 = cf_qconv_bias_at(bias, channel);
    int32_t s01 = cf_qconv_bias_at(bias, channel + 1u);
    int32_t s02 = cf_qconv_bias_at(bias, channel + 2u);
    int32_t s03 = cf_qconv_bias_at(bias, channel + 3u);
    int32_t s10 = s00, s11 = s01, s12 = s02, s13 = s03;
    int32_t s20 = s00, s21 = s01, s22 = s02, s23 = s03;
    int32_t s30 = s00, s31 = s01, s32 = s02, s33 = s03;
    uint32_t word;
    uint32_t lane;

    /* Load the four pixels once; sweep the four weight rows past them. */
#define CF_QCONV_ACCUMULATE_4X4()                                             \
  do {                                                                        \
    const v4u x0 = *((const v4u *)(in));                                      \
    const v4u x1 = *((const v4u *)(in + cs));                                 \
    const v4u x2 = *((const v4u *)(in + 2u * cs));                            \
    const v4u x3 = *((const v4u *)(in + 3u * cs));                            \
    v4s w;                                                                    \
    w = *((const v4s *)(weight));                                             \
    s00 = CF_QCONV_DOTP4(x0, w, s00);                                         \
    s10 = CF_QCONV_DOTP4(x1, w, s10);                                         \
    s20 = CF_QCONV_DOTP4(x2, w, s20);                                         \
    s30 = CF_QCONV_DOTP4(x3, w, s30);                                         \
    w = *((const v4s *)(weight + cs));                                        \
    s01 = CF_QCONV_DOTP4(x0, w, s01);                                         \
    s11 = CF_QCONV_DOTP4(x1, w, s11);                                         \
    s21 = CF_QCONV_DOTP4(x2, w, s21);                                         \
    s31 = CF_QCONV_DOTP4(x3, w, s31);                                         \
    w = *((const v4s *)(weight + 2u * cs));                                   \
    s02 = CF_QCONV_DOTP4(x0, w, s02);                                         \
    s12 = CF_QCONV_DOTP4(x1, w, s12);                                         \
    s22 = CF_QCONV_DOTP4(x2, w, s22);                                         \
    s32 = CF_QCONV_DOTP4(x3, w, s32);                                         \
    w = *((const v4s *)(weight + 3u * cs));                                   \
    s03 = CF_QCONV_DOTP4(x0, w, s03);                                         \
    s13 = CF_QCONV_DOTP4(x1, w, s13);                                         \
    s23 = CF_QCONV_DOTP4(x2, w, s23);                                         \
    s33 = CF_QCONV_DOTP4(x3, w, s33);                                         \
    weight += 4;                                                              \
    in += 4;                                                                  \
  } while (0)

    for (word = 0u; word < words; ++word) {
      CF_QCONV_ACCUMULATE_4X4();
    }
#undef CF_QCONV_ACCUMULATE_4X4

    for (lane = 0u; lane < tail; ++lane) {
      const int32_t x0 = in[0];
      const int32_t x1 = in[cs];
      const int32_t x2 = in[2u * cs];
      const int32_t x3 = in[3u * cs];
      const int32_t w0 = weight[0];
      const int32_t w1 = weight[cs];
      const int32_t w2 = weight[2u * cs];
      const int32_t w3 = weight[3u * cs];
      s00 += x0 * w0; s01 += x0 * w1; s02 += x0 * w2; s03 += x0 * w3;
      s10 += x1 * w0; s11 += x1 * w1; s12 += x1 * w2; s13 += x1 * w3;
      s20 += x2 * w0; s21 += x2 * w1; s22 += x2 * w2; s23 += x2 * w3;
      s30 += x3 * w0; s31 += x3 * w1; s32 += x3 * w2; s33 += x3 * w3;
      ++weight;
      ++in;
    }

#define CF_QCONV_Q4X4(value, lane_index)                                      \
  cf_qconv_quantize(mode, (value), output_multiplier, output_shift,           \
                    batch_norm_multiplier, batch_norm_bias,                   \
                    channel + (lane_index))

    if (packed_store) {
      *((v4u *)(out0 + channel)) = __builtin_pulp_pack4(
          CF_QCONV_Q4X4(s00, 0u), CF_QCONV_Q4X4(s01, 1u),
          CF_QCONV_Q4X4(s02, 2u), CF_QCONV_Q4X4(s03, 3u));
      *((v4u *)(out1 + channel)) = __builtin_pulp_pack4(
          CF_QCONV_Q4X4(s10, 0u), CF_QCONV_Q4X4(s11, 1u),
          CF_QCONV_Q4X4(s12, 2u), CF_QCONV_Q4X4(s13, 3u));
      *((v4u *)(out2 + channel)) = __builtin_pulp_pack4(
          CF_QCONV_Q4X4(s20, 0u), CF_QCONV_Q4X4(s21, 1u),
          CF_QCONV_Q4X4(s22, 2u), CF_QCONV_Q4X4(s23, 3u));
      *((v4u *)(out3 + channel)) = __builtin_pulp_pack4(
          CF_QCONV_Q4X4(s30, 0u), CF_QCONV_Q4X4(s31, 1u),
          CF_QCONV_Q4X4(s32, 2u), CF_QCONV_Q4X4(s33, 3u));
    } else {
      out0[channel] = CF_QCONV_Q4X4(s00, 0u);
      out0[channel + 1u] = CF_QCONV_Q4X4(s01, 1u);
      out0[channel + 2u] = CF_QCONV_Q4X4(s02, 2u);
      out0[channel + 3u] = CF_QCONV_Q4X4(s03, 3u);
      out1[channel] = CF_QCONV_Q4X4(s10, 0u);
      out1[channel + 1u] = CF_QCONV_Q4X4(s11, 1u);
      out1[channel + 2u] = CF_QCONV_Q4X4(s12, 2u);
      out1[channel + 3u] = CF_QCONV_Q4X4(s13, 3u);
      out2[channel] = CF_QCONV_Q4X4(s20, 0u);
      out2[channel + 1u] = CF_QCONV_Q4X4(s21, 1u);
      out2[channel + 2u] = CF_QCONV_Q4X4(s22, 2u);
      out2[channel + 3u] = CF_QCONV_Q4X4(s23, 3u);
      out3[channel] = CF_QCONV_Q4X4(s30, 0u);
      out3[channel + 1u] = CF_QCONV_Q4X4(s31, 1u);
      out3[channel + 2u] = CF_QCONV_Q4X4(s32, 2u);
      out3[channel + 3u] = CF_QCONV_Q4X4(s33, 3u);
    }
  }

  /* Channel tail: one channel by four pixels. Four dot products per two loads,
   * which is a better ratio than the block above and worse throughput, because
   * it runs at most three times. */
  for (; channel < output_channels; ++channel) {
    const int8_t *weight = weights + (size_t)channel * cs;
    const uint8_t *in = input;
    int32_t s0 = cf_qconv_bias_at(bias, channel);
    int32_t s1 = s0, s2 = s0, s3 = s0;
    uint32_t word;
    uint32_t lane;
    for (word = 0u; word < words; ++word) {
      const v4s w = *((const v4s *)weight);
      s0 = CF_QCONV_DOTP4(*((const v4u *)(in)), w, s0);
      s1 = CF_QCONV_DOTP4(*((const v4u *)(in + cs)), w, s1);
      s2 = CF_QCONV_DOTP4(*((const v4u *)(in + 2u * cs)), w, s2);
      s3 = CF_QCONV_DOTP4(*((const v4u *)(in + 3u * cs)), w, s3);
      weight += 4;
      in += 4;
    }
    for (lane = 0u; lane < tail; ++lane) {
      const int32_t w = *weight++;
      s0 += (int32_t)in[0] * w;
      s1 += (int32_t)in[cs] * w;
      s2 += (int32_t)in[2u * cs] * w;
      s3 += (int32_t)in[3u * cs] * w;
      ++in;
    }
    out0[channel] = CF_QCONV_Q4X4(s0, 0u);
    out1[channel] = CF_QCONV_Q4X4(s1, 0u);
    out2[channel] = CF_QCONV_Q4X4(s2, 0u);
    out3[channel] = CF_QCONV_Q4X4(s3, 0u);
  }
#undef CF_QCONV_Q4X4
  return (uint8_t *)output_start + 4u * (uint32_t)output_stride;
}

uint8_t *cf_qconv_matmul4x4(
    const int8_t *weights, const uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, uint16_t output_stride,
    int flag_relu, int flag_batch_norm) {
  if (flag_batch_norm && flag_relu) {
    return cf_qconv_matmul4x4_impl(
        weights, input, output_channels, columns, output_shift,
        output_multiplier, batch_norm_multiplier, batch_norm_bias, bias,
        output, output_stride, CF_QCONV_U8_BATCHNORM_RELU);
  }
  if (flag_relu) {
    return cf_qconv_matmul4x4_impl(
        weights, input, output_channels, columns, output_shift,
        output_multiplier, batch_norm_multiplier, batch_norm_bias, bias,
        output, output_stride, CF_QCONV_U8_REQUANT_RELU);
  }
  return cf_qconv_matmul4x4_impl(
      weights, input, output_channels, columns, output_shift,
      output_multiplier, batch_norm_multiplier, batch_norm_bias, bias, output,
      output_stride, CF_QCONV_U8_SHIFT_CLIP);
}
