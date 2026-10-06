/*
 * 4x3-blocked convolution matmul: four output channels by *three* output pixels.
 *
 * This exists because 4x4 lost, and the disassembly said why rather than
 * leaving it a mystery. Sixteen live int32 accumulators plus four input words,
 * a weight word, two running pointers and three stride constants is 27 values
 * against RI5CY's 27 usable general-purpose registers, and GCC spills: the 4x4
 * inner loop compiles to 39 instructions for 16 dot products with **13 of them
 * stack-relative**. The 4x2 loop has none. Since the stack is in TCDM, those
 * spills are also L1 traffic -- which is the same measurement as the 8x rise in
 * `tcdm_contentions`, not a second effect.
 *
 * Three pixels drops the accumulator count to twelve and the live set to about
 * twenty-one, which fits. The issue-slot arithmetic is still better than 4x2:
 *
 *     4x2   6 loads +  8 dotp = 14 slots for 32 MACs -> 2.29 MAC/cycle/core
 *     4x3   7 loads + 12 dotp = 19 slots for 48 MACs -> 2.53 MAC/cycle/core
 *     4x4   8 loads + 16 dotp = 24 slots for 64 MACs -> 2.67, unreachable
 *
 * so the question this answers is whether the ceiling that 4x4 could not reach
 * is reachable one pixel lower down. Derived from cf_qconv_matmul4x4.c by
 * dropping the fourth pixel and nothing else, so any difference between them is
 * the register pressure and not the arithmetic.
 */
#include <stdint.h>

#include "pmsis.h"

#include "cf_qconv.h"
#include "cf_qconv_internal.h"

CF_QCONV_ALWAYS_INLINE uint8_t *cf_qconv_matmul4x3_impl(
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
  const uint8_t packed_store =
      ((((uintptr_t)out0 | (uintptr_t)out1 | (uintptr_t)out2) & 3u) == 0u);
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
    uint32_t word;
    uint32_t lane;

    /* Load the four pixels once; sweep the four weight rows past them. */
#define CF_QCONV_ACCUMULATE_4X3()                                             \
  do {                                                                        \
    const v4u x0 = *((const v4u *)(in));                                      \
    const v4u x1 = *((const v4u *)(in + cs));                                 \
    const v4u x2 = *((const v4u *)(in + 2u * cs));                            \
    v4s w;                                                                    \
    w = *((const v4s *)(weight));                                             \
    s00 = CF_QCONV_DOTP4(x0, w, s00);                                         \
    s10 = CF_QCONV_DOTP4(x1, w, s10);                                         \
    s20 = CF_QCONV_DOTP4(x2, w, s20);                                         \
    w = *((const v4s *)(weight + cs));                                        \
    s01 = CF_QCONV_DOTP4(x0, w, s01);                                         \
    s11 = CF_QCONV_DOTP4(x1, w, s11);                                         \
    s21 = CF_QCONV_DOTP4(x2, w, s21);                                         \
    w = *((const v4s *)(weight + 2u * cs));                                   \
    s02 = CF_QCONV_DOTP4(x0, w, s02);                                         \
    s12 = CF_QCONV_DOTP4(x1, w, s12);                                         \
    s22 = CF_QCONV_DOTP4(x2, w, s22);                                         \
    w = *((const v4s *)(weight + 3u * cs));                                   \
    s03 = CF_QCONV_DOTP4(x0, w, s03);                                         \
    s13 = CF_QCONV_DOTP4(x1, w, s13);                                         \
    s23 = CF_QCONV_DOTP4(x2, w, s23);                                         \
    weight += 4;                                                              \
    in += 4;                                                                  \
  } while (0)

    for (word = 0u; word < words; ++word) {
      CF_QCONV_ACCUMULATE_4X3();
    }
#undef CF_QCONV_ACCUMULATE_4X3

    for (lane = 0u; lane < tail; ++lane) {
      const int32_t x0 = in[0];
      const int32_t x1 = in[cs];
      const int32_t x2 = in[2u * cs];
      const int32_t w0 = weight[0];
      const int32_t w1 = weight[cs];
      const int32_t w2 = weight[2u * cs];
      const int32_t w3 = weight[3u * cs];
      s00 += x0 * w0; s01 += x0 * w1; s02 += x0 * w2; s03 += x0 * w3;
      s10 += x1 * w0; s11 += x1 * w1; s12 += x1 * w2; s13 += x1 * w3;
      s20 += x2 * w0; s21 += x2 * w1; s22 += x2 * w2; s23 += x2 * w3;
      ++weight;
      ++in;
    }

#define CF_QCONV_Q4X3(value, lane_index)                                      \
  cf_qconv_quantize(mode, (value), output_multiplier, output_shift,           \
                    batch_norm_multiplier, batch_norm_bias,                   \
                    channel + (lane_index))

    if (packed_store) {
      *((v4u *)(out0 + channel)) = __builtin_pulp_pack4(
          CF_QCONV_Q4X3(s00, 0u), CF_QCONV_Q4X3(s01, 1u),
          CF_QCONV_Q4X3(s02, 2u), CF_QCONV_Q4X3(s03, 3u));
      *((v4u *)(out1 + channel)) = __builtin_pulp_pack4(
          CF_QCONV_Q4X3(s10, 0u), CF_QCONV_Q4X3(s11, 1u),
          CF_QCONV_Q4X3(s12, 2u), CF_QCONV_Q4X3(s13, 3u));
      *((v4u *)(out2 + channel)) = __builtin_pulp_pack4(
          CF_QCONV_Q4X3(s20, 0u), CF_QCONV_Q4X3(s21, 1u),
          CF_QCONV_Q4X3(s22, 2u), CF_QCONV_Q4X3(s23, 3u));
    } else {
      out0[channel] = CF_QCONV_Q4X3(s00, 0u);
      out0[channel + 1u] = CF_QCONV_Q4X3(s01, 1u);
      out0[channel + 2u] = CF_QCONV_Q4X3(s02, 2u);
      out0[channel + 3u] = CF_QCONV_Q4X3(s03, 3u);
      out1[channel] = CF_QCONV_Q4X3(s10, 0u);
      out1[channel + 1u] = CF_QCONV_Q4X3(s11, 1u);
      out1[channel + 2u] = CF_QCONV_Q4X3(s12, 2u);
      out1[channel + 3u] = CF_QCONV_Q4X3(s13, 3u);
      out2[channel] = CF_QCONV_Q4X3(s20, 0u);
      out2[channel + 1u] = CF_QCONV_Q4X3(s21, 1u);
      out2[channel + 2u] = CF_QCONV_Q4X3(s22, 2u);
      out2[channel + 3u] = CF_QCONV_Q4X3(s23, 3u);
    }
  }

  /* Channel tail: one channel by four pixels. Four dot products per two loads,
   * which is a better ratio than the block above and worse throughput, because
   * it runs at most three times. */
  for (; channel < output_channels; ++channel) {
    const int8_t *weight = weights + (size_t)channel * cs;
    const uint8_t *in = input;
    int32_t s0 = cf_qconv_bias_at(bias, channel);
    int32_t s1 = s0, s2 = s0;
    uint32_t word;
    uint32_t lane;
    for (word = 0u; word < words; ++word) {
      const v4s w = *((const v4s *)weight);
      s0 = CF_QCONV_DOTP4(*((const v4u *)(in)), w, s0);
      s1 = CF_QCONV_DOTP4(*((const v4u *)(in + cs)), w, s1);
      s2 = CF_QCONV_DOTP4(*((const v4u *)(in + 2u * cs)), w, s2);
      weight += 4;
      in += 4;
    }
    for (lane = 0u; lane < tail; ++lane) {
      const int32_t w = *weight++;
      s0 += (int32_t)in[0] * w;
      s1 += (int32_t)in[cs] * w;
      s2 += (int32_t)in[2u * cs] * w;
      ++in;
    }
    out0[channel] = CF_QCONV_Q4X3(s0, 0u);
    out1[channel] = CF_QCONV_Q4X3(s1, 0u);
    out2[channel] = CF_QCONV_Q4X3(s2, 0u);
  }
#undef CF_QCONV_Q4X3
  return (uint8_t *)output_start + 3u * (uint32_t)output_stride;
}

uint8_t *cf_qconv_matmul4x3(
    const int8_t *weights, const uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, uint16_t output_stride,
    int flag_relu, int flag_batch_norm) {
  if (flag_batch_norm && flag_relu) {
    return cf_qconv_matmul4x3_impl(
        weights, input, output_channels, columns, output_shift,
        output_multiplier, batch_norm_multiplier, batch_norm_bias, bias,
        output, output_stride, CF_QCONV_U8_BATCHNORM_RELU);
  }
  if (flag_relu) {
    return cf_qconv_matmul4x3_impl(
        weights, input, output_channels, columns, output_shift,
        output_multiplier, batch_norm_multiplier, batch_norm_bias, bias,
        output, output_stride, CF_QCONV_U8_REQUANT_RELU);
  }
  return cf_qconv_matmul4x3_impl(
      weights, input, output_channels, columns, output_shift,
      output_multiplier, batch_norm_multiplier, batch_norm_bias, bias, output,
      output_stride, CF_QCONV_U8_SHIFT_CLIP);
}
