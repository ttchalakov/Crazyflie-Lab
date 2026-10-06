#include <stdint.h>

#include "pmsis.h"

#include "cf_qconv.h"
#include "cf_qconv_internal.h"

CF_QCONV_ALWAYS_INLINE uint8_t *cf_qconv_matmul4x2_impl(
    const int8_t *weights, const uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    const int32_t *batch_norm_multiplier, const int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, uint8_t *output2,
    cf_qconv_u8_epilogue_t mode, uint8_t rotate_k) {
  const uint8_t *const output_start = output;
  const uint32_t full_channels = (uint32_t)output_channels & ~3u;
  const uint32_t words = (uint32_t)columns >> 2;
  const uint32_t tail = (uint32_t)columns & 3u;
  const uint8_t packed_store =
      ((((uintptr_t)output | (uintptr_t)output2) & 3u) == 0u);
  uint32_t rotation_words = 0u;
  uint32_t channel;

  if (rotate_k && words > 1u) {
    rotation_words = (uint32_t)pi_core_id();
    if (rotation_words >= words) rotation_words %= words;
  }

  for (channel = 0u; channel < full_channels; channel += 4u) {
    const int8_t *const weight0_base = weights + channel * columns;
    const int8_t *const weight1_base = weight0_base + columns;
    const int8_t *const weight2_base = weight1_base + columns;
    const int8_t *const weight3_base = weight2_base + columns;
    const uint8_t *const input0_base = input;
    const uint8_t *const input1_base = input + columns;
    const int8_t *weight0 = weight0_base + (rotation_words << 2);
    const int8_t *weight1 = weight1_base + (rotation_words << 2);
    const int8_t *weight2 = weight2_base + (rotation_words << 2);
    const int8_t *weight3 = weight3_base + (rotation_words << 2);
    const uint8_t *input0 = input0_base + (rotation_words << 2);
    const uint8_t *input1 = input1_base + (rotation_words << 2);
    int32_t sum0 = cf_qconv_bias_at(bias, channel);
    int32_t sum1 = cf_qconv_bias_at(bias, channel + 1u);
    int32_t sum2 = cf_qconv_bias_at(bias, channel + 2u);
    int32_t sum3 = cf_qconv_bias_at(bias, channel + 3u);
    int32_t sum4 = sum0;
    int32_t sum5 = sum1;
    int32_t sum6 = sum2;
    int32_t sum7 = sum3;
    uint32_t word;
    uint32_t lane;
    uint8_t out0;
    uint8_t out1;
    uint8_t out2_value;
    uint8_t out3;
    uint8_t out4;
    uint8_t out5;
    uint8_t out6;
    uint8_t out7;

#define CF_QCONV_ACCUMULATE_4X2()                                             \
  do {                                                                        \
    const v4s w0 = *((const v4s *)weight0);                                   \
    const v4s w1 = *((const v4s *)weight1);                                   \
    const v4s w2 = *((const v4s *)weight2);                                   \
    const v4s w3 = *((const v4s *)weight3);                                   \
    const v4u x0 = *((const v4u *)input0);                                    \
    const v4u x1 = *((const v4u *)input1);                                    \
    sum0 = CF_QCONV_DOTP4(x0, w0, sum0);                                     \
    sum1 = CF_QCONV_DOTP4(x0, w1, sum1);                                     \
    sum2 = CF_QCONV_DOTP4(x0, w2, sum2);                                     \
    sum3 = CF_QCONV_DOTP4(x0, w3, sum3);                                     \
    sum4 = CF_QCONV_DOTP4(x1, w0, sum4);                                     \
    sum5 = CF_QCONV_DOTP4(x1, w1, sum5);                                     \
    sum6 = CF_QCONV_DOTP4(x1, w2, sum6);                                     \
    sum7 = CF_QCONV_DOTP4(x1, w3, sum7);                                     \
    weight0 += 4;                                                             \
    weight1 += 4;                                                             \
    weight2 += 4;                                                             \
    weight3 += 4;                                                             \
    input0 += 4;                                                              \
    input1 += 4;                                                              \
  } while (0)

    for (word = rotation_words; word < words; ++word) {
      CF_QCONV_ACCUMULATE_4X2();
    }
    if (rotation_words != 0u) {
      weight0 = weight0_base;
      weight1 = weight1_base;
      weight2 = weight2_base;
      weight3 = weight3_base;
      input0 = input0_base;
      input1 = input1_base;
      for (word = 0u; word < rotation_words; ++word) {
        CF_QCONV_ACCUMULATE_4X2();
      }
    }
#undef CF_QCONV_ACCUMULATE_4X2
    weight0 = weight0_base + (words << 2);
    weight1 = weight1_base + (words << 2);
    weight2 = weight2_base + (words << 2);
    weight3 = weight3_base + (words << 2);
    input0 = input0_base + (words << 2);
    input1 = input1_base + (words << 2);
    for (lane = 0u; lane < tail; ++lane) {
      const int32_t x0 = *input0++;
      const int32_t x1 = *input1++;
      const int32_t w0 = *weight0++;
      const int32_t w1 = *weight1++;
      const int32_t w2 = *weight2++;
      const int32_t w3 = *weight3++;
      sum0 += x0 * w0;
      sum1 += x0 * w1;
      sum2 += x0 * w2;
      sum3 += x0 * w3;
      sum4 += x1 * w0;
      sum5 += x1 * w1;
      sum6 += x1 * w2;
      sum7 += x1 * w3;
    }

    out0 = cf_qconv_quantize(mode, sum0, output_multiplier, output_shift,
                             batch_norm_multiplier, batch_norm_bias, channel);
    out1 = cf_qconv_quantize(mode, sum1, output_multiplier, output_shift,
                             batch_norm_multiplier, batch_norm_bias,
                             channel + 1u);
    out2_value = cf_qconv_quantize(
        mode, sum2, output_multiplier, output_shift, batch_norm_multiplier,
        batch_norm_bias, channel + 2u);
    out3 = cf_qconv_quantize(mode, sum3, output_multiplier, output_shift,
                             batch_norm_multiplier, batch_norm_bias,
                             channel + 3u);
    out4 = cf_qconv_quantize(mode, sum4, output_multiplier, output_shift,
                             batch_norm_multiplier, batch_norm_bias, channel);
    out5 = cf_qconv_quantize(mode, sum5, output_multiplier, output_shift,
                             batch_norm_multiplier, batch_norm_bias,
                             channel + 1u);
    out6 = cf_qconv_quantize(mode, sum6, output_multiplier, output_shift,
                             batch_norm_multiplier, batch_norm_bias,
                             channel + 2u);
    out7 = cf_qconv_quantize(mode, sum7, output_multiplier, output_shift,
                             batch_norm_multiplier, batch_norm_bias,
                             channel + 3u);

    if (packed_store) {
      *((v4u *)(output + channel)) =
          __builtin_pulp_pack4(out0, out1, out2_value, out3);
      *((v4u *)(output2 + channel)) =
          __builtin_pulp_pack4(out4, out5, out6, out7);
    } else {
      output[channel] = out0;
      output[channel + 1u] = out1;
      output[channel + 2u] = out2_value;
      output[channel + 3u] = out3;
      output2[channel] = out4;
      output2[channel + 1u] = out5;
      output2[channel + 2u] = out6;
      output2[channel + 3u] = out7;
    }
  }

  for (; channel < output_channels; ++channel) {
    const int8_t *const weight_base = weights + channel * columns;
    const uint8_t *const input0_base = input;
    const uint8_t *const input1_base = input + columns;
    const int8_t *weight = weight_base + (rotation_words << 2);
    const uint8_t *input0 = input0_base + (rotation_words << 2);
    const uint8_t *input1 = input1_base + (rotation_words << 2);
    int32_t sum0 = cf_qconv_bias_at(bias, channel);
    int32_t sum1 = sum0;
    uint32_t word;
    uint32_t lane;
#define CF_QCONV_ACCUMULATE_1X2()                                             \
  do {                                                                        \
    const v4s w = *((const v4s *)weight);                                     \
    sum0 = CF_QCONV_DOTP4(*((const v4u *)input0), w, sum0);                   \
    sum1 = CF_QCONV_DOTP4(*((const v4u *)input1), w, sum1);                   \
    weight += 4;                                                              \
    input0 += 4;                                                              \
    input1 += 4;                                                              \
  } while (0)

    for (word = rotation_words; word < words; ++word) {
      CF_QCONV_ACCUMULATE_1X2();
    }
    if (rotation_words != 0u) {
      weight = weight_base;
      input0 = input0_base;
      input1 = input1_base;
      for (word = 0u; word < rotation_words; ++word) {
        CF_QCONV_ACCUMULATE_1X2();
      }
    }
#undef CF_QCONV_ACCUMULATE_1X2
    weight = weight_base + (words << 2);
    input0 = input0_base + (words << 2);
    input1 = input1_base + (words << 2);
    for (lane = 0u; lane < tail; ++lane) {
      const int32_t value = *weight++;
      sum0 += (int32_t)*input0++ * value;
      sum1 += (int32_t)*input1++ * value;
    }
    output[channel] = cf_qconv_quantize(
        mode, sum0, output_multiplier, output_shift, batch_norm_multiplier,
        batch_norm_bias, channel);
    output2[channel] = cf_qconv_quantize(
        mode, sum1, output_multiplier, output_shift, batch_norm_multiplier,
        batch_norm_bias, channel);
  }
  return (uint8_t *)output_start + 2u * output_channels;
}

uint8_t *cf_qconv_matmul4x2(
    const int8_t *weights, uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, uint8_t *output2, int flag_relu,
    int flag_batch_norm) {
  if (flag_batch_norm && flag_relu) {
    return cf_qconv_matmul4x2_impl(
        weights, input, output_channels, columns, output_shift,
        output_multiplier, batch_norm_multiplier, batch_norm_bias, bias,
        output, output2, CF_QCONV_U8_BATCHNORM_RELU, 0u);
  }
  if (flag_relu) {
    return cf_qconv_matmul4x2_impl(
        weights, input, output_channels, columns, output_shift,
        output_multiplier, batch_norm_multiplier, batch_norm_bias, bias,
        output, output2, CF_QCONV_U8_REQUANT_RELU, 0u);
  }
  return cf_qconv_matmul4x2_impl(
      weights, input, output_channels, columns, output_shift,
      output_multiplier, batch_norm_multiplier, batch_norm_bias, bias, output,
      output2, CF_QCONV_U8_SHIFT_CLIP, 0u);
}

uint8_t *cf_qconv_matmul4x2_krotate(
    const int8_t *weights, uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, uint8_t *output2, int flag_relu,
    int flag_batch_norm) {
  if (flag_batch_norm && flag_relu) {
    return cf_qconv_matmul4x2_impl(
        weights, input, output_channels, columns, output_shift,
        output_multiplier, batch_norm_multiplier, batch_norm_bias, bias,
        output, output2, CF_QCONV_U8_BATCHNORM_RELU, 1u);
  }
  if (flag_relu) {
    return cf_qconv_matmul4x2_impl(
        weights, input, output_channels, columns, output_shift,
        output_multiplier, batch_norm_multiplier, batch_norm_bias, bias,
        output, output2, CF_QCONV_U8_REQUANT_RELU, 1u);
  }
  return cf_qconv_matmul4x2_impl(
      weights, input, output_channels, columns, output_shift,
      output_multiplier, batch_norm_multiplier, batch_norm_bias, bias, output,
      output2, CF_QCONV_U8_SHIFT_CLIP, 1u);
}

#ifdef CF_QCONV_REPLACE_PULP_NN
uint8_t *pulp_nn_matmul(
    const int8_t *weights, uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, uint8_t *output2, int flag_relu,
    int flag_batch_norm) {
  return cf_qconv_matmul4x2(
      weights, input, output_channels, columns, output_shift,
      output_multiplier, batch_norm_multiplier, batch_norm_bias, bias, output,
      output2, flag_relu, flag_batch_norm);
}
#endif
