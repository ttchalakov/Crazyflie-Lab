#ifndef CF_GAP8_QCONV_INTERNAL_H
#define CF_GAP8_QCONV_INTERNAL_H

#include <stdint.h>

#include "pmsis.h"

#include "cf_qconv.h"
#include "cf_qconv_bias.h"

#define CF_QCONV_ALWAYS_INLINE static inline __attribute__((always_inline))
#define CF_QCONV_DOTP4(a, b, c) __builtin_pulp_sdotusp4((a), (b), (c))
#define CF_QCONV_CLIP_U8(value) __builtin_pulp_clipu_r((value), 255)

CF_QCONV_ALWAYS_INLINE uint32_t cf_qconv_min_u32(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}

CF_QCONV_ALWAYS_INLINE uint8_t cf_qconv_quantize(
    cf_qconv_u8_epilogue_t mode, int32_t accumulator,
    uint16_t output_multiplier, uint16_t output_shift,
    const int32_t *batch_norm_multiplier, const int32_t *batch_norm_bias,
    uint32_t output_channel) {
  switch (mode) {
    case CF_QCONV_U8_SHIFT_CLIP:
      return CF_QCONV_CLIP_U8(accumulator >> output_shift);
    case CF_QCONV_U8_REQUANT_RELU: {
      const int32_t multiplier = (int16_t)output_multiplier;
      const int16_t result =
          (int16_t)((multiplier * accumulator) >> output_shift);
      return CF_QCONV_CLIP_U8(result);
    }
    case CF_QCONV_U8_BATCHNORM_RELU:
      return CF_QCONV_CLIP_U8(
          (batch_norm_multiplier[output_channel] * accumulator +
           batch_norm_bias[output_channel]) >>
          output_shift);
    default:
      return 0u;
  }
}

CF_QCONV_ALWAYS_INLINE void cf_qconv_copy_u8(uint8_t *destination,
                                              const uint8_t *source,
                                              uint32_t bytes) {
  const uint32_t words = bytes >> 2;
  uint32_t index;

  for (index = 0u; index < words; ++index) {
    *((v4u *)destination) = *((const v4u *)source);
    destination += 4;
    source += 4;
  }
  for (index = words << 2; index < bytes; ++index) {
    *destination++ = *source++;
  }
}

CF_QCONV_ALWAYS_INLINE void cf_qconv_zero_u8(uint8_t *destination,
                                              uint32_t bytes) {
  const v4u zero = (v4u){0u, 0u, 0u, 0u};
  const uint32_t words = bytes >> 2;
  uint32_t index;

  for (index = 0u; index < words; ++index) {
    *((v4u *)destination) = zero;
    destination += 4;
  }
  for (index = words << 2; index < bytes; ++index) {
    *destination++ = 0u;
  }
}

/* Build one compact K column in exactly DORY/PULP-NN's ky,kx,channel order. */
CF_QCONV_ALWAYS_INLINE void cf_qconv_im2col_patch(
    const uint8_t *input, uint8_t *column, uint32_t output_pixel,
    uint16_t input_width, uint16_t input_height, uint16_t input_channels,
    uint16_t output_width, uint16_t kernel_width, uint16_t kernel_height,
    uint16_t padding_top, uint16_t padding_left, uint16_t stride_width,
    uint16_t stride_height) {
  const uint32_t output_y = output_pixel / output_width;
  const uint32_t output_x = output_pixel - output_y * output_width;
  const int32_t origin_y =
      (int32_t)output_y * stride_height - padding_top;
  const int32_t origin_x =
      (int32_t)output_x * stride_width - padding_left;
  const uint8_t interior =
      origin_y >= 0 && origin_x >= 0 &&
      origin_y + kernel_height <= input_height &&
      origin_x + kernel_width <= input_width;
  uint16_t kernel_y;

  if (interior) {
    const uint32_t row_bytes = (uint32_t)kernel_width * input_channels;
    const uint8_t *row =
        input + ((uint32_t)origin_y * input_width + (uint32_t)origin_x) *
                    input_channels;
    for (kernel_y = 0u; kernel_y < kernel_height; ++kernel_y) {
      cf_qconv_copy_u8(column, row, row_bytes);
      column += row_bytes;
      row += (uint32_t)input_width * input_channels;
    }
    return;
  }

  for (kernel_y = 0u; kernel_y < kernel_height; ++kernel_y) {
    const int32_t input_y = origin_y + kernel_y;
    uint16_t kernel_x;
    for (kernel_x = 0u; kernel_x < kernel_width; ++kernel_x) {
      const int32_t input_x = origin_x + kernel_x;
      if (input_y < 0 || input_y >= input_height || input_x < 0 ||
          input_x >= input_width) {
        cf_qconv_zero_u8(column, input_channels);
      } else {
        const uint8_t *source =
            input + ((uint32_t)input_y * input_width + (uint32_t)input_x) *
                        input_channels;
        cf_qconv_copy_u8(column, source, input_channels);
      }
      column += input_channels;
    }
  }
}

uint8_t *cf_qconv_matmul4x1_internal(
    const int8_t *weights, const uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    const int32_t *batch_norm_multiplier, const int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, cf_qconv_u8_epilogue_t mode);

#endif /* CF_GAP8_QCONV_INTERNAL_H */
