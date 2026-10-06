#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "cf_qlinear.h"

uint16_t cf_qlinear_u8s8_i32_padded_input_dim(uint16_t input_dim) {
  if (input_dim == 0u || input_dim > UINT16_MAX - 3u) {
    return 0u;
  }
  return (uint16_t)((input_dim + 3u) & ~3u);
}

int cf_qlinear_u8s8_i32_prepare_input(uint8_t *prepared,
                                     uint16_t prepared_dim,
                                     const uint8_t *source,
                                     uint16_t input_dim) {
  const uint16_t required =
      cf_qlinear_u8s8_i32_padded_input_dim(input_dim);

  if (prepared == NULL || source == NULL || required == 0u ||
      prepared_dim < required || (prepared_dim & 3u) != 0u) {
    return -1;
  }
  memcpy(prepared, source, input_dim);
  memset(prepared + input_dim, 0, (size_t)prepared_dim - input_dim);
  return 0;
}

int cf_qlinear_u8s8_i32_prepare_weights(int8_t *prepared,
                                       uint16_t prepared_stride_bytes,
                                       const int8_t *source,
                                       uint16_t source_stride_bytes,
                                       uint16_t input_dim,
                                       uint16_t output_dim) {
  const uint16_t required =
      cf_qlinear_u8s8_i32_padded_input_dim(input_dim);
  uint16_t output_index;

  if (prepared == NULL || source == NULL || required == 0u ||
      output_dim == 0u || source_stride_bytes < input_dim ||
      prepared_stride_bytes < required ||
      (prepared_stride_bytes & 3u) != 0u) {
    return -1;
  }
  for (output_index = 0u; output_index < output_dim; ++output_index) {
    int8_t *destination_row =
        prepared + (size_t)output_index * prepared_stride_bytes;
    const int8_t *source_row =
        source + (size_t)output_index * source_stride_bytes;

    memcpy(destination_row, source_row, input_dim);
    memset(destination_row + input_dim, 0,
           (size_t)prepared_stride_bytes - input_dim);
  }
  return 0;
}
