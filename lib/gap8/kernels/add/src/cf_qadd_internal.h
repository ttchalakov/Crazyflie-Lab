#ifndef CF_GAP8_QADD_INTERNAL_H
#define CF_GAP8_QADD_INTERNAL_H

#include <stdint.h>

static inline uint8_t cf_qadd_quantize(uint8_t input1, uint8_t input2,
                                       uint16_t multiplier1,
                                       uint16_t multiplier2,
                                       uint16_t output_shift) {
  const int32_t weighted =
      (int32_t)input1 * (int16_t)multiplier1 +
      (int32_t)input2 * (int16_t)multiplier2;
  const uint32_t integer_image = (uint32_t)weighted;
  const uint16_t shifted = (uint16_t)(integer_image >> output_shift);
  return shifted > UINT8_MAX ? UINT8_MAX : (uint8_t)shifted;
}

#endif /* CF_GAP8_QADD_INTERNAL_H */
