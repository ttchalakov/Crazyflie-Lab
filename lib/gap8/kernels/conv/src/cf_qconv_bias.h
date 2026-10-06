#ifndef CF_GAP8_QCONV_BIAS_H
#define CF_GAP8_QCONV_BIAS_H

#include <stdint.h>

/*
 * The bias ABI on its own, split out of cf_qconv_internal.h so that the
 * reference implementation and the kernel_bench fixtures can agree with the
 * kernels without also inheriting the internal header's inline helpers -- both
 * of those translation units define their own cf_qconv_quantize().
 *
 * Every reader of the bias must go through cf_qconv_bias_at(), and every
 * writer must allocate CF_QCONV_BIAS_STRIDE bytes per output channel. A fixture
 * that writes one byte per channel while the kernel reads four does not fail
 * loudly; it just stops testing the branch that ships.
 */

#ifndef CF_QCONV_DORY_INT32_BIAS
#define CF_QCONV_DORY_INT32_BIAS 0
#endif

static inline int32_t cf_qconv_bias_at(const int8_t *bias, uint32_t channel) {
  if (bias == 0) return 0;
#if CF_QCONV_DORY_INT32_BIAS
  /* DORY keeps the historical int8_t pointer in the generated function ABI,
   * but serializes and DMAs four bytes for every convolution output channel. */
  return ((const int32_t *)(const void *)bias)[channel];
#else
  return bias[channel];
#endif
}

/* Bytes the caller must allocate and populate per output channel. */
#if CF_QCONV_DORY_INT32_BIAS
#define CF_QCONV_BIAS_STRIDE ((uint32_t)sizeof(int32_t))
#else
#define CF_QCONV_BIAS_STRIDE ((uint32_t)sizeof(int8_t))
#endif

#endif /* CF_GAP8_QCONV_BIAS_H */
