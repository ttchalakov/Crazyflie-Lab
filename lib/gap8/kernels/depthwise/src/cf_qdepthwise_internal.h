#ifndef CF_QDEPTHWISE_INTERNAL_H
#define CF_QDEPTHWISE_INTERNAL_H

#include <stdint.h>

#ifndef CF_QDEPTHWISE_DORY_INT32_BIAS
#define CF_QDEPTHWISE_DORY_INT32_BIAS 0
#endif

/*
 * Every reader of the bias must go through here, the reference implementation
 * included. DORY keeps the historical int8_t * in the generated ABI but
 * serializes four bytes per output channel, exactly as cf_qconv_bias_at()
 * documents for the convolution path.
 *
 * The oracle used to read bias[channel] directly. That silently made the
 * host-side golden disagree with the kernel under
 * CF_QDEPTHWISE_DORY_INT32_BIAS=1 -- the exactness gate then failed for a
 * reason that had nothing to do with the kernel under test.
 */
static inline int32_t cf_qdepthwise_bias_at(const int8_t *bias,
                                            uint32_t channel) {
  if (bias == 0) return 0;
#if CF_QDEPTHWISE_DORY_INT32_BIAS
  return ((const int32_t *)(const void *)bias)[channel];
#else
  return bias[channel];
#endif
}

/* Bytes the caller must allocate and populate per output channel. */
#if CF_QDEPTHWISE_DORY_INT32_BIAS
#define CF_QDEPTHWISE_BIAS_STRIDE ((uint32_t)sizeof(int32_t))
#else
#define CF_QDEPTHWISE_BIAS_STRIDE ((uint32_t)sizeof(int8_t))
#endif

#endif /* CF_QDEPTHWISE_INTERNAL_H */
