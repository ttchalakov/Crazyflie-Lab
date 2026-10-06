/*
 * Accumulator-only 4x2 convolution matmul.
 *
 * Identical arithmetic to cf_qconv_matmul4x2, stopping one step earlier: it
 * writes the int32 accumulators instead of requantising them to uint8. It
 * exists for the fused 2x2 pool of nn_bench's convpool and dwpw executors,
 * where requantisation is monotonic --
 * `bn_multiplier[c] >= 0` makes `q()` non-decreasing, so
 *
 *     max(q(a), q(b), q(c), q(d)) == q(max(a, b, c, d))
 *
 * and the four conv results in a pooled window can be reduced in int32 and
 * requantised **once** instead of four times. The output is bit-identical; only
 * the count of epilogues changes.
 *
 * Deliberately a separate function rather than a mode on the existing kernel,
 * duplicating one accumulation loop: threading an alternate output type
 * through cf_qconv_matmul4x2_impl would change the inner loop that every
 * convolution measurement in kernel_bench runs.
 *
 * The caller owns the monotonicity precondition. cf_qconv_pool_requant4()
 * below asserts it rather than assuming it, because a negative multiplier
 * silently inverts the reduction.
 */
#include <stdint.h>

#include "pmsis.h"

#include "cf_qconv.h"
#include "cf_qconv_internal.h"

void cf_qconv_matmul4x2_acc(const int8_t *weights, const uint8_t *input,
                            uint16_t output_channels, uint16_t columns,
                            const int8_t *bias, int32_t *acc0, int32_t *acc1) {
  const uint32_t full_channels = (uint32_t)output_channels & ~3u;
  const uint32_t words = (uint32_t)columns >> 2;
  const uint32_t tail = (uint32_t)columns & 3u;
  uint32_t channel;

  /* Four channels by two pixels, the same blocking as cf_qconv_matmul4x2: each
   * pair of input words feeds eight dot products, so the input loads amortise
   * 4x. A one-channel-at-a-time loop would load each input word four times
   * over -- the blocking is the point of the kernel, not an incidental detail. */
  for (channel = 0u; channel < full_channels; channel += 4u) {
    const int8_t *weight0 = weights + (size_t)channel * columns;
    const int8_t *weight1 = weight0 + columns;
    const int8_t *weight2 = weight1 + columns;
    const int8_t *weight3 = weight2 + columns;
    const uint8_t *input0 = input;
    const uint8_t *input1 = input + columns;
    int32_t sum0 = cf_qconv_bias_at(bias, channel);
    int32_t sum1 = cf_qconv_bias_at(bias, channel + 1u);
    int32_t sum2 = cf_qconv_bias_at(bias, channel + 2u);
    int32_t sum3 = cf_qconv_bias_at(bias, channel + 3u);
    int32_t sum4 = sum0, sum5 = sum1, sum6 = sum2, sum7 = sum3;
    uint32_t word;
    uint32_t lane;

    for (word = 0u; word < words; ++word) {
      const v4u x0 = *((const v4u *)input0);
      const v4u x1 = *((const v4u *)input1);
      const v4s w0 = *((const v4s *)weight0);
      const v4s w1 = *((const v4s *)weight1);
      const v4s w2 = *((const v4s *)weight2);
      const v4s w3 = *((const v4s *)weight3);
      sum0 = CF_QCONV_DOTP4(x0, w0, sum0);
      sum1 = CF_QCONV_DOTP4(x0, w1, sum1);
      sum2 = CF_QCONV_DOTP4(x0, w2, sum2);
      sum3 = CF_QCONV_DOTP4(x0, w3, sum3);
      sum4 = CF_QCONV_DOTP4(x1, w0, sum4);
      sum5 = CF_QCONV_DOTP4(x1, w1, sum5);
      sum6 = CF_QCONV_DOTP4(x1, w2, sum6);
      sum7 = CF_QCONV_DOTP4(x1, w3, sum7);
      weight0 += 4; weight1 += 4; weight2 += 4; weight3 += 4;
      input0 += 4; input1 += 4;
    }
    for (lane = 0u; lane < tail; ++lane) {
      const int32_t x0 = *input0++;
      const int32_t x1 = *input1++;
      const int32_t w0 = *weight0++;
      const int32_t w1 = *weight1++;
      const int32_t w2 = *weight2++;
      const int32_t w3 = *weight3++;
      sum0 += x0 * w0; sum1 += x0 * w1; sum2 += x0 * w2; sum3 += x0 * w3;
      sum4 += x1 * w0; sum5 += x1 * w1; sum6 += x1 * w2; sum7 += x1 * w3;
    }
    acc0[channel] = sum0;
    acc0[channel + 1u] = sum1;
    acc0[channel + 2u] = sum2;
    acc0[channel + 3u] = sum3;
    acc1[channel] = sum4;
    acc1[channel + 1u] = sum5;
    acc1[channel + 2u] = sum6;
    acc1[channel + 3u] = sum7;
  }

  for (; channel < output_channels; ++channel) {
    const int8_t *weight = weights + (size_t)channel * columns;
    const uint8_t *input0 = input;
    const uint8_t *input1 = input + columns;
    int32_t sum0 = cf_qconv_bias_at(bias, channel);
    int32_t sum1 = sum0;
    uint32_t word;
    uint32_t lane;

    for (word = 0u; word < words; ++word) {
      const v4s w = *((const v4s *)weight);
      sum0 = CF_QCONV_DOTP4(*((const v4u *)input0), w, sum0);
      sum1 = CF_QCONV_DOTP4(*((const v4u *)input1), w, sum1);
      weight += 4; input0 += 4; input1 += 4;
    }
    for (lane = 0u; lane < tail; ++lane) {
      const int32_t value = *weight++;
      sum0 += (int32_t)*input0++ * value;
      sum1 += (int32_t)*input1++ * value;
    }
    acc0[channel] = sum0;
    acc1[channel] = sum1;
  }
}

/*
 * Reduce four accumulator vectors and requantise once per channel.
 *
 * Returns 0 and writes nothing if any multiplier is negative: the identity this
 * rests on holds only for a non-decreasing q(), and a wrong answer is worse
 * than a refused one.
 */
int cf_qconv_pool_requant4(const int32_t *acc0, const int32_t *acc1,
                           const int32_t *acc2, const int32_t *acc3,
                           uint16_t output_channels, uint16_t output_shift,
                           const int32_t *batch_norm_multiplier,
                           const int32_t *batch_norm_bias, uint8_t *output) {
  uint32_t channel;

  for (channel = 0u; channel < output_channels; ++channel) {
    if (batch_norm_multiplier[channel] < 0) return 0;
  }
  for (channel = 0u; channel < output_channels; ++channel) {
    int32_t best = acc0[channel];
    if (acc1[channel] > best) best = acc1[channel];
    if (acc2[channel] > best) best = acc2[channel];
    if (acc3[channel] > best) best = acc3[channel];
    output[channel] = cf_qconv_quantize(
        CF_QCONV_U8_BATCHNORM_RELU, best, 1u, output_shift,
        batch_norm_multiplier, batch_norm_bias, channel);
  }
  return 1;
}
