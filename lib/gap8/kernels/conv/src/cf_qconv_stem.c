/*
 * Stem convolution: word-align the K axis by padding, then run the ordinary
 * flattened 4x2 matmul.
 *
 * The problem this solves: a 3x3 convolution over a 1-channel or 3-channel
 * input -- the first layer of every camera network -- has
 * K = kernel_h * kernel_w * input_channels = 9 or 27, neither a multiple of
 * four. Two things follow, and both are silent:
 *
 *   1. Consecutive weight rows sit at stride K, so every row whose index makes
 *      channel * K odd is not word aligned. cf_qconv_matmul4x2 loads them as
 *      v4s. GAP8 splits a misaligned word load in hardware at roughly double
 *      cost, so the kernel runs correctly and about half as fast. The same
 *      applies to the second im2col column at input + K.
 *   2. K & 3 is nonzero, so every output channel pays a scalar tail loop on top
 *      of its vector body -- one to three MACs at one per cycle where the body
 *      does four.
 *
 * In kernel_bench's `conv3x3_stems` the flattened path runs 0.451 MAC/cycle at
 * Cin=1 against 1.200 for the same shape at Cin=4, and the Cin=4 control is
 * what identifies alignment rather than channel count as the cause: it has
 * *more* arithmetic to do and is 2.7x faster.
 *
 * The fix is not a structural rewrite. Pad K up to a multiple of four with
 * zeros on both operands: the weight rows are rewritten once at layer entry
 * into job->weight_repack at the padded stride, and the im2col columns are
 * built at the same padded stride with their tails zeroed. Zero times anything
 * is zero, so the accumulator is bit-identical -- this is an exactness-
 * preserving transform, and the suite gates it as one.
 *
 * What it costs: 12 lanes of work where 9 were needed at Cin=1, so
 * executed_operations rises by a third while `operations` (the useful MACs)
 * stays put. That is the trade being measured -- more lanes, all of them
 * aligned and none of them scalar. The report carries both numbers so the
 * trade is visible rather than hidden in a single throughput figure.
 *
 * When K is already a multiple of four there is nothing to gain and the repack
 * would be pure overhead, so the entry point forwards to the unpadded flattened
 * path unchanged.
 */
#include <stdint.h>

#include "pmsis.h"

#include "cf_qconv.h"
#include "cf_qconv_internal.h"

void cf_qconv_u8s8_u8_flattened_team(void *argument);

static void cf_qconv_Ho_parallel_stem(cf_qconv_u8s8_u8_job_t *job) {
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t cores = (uint32_t)pi_cl_team_nb_cores();
  const uint32_t columns = (uint32_t)job->kernel_width * job->kernel_height *
                           job->input_channels;
  const uint32_t padded = (columns + 3u) & ~3u;
  const uint32_t output_channels = job->output_channels;
  const uint32_t pixels = (uint32_t)job->output_width * job->output_height;
  const uint32_t pairs = (pixels + 1u) >> 1;
  const uint32_t chunk = (pairs + cores - 1u) / cores;
  const uint32_t first = cf_qconv_min_u32(core * chunk, pairs);
  const uint32_t last = cf_qconv_min_u32(first + chunk, pairs);
  const int flag_relu = job->epilogue != CF_QCONV_U8_SHIFT_CLIP;
  const int flag_batch_norm = job->epilogue == CF_QCONV_U8_BATCHNORM_RELU;
  const cf_qconv_u8_epilogue_t mode = job->epilogue;
  int8_t *const repack = job->weight_repack;
  uint8_t *const columns_buffer = job->im2col + 2u * core * padded;
  uint32_t pair;

  /*
   * Repack the weight rows, split across the team by output channel.
   *
   * The source rows are misaligned by construction -- that is the whole point
   * -- so this copies bytes. It is O(output_channels * K) once per call against
   * O(output_channels * K * pixels) of arithmetic, so for anything but a
   * one-pixel tile it disappears. It is deliberately inside the timed region:
   * a DORY layer would pay it per tile, and a speedup that only holds when the
   * repack is free would not be a real one.
   */
  {
    const uint32_t channel_chunk = (output_channels + cores - 1u) / cores;
    const uint32_t channel_first =
        cf_qconv_min_u32(core * channel_chunk, output_channels);
    const uint32_t channel_last =
        cf_qconv_min_u32(channel_first + channel_chunk, output_channels);
    uint32_t channel;

    for (channel = channel_first; channel < channel_last; ++channel) {
      const int8_t *source = job->weights + channel * columns;
      int8_t *destination = repack + channel * padded;
      uint32_t index;

      for (index = 0u; index < columns; ++index) destination[index] = source[index];
      for (; index < padded; ++index) destination[index] = 0;
    }
  }

  /*
   * Zero this core's two column tails once, not once per patch.
   * cf_qconv_im2col_patch writes exactly `columns` bytes from the slot base, so
   * bytes [columns, padded) of each slot are never touched again for the rest
   * of the call and stay zero.
   */
  if (padded != columns) {
    cf_qconv_zero_u8(columns_buffer + columns, padded - columns);
    cf_qconv_zero_u8(columns_buffer + padded + columns, padded - columns);
  }
  pi_cl_team_barrier(0);

  for (pair = first; pair < last; ++pair) {
    const uint32_t pixel0 = pair << 1;
    const uint32_t pixel1 = pixel0 + 1u;
    uint8_t *const output0 = job->output + pixel0 * output_channels;

    cf_qconv_im2col_patch(job->input, columns_buffer, pixel0, job->input_width,
                          job->input_height, job->input_channels,
                          job->output_width, job->kernel_width,
                          job->kernel_height, job->padding_top,
                          job->padding_left, job->stride_width,
                          job->stride_height);
    if (pixel1 < pixels) {
      uint8_t *const output1 = job->output + pixel1 * output_channels;
      cf_qconv_im2col_patch(job->input, columns_buffer + padded, pixel1,
                            job->input_width, job->input_height,
                            job->input_channels, job->output_width,
                            job->kernel_width, job->kernel_height,
                            job->padding_top, job->padding_left,
                            job->stride_width, job->stride_height);
      (void)cf_qconv_matmul4x2(
          repack, columns_buffer, (uint16_t)output_channels, (uint16_t)padded,
          job->output_shift, job->output_multiplier,
          (int32_t *)job->batch_norm_multiplier,
          (int32_t *)job->batch_norm_bias, job->bias, output0, output1,
          flag_relu, flag_batch_norm);
    } else {
      (void)cf_qconv_matmul4x1_internal(
          repack, columns_buffer, (uint16_t)output_channels, (uint16_t)padded,
          job->output_shift, job->output_multiplier,
          job->batch_norm_multiplier, job->batch_norm_bias, job->bias, output0,
          mode);
    }
  }
  pi_cl_team_barrier(0);
}

void cf_qconv_u8s8_u8_stem_repack_team(void *argument) {
  cf_qconv_u8s8_u8_job_t *job = (cf_qconv_u8s8_u8_job_t *)argument;
  const uint32_t columns = (uint32_t)job->kernel_width * job->kernel_height *
                           job->input_channels;

  /* Nothing to align and nothing to pad: the repack would be cost with no
   * benefit, so run the path this one is a variant of. */
  if ((columns & 3u) == 0u || job->weight_repack == NULL) {
    cf_qconv_u8s8_u8_flattened_team(argument);
    return;
  }
  cf_qconv_Ho_parallel_stem(job);
}
