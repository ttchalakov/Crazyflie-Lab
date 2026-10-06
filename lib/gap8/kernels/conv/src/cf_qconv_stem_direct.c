/*
 * Direct 3x3 convolution for a single-channel input: no im2col.
 *
 * A camera stem has K = 9, so the flattened path's weight rows are not word
 * aligned. `stem_repack4` (cf_qconv_stem.c) pads K to 12 to align them, but
 * alignment is worth ~1.2x of lane throughput against 1.33x more lanes: 0.89x at
 * Cin=1 (it does win at Cin=3, where the padding is only 4%). This variant drops
 * im2col instead, keeping the weight reuse that makes `cf_qconv_matmul4x2` fast:
 * a loop over one pixel would reload three weight words per channel for three
 * dot products, where matmul4x2 shares each weight load across two pixels.
 *
 * What it does, and why each part is there:
 *
 *   - **two pixels per weight load.** A pair of adjacent output pixels is
 *     computed together, so the three weight words for a channel feed six dot
 *     products. With the six input words held live across every output channel
 *     the ratio is 30 loads to 48 dot products per pair, against matmul4x2's
 *     ~24 to 32 -- and without im2col's copy;
 *   - **weights pre-arranged** once per call as int8[Cout][3][4], each row
 *     (w0, w1, w2, 0). The zero lane keeps every weight load word-aligned and
 *     removes the per-channel scalar tail;
 *   - **borders share the arithmetic path.** A window that leaves the image is
 *     assembled zero-padded into a 4-byte scratch and then runs the *same* dot
 *     products. Borders are 25% of a 20x12 output against 7.4% of a 75x41 one,
 *     so a slower border path would dominate the small tile.
 *
 * The pad lane multiplies a byte that is read but not used, so the result is
 * bit-identical to the reference. The interior fast path reads four bytes and
 * uses three, so it needs one byte of slack past the third column; pixels whose
 * window ends at the last input column take the padded path. That is a
 * correctness requirement, not a tuning knob -- getting it wrong reads past the
 * input buffer.
 *
 * Reuses job->weight_repack (Cout * 12 bytes at Cin=1), the same buffer
 * `stem_repack4` uses, with a different interior layout.
 */
#include <stdint.h>

#include "pmsis.h"

#include "cf_qconv.h"
#include "cf_qconv_internal.h"

void cf_qconv_u8s8_u8_flattened_team(void *argument);

#define CF_QSTEM_TAPS 4u   /* 3 real taps plus the zero lane */
#define CF_QSTEM_ROWS 3u

typedef union {
  uint8_t byte[4];
  v4u word;
} cf_qstem_window_t;

/*
 * The three input words feeding one output pixel, zero-padded where the window
 * leaves the image. The caller decides `safe` once per pixel rather than per
 * row, so the interior path is three loads and no branching.
 */
CF_QCONV_ALWAYS_INLINE void cf_qstem_gather(const uint8_t *input,
                                            int32_t origin_y, int32_t origin_x,
                                            int32_t width, int32_t height,
                                            uint8_t safe, v4u *rows) {
  if (safe) {
    const uint8_t *base = input + (size_t)origin_y * width + origin_x;
    rows[0] = *(const v4u *)(base);
    rows[1] = *(const v4u *)(base + width);
    rows[2] = *(const v4u *)(base + 2 * width);
    return;
  }
  {
    uint32_t ky;
    for (ky = 0u; ky < CF_QSTEM_ROWS; ++ky) {
      cf_qstem_window_t window;
      const int32_t iy = origin_y + (int32_t)ky;
      uint32_t kx;

      window.word = (v4u){0u, 0u, 0u, 0u};
      if (iy >= 0 && iy < height) {
        for (kx = 0u; kx < 3u; ++kx) {
          const int32_t ix = origin_x + (int32_t)kx;
          if (ix >= 0 && ix < width) {
            window.byte[kx] = input[(size_t)iy * width + ix];
          }
        }
      }
      rows[ky] = window.word;
    }
  }
}

/*
 * One pair of output pixels. Channels are the inner loop so the six gathered
 * input words stay live across all of them, and the three weight words for a
 * channel are loaded once and feed both pixels.
 */
CF_QCONV_ALWAYS_INLINE void cf_qstem_pair(const cf_qconv_u8s8_u8_job_t *job,
                                          const int8_t *taps, const v4u *a,
                                          const v4u *b, uint8_t *out_a,
                                          uint8_t *out_b, uint32_t out_c,
                                          uint8_t have_b,
                                          cf_qconv_u8_epilogue_t mode) {
  uint32_t channel;

  for (channel = 0u; channel < out_c; ++channel) {
    const int8_t *const w =
        taps + (size_t)channel * CF_QSTEM_ROWS * CF_QSTEM_TAPS;
    const v4s w0 = *(const v4s *)(w);
    const v4s w1 = *(const v4s *)(w + CF_QSTEM_TAPS);
    const v4s w2 = *(const v4s *)(w + 2u * CF_QSTEM_TAPS);
    const int32_t seed = cf_qconv_bias_at(job->bias, channel);
    int32_t sum_a = seed;

    sum_a = CF_QCONV_DOTP4(a[0], w0, sum_a);
    sum_a = CF_QCONV_DOTP4(a[1], w1, sum_a);
    sum_a = CF_QCONV_DOTP4(a[2], w2, sum_a);
    out_a[channel] = cf_qconv_quantize(mode, sum_a, job->output_multiplier,
                                       job->output_shift,
                                       job->batch_norm_multiplier,
                                       job->batch_norm_bias, channel);
    if (have_b) {
      int32_t sum_b = seed;
      sum_b = CF_QCONV_DOTP4(b[0], w0, sum_b);
      sum_b = CF_QCONV_DOTP4(b[1], w1, sum_b);
      sum_b = CF_QCONV_DOTP4(b[2], w2, sum_b);
      out_b[channel] = cf_qconv_quantize(mode, sum_b, job->output_multiplier,
                                         job->output_shift,
                                         job->batch_norm_multiplier,
                                         job->batch_norm_bias, channel);
    }
  }
}

static void cf_qconv_stem_direct_run(cf_qconv_u8s8_u8_job_t *job) {
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t cores = (uint32_t)pi_cl_team_nb_cores();
  const uint32_t out_c = job->output_channels;
  const uint32_t out_w = job->output_width;
  const uint32_t pixels = (uint32_t)job->output_height * out_w;
  const uint32_t pairs = (pixels + 1u) >> 1;
  const int32_t width = (int32_t)job->input_width;
  const int32_t height = (int32_t)job->input_height;
  const int32_t stride_h = (int32_t)job->stride_height;
  const int32_t stride_w = (int32_t)job->stride_width;
  const int32_t pad_t = (int32_t)job->padding_top;
  const int32_t pad_l = (int32_t)job->padding_left;
  const cf_qconv_u8_epilogue_t mode = job->epilogue;
  int8_t *const taps = job->weight_repack;
  /* Split by pixel pair, matching the flattened traversal: splitting by output
   * row would leave 2 of 8 cores idle on a 12-row stem. */
  const uint32_t chunk = (pairs + cores - 1u) / cores;
  const uint32_t first = cf_qconv_min_u32(core * chunk, pairs);
  const uint32_t last = cf_qconv_min_u32(first + chunk, pairs);
  uint32_t pair;

  /*
   * Rearrange the compact 9-byte weight rows into Cout x 3 x (3 taps + zero),
   * split across the team by output channel. 96 bytes at Cout=8, once per call,
   * inside the timed region because a DORY layer would pay it per tile.
   */
  {
    const uint32_t channel_chunk = (out_c + cores - 1u) / cores;
    const uint32_t channel_first = cf_qconv_min_u32(core * channel_chunk, out_c);
    const uint32_t channel_last =
        cf_qconv_min_u32(channel_first + channel_chunk, out_c);
    uint32_t channel;

    for (channel = channel_first; channel < channel_last; ++channel) {
      const int8_t *source = job->weights + (size_t)channel * 9u;
      int8_t *destination =
          taps + (size_t)channel * CF_QSTEM_ROWS * CF_QSTEM_TAPS;
      uint32_t ky;
      for (ky = 0u; ky < CF_QSTEM_ROWS; ++ky) {
        destination[ky * CF_QSTEM_TAPS + 0u] = source[ky * 3u + 0u];
        destination[ky * CF_QSTEM_TAPS + 1u] = source[ky * 3u + 1u];
        destination[ky * CF_QSTEM_TAPS + 2u] = source[ky * 3u + 2u];
        destination[ky * CF_QSTEM_TAPS + 3u] = 0;
      }
    }
  }
  pi_cl_team_barrier(0);

  for (pair = first; pair < last; ++pair) {
    const uint32_t pixel_a = pair << 1;
    const uint32_t pixel_b = pixel_a + 1u;
    const uint32_t row_a = pixel_a / out_w;
    const uint32_t column_a = pixel_a - row_a * out_w;
    const int32_t origin_ya = (int32_t)row_a * stride_h - pad_t;
    const int32_t origin_xa = (int32_t)column_a * stride_w - pad_l;
    const uint8_t rows_safe = origin_ya >= 0 && origin_ya + 3 <= height;
    const uint8_t have_b = pixel_b < pixels;
    v4u a[CF_QSTEM_ROWS];
    v4u b[CF_QSTEM_ROWS];

    cf_qstem_gather(job->input, origin_ya, origin_xa, width, height,
                    (uint8_t)(rows_safe && origin_xa >= 0 &&
                              origin_xa + 4 <= width),
                    a);
    if (have_b) {
      /* b is the next pixel in flattened order, so it is the next column except
       * at a row boundary. Deriving it keeps one integer division per pair,
       * matching what the flattened path pays. */
      const uint32_t row_b = column_a + 1u < out_w ? row_a : row_a + 1u;
      const uint32_t column_b = column_a + 1u < out_w ? column_a + 1u : 0u;
      const int32_t origin_yb = (int32_t)row_b * stride_h - pad_t;
      const int32_t origin_xb = (int32_t)column_b * stride_w - pad_l;

      cf_qstem_gather(job->input, origin_yb, origin_xb, width, height,
                      (uint8_t)(origin_yb >= 0 && origin_yb + 3 <= height &&
                                origin_xb >= 0 && origin_xb + 4 <= width),
                      b);
    } else {
      b[0] = a[0];
      b[1] = a[1];
      b[2] = a[2];
    }
    cf_qstem_pair(job, taps, a, b, job->output + (size_t)pixel_a * out_c,
                  job->output + (size_t)pixel_b * out_c, out_c, have_b, mode);
  }
  pi_cl_team_barrier(0);
}

void cf_qconv_u8s8_u8_stem_direct_team(void *argument) {
  cf_qconv_u8s8_u8_job_t *job = (cf_qconv_u8s8_u8_job_t *)argument;

  /* Single-channel 3x3 only. Silently producing a wrong answer for a shape this
   * does not model would be far worse than declining it. */
  if (job->input_channels != 1u || job->kernel_width != 3u ||
      job->kernel_height != 3u || job->weight_repack == NULL) {
    cf_qconv_u8s8_u8_flattened_team(argument);
    return;
  }
  cf_qconv_stem_direct_run(job);
}
