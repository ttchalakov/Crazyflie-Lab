/*
 * Fused depthwise/pointwise: blocks of a 3x3 depthwise, a 1x1 pointwise and a
 * 2x2 max pool, in one cluster team.
 *
 * A depthwise output feeds its pointwise at the same resolution, so the
 * intermediate cannot be materialised in L1 at all (the first block of
 * dwpw-24x24c16-c32mp-c64mp-c96mp-out4 alone needs more than L1 holds). It is
 * fused per pixel instead: for each pooled output pixel the four depthwise
 * vectors feeding it go into a small scratch, which is already the pointwise's
 * im2col (a 1x1 convolution over C inputs is a matmul with C columns), and the
 * four pointwise results are pooled in int32 and requantised once. Every tensor
 * is HWC, so no layout transpose happens between blocks (DORY's depthwise moves
 * its input HWC to CHW per channel). Activations stay in L1 for the whole
 * inference.
 */
#include "pmsis.h"

#include <string.h>

#include "cf_qconv.h"
#include "cf_qconv_internal.h"
#include "tail.h"

typedef struct {
  uint32_t in_h, in_w, in_c, out_c, dw_offset, pw_offset, mid_offset, out_offset, buffer, layer;
} block_t;

#define CHUNK 32u /* pointwise channels accumulated at once, 4 pixels x int32 */

static uint8_t gInput[NNBENCH_INPUT_LEN] __attribute__((aligned(4)));
static uint8_t *gAct[2], *gDw, *gGap;
static int8_t *gWeights;
static int32_t *gDwMult, *gPwMult, *gZero, *gAcc, *gLogits;

uint8_t *fused_input(void) { return gInput; }

int fused_init(void) {
  const uint32_t zero = FUSED_MAX_MID > FUSED_MAX_OUT ? FUSED_MAX_MID : FUSED_MAX_OUT;
  gAct[0] = pmsis_l1_malloc(FUSED_ACT0_BYTES);
  gAct[1] = pmsis_l1_malloc(FUSED_ACT1_BYTES);
  gWeights = pmsis_l1_malloc(FUSED_WEIGHT_BYTES);
  gDwMult = pmsis_l1_malloc(FUSED_MID_CHANNELS * 4);
  gPwMult = pmsis_l1_malloc(FUSED_OUT_CHANNELS * 4);
  gZero = pmsis_l1_malloc(zero * 4);
  gDw = pmsis_l1_malloc(4 * FUSED_CORES * FUSED_MAX_MID);
  gAcc = pmsis_l1_malloc(4 * FUSED_CORES * CHUNK * 4);
  gGap = pmsis_l1_malloc(FUSED_FC_IN);
  gLogits = pmsis_l1_malloc(FUSED_CLASSES * 4);
  if (!gAct[0] || !gAct[1] || !gWeights || !gDwMult || !gPwMult || !gZero || !gDw || !gAcc || !gGap || !gLogits)
    return -1;
  memcpy(gDwMult, fusedDwMult, FUSED_MID_CHANNELS * 4);
  memcpy(gPwMult, fusedPwMult, FUSED_OUT_CHANNELS * 4);
  memset(gZero, 0, zero * 4);
  return 0;
}

/* The depthwise output at pixel (y, x), every channel, HWC in and out. The
 * weights are tap-major (tap * C + channel), so four channels of one input
 * pixel meet their weights in adjacent bytes; four channels at a time. */
static void depthwise_pixel(const uint8_t *in, const block_t *b, const int8_t *taps, const int32_t *mult,
                            uint32_t y, uint32_t x, uint8_t *out) {
  const uint32_t channels = b->in_c;
  for (uint32_t c = 0; c < channels; c += 4) {
    int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int32_t ky = 0; ky < 3; ky++) {
      const int32_t iy = (int32_t)y + ky - 1;
      if (iy < 0 || iy >= (int32_t)b->in_h) continue;
      for (int32_t kx = 0; kx < 3; kx++) {
        const int32_t ix = (int32_t)x + kx - 1;
        if (ix < 0 || ix >= (int32_t)b->in_w) continue;
        const uint8_t *p = in + ((uint32_t)iy * b->in_w + (uint32_t)ix) * channels + c;
        const int8_t *w = taps + (ky * 3 + kx) * channels + c;
        a0 += (int32_t)p[0] * w[0];
        a1 += (int32_t)p[1] * w[1];
        a2 += (int32_t)p[2] * w[2];
        a3 += (int32_t)p[3] * w[3];
      }
    }
    out[c] = CF_QCONV_CLIP_U8((mult[c] * a0) >> FUSED_SHIFT);
    out[c + 1] = CF_QCONV_CLIP_U8((mult[c + 1] * a1) >> FUSED_SHIFT);
    out[c + 2] = CF_QCONV_CLIP_U8((mult[c + 2] * a2) >> FUSED_SHIFT);
    out[c + 3] = CF_QCONV_CLIP_U8((mult[c + 3] * a3) >> FUSED_SHIFT);
  }
}

static void block_run(const block_t *b, const uint8_t *in, uint8_t *out, uint32_t core) {
  const uint32_t mid = b->in_c, out_c = b->out_c, pooled_w = b->in_w >> 1;
  const uint32_t total = (b->in_h >> 1) * pooled_w, chunk = (total + FUSED_CORES - 1) / FUSED_CORES;
  const uint32_t first = cf_qconv_min_u32(core * chunk, total), last = cf_qconv_min_u32(first + chunk, total);
  uint8_t *const dw = gDw + 4 * core * FUSED_MAX_MID;
  int32_t *const acc = gAcc + 4 * core * CHUNK;
  const int8_t *const taps = gWeights + b->dw_offset;
  const int8_t *const pw = gWeights + FUSED_DW_BYTES + b->pw_offset;
  for (uint32_t u = first; u < last; u++) {
    const uint32_t y = 2 * (u / pooled_w), x = 2 * (u % pooled_w);
    for (uint32_t k = 0; k < 4; k++)
      depthwise_pixel(in, b, taps, gDwMult + b->mid_offset, y + (k >> 1), x + (k & 1), dw + k * mid);
    for (uint32_t base = 0; base < out_c; base += CHUNK) {
      const uint32_t span = cf_qconv_min_u32(CHUNK, out_c - base);
      cf_qconv_matmul4x2_acc(pw + base * mid, dw, span, mid, NULL, acc, acc + CHUNK);
      cf_qconv_matmul4x2_acc(pw + base * mid, dw + 2 * mid, span, mid, NULL, acc + 2 * CHUNK, acc + 3 * CHUNK);
      cf_qconv_pool_requant4(acc, acc + CHUNK, acc + 2 * CHUNK, acc + 3 * CHUNK, span, FUSED_SHIFT,
                             gPwMult + b->out_offset + base, gZero, out + u * out_c + base);
    }
  }
  pi_cl_team_barrier(0);
}

static void worker(void *arg) {
  (void)arg;
  const uint32_t core = pi_core_id();
  const block_t *b = NULL;
  if (core == 0u) {
    fused_copy(gWeights, fusedWeights, FUSED_WEIGHT_BYTES);
    fused_copy(gAct[0], gInput, NNBENCH_INPUT_LEN);
  }
  pi_cl_team_barrier(0);
  for (uint32_t i = 0; i < FUSED_BLOCKS; i++) {
    b = (const block_t *)fusedBlock[i];
    block_run(b, gAct[b->buffer ^ 1], gAct[b->buffer], core);
    if (core == 0u) nnbench_layer(b->layer, gAct[b->buffer], (b->in_h >> 1) * (b->in_w >> 1) * b->out_c);
  }
  fused_tail(gAct[b->buffer], (b->in_h >> 1) * (b->in_w >> 1), gGap, gLogits,
             gWeights + FUSED_WEIGHT_BYTES - FUSED_CLASSES * FUSED_FC_IN, core);
}

void fused_run(void) { pi_cl_team_fork(FUSED_CORES, worker, NULL); }
