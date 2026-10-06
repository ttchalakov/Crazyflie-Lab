/*
 * Fused convpool: 3x3 convolutions, each optionally followed by a 2x2 max pool,
 * in one cluster team.
 *
 * Activations live in two L1 buffers for the whole inference; DORY writes every
 * layer's output to L2 and reads it back. The pool is fused into its
 * convolution: for each pooled pixel the four convolution outputs feeding it are
 * accumulated in int32 and requantised once, so the unpooled output never
 * exists (exact, because the requantisation is non-decreasing for the
 * non-negative multipliers the graph uses). The weights are staged into L1 once
 * per inference. A first stage that pools a stride-2 convolution of a camera
 * frame reads the frame from L2 in bands of rows into the second activation
 * buffer, idle until then, so the frame never needs L1 of its own.
 */
#include "pmsis.h"

#include <string.h>

#include "cf_qconv.h"
#include "cf_qconv_internal.h"
#include "tail.h"

typedef struct {
  uint32_t in_h, in_w, in_c, out_h, out_w, out_c, stride, pool, weight_offset, channel_offset, buffer, layer;
} stage_t;

#define MAX_C FUSED_MAX_CHANNELS

static uint8_t gInput[NNBENCH_INPUT_LEN] __attribute__((aligned(4))); /* the frame, L2 */
static uint8_t *gAct[2];
static int8_t *gWeights;
static int32_t *gMult, *gZero, *gAcc, *gLogits;
static uint8_t *gIm2col, *gGap;

uint8_t *fused_input(void) { return gInput; }

int fused_init(void) {
  gAct[0] = pmsis_l1_malloc(FUSED_ACT0_BYTES);
  gAct[1] = pmsis_l1_malloc(FUSED_ACT1_BYTES);
  gWeights = pmsis_l1_malloc(FUSED_WEIGHT_BYTES);
  gMult = pmsis_l1_malloc(FUSED_CHANNELS * 4);
  gZero = pmsis_l1_malloc(MAX_C * 4);
  gAcc = pmsis_l1_malloc(4 * FUSED_CORES * MAX_C * 4);
  gIm2col = pmsis_l1_malloc(2 * FUSED_CORES * FUSED_MAX_COLUMNS);
  gGap = pmsis_l1_malloc(FUSED_FC_IN);
  gLogits = pmsis_l1_malloc(FUSED_CLASSES * 4);
  if (!gAct[0] || !gAct[1] || !gWeights || !gMult || !gZero || !gAcc || !gIm2col || !gGap || !gLogits) return -1;
  memcpy(gMult, fusedMult, FUSED_CHANNELS * 4);
  memset(gZero, 0, MAX_C * 4);
  return 0;
}

/* Output units [u0, u1) of a stage, split across the team: a pooled pixel, or
 * two adjacent pixels of an unpooled stage. `in` is addressed in whole-frame
 * coordinates (a band passes a base biased below its buffer). */
static void units(const stage_t *st, const uint8_t *in, uint8_t *out, uint32_t u0, uint32_t u1, uint32_t core) {
  const uint32_t cols = 9 * st->in_c, out_c = st->out_c, pooled_w = st->out_w >> 1;
  const uint32_t chunk = (u1 - u0 + FUSED_CORES - 1) / FUSED_CORES;
  const uint32_t first = u0 + cf_qconv_min_u32(core * chunk, u1 - u0), last = cf_qconv_min_u32(first + chunk, u1);
  uint8_t *const col = gIm2col + 2 * core * FUSED_MAX_COLUMNS;
  int32_t *const acc = gAcc + 4 * core * MAX_C;
  const int8_t *const w = gWeights + st->weight_offset;
  int32_t *const mult = gMult + st->channel_offset;
#define PATCH(dst, pixel) \
  cf_qconv_im2col_patch(in, dst, pixel, st->in_w, st->in_h, st->in_c, st->out_w, 3, 3, 1, 1, st->stride, st->stride)
  for (uint32_t u = first; u < last; u++) {
    if (st->pool) {
      const uint32_t py = u / pooled_w, px = u - py * pooled_w, row0 = 2 * py * st->out_w + 2 * px;
      PATCH(col, row0);
      PATCH(col + cols, row0 + 1);
      cf_qconv_matmul4x2_acc(w, col, out_c, cols, NULL, acc, acc + MAX_C);
      PATCH(col, row0 + st->out_w);
      PATCH(col + cols, row0 + st->out_w + 1);
      cf_qconv_matmul4x2_acc(w, col, out_c, cols, NULL, acc + 2 * MAX_C, acc + 3 * MAX_C);
      cf_qconv_pool_requant4(acc, acc + MAX_C, acc + 2 * MAX_C, acc + 3 * MAX_C, out_c, FUSED_SHIFT, mult, gZero,
                             out + u * out_c);
    } else {
      const uint32_t p0 = 2 * u;
      PATCH(col, p0);
      if (p0 + 1 < st->out_h * st->out_w) {
        PATCH(col + cols, p0 + 1);
        cf_qconv_matmul4x2(w, col, out_c, cols, FUSED_SHIFT, 1, mult, gZero, NULL, out + p0 * out_c,
                           out + (p0 + 1) * out_c, 1, 1);
      } else {
        cf_qconv_matmul4x1_internal(w, col, out_c, cols, FUSED_SHIFT, 1, mult, gZero, NULL, out + p0 * out_c,
                                    CF_QCONV_U8_BATCHNORM_RELU);
      }
    }
  }
#undef PATCH
}

/* The first stage, its frame read from L2 a band of FUSED_BAND_ROWS pooled rows
 * at a time into act[0]. Pooled row py reads input rows 4py-1 .. 4py+3. */
static void banded(const stage_t *st, uint8_t *out, uint32_t core) {
  const uint32_t pooled_w = st->out_w >> 1, pooled_h = st->out_h >> 1;
  for (uint32_t py0 = 0; py0 < pooled_h; py0 += FUSED_BAND_ROWS) {
    const uint32_t py1 = cf_qconv_min_u32(py0 + FUSED_BAND_ROWS, pooled_h);
    const uint32_t r0 = py0 ? 4 * py0 - 1 : 0, r1 = cf_qconv_min_u32(4 * py1, st->in_h);
    if (core == 0u) fused_copy(gAct[0], gInput + r0 * st->in_w, (r1 - r0) * st->in_w);
    pi_cl_team_barrier(0);
    units(st, gAct[0] - r0 * st->in_w, out, py0 * pooled_w, py1 * pooled_w, core);
    pi_cl_team_barrier(0); /* done with the band before the next overwrites it */
  }
}

static void worker(void *arg) {
  (void)arg;
  const uint32_t core = pi_core_id();
  const stage_t *st = NULL;
  if (core == 0u) fused_copy(gWeights, fusedWeights, FUSED_WEIGHT_BYTES);
  pi_cl_team_barrier(0);
  for (uint32_t i = 0; i < FUSED_STAGES; i++) {
    st = (const stage_t *)fusedStage[i];
    uint8_t *out = gAct[st->buffer];
    const uint32_t h = st->pool ? st->out_h >> 1 : st->out_h, w = st->pool ? st->out_w >> 1 : st->out_w;
    if (i == 0 && FUSED_BAND_ROWS) {
      banded(st, out, core);
    } else {
      units(st, i ? gAct[st->buffer ^ 1] : gInput, out, 0, st->pool ? h * w : (h * w + 1) / 2, core);
      pi_cl_team_barrier(0);
    }
    if (core == 0u) nnbench_layer(st->layer, out, h * w * st->out_c);
  }
  fused_tail(gAct[st->buffer], (st->pool ? (st->out_h >> 1) * (st->out_w >> 1) : st->out_h * st->out_w), gGap,
             gLogits, gWeights + FUSED_WEIGHT_BYTES - FUSED_CLASSES * FUSED_FC_IN, core);
}

void fused_run(void) { pi_cl_team_fork(FUSED_CORES, worker, NULL); }
