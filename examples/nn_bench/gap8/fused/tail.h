/* What the convpool and dwpw executors share: staging into L1, and the network's tail. */
#ifndef NNBENCH_FUSED_TAIL_H
#define NNBENCH_FUSED_TAIL_H

#include "pmsis.h"

#include "fused.h"
#include "nn_bench_fused.h"

/* L2 -> L1 from core 0, waited for. In 32 KB pieces: a longer MCHAN command
 * moves only `size & 0x7FFF` bytes and reports success. */
static inline void fused_copy(void *l1, const void *l2, uint32_t bytes) {
  pi_cl_dma_cmd_t dma;
  for (uint32_t done = 0, n; done < bytes; done += n) {
    n = bytes - done < 32768u ? bytes - done : 32768u;
    pi_cl_dma_cmd((uint32_t)l2 + done, (uint32_t)l1 + done, n, PI_CL_DMA_DIR_EXT2LOC, &dma);
    pi_cl_dma_cmd_wait(&dma);
  }
}

/* The global average over `positions` HWC pixels (an exact integer mean, as the
 * graph defines it), then the int32 output layer; each reported to nn_bench.c. */
static inline void fused_tail(const uint8_t *in, uint32_t positions, uint8_t *gap, int32_t *logits,
                              const int8_t *fc, uint32_t core) {
  const uint32_t chunk = (FUSED_FC_IN + FUSED_CORES - 1u) / FUSED_CORES;
  for (uint32_t c = core * chunk; c < FUSED_FC_IN && c < (core + 1u) * chunk; c++) {
    int32_t sum = 0;
    for (uint32_t p = 0; p < positions; p++) sum += in[p * FUSED_FC_IN + c];
    gap[c] = (uint8_t)(sum / (int32_t)positions);
  }
  pi_cl_team_barrier(0);
  if (core == 0u) nnbench_layer(FUSED_GAP_LAYER, gap, FUSED_FC_IN);
  for (uint32_t k = core; k < FUSED_CLASSES; k += FUSED_CORES) {
    const int8_t *row = fc + k * FUSED_FC_IN;
    int32_t acc = fusedFcBias[k];
    uint32_t i = 0;
    for (; i + 4u <= FUSED_FC_IN; i += 4u)
      acc = __builtin_pulp_sdotusp4(*(const v4u *)(gap + i), *(const v4s *)(row + i), acc);
    for (; i < FUSED_FC_IN; i++) acc += (int32_t)gap[i] * row[i];
    logits[k] = acc;
  }
  pi_cl_team_barrier(0);
  if (core == 0u) nnbench_layer(FUSED_FC_LAYER, logits, FUSED_CLASSES * 4u);
}

#endif
