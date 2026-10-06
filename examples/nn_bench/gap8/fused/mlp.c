/*
 * Fused MLP: the whole network in one cluster team.
 *
 * The activations ping-pong between two L1 buffers and never touch L2; DORY
 * writes every layer's output to L2 and reads it back. The weights stream into
 * L1 in double-buffered tiles: core 0 issues the DMA for tile t+1 before the
 * team computes tile t, so the transfer runs under the arithmetic. On l3 the
 * weights are DORY's own files, loaded into HyperRAM at boot, and core 0 reads
 * them into L2 a row block ahead of the L1 tiles, across layer boundaries, so
 * the network need not fit L2.
 */
#include "pmsis.h"

#include <string.h>

#include "fused.h"
#include "mlp_tile.h"
#include "nn_bench_fused.h"
#if FUSED_L3
#include "bsp/ram.h"
#include "mem.h"
#define BLOCK_TILES 4u /* L1 tiles per L2 row block */
#endif

static uint8_t gInput[NNBENCH_INPUT_LEN] __attribute__((aligned(4)));
static uint8_t *gAct[2];    /* L1 */
static int8_t *gTile[2];    /* L1 */
static pi_cl_dma_cmd_t gDma[2];
#if FUSED_L3
static uint32_t gL3[FUSED_LAYERS]; /* each layer's weights in HyperRAM */
static int8_t *gBlock[2];          /* L2 */
static pi_cl_ram_req_t gReq[2];
#endif

static uint32_t tile_bytes(uint32_t li) { return mlp_tile_rows(fusedIn[li]) * fusedIn[li]; }

uint8_t *fused_input(void) { return gInput; }

int fused_init(void) {
  uint32_t tile = 0u, block = 0u;
  for (uint32_t li = 0; li < FUSED_LAYERS; li++) {
    const uint32_t rows = mlp_tile_rows(fusedIn[li]);
    if (tile_bytes(li) > tile) tile = tile_bytes(li);
    if ((rows * 4u < fusedOut[li] ? rows * 4u : fusedOut[li]) * fusedIn[li] > block)
      block = (rows * 4u < fusedOut[li] ? rows * 4u : fusedOut[li]) * fusedIn[li];
  }
  gAct[0] = pmsis_l1_malloc(FUSED_WIDTH);
  gAct[1] = pmsis_l1_malloc(FUSED_WIDTH);
  gTile[0] = pmsis_l1_malloc(tile);
  gTile[1] = pmsis_l1_malloc(tile);
  if (!gAct[0] || !gAct[1] || !gTile[0] || !gTile[1]) return -1;
#if FUSED_L3
  mem_init();
  for (uint32_t li = 0; li < FUSED_LAYERS; li++) {
    gL3[li] = (uint32_t)ram_malloc(fusedFileBytes[li]);
    if (!gL3[li] || load_file_to_ram((void *)gL3[li], fusedFiles[li]) != fusedFileBytes[li]) return -2;
  }
  gBlock[0] = pi_l2_malloc(block);
  gBlock[1] = pi_l2_malloc(block);
  if (!gBlock[0] || !gBlock[1]) return -3;
#endif
  return 0;
}

/* `rows` rows of weights in L2 through the L1 tiles: out[base + r] for r < rows. */
static void rows_through_tiles(uint32_t li, const int8_t *w, uint32_t base, uint32_t rows, const uint8_t *in,
                               uint8_t *out, uint32_t core) {
  const uint32_t k = fusedIn[li], per = mlp_tile_rows(k), tiles = (rows + per - 1u) / per;
  const int32_t *bias = fusedBias + fusedBiasOffset[li] + base;
  if (core == 0u)
    pi_cl_dma_cmd((uint32_t)w, (uint32_t)gTile[0], (rows < per ? rows : per) * k, PI_CL_DMA_DIR_EXT2LOC, &gDma[0]);
  for (uint32_t t = 0; t < tiles; t++) {
    const uint32_t off = t * per, n = rows - off < per ? rows - off : per, buf = t & 1u;
    if (core == 0u) {
      pi_cl_dma_cmd_wait(&gDma[buf]);
      if (t + 1u < tiles) {
        const uint32_t next = rows - off - n < per ? rows - off - n : per;
        pi_cl_dma_cmd((uint32_t)(w + (size_t)(off + n) * k), (uint32_t)gTile[buf ^ 1u], next * k,
                      PI_CL_DMA_DIR_EXT2LOC, &gDma[buf ^ 1u]);
      }
    }
    pi_cl_team_barrier(0); /* the tile is in L1 */
    const uint32_t chunk = (n + FUSED_CORES - 1u) / FUSED_CORES;
    const uint32_t first = core * chunk < n ? core * chunk : n, last = first + chunk < n ? first + chunk : n;
    if (fusedMult[li])
      mlp_tile(in, gTile[buf], bias + off, out + base + off, k, first, last, fusedMult[li], FUSED_SHIFT);
    else
      mlp_tile_i32(in, gTile[buf], bias + off, (int32_t *)out + base + off, k, first, last);
    pi_cl_team_barrier(0); /* no core still reads the tile */
  }
}

#if FUSED_L3
static uint32_t block_rows(uint32_t li) { return mlp_tile_rows(fusedIn[li]) * BLOCK_TILES; }

/* The g-th row block of the whole network, flattened across layers so the read
 * for the next layer's first block overlaps this layer's last. */
static int next_block(uint32_t *li, uint32_t *base) {
  *base += block_rows(*li);
  if (*base >= fusedOut[*li]) {
    *base = 0;
    if (++*li == FUSED_LAYERS) return 0;
  }
  return 1;
}

static void read_block(uint32_t li, uint32_t base, uint32_t buf) {
  const uint32_t n = fusedOut[li] - base < block_rows(li) ? fusedOut[li] - base : block_rows(li);
  pi_cl_ram_read(get_ram_ptr(), gL3[li] + base * fusedIn[li], gBlock[buf], n * fusedIn[li], &gReq[buf]);
}
#endif

static void worker(void *arg) {
  (void)arg;
  const uint32_t core = pi_core_id();
#if FUSED_L3
  uint32_t li = 0, base = 0, buf = 0, cur = 0;
  if (core == 0u) read_block(0, 0, 0);
  for (;;) {
    const uint32_t n = fusedOut[li] - base < block_rows(li) ? fusedOut[li] - base : block_rows(li);
    uint32_t nli = li, nbase = base;
    const int more = next_block(&nli, &nbase);
    if (core == 0u) {
      pi_cl_ram_read_wait(&gReq[buf]);
      if (more) read_block(nli, nbase, buf ^ 1u);
    }
    pi_cl_team_barrier(0);
    rows_through_tiles(li, gBlock[buf], base, n, gAct[cur], gAct[cur ^ 1u], core);
    if (nli != li || !more) { /* the layer is complete */
      if (core == 0u) nnbench_layer(li, gAct[cur ^ 1u], fusedOut[li] * (fusedMult[li] ? 1u : 4u));
      cur ^= 1u;
    }
    if (!more) break;
    li = nli, base = nbase, buf ^= 1u;
  }
#else
  for (uint32_t li = 0, cur = 0; li < FUSED_LAYERS; li++, cur ^= 1u) {
    rows_through_tiles(li, fusedWeights + fusedWeightOffset[li], 0, fusedOut[li], gAct[cur], gAct[cur ^ 1u], core);
    if (core == 0u) nnbench_layer(li, gAct[cur ^ 1u], fusedOut[li] * (fusedMult[li] ? 1u : 4u));
  }
#endif
}

void fused_run(void) {
  memcpy(gAct[0], gInput, NNBENCH_INPUT_LEN);
  pi_cl_team_fork(FUSED_CORES, worker, NULL);
}
