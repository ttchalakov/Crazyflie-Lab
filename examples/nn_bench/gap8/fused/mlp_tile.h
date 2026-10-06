/* The fused MLP's arithmetic: one tile of output rows, the epilogue fused. */
#ifndef NNBENCH_MLP_TILE_H
#define NNBENCH_MLP_TILE_H

#include <stddef.h>
#include <stdint.h>

#include "pmsis.h"

#define MLP_DOTP4(a, b, c) __builtin_pulp_sdotusp4((a), (b), (c))
#define MLP_CLIP_U8(v) __builtin_pulp_clipu_r((v), 255)

/* One output row: dot(input[k], weights[k]), the tail scalar. */
static inline int32_t mlp_row(const uint8_t *input, const int8_t *weights, uint32_t k, int32_t acc) {
  uint32_t i = 0u;
  for (; i + 4u <= k; i += 4u) acc = MLP_DOTP4(*(const v4u *)(input + i), *(const v4s *)(weights + i), acc);
  for (; i < k; ++i) acc += (int32_t)input[i] * weights[i];
  return acc;
}

/* Rows [first, last) of a tile, requantised: clip((acc * mult) >> shift, 0, 255).
 * Four rows at a time, so the accumulators never reach memory. */
static inline void mlp_tile(const uint8_t *input, const int8_t *tile, const int32_t *bias, uint8_t *out,
                            uint32_t k, uint32_t first, uint32_t last, int32_t mult, uint16_t shift) {
  uint32_t r = first;
  for (; r + 4u <= last; r += 4u) {
    int32_t a0 = mlp_row(input, tile + (size_t)r * k, k, bias[r]);
    int32_t a1 = mlp_row(input, tile + (size_t)(r + 1u) * k, k, bias[r + 1u]);
    int32_t a2 = mlp_row(input, tile + (size_t)(r + 2u) * k, k, bias[r + 2u]);
    int32_t a3 = mlp_row(input, tile + (size_t)(r + 3u) * k, k, bias[r + 3u]);
    out[r] = MLP_CLIP_U8((int32_t)(((int64_t)a0 * mult) >> shift));
    out[r + 1u] = MLP_CLIP_U8((int32_t)(((int64_t)a1 * mult) >> shift));
    out[r + 2u] = MLP_CLIP_U8((int32_t)(((int64_t)a2 * mult) >> shift));
    out[r + 3u] = MLP_CLIP_U8((int32_t)(((int64_t)a3 * mult) >> shift));
  }
  for (; r < last; ++r)
    out[r] = MLP_CLIP_U8((int32_t)(((int64_t)mlp_row(input, tile + (size_t)r * k, k, bias[r]) * mult) >> shift));
}

/* The same, raw int32: the output layer. */
static inline void mlp_tile_i32(const uint8_t *input, const int8_t *tile, const int32_t *bias, int32_t *out,
                                uint32_t k, uint32_t first, uint32_t last) {
  for (uint32_t r = first; r < last; ++r) out[r] = mlp_row(input, tile + (size_t)r * k, k, bias[r]);
}

/* Rows per L1 weight tile: as many as fit 12 KB, at most 64, a multiple of 4. */
static inline uint32_t mlp_tile_rows(uint32_t k) {
  uint32_t rows = k ? 12288u / k : 0u;
  if (rows > 64u) rows = 64u;
  rows &= ~3u;
  return rows ? rows : 4u;
}

#endif
