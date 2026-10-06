#ifndef CF_KERNEL_BENCH_FIXTURE_H
#define CF_KERNEL_BENCH_FIXTURE_H

/*
 * Scaffolding shared by every benchmark adapter: bump allocation out of the
 * harness-owned arenas, deterministic fixture generation, output checksums, and
 * the per-core cluster performance counters.
 *
 * Every adapter previously carried its own byte-identical copy of these. Keep
 * new adapters on these helpers so a change to the measurement protocol is made
 * once rather than nine times.
 */

#include <stddef.h>
#include <stdint.h>

#include "pmsis.h"

#include "kernel_bench/kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/* pi_cl_team_fork() configures the contiguous low-core mask (1 << N) - 1. */
#define KBENCH_MAX_CORES 8u

/* --- bump allocation ---------------------------------------------------- */

typedef struct {
  uintptr_t cursor;
  uintptr_t end;
} kbench_arena_t;

static inline uintptr_t kbench_align_up(uintptr_t value, size_t alignment) {
  return (value + alignment - 1u) & ~((uintptr_t)alignment - 1u);
}

static inline void kbench_arena_init(kbench_arena_t *arena, void *base,
                                     size_t bytes) {
  arena->cursor = (uintptr_t)base;
  arena->end = arena->cursor + bytes;
}

/* Returns NULL when the arena cannot satisfy the request; callers report
 * KBENCH_ERR_NO_MEMORY rather than silently shrinking the fixture. */
static inline void *kbench_arena_take(kbench_arena_t *arena, size_t bytes,
                                      size_t alignment) {
  const uintptr_t start = kbench_align_up(arena->cursor, alignment);

  if (start > arena->end || bytes > (size_t)(arena->end - start)) return NULL;
  arena->cursor = start + bytes;
  return (void *)start;
}

/* --- deterministic fixtures --------------------------------------------- */

/* Numerical Recipes LCG. Seeded from kbench_case_t::seed so a reported result is
 * reproducible from the manifest alone. */
static inline uint32_t kbench_lcg(uint32_t *state) {
  *state = *state * 1664525u + 1013904223u;
  return *state;
}

/* FNV-1a over the output tensor. The checksum is a cheap report field; exactness
 * is always decided by a full element compare in check(). */
uint32_t kbench_checksum_u8(const uint8_t *values, uint32_t count);
uint32_t kbench_checksum_i32(const int32_t *values, uint32_t count);

/* --- cluster performance counters ---------------------------------------- */

typedef struct {
  uint32_t cycles[KBENCH_MAX_CORES];
  uint32_t tcdm_contentions[KBENCH_MAX_CORES];
  uint32_t load_stalls[KBENCH_MAX_CORES];
} kbench_counters_t;

/*
 * Called from inside the forked team, immediately around the measured work:
 *
 *   kbench_perf_begin();
 *   ... repetitions of the kernel under test ...
 *   kbench_perf_end(&state->counters, pi_core_id());
 *
 * begin() reaches a team barrier before starting the counters so every core
 * measures the same interval; the fork's join after end() is what makes the
 * complete set visible to the harness.
 */
void kbench_perf_begin(void);
void kbench_perf_end(kbench_counters_t *counters, uint32_t core);

/*
 * Fold per-core counters into a check record: kernel_cycles is the maximum over
 * participating cores (the team's makespan), while contention and stall events
 * accumulate across them.
 */
void kbench_counters_reduce(const kbench_counters_t *counters, uint32_t cores,
                            kbench_check_t *check);

#ifdef __cplusplus
}
#endif

#endif /* CF_KERNEL_BENCH_FIXTURE_H */
