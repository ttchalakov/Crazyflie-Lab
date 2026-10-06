#include "kernel_bench/fixture.h"

uint32_t kbench_checksum_u8(const uint8_t *values, uint32_t count) {
  uint32_t hash = 2166136261u;
  uint32_t index;

  for (index = 0u; index < count; ++index) {
    hash ^= values[index];
    hash *= 16777619u;
  }
  return hash;
}

uint32_t kbench_checksum_i32(const int32_t *values, uint32_t count) {
  uint32_t hash = 2166136261u;
  uint32_t index;

  for (index = 0u; index < count; ++index) {
    const uint32_t value = (uint32_t)values[index];
    uint32_t byte_index;

    for (byte_index = 0u; byte_index < 4u; ++byte_index) {
      hash ^= (value >> (byte_index * 8u)) & 0xffu;
      hash *= 16777619u;
    }
  }
  return hash;
}

void kbench_perf_begin(void) {
  pi_perf_conf((1u << PI_PERF_CYCLES) | (1u << PI_PERF_TCDM_CONT) |
               (1u << PI_PERF_LD_STALL));
  pi_perf_reset();
  pi_perf_stop();
  /* Start every core's counters from the same point so the maximum below is a
   * makespan rather than a fork-skew artifact. No trailing barrier: the fork's
   * own join already implies one. */
  pi_cl_team_barrier(0);
  pi_perf_start();
}

void kbench_perf_end(kbench_counters_t *counters, uint32_t core) {
  pi_perf_stop();
  if (counters != NULL && core < KBENCH_MAX_CORES) {
    counters->cycles[core] = pi_perf_read(PI_PERF_CYCLES);
    counters->tcdm_contentions[core] = pi_perf_read(PI_PERF_TCDM_CONT);
    counters->load_stalls[core] = pi_perf_read(PI_PERF_LD_STALL);
  }
}

void kbench_counters_reduce(const kbench_counters_t *counters, uint32_t cores,
                            kbench_check_t *check) {
  uint32_t core;

  if (counters == NULL || check == NULL) return;
  if (cores > KBENCH_MAX_CORES) cores = KBENCH_MAX_CORES;
  for (core = 0u; core < cores; ++core) {
    if (counters->cycles[core] > check->kernel_cycles) {
      check->kernel_cycles = counters->cycles[core];
    }
    check->tcdm_contentions += counters->tcdm_contentions[core];
    check->load_stalls += counters->load_stalls[core];
  }
}
