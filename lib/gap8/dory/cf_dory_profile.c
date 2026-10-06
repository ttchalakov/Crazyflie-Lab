#include "cf_dory_profile.h"

#if CF_DORY_PROFILE

/* pmsis.h first, and it is not optional: it remaps printf/snprintf onto the
 * SDK's shim. Reaching <stdio.h> first binds them to newlib symbols that are
 * never linked into a GAP8 image, which is a link error that only appears once
 * something actually calls the function -- so it hides behind --gc-sections
 * until the day the profiler is switched on. */
#include "pmsis.h"
#include <stdio.h>

/* PI_L2 so both the FC (which prints) and the cluster (which accumulates) see
 * the same storage. */
PI_L2 uint32_t cf_dory_prof_kernel_cycles = 0u;
PI_L2 uint32_t cf_dory_prof_kernel_calls = 0u;
PI_L2 uint32_t cf_dory_prof_entry[CF_DORY_PROF_MAX_CORES] = {0u};
PI_L2 uint32_t cf_dory_prof_skew_cycles = 0u;
PI_L2 uint32_t cf_dory_prof_skew_max = 0u;

void cf_dory_prof_reset(void) {
  cf_dory_prof_kernel_cycles = 0u;
  cf_dory_prof_kernel_calls = 0u;
  cf_dory_prof_skew_cycles = 0u;
  cf_dory_prof_skew_max = 0u;
}

/*
 * One machine-readable line, matching kernel_bench's KBENCH| record convention so
 * the host parser can pick it out of a mixed log.
 *
 * `total_cycles` is DORY's own figure for the same span. The residual is
 * everything that is not the overlaid kernel: DMA enqueue and wait, tile
 * pointer arithmetic, cluster dispatch, and barriers.
 */
void cf_dory_prof_report(uint32_t total_cycles) {
  const uint32_t kernel = cf_dory_prof_kernel_cycles;
  const uint32_t other = total_cycles > kernel ? total_cycles - kernel : 0u;
  const uint32_t permille =
      total_cycles != 0u ? (uint32_t)(((uint64_t)other * 1000u) / total_cycles)
                         : 0u;
  printf("CFPROF|kernel_cycles=%u|total_cycles=%u|other_cycles=%u"
         "|orchestration_permille=%u|kernel_calls=%u"
         "|team_skew_cycles=%u|team_skew_max=%u\n",
         (unsigned)kernel, (unsigned)total_cycles, (unsigned)other,
         (unsigned)permille, (unsigned)cf_dory_prof_kernel_calls,
         (unsigned)cf_dory_prof_skew_cycles,
         (unsigned)cf_dory_prof_skew_max);
}

#endif /* CF_DORY_PROFILE */
