#ifndef CF_GAP8_DORY_PROFILE_H
#define CF_GAP8_DORY_PROFILE_H

#include <stdint.h>

/*
 * Compute-versus-orchestration decomposition for DORY graphs.
 *
 * Every overlaid kernel accumulates the cluster cycles it actually spends, and
 * DORY's own per-layer counter reports each layer's total. The difference is
 * DMA wait, tiling, dispatch and barriers: the part a compiler change could
 * address and a kernel change cannot. examples/kernel_bench/DORY_AND_DEEPLOY.md
 * gives the split for the networks nn_bench runs.
 *
 * The harness drives it: it calls cf_dory_prof_reset() before the network runs
 * and cf_dory_prof_report(total) after it. examples/nn_bench does, when built
 * with CF_DORY_PROFILE=1.
 *
 * `total_cycles` must have the caller's own verification subtracted from it:
 * DORY reads its per-layer counter after dory_layer_done() returns, and that
 * hook is where a harness checks the layer's output against a golden, which can
 * take most of a layer's cycles.
 *
 * Deliberately cheap: two counter reads and an add per kernel invocation, and
 * the whole thing compiles to nothing unless CF_DORY_PROFILE=1. It reads
 * PI_PERF_CYCLES without reconfiguring or resetting it, so it does not disturb
 * the layer-level measurement DORY is taking around it.
 */

#ifndef CF_DORY_PROFILE
#define CF_DORY_PROFILE 0
#endif

#if CF_DORY_PROFILE

#include "pmsis.h"

/* Cluster-side accumulators. Only core 0 writes, inside an active team. */
extern uint32_t cf_dory_prof_kernel_cycles;
extern uint32_t cf_dory_prof_kernel_calls;

/*
 * Team arrival skew: how far apart the eight cores enter the kernel.
 *
 * DORY forks the team and every core calls the overlaid kernel, so the spread
 * of entry timestamps across cores is the skew with which the team arrives,
 * measured without a DORY change. It is one of the candidate causes of a tiled
 * layer's bimodal timing.
 *
 * Each core stamps its own slot; core 0 reduces at END, by which point the
 * team's internal barrier has guaranteed every slot is written for this call.
 */
#define CF_DORY_PROF_MAX_CORES 8u
extern uint32_t cf_dory_prof_entry[CF_DORY_PROF_MAX_CORES];
extern uint32_t cf_dory_prof_skew_cycles;
extern uint32_t cf_dory_prof_skew_max;

/* The counter is free-running and owned by the caller's measurement, so this
 * samples rather than configures. Every core reads it -- one extra read per
 * core per call -- but only core 0 accumulates the elapsed time, because the
 * others are inside the same team and would multiply-count the same wall. */
#define CF_DORY_PROF_BEGIN()                                                  \
  const uint32_t cf_dory_prof_core_ = (uint32_t)pi_core_id();                 \
  const uint32_t cf_dory_prof_t0_ = (uint32_t)pi_perf_read(PI_PERF_CYCLES);   \
  const int cf_dory_prof_master_ = (cf_dory_prof_core_ == 0u);                \
  if (cf_dory_prof_core_ < CF_DORY_PROF_MAX_CORES) {                          \
    cf_dory_prof_entry[cf_dory_prof_core_] = cf_dory_prof_t0_;                \
  }

#define CF_DORY_PROF_END()                                                    \
  do {                                                                        \
    if (cf_dory_prof_master_) {                                               \
      uint32_t cf_dory_prof_n_ = (uint32_t)pi_cl_team_nb_cores();             \
      uint32_t cf_dory_prof_lo_ = cf_dory_prof_entry[0];                      \
      uint32_t cf_dory_prof_hi_ = cf_dory_prof_lo_;                           \
      uint32_t cf_dory_prof_i_;                                               \
      if (cf_dory_prof_n_ > CF_DORY_PROF_MAX_CORES) {                         \
        cf_dory_prof_n_ = CF_DORY_PROF_MAX_CORES;                             \
      }                                                                       \
      for (cf_dory_prof_i_ = 1u; cf_dory_prof_i_ < cf_dory_prof_n_;           \
           ++cf_dory_prof_i_) {                                               \
        const uint32_t cf_dory_prof_v_ = cf_dory_prof_entry[cf_dory_prof_i_]; \
        if (cf_dory_prof_v_ < cf_dory_prof_lo_) {                             \
          cf_dory_prof_lo_ = cf_dory_prof_v_;                                 \
        }                                                                     \
        if (cf_dory_prof_v_ > cf_dory_prof_hi_) {                             \
          cf_dory_prof_hi_ = cf_dory_prof_v_;                                 \
        }                                                                     \
      }                                                                       \
      cf_dory_prof_skew_cycles += cf_dory_prof_hi_ - cf_dory_prof_lo_;        \
      if (cf_dory_prof_hi_ - cf_dory_prof_lo_ > cf_dory_prof_skew_max) {      \
        cf_dory_prof_skew_max = cf_dory_prof_hi_ - cf_dory_prof_lo_;          \
      }                                                                       \
      cf_dory_prof_kernel_cycles +=                                           \
          pi_perf_read(PI_PERF_CYCLES) - cf_dory_prof_t0_;                    \
      cf_dory_prof_kernel_calls += 1u;                                        \
    }                                                                         \
  } while (0)

void cf_dory_prof_reset(void);
void cf_dory_prof_report(uint32_t total_cycles);

#else

#define CF_DORY_PROF_BEGIN() ((void)0)
#define CF_DORY_PROF_END() ((void)0)
#define cf_dory_prof_reset() ((void)0)
#define cf_dory_prof_report(total_cycles) ((void)(total_cycles))

#endif /* CF_DORY_PROFILE */

#endif /* CF_GAP8_DORY_PROFILE_H */
