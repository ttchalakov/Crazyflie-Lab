#include <stdint.h>

#include "pmsis.h"

#include "cf_qactivation.h"

static int cf_qsoftmax_job_valid(const cf_qsoftmax_job_t *job) {
  return job != NULL && job->input != NULL && job->output != NULL &&
         job->exp_lut != NULL && job->rows != 0u && job->columns != 0u &&
         job->columns <= CF_QSOFTMAX_MAX_COLUMNS && job->cores != 0u &&
         job->cores <= CF_QACTIVATION_MAX_CORES;
}

void cf_qsoftmax_scalar_team(void *argument) {
  cf_qsoftmax_job_t *job = (cf_qsoftmax_job_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();

  if (!cf_qsoftmax_job_valid(job) || core >= job->cores) return;
  if (core == 0u) {
    /* Core 0 normalizes every row; the team meets at the barrier below. */
    cf_qsoftmax_reference(job);
  }
  pi_cl_team_barrier();
}
