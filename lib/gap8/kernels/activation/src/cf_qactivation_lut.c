#include <stdint.h>

#include "pmsis.h"

#include "cf_qactivation.h"

static void cf_qactivation_partition(uint32_t elements, uint32_t core,
                                     uint32_t cores, uint32_t *start,
                                     uint32_t *end) {
  *start = (uint32_t)(((uint64_t)elements * core) / cores);
  *end = (uint32_t)(((uint64_t)elements * (core + 1u)) / cores);
}

void cf_qactivation_lut_scalar_team(void *argument) {
  cf_qactivation_lut_job_t *job = (cf_qactivation_lut_job_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();
  uint32_t start;
  uint32_t end;
  uint32_t index;

  if (job == NULL || job->input == NULL || job->output == NULL ||
      job->lut == NULL || job->cores == 0u ||
      job->cores > CF_QACTIVATION_MAX_CORES || core >= job->cores) {
    return;
  }
  cf_qactivation_partition(job->elements, core, job->cores, &start, &end);
  for (index = start; index < end; ++index) {
    job->output[index] = job->lut[job->input[index]];
  }
}

void cf_qactivation_lut_packed4_team(void *argument) {
  cf_qactivation_lut_job_t *job = (cf_qactivation_lut_job_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t words = job != NULL ? job->elements >> 2 : 0u;
  const uint32_t tail_start = words << 2;
  uint32_t word_start;
  uint32_t word_end;

  if (job == NULL || job->input == NULL || job->output == NULL ||
      job->lut == NULL || job->cores == 0u ||
      job->cores > CF_QACTIVATION_MAX_CORES || core >= job->cores) {
    return;
  }
  if ((((uintptr_t)job->input | (uintptr_t)job->output) & 3u) != 0u) {
    cf_qactivation_lut_scalar_team(argument);
    return;
  }

  cf_qactivation_partition(words, core, job->cores, &word_start, &word_end);
  {
    const v4u *input = (const v4u *)job->input + word_start;
    v4u *output = (v4u *)job->output + word_start;
    uint32_t word;
    for (word = word_start; word < word_end; ++word) {
      const v4u values = *input++;
      *output++ = __builtin_pulp_pack4(
          job->lut[values[0]], job->lut[values[1]], job->lut[values[2]],
          job->lut[values[3]]);
    }
    if (core == 0u) {
      uint32_t index;
      for (index = tail_start; index < job->elements; ++index) {
        job->output[index] = job->lut[job->input[index]];
      }
    }
  }
}
