#include <stdint.h>

#include "pmsis.h"

#include "cf_qadd.h"

extern void pulp_nn_add(uint8_t *input1, uint8_t *input2, uint8_t *output,
                        uint16_t input1_multiplier,
                        uint16_t input2_multiplier, uint16_t output_shift,
                        uint16_t height, uint16_t width, uint16_t channels);

/* Single-invocation team entry: called by every core of an already-forked
 * team so the benchmark adapter can place performance counters around the
 * arithmetic without measuring the fork. */
void cf_qadd_u8_pulp_nn_team(void *argument) {
  cf_qadd_u8_job_t *job = (cf_qadd_u8_job_t *)argument;

  pulp_nn_add((uint8_t *)job->input1, (uint8_t *)job->input2, job->output,
              job->input1_multiplier, job->input2_multiplier,
              job->output_shift, job->height, job->width, job->channels);
}
