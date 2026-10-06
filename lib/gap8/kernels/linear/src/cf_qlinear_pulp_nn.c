#include "pmsis.h"

#include "cf_qlinear.h"

/* Exact symbol supplied by the pinned PULP-NN source in linear.mk. Its bias
 * parameter remains int8_t * for ABI compatibility even though it reads int32. */
extern void pulp_nn_linear_out_32(uint8_t *input, int8_t *bias,
                                  int32_t *output, int8_t *weights,
                                  uint16_t input_dim,
                                  uint16_t output_dim);

/* Called by every core of an already-forked team. The benchmark adapter needs
 * this split so it can place performance counters around the arithmetic without
 * measuring the fork itself. */
void cf_qlinear_u8s8_i32_pulp_nn_team(void *argument) {
  cf_qlinear_u8s8_i32_job_t *job = (cf_qlinear_u8s8_i32_job_t *)argument;

  pulp_nn_linear_out_32((uint8_t *)job->input, (int8_t *)job->bias,
                        job->output, (int8_t *)job->weights,
                        job->packed_input_dim, job->output_dim);
}
