#include <stdint.h>

#include "pmsis.h"

#include "cf_qlinear.h"

extern void pulp_nn_linear(uint8_t *input, int8_t *bias, uint8_t *output,
                           int8_t *weights, int32_t *k, int32_t *lambda,
                           uint16_t output_multiplier, int8_t output_shift,
                           uint16_t input_dim, uint16_t output_dim,
                           int flag_relu, int flag_batch_norm);

/* Team-level entry so the benchmark adapter can place performance counters
 * around the arithmetic without measuring the fork itself. */
void cf_qlinear_u8s8_u8_pulp_nn_team(void *argument) {
  cf_qlinear_u8s8_u8_job_t *job = (cf_qlinear_u8s8_u8_job_t *)argument;
  const int flag_relu = job->epilogue != CF_QLINEAR_U8_SHIFT_CLIP;
  const int flag_batch_norm =
      job->epilogue == CF_QLINEAR_U8_BATCHNORM_RELU;

  pulp_nn_linear((uint8_t *)job->input, (int8_t *)job->bias, job->output,
                 (int8_t *)job->weights,
                 (int32_t *)job->batch_norm_multiplier,
                 (int32_t *)job->batch_norm_bias, job->output_multiplier,
                 job->output_shift, job->packed_input_dim, job->output_dim,
                 flag_relu, flag_batch_norm);
}
