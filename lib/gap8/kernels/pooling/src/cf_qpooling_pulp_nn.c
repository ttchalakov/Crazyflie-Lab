#include <stdint.h>

#include "pmsis.h"
#include "pulp_nn_kernels.h"

#include "cf_qpooling.h"

/* Single-invocation team entry: called by every core of an already-forked
 * team so the benchmark adapter can place performance counters around the
 * arithmetic without measuring the fork. */
void cf_qpooling_pulp_nn_team(void *argument) {
  cf_qpooling_job_t *job = (cf_qpooling_job_t *)argument;
  if (job == NULL || job->input == NULL || job->output == NULL ||
      job->cores != 8u) {
    return;
  }
  {
    if (job->kind == CF_QPOOLING_MAX) {
      pulp_nn_maxpool(
          job->input, job->output, job->input_width, job->input_height,
          job->channels, job->output_width, job->output_height,
          job->kernel_width, job->kernel_height, job->padding_top,
          job->padding_bottom, job->padding_left, job->padding_right,
          job->stride_width, job->stride_height);
    } else if (job->kind == CF_QPOOLING_AVERAGE && job->count_include_pad) {
      pulp_nn_avgpool(
          job->input, job->output, job->output_multiplier,
          job->output_shift, job->output_add, job->input_width,
          job->input_height, job->channels, job->output_width,
          job->output_height, job->kernel_width, job->kernel_height,
          job->padding_top, job->padding_bottom, job->padding_left,
          job->padding_right, job->stride_width, job->stride_height,
          job->requantize);
    }
  }
}
