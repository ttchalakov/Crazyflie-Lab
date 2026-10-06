#include <stdint.h>

#include "cf_qadd.h"
#include "cf_qadd_internal.h"

void cf_qadd_u8_reference(const cf_qadd_u8_job_t *job) {
  uint32_t elements;
  uint32_t index;

  if (job == 0 || job->input1 == 0 || job->input2 == 0 || job->output == 0 ||
      job->height == 0u || job->width == 0u || job->channels == 0u ||
      job->output_shift > 31u) {
    return;
  }
  elements = (uint32_t)job->height * job->width * job->channels;
  for (index = 0u; index < elements; ++index) {
    job->output[index] = cf_qadd_quantize(
        job->input1[index], job->input2[index], job->input1_multiplier,
        job->input2_multiplier, job->output_shift);
  }
}
