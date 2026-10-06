#include <stdint.h>

#include "pmsis.h"

#include "cf_dory_profile.h"
#include "cf_qadd.h"
#include "cf_qadd_internal.h"

extern void cf_qadd_vendor(uint8_t *input1, uint8_t *input2, uint8_t *output,
                           uint16_t input1_multiplier,
                           uint16_t input2_multiplier, uint16_t output_shift,
                           uint16_t height, uint16_t width, uint16_t channels);

/* The packed traversal casts all three tiles to v4u. DORY's generated L1
 * offsets are normally word aligned, but nothing in its contract promises it,
 * and kernel_bench always allocates aligned so it can never catch a violation. */
static inline int cf_qadd_word_aligned(const cf_qadd_u8_job_t *job) {
  return ((((uintptr_t)job->input1 | (uintptr_t)job->input2 |
            (uintptr_t)job->output) &
           3u) == 0u);
}

static inline v4u cf_qadd_word(v4u input1, v4u input2,
                               uint16_t multiplier1,
                               uint16_t multiplier2,
                               uint16_t output_shift) {
  return __builtin_pulp_pack4(
      cf_qadd_quantize(input1[0], input2[0], multiplier1, multiplier2,
                       output_shift),
      cf_qadd_quantize(input1[1], input2[1], multiplier1, multiplier2,
                       output_shift),
      cf_qadd_quantize(input1[2], input2[2], multiplier1, multiplier2,
                       output_shift),
      cf_qadd_quantize(input1[3], input2[3], multiplier1, multiplier2,
                       output_shift));
}

void cf_qadd_u8_packed4_team(void *argument) {
  cf_qadd_u8_job_t *job = (cf_qadd_u8_job_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t cores = (uint32_t)pi_cl_team_nb_cores();
  const uint32_t elements =
      (uint32_t)job->height * job->width * job->channels;
  const uint32_t words = elements >> 2;
  const uint32_t chunk = (words + cores - 1u) / cores;
  const uint32_t first = core * chunk < words ? core * chunk : words;
  const uint32_t last = first + chunk < words ? first + chunk : words;
  const v4u *input1;
  const v4u *input2;
  v4u *output;
  uint32_t word = first;

  if (!cf_qadd_word_aligned(job)) {
    /* Scalar per-element fallback, matching cf_qactivation_lut_packed4_team
     * and cf_qpooling_channel4_team rather than reading past an odd base. */
    uint32_t index;
    const uint32_t start = (uint32_t)(((uint64_t)elements * core) / cores);
    const uint32_t stop = (uint32_t)(((uint64_t)elements * (core + 1u)) / cores);
    for (index = start; index < stop; ++index) {
      job->output[index] = cf_qadd_quantize(
          job->input1[index], job->input2[index], job->input1_multiplier,
          job->input2_multiplier, job->output_shift);
    }
    pi_cl_team_barrier(0);
    return;
  }
  input1 = (const v4u *)job->input1 + first;
  input2 = (const v4u *)job->input2 + first;
  output = (v4u *)job->output + first;

  for (; word + 1u < last; word += 2u) {
    const v4u input10 = *input1++;
    const v4u input20 = *input2++;
    const v4u input11 = *input1++;
    const v4u input21 = *input2++;
    *output++ = cf_qadd_word(input10, input20, job->input1_multiplier,
                             job->input2_multiplier, job->output_shift);
    *output++ = cf_qadd_word(input11, input21, job->input1_multiplier,
                             job->input2_multiplier, job->output_shift);
  }
  if (word < last) {
    *output = cf_qadd_word(*input1, *input2, job->input1_multiplier,
                           job->input2_multiplier, job->output_shift);
  }
  if (core == 0u) {
    uint32_t index;
    for (index = words << 2; index < elements; ++index) {
      job->output[index] = cf_qadd_quantize(
          job->input1[index], job->input2[index], job->input1_multiplier,
          job->input2_multiplier, job->output_shift);
    }
  }
  pi_cl_team_barrier(0);
}

#ifdef CF_QADD_REPLACE_PULP_NN
void pulp_nn_add(uint8_t *input1, uint8_t *input2, uint8_t *output,
                 uint16_t input1_multiplier, uint16_t input2_multiplier,
                 uint16_t output_shift, uint16_t height, uint16_t width,
                 uint16_t channels) {
  /* The residual add is 9-11% of the ResNet, so leaving it uninstrumented
   * moved that much of the graph from `kernel` into `other`. */
  CF_DORY_PROF_BEGIN();
  if (input1 == 0 || input2 == 0 || output == 0 || height == 0u ||
      width == 0u || channels == 0u || output_shift > 31u) {
    cf_qadd_vendor(input1, input2, output, input1_multiplier,
                   input2_multiplier, output_shift, height, width, channels);
    CF_DORY_PROF_END();
    return;
  }
  cf_qadd_u8_job_t job = {
      .input1 = input1,
      .input2 = input2,
      .output = output,
      .input1_multiplier = input1_multiplier,
      .input2_multiplier = input2_multiplier,
      .output_shift = output_shift,
      .height = height,
      .width = width,
      .channels = channels,
      .cores = (uint8_t)pi_cl_team_nb_cores(),
  };
  cf_qadd_u8_packed4_team(&job);
  CF_DORY_PROF_END();
}
#endif
