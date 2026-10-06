#ifndef CF_GAP8_QADD_H
#define CF_GAP8_QADD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CF_QADD_MAX_CORES 8u

/* Exact batch-1 DORY/PULP-NN uint8 residual-add contract. */
typedef struct {
  const uint8_t *input1;
  const uint8_t *input2;
  uint8_t *output;
  uint16_t input1_multiplier;
  uint16_t input2_multiplier;
  uint16_t output_shift;
  uint16_t height;
  uint16_t width;
  uint16_t channels;
  uint8_t cores;
} cf_qadd_u8_job_t;

void cf_qadd_u8_reference(const cf_qadd_u8_job_t *job);

/* Called by every core of an already-active cluster team. */
void cf_qadd_u8_pulp_nn_team(void *argument);
void cf_qadd_u8_packed4_team(void *argument);

#ifdef CF_QADD_REPLACE_PULP_NN
void pulp_nn_add(uint8_t *input1, uint8_t *input2, uint8_t *output,
                 uint16_t input1_multiplier, uint16_t input2_multiplier,
                 uint16_t output_shift, uint16_t height, uint16_t width,
                 uint16_t channels);
#endif

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_QADD_H */
