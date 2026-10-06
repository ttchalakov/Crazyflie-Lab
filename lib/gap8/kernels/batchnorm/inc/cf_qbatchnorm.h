#ifndef CF_GAP8_QBATCHNORM_H
#define CF_GAP8_QBATCHNORM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CF_QBATCHNORM_MAX_CORES 8u
#define CF_QBATCHNORM_Q16_ONE 65536u

/*
 * Prepared, fixed-statistics inference BatchNorm.
 *
 * Parameters are prepared per channel and use the same affine shape as the
 * DORY/PULP-NN kappa/lambda epilogue:
 *
 *   output = clip_u8((multiplier[channel] * input + bias[channel]) >> shift)
 *
 * Folding this expression into the producer remains faster. This standalone
 * form is for graph boundaries and tensors whose producer cannot absorb it.
 */
typedef struct {
  const uint8_t *input;
  uint8_t *output;
  const int32_t *multiplier;
  const int32_t *bias;
  uint32_t elements;
  uint16_t channels;
  uint8_t shift;
  uint8_t cores;
} cf_qbatchnorm_fixed_job_t;

typedef enum {
  /* Normalize with current activation statistics; do not change state. */
  CF_QBATCHNORM_BATCH_STATS = 0,
  /* Normalize with current statistics, then update running state. */
  CF_QBATCHNORM_UPDATE_RUNNING = 1,
  /* Update running state without producing an output tensor. */
  CF_QBATCHNORM_CALIBRATE_ONLY = 2,
} cf_qbatchnorm_update_mode_t;

/*
 * Forward-only, batch-one HWC BatchNorm with mutable running statistics.
 * There are no gradients, backward pass, or optimizer-owned parameters.
 *
 * Numeric state:
 *   running_mean_q16       Q16.16 in raw input-code units
 *   running_variance_q16   unsigned Q16.16 in raw input-code units squared
 *   gamma_q16              signed Q16.16 output-code units per normalized unit
 *   beta_q16               signed Q16.16 output-code units
 *   epsilon_q16            unsigned Q16.16 input-code units squared
 *   momentum_q16           [0,65535], weight assigned to new statistics
 *
 * The forward pass uses the biased population variance. Running variance is
 * updated with the unbiased estimate, matching conventional BatchNorm
 * training behavior when spatial_samples > 1. Gamma and beta are immutable.
 *
 * All storage is borrowed. Team functions are called by every core in an
 * already-active team and do not allocate, launch DMA, fork, or own a cluster.
 * UPDATE_RUNNING is one run-to-completion state transaction; callers must not
 * retry a partially executed call as though it had never committed.
 */
typedef struct {
  const uint8_t *input;
  uint8_t *output;
  const int32_t *gamma_q16;
  const int32_t *beta_q16;
  int32_t *running_mean_q16;
  uint32_t *running_variance_q16;
  uint32_t spatial_samples;
  uint16_t channels;
  uint32_t epsilon_q16;
  uint16_t momentum_q16;
  uint8_t output_zero_point;
  uint8_t cores;
  cf_qbatchnorm_update_mode_t mode;
} cf_qbatchnorm_update_job_t;

void cf_qbatchnorm_fixed_reference(const cf_qbatchnorm_fixed_job_t *job);
void cf_qbatchnorm_fixed_scalar_team(void *argument);
void cf_qbatchnorm_fixed_channel4_team(void *argument);

void cf_qbatchnorm_update_reference(cf_qbatchnorm_update_job_t *job);
void cf_qbatchnorm_update_channel_team(void *argument);

/* Prepare the fixed affine contract from Q16.16 running state and immutable
 * gamma/beta. Returns zero on success and -1 if an argument or range is bad. */
int cf_qbatchnorm_prepare_fixed(const cf_qbatchnorm_update_job_t *job,
                                int32_t *multiplier, int32_t *bias,
                                uint8_t shift);

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_QBATCHNORM_H */
