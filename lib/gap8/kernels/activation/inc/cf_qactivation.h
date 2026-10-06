#ifndef CF_GAP8_QACTIVATION_H
#define CF_GAP8_QACTIVATION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CF_QACTIVATION_LUT_ENTRIES 256u
#define CF_QACTIVATION_MAX_CORES 8u
#define CF_QSOFTMAX_MAX_COLUMNS 1024u

typedef enum {
  CF_QACTIVATION_SIGMOID = 0,
  CF_QACTIVATION_TANH = 1,
  CF_QACTIVATION_LEAKY_RELU = 2,
  CF_QACTIVATION_HARD_SIGMOID = 3,
  CF_QACTIVATION_HARD_SWISH = 4,
  CF_QACTIVATION_SILU = 5,
  CF_QACTIVATION_GELU = 6,
} cf_qactivation_kind_t;

/*
 * Generic quantized elementwise contract.
 *
 * The LUT is indexed by the input byte's raw bit pattern and returns the output
 * byte's raw bit pattern. Signedness, zero points, and scales are therefore
 * encoded when the 256-byte table is prepared instead of branching in the
 * kernel. This supports uint8 DORY activations and signed recurrent state with
 * the same arithmetic entry point.
 *
 * All pointers are borrowed. Team functions are called by every core of an
 * already-active team and never allocate, launch DMA, fork, or own the cluster.
 */
typedef struct {
  const uint8_t *input;
  uint8_t *output;
  const uint8_t *lut;
  uint32_t elements;
  uint8_t cores;
} cf_qactivation_lut_job_t;

/* Scalar oracle. */
void cf_qactivation_lut_reference(const cf_qactivation_lut_job_t *job);

/* One scalar LUT lookup per element, partitioned over the active team. */
void cf_qactivation_lut_scalar_team(void *argument);

/* Four-byte aligned loads/stores with four scalar LUT gathers per iteration. */
void cf_qactivation_lut_packed4_team(void *argument);

/*
 * Quantized softmax contract. Input elements are integer logits separated by
 * input quantum epsilon. The exp LUT contains round(exp(-d*epsilon) * 32768)
 * for d in [0,255]. Output is uint8 probability mass whose row sums to 255.
 */
typedef struct {
  const uint8_t *input;
  uint8_t *output;
  const uint16_t *exp_lut;
  uint16_t rows;
  uint16_t columns;
  uint8_t cores;
  uint8_t input_signed;
} cf_qsoftmax_job_t;

/* Scalar oracle; requires no scratch. */
void cf_qsoftmax_reference(const cf_qsoftmax_job_t *job);

/* DORY-shaped worker called by every core of an already-active team. Core 0
 * normalizes the rows and all participating cores complete at its barrier. */
void cf_qsoftmax_scalar_team(void *argument);

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_QACTIVATION_H */
