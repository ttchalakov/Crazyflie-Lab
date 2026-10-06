#ifndef CF_GAP8_QPOOLING_H
#define CF_GAP8_QPOOLING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CF_QPOOLING_MAX_CORES 8u

typedef enum {
  CF_QPOOLING_MAX = 0,
  CF_QPOOLING_AVERAGE = 1,
  CF_QPOOLING_ADAPTIVE_AVERAGE = 2,
} cf_qpooling_kind_t;

/*
 * Batch-one compact HWC uint8 pooling contract. All buffers are borrowed.
 * Average pooling rounds down, matching the pinned PULP-NN uint8 kernel.
 * When requantize is set, average-family outputs instead apply
 * ((sum * output_multiplier) / divisor + output_add) >> output_shift and
 * saturate to uint8, matching the operation order in the pinned DORY kernel.
 * Adaptive average pooling uses floor/ceil bin boundaries equivalent to the
 * common framework definition and ignores kernel/stride/padding fields.
 *
 * Team functions are entered by every active core and never allocate, launch
 * DMA, fork a nested team, or manage cluster ownership.
 */
typedef struct {
  uint8_t *input;
  uint8_t *output;
  uint16_t input_width;
  uint16_t input_height;
  uint16_t channels;
  uint16_t output_width;
  uint16_t output_height;
  uint16_t kernel_width;
  uint16_t kernel_height;
  uint16_t stride_width;
  uint16_t stride_height;
  uint16_t padding_top;
  uint16_t padding_bottom;
  uint16_t padding_left;
  uint16_t padding_right;
  int32_t output_multiplier;
  int32_t output_add;
  uint8_t output_shift;
  uint8_t cores;
  uint8_t count_include_pad;
  uint8_t requantize;
  cf_qpooling_kind_t kind;
} cf_qpooling_job_t;

void cf_qpooling_reference(const cf_qpooling_job_t *job);

/* Generic output-pixel partitioning with scalar channel loops. */
void cf_qpooling_spatial_team(void *argument);
/* Four adjacent HWC channels are accumulated together; arbitrary tails fall
 * back to the generic team path without changing the public contract. */
void cf_qpooling_channel4_team(void *argument);
/* Global-average specialization: channels are independently assigned to the
 * team and each input element is read exactly once. */
void cf_qglobal_avgpool_channel_team(void *argument);

/* Pinned DORY/PULP-NN baseline adapter. It is valid only for MAX/AVERAGE,
 * eight cores, fixed windows, and count_include_pad average semantics. The
 * upstream max-pool routine uses its input tile as workspace. */
void cf_qpooling_pulp_nn_team(void *argument);

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_QPOOLING_H */
