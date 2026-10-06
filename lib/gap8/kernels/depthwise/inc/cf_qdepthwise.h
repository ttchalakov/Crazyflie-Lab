#ifndef CF_GAP8_QDEPTHWISE_H
#define CF_GAP8_QDEPTHWISE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CF_QDEPTHWISE_MAX_CORES 8u

typedef enum {
  CF_QDEPTHWISE_U8_SHIFT_CLIP = 0,
  CF_QDEPTHWISE_U8_REQUANT_RELU = 1,
  CF_QDEPTHWISE_U8_BATCHNORM_RELU = 2,
} cf_qdepthwise_u8_epilogue_t;

/*
 * Exact data contract used by DORY's pinned pulp_nn_depthwise_generic call.
 *
 * Input is CHW, output is HWC, and weights are compact [C][KH][KW]. The
 * channel multiplier is one, so input_channels must equal output_channels.
 * `scratch` and `weight_buffer` remain in the job because the stock ABI owns
 * both pointers. The direct 3x3 kernel does not dereference either one.
 *
 * Team functions run inside an already-active cluster team and never allocate,
 * fork, or issue DMA.
 */
typedef struct {
  const uint8_t *input;
  uint8_t *scratch;
  const int8_t *bias;
  uint8_t *output;
  const int8_t *weights;
  uint8_t *weight_buffer;
  const int32_t *batch_norm_multiplier;
  const int32_t *batch_norm_bias;
  uint16_t output_multiplier;
  uint16_t output_shift;
  uint16_t input_width;
  uint16_t input_height;
  uint16_t input_channels;
  uint16_t output_width;
  uint16_t output_height;
  uint16_t output_channels;
  uint16_t kernel_width;
  uint16_t kernel_height;
  uint16_t padding_top;
  uint16_t padding_bottom;
  uint16_t padding_left;
  uint16_t padding_right;
  uint16_t stride_width;
  uint16_t stride_height;
  cf_qdepthwise_u8_epilogue_t epilogue;
  uint8_t cores;
} cf_qdepthwise_u8s8_u8_job_t;

/* Scalar FC-side oracle. Invalid jobs are ignored. */
void cf_qdepthwise_u8s8_u8_reference(
    const cf_qdepthwise_u8s8_u8_job_t *job);

/* Stock DORY/PULP-NN generic kernel; its work split requires eight cores. */
void cf_qdepthwise_u8s8_u8_pulp_nn_team(void *argument);

/* Direct 3x3 micro-kernel, with paired horizontal outputs for stride one. */
void cf_qdepthwise_u8s8_u8_direct3x3_team(void *argument);

/*
 * Channel-blocked 3x3. Requires output_channels % 4 == 0: it stores four
 * channels of one output pixel as a single aligned word, which is the point.
 * See cf_qdepthwise_channel4.c for why the per-channel kernel is store-bound.
 */
void cf_qdepthwise_u8s8_u8_channel4_team(void *argument);

/* Stock scratch contract, useful to a caller retaining runtime dispatch. */
uint32_t cf_qdepthwise_pulp_nn_scratch_bytes(
    uint16_t input_height, uint16_t kernel_width, uint16_t padding_top,
    uint16_t padding_bottom, uint8_t cores);

/*
 * Exact pulp_nn_depthwise_generic ABI. For 3x3 it executes the direct kernel;
 * other kernel sizes fall back to a privately renamed copy of the pinned
 * implementation so the replacement remains safe for a whole DORY network.
 */
void cf_qdepthwise_generic_direct(
    const uint8_t *input, uint8_t *scratch, const int8_t *bias,
    uint8_t *output, const int8_t *weights, uint8_t *weight_buffer,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    uint16_t output_multiplier, uint16_t output_shift, uint16_t input_width,
    uint16_t input_height, uint16_t input_channels, uint16_t output_width,
    uint16_t output_height, uint16_t output_channels, uint16_t kernel_width,
    uint16_t kernel_height, uint16_t padding_top, uint16_t padding_bottom,
    uint16_t padding_left, uint16_t padding_right, uint16_t stride_width,
    uint16_t stride_height, int8_t flag_relu, int8_t flag_batch_norm);

#ifdef CF_QDEPTHWISE_REPLACE_PULP_NN
void pulp_nn_depthwise_generic(
    const uint8_t *input, uint8_t *scratch, const int8_t *bias,
    uint8_t *output, const int8_t *weights, uint8_t *weight_buffer,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    uint16_t output_multiplier, uint16_t output_shift, uint16_t input_width,
    uint16_t input_height, uint16_t input_channels, uint16_t output_width,
    uint16_t output_height, uint16_t output_channels, uint16_t kernel_width,
    uint16_t kernel_height, uint16_t padding_top, uint16_t padding_bottom,
    uint16_t padding_left, uint16_t padding_right, uint16_t stride_width,
    uint16_t stride_height, int8_t flag_relu, int8_t flag_batch_norm);
#endif

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_QDEPTHWISE_H */
