#ifndef CF_GAP8_Q_LINEAR_H
#define CF_GAP8_Q_LINEAR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CF_QLINEAR_MAX_CORES 8u

/*
 * Batch-1 inference projection:
 *
 *   output[o] = bias[o] + sum_k input[k] *
 *                                  weights[o * weight_row_stride_bytes + k]
 *
 * Inputs are unsigned activations, weights are signed and row-major, biases and
 * outputs are int32. input_dim is the logical K used by the operation.
 * packed_input_dim is the zero-padded SIMD extent and weight_row_stride_bytes is
 * the physical distance between weight rows. The packed extent must be a
 * multiple of four and both padding regions must be zero. This lets arbitrary K
 * use the aligned packed kernel without unaligned row loads.
 *
 * The pointed-to buffers may live in L1 or L2, but they must remain valid until
 * the team function returns. Fast implementations require four-byte-aligned
 * input/weights. Dimensions must be nonzero, and cores must be one of 1/2/4/8
 * (the PULP-NN adapter specifically requires 8). The caller must choose
 * quantization bounds for which every bias-plus-dot result fits int32; signed
 * accumulator overflow is outside this contract.
 */
typedef struct {
  const uint8_t *input;
  const int8_t *weights;
  const int32_t *bias;
  int32_t *output;
  /* Logical K: padding is excluded from correctness and operation counts. */
  uint16_t input_dim;
  /* Physical zero-padded K consumed by SIMD. */
  uint16_t packed_input_dim;
  /* Physical bytes between row-major weight rows. */
  uint16_t weight_row_stride_bytes;
  uint16_t output_dim;
  uint8_t cores;
} cf_qlinear_u8s8_i32_job_t;

typedef enum {
  /* Matches pulp_nn_linear(flag_relu=0, flag_batch_norm=0). */
  CF_QLINEAR_U8_SHIFT_CLIP = 0,
  /* Matches pulp_nn_linear(flag_relu=1, flag_batch_norm=0). */
  CF_QLINEAR_U8_REQUANT_RELU = 1,
  /* Matches pulp_nn_linear(flag_relu=1, flag_batch_norm=1). */
  CF_QLINEAR_U8_BATCHNORM_RELU = 2,
} cf_qlinear_u8_epilogue_t;

/*
 * DORY-compatible fused batch-1 projection. The input/weight layout and core
 * contract are identical to cf_qlinear_u8s8_i32_job_t, but each accumulator is
 * requantized and written directly as uint8_t.
 *
 * SHIFT_CLIP computes clip_u8(accumulator >> output_shift).
 * REQUANT_RELU computes the pinned PULP-NN operation
 *   clip_u8((int16_t)((int16_t(output_multiplier) * accumulator)
 *                     >> output_shift)).
 * BATCHNORM_RELU computes
 *   clip_u8((batch_norm_multiplier[o] * accumulator
 *            + batch_norm_bias[o]) >> output_shift).
 *
 * output_shift must be in [0, 31]. The multiplier, accumulator, and
 * batch-normalization parameters must be chosen so the signed intermediate
 * arithmetic fits int32; REQUANT_RELU's shifted value must also fit int16.
 * Those are the numeric assumptions made by the pinned pulp_nn_linear ABI.
 * Batch-normalization arrays are required only for BATCHNORM_RELU. Bias remains
 * optional. Output must be four-byte aligned so a complete four-row tile can be
 * committed with one packed store. The PULP-NN benchmark adapter uses non-null
 * batch-normalization arrays for every mode to avoid null-pointer arithmetic in
 * the vendor source.
 */
typedef struct {
  const uint8_t *input;
  const int8_t *weights;
  const int32_t *bias;
  uint8_t *output;
  const int32_t *batch_norm_multiplier;
  const int32_t *batch_norm_bias;
  uint16_t input_dim;
  uint16_t packed_input_dim;
  uint16_t weight_row_stride_bytes;
  uint16_t output_dim;
  uint16_t output_multiplier;
  int8_t output_shift;
  cf_qlinear_u8_epilogue_t epilogue;
  uint8_t cores;
} cf_qlinear_u8s8_u8_job_t;

/* Caller-owned preparation helpers. Source and destination must not overlap.
 * They allocate no memory and return 0 on success, -1 for invalid dimensions,
 * strides, or pointers. A zero padded dimension means the logical K cannot be
 * represented by this uint16 ABI. */
uint16_t cf_qlinear_u8s8_i32_padded_input_dim(uint16_t input_dim);
int cf_qlinear_u8s8_i32_prepare_input(uint8_t *prepared,
                                     uint16_t prepared_dim,
                                     const uint8_t *source,
                                     uint16_t input_dim);
int cf_qlinear_u8s8_i32_prepare_weights(int8_t *prepared,
                                       uint16_t prepared_stride_bytes,
                                       const int8_t *source,
                                       uint16_t source_stride_bytes,
                                       uint16_t input_dim,
                                       uint16_t output_dim);

/* FC-side scalar oracle. This is correctness code, not a timed kernel. Invalid
 * null-buffer jobs are ignored. */
void cf_qlinear_u8s8_i32_reference(const cf_qlinear_u8s8_i32_job_t *job);

/* FC-side scalar oracle for the fused uint8 contract. */
void cf_qlinear_u8s8_u8_reference(const cf_qlinear_u8s8_u8_job_t *job);

/*
 * Team entry points. Every already-active cluster core must call the same
 * function with the same prevalidated job. They perform exactly one inference
 * and do not nest pi_cl_team_fork().
 *
 * The PULP-NN adapters wrap the exact pinned pulp_nn_linear_out_32 and
 * pulp_nn_linear implementations and consequently require cores == 8.
 */
void cf_qlinear_u8s8_i32_pulp_nn_team(void *argument);
void cf_qlinear_u8s8_u8_pulp_nn_team(void *argument);

/* DORY-compatible register-4 team entry points: runtime team widths 1/2/4/8,
 * each packed input load reused across as many as four output accumulators.
 * They do not add an internal barrier: DORY's generated layer wrapper already
 * barriers after its compute call, and any other team caller must do the
 * same. */
void cf_qlinear_u8s8_i32_register4_team(void *argument);
void cf_qlinear_u8s8_i32_register4_unroll2_team(void *argument);

/* DORY-compatible fused team entry. It performs one inference, does not fork,
 * allocate, DMA, or add a barrier; the generated DORY wrapper owns those. */
void cf_qlinear_u8s8_u8_register4_team(void *argument);

/* Exact DORY/PULP-NN symbols are emitted only for a replacement build. The
 * wrappers use register-4 for compact, four-byte-aligned tiles and retain the
 * pinned implementation as a private fallback for other generated tiles. */
#ifdef CF_QLINEAR_REPLACE_PULP_NN
void pulp_nn_linear(uint8_t *input, int8_t *bias, uint8_t *output,
                    int8_t *weights, int32_t *k, int32_t *lambda,
                    uint16_t output_multiplier, int8_t output_shift,
                    uint16_t input_dim, uint16_t output_dim, int flag_relu,
                    int flag_batch_norm);
void pulp_nn_linear_out_32(uint8_t *input, int8_t *bias, int32_t *output,
                           int8_t *weights, uint16_t input_dim,
                           uint16_t output_dim);
#endif

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_Q_LINEAR_H */
