#ifndef CF_GAP8_QCONV_H
#define CF_GAP8_QCONV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CF_QCONV_MAX_CORES 8u

typedef enum {
  /* Matches flag_relu=0, flag_batch_norm=0. */
  CF_QCONV_U8_SHIFT_CLIP = 0,
  /* Matches flag_relu=1, flag_batch_norm=0. */
  CF_QCONV_U8_REQUANT_RELU = 1,
  /* Matches flag_relu=1, flag_batch_norm=1. */
  CF_QCONV_U8_BATCHNORM_RELU = 2,
} cf_qconv_u8_epilogue_t;

/*
 * Batch-1 NHWC uint8 x OHWI int8 convolution with a fused uint8 epilogue.
 *
 * This deliberately mirrors the buffers and dimensions supplied to the pinned
 * GAP8 DORY pulp_nn_conv_Ho_parallel call. Input and output are NHWC. Each
 * output-channel weight row is compact and contains
 * kernel_height * kernel_width * input_channels int8 values. im2col must point
 * to caller-owned storage of at least
 *   2 * cores * kernel_height * kernel_width * input_channels bytes.
 *
 * The kernel never allocates, forks, or performs DMA. Every function below is
 * called by each core of an already-active team; a DORY layer wrapper owns the
 * team, L1 layout, DMA, and the barrier after compute.
 */
typedef struct {
  const uint8_t *input;
  uint8_t *im2col;
  const int8_t *bias;
  uint8_t *output;
  const int8_t *weights;
  /*
   * Scratch for the stem variant only; every other function ignores it and
   * accepts NULL. cf_qconv_u8s8_u8_stem_repack_* rewrites the weight rows into
   * it at a word-aligned stride, so it must hold
   *   output_channels * align_up(kernel_height * kernel_width * input_channels, 4)
   * bytes and, like every other operand, start on a 4-byte boundary. The same
   * padding applies to im2col, which must be sized at the padded stride too.
   */
  int8_t *weight_repack;
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
  cf_qconv_u8_epilogue_t epilogue;
  uint8_t cores;
} cf_qconv_u8s8_u8_job_t;

/* Scalar FC-side oracle. Invalid jobs are ignored. */
void cf_qconv_u8s8_u8_reference(const cf_qconv_u8s8_u8_job_t *job);

/* Exact pinned PULP-NN baseline; requires cores == 8. */
void cf_qconv_u8s8_u8_pulp_nn_team(void *argument);

/* Stock spatial traversal and im2col with the optimized 4x2 matmul. */
void cf_qconv_u8s8_u8_matmul4x2_team(void *argument);

/* Flattened spatial traversal plus optimized im2col and 4x2 matmul. */
void cf_qconv_u8s8_u8_flattened_team(void *argument);

/*
 * Three output pixels per matmul block instead of two, using
 * cf_qconv_matmul4x3. Requires an im2col buffer of `3 * cores * columns`
 * bytes -- half again what PULP-NN's contract reserves -- so the caller must
 * own the allocation. Not reachable from the DORY overlay; see the note in
 * cf_qconv_flattened.c.
 */
void cf_qconv_u8s8_u8_flattened3_team(void *argument);

/*
 * Stem variant: pads the K axis to a word boundary so the weight rows and the
 * second im2col column are aligned and the scalar tail disappears. Requires
 * job->weight_repack and a padded-stride im2col; see the struct comment. Exact
 * by construction (the pad lanes are zero) and forwards to the flattened path
 * when K is already a multiple of four.
 */
void cf_qconv_u8s8_u8_stem_repack_team(void *argument);

/*
 * Direct single-channel 3x3 stem: no im2col, weights pre-arranged as
 * Cout x 3 x (3 taps + zero lane) in job->weight_repack (Cout * 12 bytes).
 * Reads the input in place, so the three word loads per output pixel amortise
 * across every output channel. Falls back to the flattened path for anything
 * that is not Cin=1 3x3. Exact.
 */
void cf_qconv_u8s8_u8_stem_direct_team(void *argument);

/* Exact DORY 1x1 pointwise baseline and flattened direct-input candidate. */
void cf_qconv_u8s8_u8_pointwise_pulp_nn_team(void *argument);
void cf_qconv_u8s8_u8_pointwise_flattened_team(void *argument);

/*
 * Shared two-column microkernel. Its pointer ABI matches pulp_nn_matmul exactly.
 * Direct users get the documented int8 bias layout; the DORY overlay defines
 * CF_QCONV_DORY_INT32_BIAS because generated NEMO blobs contain int32 bias even
 * though the historical function parameter remained int8_t *. It consumes two
 * compact K columns and writes four-channel register tiles to output/output2.
 */
/*
 * Accumulator-only 4x2 matmul, and the reduce that pairs with it.
 *
 * Requantisation with a non-negative batch-norm multiplier is monotonic, so a
 * fused 2x2 max pool can keep four int32 accumulator vectors, reduce them, and
 * requantise **once** per pooled channel instead of four times -- bit-identical
 * output, one epilogue and one store instead of four.
 *
 * cf_qconv_pool_requant4() returns 0 without writing if any multiplier is
 * negative, because the identity does not hold there.
 */
void cf_qconv_matmul4x2_acc(const int8_t *weights, const uint8_t *input,
                            uint16_t output_channels, uint16_t columns,
                            const int8_t *bias, int32_t *acc0, int32_t *acc1);

int cf_qconv_pool_requant4(const int32_t *acc0, const int32_t *acc1,
                           const int32_t *acc2, const int32_t *acc3,
                           uint16_t output_channels, uint16_t output_shift,
                           const int32_t *batch_norm_multiplier,
                           const int32_t *batch_norm_bias, uint8_t *output);

uint8_t *cf_qconv_matmul4x2(
    const int8_t *weights, uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, uint8_t *output2, int flag_relu,
    int flag_batch_norm);

/*
 * Exact-ABI TCDM experiment. Each core rotates its aligned K-word traversal,
 * then wraps to word zero. Layout and arithmetic are unchanged; only the order
 * of the int32 additions and L1 bank phase differ.
 */
uint8_t *cf_qconv_matmul4x2_krotate(
    const int8_t *weights, uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, uint8_t *output2, int flag_relu,
    int flag_batch_norm);

/*
 * Four output pixels by four output channels. Raises the inner loop's
 * load-to-dot-product ratio from 8/6 to 16/8 -- about 17% more arithmetic per
 * issue slot -- at the cost of sixteen live accumulators. `output_stride` is
 * the distance between consecutive output pixels, normally output_channels.
 *
 * Not ABI-compatible with pulp_nn_matmul, which is fixed at two pixels, so it
 * is reachable only through a caller that builds four im2col columns.
 */
/* Same idea one pixel lower, where the accumulators still fit the register
 * file. See the header comment in cf_qconv_matmul4x3.c. */
uint8_t *cf_qconv_matmul4x3(
    const int8_t *weights, const uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, uint16_t output_stride,
    int flag_relu, int flag_batch_norm);

uint8_t *cf_qconv_matmul4x4(
    const int8_t *weights, const uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, uint16_t output_stride,
    int flag_relu, int flag_batch_norm);

#ifdef CF_QCONV_REPLACE_PULP_NN
/*
 * Defining CF_QCONV_REPLACE_PULP_NN turns this library into an exact-symbol
 * source replacement for a DORY-generated pulp_nn_matmul.c and
 * pulp_nn_conv_Ho_parallel.c. Do not compile the vendor definitions alongside
 * these aliases. Use CF_QCONV_DORY_INT32_BIAS only when the caller supplies
 * DORY's four-byte-per-output-channel bias blob.
 */
uint8_t *pulp_nn_matmul(
    const int8_t *weights, uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, uint8_t *output2, int flag_relu,
    int flag_batch_norm);

void pulp_nn_conv_Ho_parallel(
    const uint8_t *input, uint8_t *im2col, const int8_t *bias,
    uint8_t *output, const int8_t *weights, int32_t *batch_norm_multiplier,
    int32_t *batch_norm_bias, uint16_t output_multiplier,
    uint16_t output_shift, uint16_t input_width, uint16_t input_height,
    uint16_t input_channels, uint16_t output_width, uint16_t output_height,
    uint16_t output_channels, uint16_t kernel_width, uint16_t kernel_height,
    uint16_t padding_top, uint16_t padding_bottom, uint16_t padding_left,
    uint16_t padding_right, uint16_t stride_width, uint16_t stride_height,
    int flag_relu, int flag_batch_norm);

void pulp_nn_pointwise_HoWo_parallel(
    const uint8_t *input, uint8_t *im2col, const int8_t *bias,
    uint8_t *output, const int8_t *weights, int32_t *batch_norm_multiplier,
    int32_t *batch_norm_bias, uint16_t output_multiplier,
    uint16_t output_shift, uint16_t input_width, uint16_t input_height,
    uint16_t input_channels, uint16_t output_width, uint16_t output_height,
    uint16_t output_channels, uint16_t kernel_width, uint16_t kernel_height,
    uint16_t padding_top, uint16_t padding_bottom, uint16_t padding_left,
    uint16_t padding_right, uint16_t stride_width, uint16_t stride_height,
    int flag_relu, int flag_batch_norm);
#endif

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_QCONV_H */
