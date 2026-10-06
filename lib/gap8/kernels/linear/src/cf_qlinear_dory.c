#include <stdint.h>

#include "pmsis.h"

#include "cf_qlinear.h"
#include "cf_dory_profile.h"

extern void cf_qlinear_vendor(
    uint8_t *input, int8_t *bias, uint8_t *output, int8_t *weights,
    int32_t *k, int32_t *lambda, uint16_t output_multiplier,
    int8_t output_shift, uint16_t input_dim, uint16_t output_dim,
    int flag_relu, int flag_batch_norm);
extern void cf_qlinear_out32_vendor(
    uint8_t *input, int8_t *bias, int32_t *output, int8_t *weights,
    uint16_t input_dim, uint16_t output_dim);

static int cf_qlinear_dory_aligned(const void *input, const void *weights,
                                   const void *output, const void *bias,
                                   uint16_t input_dim) {
  return input != 0 && weights != 0 && output != 0 && input_dim != 0u &&
         (input_dim & 3u) == 0u && ((uintptr_t)input & 3u) == 0u &&
         ((uintptr_t)weights & 3u) == 0u && ((uintptr_t)output & 3u) == 0u &&
         (bias == 0 || ((uintptr_t)bias & 3u) == 0u);
}

#ifdef CF_QLINEAR_REPLACE_PULP_NN
void pulp_nn_linear(uint8_t *input, int8_t *bias, uint8_t *output,
                    int8_t *weights, int32_t *k, int32_t *lambda,
                    uint16_t output_multiplier, int8_t output_shift,
                    uint16_t input_dim, uint16_t output_dim, int flag_relu,
                    int flag_batch_norm) {
  CF_DORY_PROF_BEGIN();
  if (!cf_qlinear_dory_aligned(input, weights, output, bias, input_dim) ||
      output_dim == 0u || output_shift < 0 || output_shift > 31 ||
      (flag_batch_norm && (!flag_relu || k == 0 || lambda == 0))) {
    cf_qlinear_vendor(input, bias, output, weights, k, lambda,
                      output_multiplier, output_shift, input_dim, output_dim,
                      flag_relu, flag_batch_norm);
    CF_DORY_PROF_END();
    return;
  }

  cf_qlinear_u8s8_u8_job_t job = {
      .input = input,
      .weights = weights,
      .bias = (const int32_t *)bias,
      .output = output,
      .batch_norm_multiplier = k,
      .batch_norm_bias = lambda,
      .input_dim = input_dim,
      .packed_input_dim = input_dim,
      .weight_row_stride_bytes = input_dim,
      .output_dim = output_dim,
      .output_multiplier = output_multiplier,
      .output_shift = output_shift,
      .epilogue = flag_batch_norm ? CF_QLINEAR_U8_BATCHNORM_RELU
                                  : flag_relu ? CF_QLINEAR_U8_REQUANT_RELU
                                              : CF_QLINEAR_U8_SHIFT_CLIP,
      .cores = (uint8_t)pi_cl_team_nb_cores(),
  };
  cf_qlinear_u8s8_u8_register4_team(&job);
  /* The optimized path fell off the end without closing the bracket, so
   * CF_DORY_PROF_BEGIN() ran on every call and CF_DORY_PROF_END() only on the
   * vendor fallback. The accumulator increments on END, which is why every
   * CFPROF line reported kernel_cycles=0 and kernel_calls=0 -- the profiler
   * was measuring the one path it was not supposed to be measuring. */
  CF_DORY_PROF_END();
}

/* Instrumented like pulp_nn_linear above. It was not, and it is the layer a
 * NEMO MLP ends in -- so a CFPROF split over an MLP graph was reporting the
 * hidden layers' compute against the whole network's total, understating the
 * compute fraction by exactly the classifier. */
void pulp_nn_linear_out_32(uint8_t *input, int8_t *bias, int32_t *output,
                           int8_t *weights, uint16_t input_dim,
                           uint16_t output_dim) {
  CF_DORY_PROF_BEGIN();
  if (!cf_qlinear_dory_aligned(input, weights, output, bias, input_dim) ||
      output_dim == 0u) {
    cf_qlinear_out32_vendor(input, bias, output, weights, input_dim,
                            output_dim);
    CF_DORY_PROF_END();
    return;
  }

  cf_qlinear_u8s8_i32_job_t job = {
      .input = input,
      .weights = weights,
      .bias = (const int32_t *)bias,
      .output = output,
      .input_dim = input_dim,
      .packed_input_dim = input_dim,
      .weight_row_stride_bytes = input_dim,
      .output_dim = output_dim,
      .cores = (uint8_t)pi_cl_team_nb_cores(),
  };
  cf_qlinear_u8s8_i32_register4_team(&job);
  CF_DORY_PROF_END();
}
#endif
