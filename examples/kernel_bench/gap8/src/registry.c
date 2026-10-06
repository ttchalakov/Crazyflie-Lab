#include "kernel_bench/registry.h"

extern const kbench_kernel_t kbench_linear_u8s8_i32_pulp_nn;
extern const kbench_kernel_t kbench_linear_u8s8_i32_register4;
extern const kbench_kernel_t kbench_linear_u8s8_i32_register4_unroll2;
extern const kbench_kernel_t kbench_linear_u8s8_u8_pulp_nn;
extern const kbench_kernel_t kbench_linear_u8s8_u8_register4;
extern const kbench_kernel_t kbench_conv2d_u8s8_u8_pulp_nn;
extern const kbench_kernel_t kbench_conv2d_u8s8_u8_matmul4x2;
extern const kbench_kernel_t kbench_conv2d_u8s8_u8_flattened3;
extern const kbench_kernel_t kbench_conv2d_u8s8_u8_flattened;
extern const kbench_kernel_t kbench_conv2d_u8s8_u8_stem_repack;
extern const kbench_kernel_t kbench_conv2d_u8s8_u8_stem_direct;
extern const kbench_kernel_t kbench_pointwise_u8s8_u8_pulp_nn;
extern const kbench_kernel_t kbench_pointwise_u8s8_u8_flattened;
extern const kbench_kernel_t kbench_depthwise_u8s8_u8_pulp_nn;
extern const kbench_kernel_t kbench_depthwise_u8s8_u8_direct3x3;
extern const kbench_kernel_t kbench_depthwise_u8s8_u8_channel4;
extern const kbench_kernel_t kbench_matmul4x2_u8s8_u8_pulp_nn;
extern const kbench_kernel_t kbench_matmul4x2_u8s8_u8_sequential;
extern const kbench_kernel_t kbench_matmul4x2_u8s8_u8_krotate;
extern const kbench_kernel_t kbench_matmul4x4_u8s8_u8_sequential;
extern const kbench_kernel_t kbench_matmul4x3_u8s8_u8_sequential;
extern const kbench_kernel_t kbench_activation_lut_scalar;
extern const kbench_kernel_t kbench_activation_lut_packed4;
extern const kbench_kernel_t kbench_softmax_s8_u8_scalar;
extern const kbench_kernel_t kbench_batchnorm_u8_fixed_scalar;
extern const kbench_kernel_t kbench_batchnorm_u8_fixed_channel4;
extern const kbench_kernel_t kbench_batchnorm_u8_update_channel;
extern const kbench_kernel_t kbench_maxpool2d_u8_pulp_nn;
extern const kbench_kernel_t kbench_maxpool2d_u8_channel4;
extern const kbench_kernel_t kbench_avgpool2d_u8_pulp_nn;
extern const kbench_kernel_t kbench_avgpool2d_u8_channel4;
extern const kbench_kernel_t kbench_global_avgpool_u8_pulp_nn;
extern const kbench_kernel_t kbench_global_avgpool_u8_channel;
extern const kbench_kernel_t kbench_adaptive_avgpool_u8_scalar;
extern const kbench_kernel_t kbench_adaptive_avgpool_u8_channel4;
extern const kbench_kernel_t kbench_add_u8_pulp_nn;
extern const kbench_kernel_t kbench_add_u8_packed4;

/* Explicit registration keeps every benchmark binary auditable. */
static const kbench_kernel_t *const kernels[] = {
    &kbench_linear_u8s8_i32_pulp_nn,
    &kbench_linear_u8s8_i32_register4,
    &kbench_linear_u8s8_i32_register4_unroll2,
    &kbench_linear_u8s8_u8_pulp_nn,
    &kbench_linear_u8s8_u8_register4,
    &kbench_conv2d_u8s8_u8_pulp_nn,
    &kbench_conv2d_u8s8_u8_matmul4x2,
    &kbench_conv2d_u8s8_u8_flattened3,
    &kbench_conv2d_u8s8_u8_flattened,
    &kbench_conv2d_u8s8_u8_stem_repack,
    &kbench_conv2d_u8s8_u8_stem_direct,
    &kbench_pointwise_u8s8_u8_pulp_nn,
    &kbench_pointwise_u8s8_u8_flattened,
    &kbench_depthwise_u8s8_u8_pulp_nn,
    &kbench_depthwise_u8s8_u8_direct3x3,
    &kbench_depthwise_u8s8_u8_channel4,
    &kbench_matmul4x2_u8s8_u8_pulp_nn,
    &kbench_matmul4x2_u8s8_u8_sequential,
    &kbench_matmul4x2_u8s8_u8_krotate,
    &kbench_matmul4x4_u8s8_u8_sequential,
    &kbench_matmul4x3_u8s8_u8_sequential,
    &kbench_activation_lut_scalar,
    &kbench_activation_lut_packed4,
    &kbench_batchnorm_u8_fixed_scalar,
    &kbench_batchnorm_u8_fixed_channel4,
    &kbench_batchnorm_u8_update_channel,
    &kbench_maxpool2d_u8_pulp_nn,
    &kbench_maxpool2d_u8_channel4,
    &kbench_avgpool2d_u8_pulp_nn,
    &kbench_avgpool2d_u8_channel4,
    &kbench_global_avgpool_u8_pulp_nn,
    &kbench_global_avgpool_u8_channel,
    &kbench_adaptive_avgpool_u8_scalar,
    &kbench_adaptive_avgpool_u8_channel4,
    &kbench_add_u8_pulp_nn,
    &kbench_add_u8_packed4,
    &kbench_softmax_s8_u8_scalar,
};

const kbench_kernel_t *const *kbench_registry(size_t *count) {
  if (count != NULL) {
    *count = sizeof(kernels) / sizeof(kernels[0]);
  }
  return kernels;
}
