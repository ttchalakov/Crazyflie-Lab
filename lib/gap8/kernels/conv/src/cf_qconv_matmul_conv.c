/*
 * Compile the exact pinned spatial traversal under a private name while routing
 * only its shared matmul call to our optimized microkernel. This keeps the
 * matmul ablation honest without copying or modifying the vendor source.
 */
#define pulp_nn_conv_Ho_parallel cf_qconv_conv_Ho_parallel_matmul4x2
#define pulp_nn_matmul cf_qconv_matmul4x2
#include "pulp_nn_conv_Ho_parallel.c"
