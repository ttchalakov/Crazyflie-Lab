/* Private pinned fallbacks for DORY tiles that cannot use the packed layout. */
#define pulp_nn_linear cf_qlinear_vendor
#include "pulp_nn_linear.c"
#undef pulp_nn_linear

#undef log2
#undef min
#undef SumDotp
#undef clip8

#define pulp_nn_linear_out_32 cf_qlinear_out32_vendor
#include "pulp_nn_linear_out_32.c"
