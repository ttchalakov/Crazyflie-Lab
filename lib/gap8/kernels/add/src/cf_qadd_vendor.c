/* Private pinned fallback for tiles the packed traversal cannot claim.
 *
 * Mirrors cf_qlinear_vendor.c and cf_qdepthwise_vendor.c: a renamed copy of
 * the pinned kernel so that enabling CF_QADD_REPLACE_PULP_NN still leaves a
 * byte-exact upstream implementation to fall back to. Without it the overlay
 * had no retreat path at all. */
#define pulp_nn_add cf_qadd_vendor
#include "pulp_nn_add.c"
