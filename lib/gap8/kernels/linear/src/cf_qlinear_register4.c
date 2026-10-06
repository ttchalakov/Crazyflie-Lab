#include <stdint.h>

#include "pmsis.h"

#include "cf_qlinear.h"

#define CF_QLINEAR_DOTP4(a, b, c) __builtin_pulp_sdotusp4((a), (b), (c))
#define CF_QLINEAR_ALWAYS_INLINE static inline __attribute__((always_inline))

typedef void (*cf_qlinear_worker_t)(void *argument);

CF_QLINEAR_ALWAYS_INLINE uint32_t cf_qlinear_min_u32(uint32_t a,
                                                     uint32_t b) {
  return a < b ? a : b;
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_output_range(
    uint32_t output_dim, uint32_t core, uint32_t cores, uint32_t *start,
    uint32_t *stop) {
  const uint32_t chunk = (output_dim + cores - 1u) / cores;

  *start = cf_qlinear_min_u32(core * chunk, output_dim);
  *stop = cf_qlinear_min_u32(*start + chunk, output_dim);
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_block4_range(
    uint32_t output_dim, uint32_t core, uint32_t cores, uint32_t *first,
    uint32_t *last) {
  const uint32_t blocks = output_dim >> 2;
  const uint32_t chunk = (blocks + cores - 1u) / cores;

  *first = cf_qlinear_min_u32(core * chunk, blocks);
  *last = cf_qlinear_min_u32(*first + chunk, blocks);
}

CF_QLINEAR_ALWAYS_INLINE int32_t cf_qlinear_bias(
    const cf_qlinear_u8s8_i32_job_t *job, uint32_t output_index) {
  return job->bias != 0 ? job->bias[output_index] : 0;
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_row_rolled(
    const cf_qlinear_u8s8_i32_job_t *job, uint32_t output_index) {
  const v4u *input = (const v4u *)job->input;
  const v4s *weight =
      (const v4s *)(job->weights +
                    output_index * job->weight_row_stride_bytes);
  const uint32_t words = (uint32_t)job->packed_input_dim >> 2;
  int32_t sum = cf_qlinear_bias(job, output_index);
  uint32_t packed;

  for (packed = 0u; packed < words; ++packed) {
    sum = CF_QLINEAR_DOTP4(input[packed], weight[packed], sum);
  }
  job->output[output_index] = sum;
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_row_unroll2(
    const cf_qlinear_u8s8_i32_job_t *job, uint32_t output_index) {
  const v4u *input = (const v4u *)job->input;
  const v4s *weight =
      (const v4s *)(job->weights +
                    output_index * job->weight_row_stride_bytes);
  const uint32_t words = (uint32_t)job->packed_input_dim >> 2;
  const uint32_t pairs = words >> 1;
  int32_t sum = cf_qlinear_bias(job, output_index);
  uint32_t pair;

  for (pair = 0u; pair < pairs; ++pair) {
    const uint32_t packed = pair << 1;
    sum = CF_QLINEAR_DOTP4(input[packed], weight[packed], sum);
    sum = CF_QLINEAR_DOTP4(input[packed + 1u], weight[packed + 1u], sum);
  }
  if ((words & 1u) != 0u) {
    sum = CF_QLINEAR_DOTP4(input[words - 1u], weight[words - 1u], sum);
  }
  job->output[output_index] = sum;
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_tile4_rolled(
    const cf_qlinear_u8s8_i32_job_t *job, uint32_t output_index) {
  const v4u *input = (const v4u *)job->input;
  const v4s *weight0 =
      (const v4s *)(job->weights +
                    output_index * job->weight_row_stride_bytes);
  const v4s *weight1 =
      (const v4s *)(job->weights +
                    (output_index + 1u) * job->weight_row_stride_bytes);
  const v4s *weight2 =
      (const v4s *)(job->weights +
                    (output_index + 2u) * job->weight_row_stride_bytes);
  const v4s *weight3 =
      (const v4s *)(job->weights +
                    (output_index + 3u) * job->weight_row_stride_bytes);
  const uint32_t words = (uint32_t)job->packed_input_dim >> 2;
  int32_t sum0 = cf_qlinear_bias(job, output_index);
  int32_t sum1 = cf_qlinear_bias(job, output_index + 1u);
  int32_t sum2 = cf_qlinear_bias(job, output_index + 2u);
  int32_t sum3 = cf_qlinear_bias(job, output_index + 3u);
  uint32_t packed;

  for (packed = 0u; packed < words; ++packed) {
    const v4u activation = input[packed];
    sum0 = CF_QLINEAR_DOTP4(activation, weight0[packed], sum0);
    sum1 = CF_QLINEAR_DOTP4(activation, weight1[packed], sum1);
    sum2 = CF_QLINEAR_DOTP4(activation, weight2[packed], sum2);
    sum3 = CF_QLINEAR_DOTP4(activation, weight3[packed], sum3);
  }

  job->output[output_index] = sum0;
  job->output[output_index + 1u] = sum1;
  job->output[output_index + 2u] = sum2;
  job->output[output_index + 3u] = sum3;
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_tile4_unroll2(
    const cf_qlinear_u8s8_i32_job_t *job, uint32_t output_index) {
  const v4u *input = (const v4u *)job->input;
  const v4s *weight0 =
      (const v4s *)(job->weights +
                    output_index * job->weight_row_stride_bytes);
  const v4s *weight1 =
      (const v4s *)(job->weights +
                    (output_index + 1u) * job->weight_row_stride_bytes);
  const v4s *weight2 =
      (const v4s *)(job->weights +
                    (output_index + 2u) * job->weight_row_stride_bytes);
  const v4s *weight3 =
      (const v4s *)(job->weights +
                    (output_index + 3u) * job->weight_row_stride_bytes);
  const uint32_t words = (uint32_t)job->packed_input_dim >> 2;
  const uint32_t pairs = words >> 1;
  int32_t sum0 = cf_qlinear_bias(job, output_index);
  int32_t sum1 = cf_qlinear_bias(job, output_index + 1u);
  int32_t sum2 = cf_qlinear_bias(job, output_index + 2u);
  int32_t sum3 = cf_qlinear_bias(job, output_index + 3u);
  uint32_t pair;

  for (pair = 0u; pair < pairs; ++pair) {
    const uint32_t packed = pair << 1;
    const v4u activation0 = input[packed];
    const v4u activation1 = input[packed + 1u];

    sum0 = CF_QLINEAR_DOTP4(activation0, weight0[packed], sum0);
    sum1 = CF_QLINEAR_DOTP4(activation0, weight1[packed], sum1);
    sum2 = CF_QLINEAR_DOTP4(activation0, weight2[packed], sum2);
    sum3 = CF_QLINEAR_DOTP4(activation0, weight3[packed], sum3);
    sum0 = CF_QLINEAR_DOTP4(activation1, weight0[packed + 1u], sum0);
    sum1 = CF_QLINEAR_DOTP4(activation1, weight1[packed + 1u], sum1);
    sum2 = CF_QLINEAR_DOTP4(activation1, weight2[packed + 1u], sum2);
    sum3 = CF_QLINEAR_DOTP4(activation1, weight3[packed + 1u], sum3);
  }
  if ((words & 1u) != 0u) {
    const v4u activation = input[words - 1u];
    sum0 = CF_QLINEAR_DOTP4(activation, weight0[words - 1u], sum0);
    sum1 = CF_QLINEAR_DOTP4(activation, weight1[words - 1u], sum1);
    sum2 = CF_QLINEAR_DOTP4(activation, weight2[words - 1u], sum2);
    sum3 = CF_QLINEAR_DOTP4(activation, weight3[words - 1u], sum3);
  }

  job->output[output_index] = sum0;
  job->output[output_index + 1u] = sum1;
  job->output[output_index + 2u] = sum2;
  job->output[output_index + 3u] = sum3;
}

static void cf_qlinear_register1_rolled_worker(void *argument) {
  cf_qlinear_u8s8_i32_job_t *job = (cf_qlinear_u8s8_i32_job_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t cores = (uint32_t)pi_cl_team_nb_cores();
  uint32_t start;
  uint32_t stop;
  uint32_t output_index;

  cf_qlinear_output_range(job->output_dim, core, cores, &start, &stop);
  for (output_index = start; output_index < stop; ++output_index) {
    cf_qlinear_row_rolled(job, output_index);
  }
}

static void cf_qlinear_register1_unroll2_worker(void *argument) {
  cf_qlinear_u8s8_i32_job_t *job = (cf_qlinear_u8s8_i32_job_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t cores = (uint32_t)pi_cl_team_nb_cores();
  uint32_t start;
  uint32_t stop;
  uint32_t output_index;

  cf_qlinear_output_range(job->output_dim, core, cores, &start, &stop);
  for (output_index = start; output_index < stop; ++output_index) {
    cf_qlinear_row_unroll2(job, output_index);
  }
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_register4_rolled_full(
    cf_qlinear_u8s8_i32_job_t *job, uint32_t core, uint32_t cores) {
  uint32_t first;
  uint32_t last;
  uint32_t block;

  cf_qlinear_block4_range(job->output_dim, core, cores, &first, &last);
  for (block = first; block < last; ++block) {
    cf_qlinear_tile4_rolled(job, block << 2);
  }
}

CF_QLINEAR_ALWAYS_INLINE void cf_qlinear_register4_unroll2_full(
    cf_qlinear_u8s8_i32_job_t *job, uint32_t core, uint32_t cores) {
  uint32_t first;
  uint32_t last;
  uint32_t block;

  cf_qlinear_block4_range(job->output_dim, core, cores, &first, &last);
  for (block = first; block < last; ++block) {
    cf_qlinear_tile4_unroll2(job, block << 2);
  }
}

static void cf_qlinear_register4_rolled_aligned_worker(void *argument) {
  cf_qlinear_u8s8_i32_job_t *job = (cf_qlinear_u8s8_i32_job_t *)argument;
  cf_qlinear_register4_rolled_full(job, (uint32_t)pi_core_id(),
                                  (uint32_t)pi_cl_team_nb_cores());
}

static void cf_qlinear_register4_unroll2_aligned_worker(void *argument) {
  cf_qlinear_u8s8_i32_job_t *job = (cf_qlinear_u8s8_i32_job_t *)argument;
  cf_qlinear_register4_unroll2_full(job, (uint32_t)pi_core_id(),
                                   (uint32_t)pi_cl_team_nb_cores());
}

static void cf_qlinear_register4_rolled_tail_worker(void *argument) {
  cf_qlinear_u8s8_i32_job_t *job = (cf_qlinear_u8s8_i32_job_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t cores = (uint32_t)pi_cl_team_nb_cores();

  cf_qlinear_register4_rolled_full(job, core, cores);
  if (core == 0u) {
    const uint32_t start = (uint32_t)job->output_dim & ~3u;
    uint32_t output_index;
    for (output_index = start; output_index < job->output_dim; ++output_index) {
      cf_qlinear_row_rolled(job, output_index);
    }
  }
}

static void cf_qlinear_register4_unroll2_tail_worker(void *argument) {
  cf_qlinear_u8s8_i32_job_t *job = (cf_qlinear_u8s8_i32_job_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();
  const uint32_t cores = (uint32_t)pi_cl_team_nb_cores();

  cf_qlinear_register4_unroll2_full(job, core, cores);
  if (core == 0u) {
    const uint32_t start = (uint32_t)job->output_dim & ~3u;
    uint32_t output_index;
    for (output_index = start; output_index < job->output_dim; ++output_index) {
      cf_qlinear_row_unroll2(job, output_index);
    }
  }
}

/*
 * Below 4 * cores outputs the block-of-four partition leaves most cores empty,
 * so dispatch falls back to one row per core with no register blocking. At
 * eight cores that is every layer with N < 32, including the deployed 256->4
 * head (register4 measures 0.99x there) and the wide MLP's 384->4 tail
 * (53->60us under the DORY overlay).
 *
 * Keeping the four-row tile and giving each core one whole block instead was
 * evaluated and rejected. On a single-issue core, tile4 costs one input load,
 * four weight loads and four dot products per word (9 units), while one row per
 * core costs one input load, one weight load and one dot product (3 units) and
 * the rows run concurrently. Makespan at eight cores:
 *
 *   N        4    8   12   17   24   31
 *   one-row  3    3    6    9    9   12
 *   tile4    9    9    9    9    9    9
 *
 * The tile only wins above N = 24, and loses badly exactly where the regression
 * actually is. The one-row fallback stays. Recovering these shapes needs a
 * different idea - fewer cores to cut synchronization, or accepting that a
 * four-output layer is latency-bound - not more register blocking.
 */
static cf_qlinear_worker_t cf_qlinear_select_worker(
    const cf_qlinear_u8s8_i32_job_t *job, uint8_t unroll2) {
  if (job->output_dim < 4u * job->cores) {
    return unroll2 ? cf_qlinear_register1_unroll2_worker
                   : cf_qlinear_register1_rolled_worker;
  }
  if ((job->output_dim & 3u) != 0u) {
    return unroll2 ? cf_qlinear_register4_unroll2_tail_worker
                   : cf_qlinear_register4_rolled_tail_worker;
  }
  return unroll2 ? cf_qlinear_register4_unroll2_aligned_worker
                 : cf_qlinear_register4_rolled_aligned_worker;
}

static void cf_qlinear_run_team(void *argument, uint8_t unroll2) {
  cf_qlinear_u8s8_i32_job_t *job = (cf_qlinear_u8s8_i32_job_t *)argument;
  cf_qlinear_worker_t worker = cf_qlinear_select_worker(job, unroll2);

  worker(job);
}

void cf_qlinear_u8s8_i32_register4_team(void *argument) {
  cf_qlinear_run_team(argument, 0u);
}

void cf_qlinear_u8s8_i32_register4_unroll2_team(void *argument) {
  cf_qlinear_run_team(argument, 1u);
}
