#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "pmsis.h"

#include "cf_qconv.h"
#include "cf_qconv_bias.h"
#include "kernel_bench/fixture.h"
#include "kernel_bench/kernel.h"

#define KBENCH_CONV_TARGET_MACS (1u << 20)
#define KBENCH_CONV_MAX_REPETITIONS 32u
#define KBENCH_CONV_MASTER_STACK_BYTES 4096u
#define KBENCH_CONV_SLAVE_STACK_BYTES 2048u

/* Single-invocation team function; the adapter owns the fork. */
typedef void (*conv_team_t)(void *argument);

typedef struct {
  cf_qconv_u8s8_u8_job_t job;
  uint8_t *expected;
  struct pi_cluster_task task;
  kbench_counters_t counters;
  conv_team_t team;
  void *team_arg;
  uint32_t team_repetitions;
  uint8_t team_cores;
  uint32_t checksum;
  uint32_t output_elements;
  uint32_t macs;
  uint32_t executed_macs;
  uint32_t column_stride;
  uint8_t initialized;
} conv_state_t;

/*
 * One fork per invocation with the repetitions inside, so the counters bracket
 * the arithmetic only. FC elapsed time around the cluster task remains the
 * dispatch-inclusive number reported alongside it.
 */
static void conv_worker(void *argument) {
  conv_state_t *state = (conv_state_t *)argument;
  uint32_t repetition;

  kbench_perf_begin();
  for (repetition = 0u; repetition < state->team_repetitions; ++repetition) {
    state->team(state->team_arg);
  }
  kbench_perf_end(&state->counters, (uint32_t)pi_core_id());
}

static void conv_cluster_entry(void *argument) {
  conv_state_t *state = (conv_state_t *)argument;
  pi_cl_team_fork(state->team_cores, conv_worker, state);
}

static int conv_output_dimensions(const kbench_problem_t *problem,
                                  uint32_t *output_height,
                                  uint32_t *output_width) {
  uint32_t padded_height;
  uint32_t padded_width;

  if (problem == NULL || output_height == NULL || output_width == NULL ||
      problem->height == 0u || problem->width == 0u ||
      problem->kernel_h == 0u || problem->kernel_w == 0u ||
      problem->stride_h == 0u || problem->stride_w == 0u) {
    return -1;
  }
  padded_height = problem->height + problem->padding_top +
                  problem->padding_bottom;
  padded_width =
      problem->width + problem->padding_left + problem->padding_right;
  if (padded_height < problem->kernel_h ||
      padded_width < problem->kernel_w) {
    return -1;
  }
  *output_height =
      (padded_height - problem->kernel_h) / problem->stride_h + 1u;
  *output_width =
      (padded_width - problem->kernel_w) / problem->stride_w + 1u;
  return *output_height != 0u && *output_width != 0u ? 0 : -1;
}

static int conv_map_epilogue(kbench_epilogue_t source,
                             cf_qconv_u8_epilogue_t *destination) {
  if (destination == NULL) return -1;
  switch (source) {
    case KBENCH_EPILOGUE_SHIFT_CLIP_U8:
      *destination = CF_QCONV_U8_SHIFT_CLIP;
      return 0;
    case KBENCH_EPILOGUE_REQUANT_RELU_U8:
      *destination = CF_QCONV_U8_REQUANT_RELU;
      return 0;
    case KBENCH_EPILOGUE_BATCHNORM_RELU_U8:
      *destination = CF_QCONV_U8_BATCHNORM_RELU;
      return 0;
    default:
      return -1;
  }
}

static int conv_validate_common(const kbench_case_t *test_case) {
  const kbench_problem_t *problem;
  cf_qconv_u8_epilogue_t epilogue;
  uint32_t output_height;
  uint32_t output_width;
  uint64_t columns;

  if (test_case == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  problem = &test_case->problem;
  columns = (uint64_t)problem->kernel_h * problem->kernel_w *
            problem->input_channels;
  if (problem->family != KBENCH_PROBLEM_CONV2D || problem->batch != 1u ||
      problem->input_channels == 0u || problem->output_channels == 0u ||
      problem->height > UINT16_MAX || problem->width > UINT16_MAX ||
      problem->input_channels > UINT16_MAX ||
      problem->output_channels > UINT16_MAX || columns == 0u ||
      columns > UINT16_MAX ||
      conv_output_dimensions(problem, &output_height, &output_width) != 0 ||
      output_height > UINT16_MAX || output_width > UINT16_MAX ||
      conv_map_epilogue(test_case->epilogue, &epilogue) != 0 ||
      test_case->output_shift < 0 || test_case->output_shift > 31) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  if (test_case->element_bits != 8u ||
      (test_case->memory_path != KBENCH_MEMORY_L2_DIRECT &&
       test_case->memory_path != KBENCH_MEMORY_L2_L1) ||
      test_case->misalignment_offset_bytes != 0u ||
      test_case->base_alignment_bytes < 4u || test_case->cores == 0u ||
      test_case->cores > CF_QCONV_MAX_CORES ||
      (test_case->cores & (test_case->cores - 1u)) != 0u) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  return KBENCH_OK;
}

static int conv_validate_generic(const kbench_case_t *test_case) {
  const int status = conv_validate_common(test_case);
  if (status != KBENCH_OK) return status;
  return ((uint32_t)test_case->problem.kernel_h *
                  test_case->problem.kernel_w >
              1u ||
          test_case->problem.stride_h > 1u ||
          test_case->problem.stride_w > 1u)
             ? KBENCH_OK
             : KBENCH_ERR_UNSUPPORTED;
}

/*
 * Variants that keep PULP-NN's int8 bias cannot be measured with DORY's int32
 * bias. CF_QCONV_DORY_INT32_BIAS makes the adapter write four bytes of bias per
 * output channel, which the pinned PULP-NN code reads as one. The DORY overlay
 * replaces those symbols, so they never run that way: they decline the case
 * instead of reporting `incorrect`.
 */
static int conv_validate_pinned_bias(void) {
#if CF_QCONV_DORY_INT32_BIAS
  return KBENCH_ERR_UNSUPPORTED;
#else
  return KBENCH_OK;
#endif
}

static int conv_validate_pulp(const kbench_case_t *test_case) {
  const int status = conv_validate_generic(test_case);
  if (status != KBENCH_OK) return status;
  if (conv_validate_pinned_bias() != KBENCH_OK) return KBENCH_ERR_UNSUPPORTED;
  return test_case->cores == 8u ? KBENCH_OK : KBENCH_ERR_UNSUPPORTED;
}

static int conv_validate_optimized(const kbench_case_t *test_case) {
  return conv_validate_generic(test_case);
}

/* The direct stem models one shape and declines the rest rather than silently
 * producing a wrong answer for a layout it does not implement. */
static int conv_validate_stem_direct(const kbench_case_t *test_case) {
  const int status = conv_validate_generic(test_case);
  if (status != KBENCH_OK) return status;
  return test_case->problem.input_channels == 1u &&
                 test_case->problem.kernel_h == 3u &&
                 test_case->problem.kernel_w == 3u
             ? KBENCH_OK
             : KBENCH_ERR_UNSUPPORTED;
}

/* Column stride the kernel indexes weight rows and im2col slots at: padded to
 * a multiple of four for the stem variant, compact for every other one. */
static uint32_t conv_column_stride(const kbench_problem_t *problem,
                                   uint8_t padded) {
  const uint32_t columns =
      (uint32_t)problem->kernel_h * problem->kernel_w * problem->input_channels;
  return padded ? ((columns + 3u) & ~3u) : columns;
}

/* This ablation deliberately retains the pinned wrapper, whose work partition
 * is compiled against NUM_CORES. Only the fully flattened wrapper uses the
 * active runtime team width. */
static int conv_validate_matmul4x2(const kbench_case_t *test_case) {
  const int status = conv_validate_generic(test_case);
  if (status != KBENCH_OK) return status;
  /* Retains the pinned wrapper, so it inherits the pinned bias convention. */
  if (conv_validate_pinned_bias() != KBENCH_OK) return KBENCH_ERR_UNSUPPORTED;
  return test_case->cores == 8u ? KBENCH_OK : KBENCH_ERR_UNSUPPORTED;
}

static int conv_validate_pointwise_common(const kbench_case_t *test_case) {
  const int status = conv_validate_common(test_case);
  const kbench_problem_t *problem;
  if (status != KBENCH_OK) return status;
  problem = &test_case->problem;
  return problem->kernel_h == 1u && problem->kernel_w == 1u &&
                 problem->stride_h == 1u && problem->stride_w == 1u &&
                 problem->padding_top == 0u &&
                 problem->padding_bottom == 0u &&
                 problem->padding_left == 0u &&
                 problem->padding_right == 0u
             ? KBENCH_OK
             : KBENCH_ERR_UNSUPPORTED;
}

static int conv_validate_pointwise_pulp(const kbench_case_t *test_case) {
  const int status = conv_validate_pointwise_common(test_case);
  if (status != KBENCH_OK) return status;
  if (conv_validate_pinned_bias() != KBENCH_OK) return KBENCH_ERR_UNSUPPORTED;
  return test_case->cores == 8u ? KBENCH_OK : KBENCH_ERR_UNSUPPORTED;
}

/*
 * im2col columns each core keeps live. PULP-NN's contract is two -- one per
 * output pixel in a 4x2 matmul block -- and DORY sizes its L1 allocation to
 * match. The 4x3 driver needs three, which is why it is only reachable where
 * the caller owns the buffer. See cf_qconv_flattened.c.
 */
static int conv_plan_common_cols(const kbench_case_t *test_case,
                                 kbench_memory_plan_t *plan, uint8_t padded,
                                 uint32_t im2col_columns) {
  const kbench_problem_t *problem;
  const size_t alignment = test_case != NULL &&
                                   test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;
  uint32_t output_height;
  uint32_t output_width;
  size_t input_bytes;
  size_t weight_bytes;
  size_t output_bytes;
  size_t im2col_bytes;
  size_t bytes;

  if (test_case == NULL || plan == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  if (conv_validate_common(test_case) != KBENCH_OK) return KBENCH_ERR_UNSUPPORTED;
  problem = &test_case->problem;
  if (conv_output_dimensions(problem, &output_height, &output_width) != 0) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  input_bytes = (size_t)problem->height * problem->width *
                problem->input_channels;
  weight_bytes = (size_t)problem->output_channels * problem->kernel_h *
                 problem->kernel_w * problem->input_channels;
  output_bytes = (size_t)output_height * output_width *
                 problem->output_channels;
  im2col_bytes =
      im2col_columns * test_case->cores * conv_column_stride(problem, padded);
  /* Operands the kernel reads and writes, excluding the host-side golden.
   * The bias term is separate from the two batch-norm arrays because it is
   * sized by the bias layout (CF_QCONV_BIAS_STRIDE), not by sizeof(int32_t). */
  bytes = input_bytes + weight_bytes + output_bytes + im2col_bytes +
          2u * (size_t)problem->output_channels * sizeof(int32_t) +
          (size_t)problem->output_channels * CF_QCONV_BIAS_STRIDE +
          7u * alignment;
  if (padded) {
    /* The repacked weight rows, at the padded stride. Charged only to the stem
     * descriptor, so it does not grow every other descriptor's arena. */
    bytes += (size_t)problem->output_channels *
                 conv_column_stride(problem, 1u) +
             alignment;
  }

  memset(plan, 0, sizeof(*plan));
  if (test_case->memory_path == KBENCH_MEMORY_L2_L1) {
    plan->persistent_l1_bytes = bytes;
    plan->persistent_l2_bytes = output_bytes + alignment;
  } else {
    plan->persistent_l2_bytes = bytes + output_bytes + alignment;
  }
  plan->input_alignment_bytes = 4u;
  plan->weight_alignment_bytes = 4u;
  plan->output_alignment_bytes = 4u;
  return KBENCH_OK;
}

static int conv_plan(const kbench_case_t *test_case, kbench_memory_plan_t *plan) {
  return conv_plan_common_cols(test_case, plan, 0u, 2u);
}

static int conv_plan_stem(const kbench_case_t *test_case,
                          kbench_memory_plan_t *plan) {
  return conv_plan_common_cols(test_case, plan, 1u, 2u);
}

/* Half again the im2col scratch, which is why this variant cannot be dropped
 * into the DORY overlay. Charged only to this descriptor. */
static int conv_plan3(const kbench_case_t *test_case,
                      kbench_memory_plan_t *plan) {
  return conv_plan_common_cols(test_case, plan, 0u, 3u);
}

static uint32_t conv_repetitions(uint32_t macs) {
  uint32_t repetitions = (KBENCH_CONV_TARGET_MACS + macs - 1u) / macs;
  if (repetitions == 0u) repetitions = 1u;
  if (repetitions > KBENCH_CONV_MAX_REPETITIONS) {
    repetitions = KBENCH_CONV_MAX_REPETITIONS;
  }
  return repetitions;
}

static int conv_init_common_ex(void *state_memory,
                               const kbench_case_t *test_case,
                               kbench_runtime_t *runtime,
                               conv_team_t team, uint8_t padded) {
  conv_state_t *state = (conv_state_t *)state_memory;
  const kbench_problem_t *problem = &test_case->problem;
  kbench_arena_t l2;
  kbench_arena_t l1;
  kbench_arena_t *host;
  kbench_arena_t *ops;
  const size_t alignment = test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;
  uint32_t output_height;
  uint32_t output_width;
  uint32_t random = test_case->seed;
  uint32_t input_bytes;
  uint32_t weight_bytes;
  uint32_t im2col_bytes;
  uint32_t index;

  if (state == NULL || runtime == NULL || runtime->cluster == NULL ||
      runtime->l2_arena == NULL || team == NULL ||
      conv_output_dimensions(problem, &output_height, &output_width) != 0) {
    return KBENCH_ERR_INVALID_ARGUMENT;
  }
  memset(state, 0, sizeof(*state));
  kbench_arena_init(&l2, runtime->l2_arena, runtime->l2_arena_bytes);
  host = &l2;
  /* Both pointers alias one L2 arena unless the case asks for L1 operands. */
  if (test_case->memory_path == KBENCH_MEMORY_L2_L1) {
    if (runtime->l1_arena == NULL) return KBENCH_ERR_NO_MEMORY;
    kbench_arena_init(&l1, runtime->l1_arena, runtime->l1_arena_bytes);
    ops = &l1;
  } else {
    ops = &l2;
  }
  input_bytes = problem->height * problem->width * problem->input_channels;
  weight_bytes = problem->output_channels * problem->kernel_h *
                 problem->kernel_w * problem->input_channels;
  state->output_elements =
      output_height * output_width * problem->output_channels;
  /* Keyed off the team entry rather than passed in: every other caller of this
   * function keeps PULP-NN's two-column contract. */
  im2col_bytes = (team == cf_qconv_u8s8_u8_flattened3_team ? 3u : 2u) *
                 test_case->cores * conv_column_stride(problem, padded);
  state->macs = state->output_elements * problem->kernel_h *
                problem->kernel_w * problem->input_channels;
  /* Lanes the SIMD units actually issue. Equal to `macs` everywhere except the
   * stem variant, where the zero pad lanes are executed and must be reported
   * rather than folded into the throughput figure. */
  state->executed_macs =
      state->output_elements * conv_column_stride(problem, padded);
  state->column_stride = conv_column_stride(problem, padded);

  state->job.input = kbench_arena_take(ops, input_bytes, alignment);
  state->job.weights = kbench_arena_take(ops, weight_bytes, alignment);
  /* A real bias, at the stride the kernel reads it back at: with NULL,
   * cf_qconv_bias_at() returns before its int32-bias branch, and
   * KBENCH_DORY_INT32_BIAS=1 would test nothing. */
  state->job.bias = kbench_arena_take(
      ops, (size_t)problem->output_channels * CF_QCONV_BIAS_STRIDE, alignment);
  state->job.batch_norm_multiplier = kbench_arena_take(
      ops, (size_t)problem->output_channels * sizeof(int32_t), alignment);
  state->job.batch_norm_bias = kbench_arena_take(
      ops, (size_t)problem->output_channels * sizeof(int32_t), alignment);
  state->job.output =
      kbench_arena_take(ops, state->output_elements, alignment);
  state->expected =
      kbench_arena_take(host, state->output_elements, alignment);
  state->job.im2col = kbench_arena_take(ops, im2col_bytes, alignment);
  /* Taken last so it does not move the other operands. NULL for every
   * non-stem descriptor, which is the value cf_qconv_u8s8_u8_stem_repack_team
   * treats as "no repack requested". */
  state->job.weight_repack =
      padded ? kbench_arena_take(
                   ops,
                   (size_t)problem->output_channels *
                       conv_column_stride(problem, 1u),
                   alignment)
             : NULL;
  if (state->job.input == NULL || state->job.weights == NULL ||
      state->job.bias == NULL ||
      state->job.batch_norm_multiplier == NULL ||
      state->job.batch_norm_bias == NULL || state->job.output == NULL ||
      state->expected == NULL || state->job.im2col == NULL ||
      (padded && state->job.weight_repack == NULL)) {
    return KBENCH_ERR_NO_MEMORY;
  }

  state->job.output_multiplier = test_case->output_multiplier;
  state->job.output_shift = (uint16_t)test_case->output_shift;
  state->job.input_width = (uint16_t)problem->width;
  state->job.input_height = (uint16_t)problem->height;
  state->job.input_channels = (uint16_t)problem->input_channels;
  state->job.output_width = (uint16_t)output_width;
  state->job.output_height = (uint16_t)output_height;
  state->job.output_channels = (uint16_t)problem->output_channels;
  state->job.kernel_width = problem->kernel_w;
  state->job.kernel_height = problem->kernel_h;
  state->job.padding_top = problem->padding_top;
  state->job.padding_bottom = problem->padding_bottom;
  state->job.padding_left = problem->padding_left;
  state->job.padding_right = problem->padding_right;
  state->job.stride_width = problem->stride_w;
  state->job.stride_height = problem->stride_h;
  state->job.cores = test_case->cores;
  state->team_repetitions =
      test_case->warmups == 0u && test_case->runs == 1u
          ? 1u
          : conv_repetitions(state->macs);
  state->team = team;
  if (conv_map_epilogue(test_case->epilogue, &state->job.epilogue) != 0) {
    return KBENCH_ERR_UNSUPPORTED;
  }

  for (index = 0u; index < input_bytes; ++index) {
    ((uint8_t *)state->job.input)[index] =
        (uint8_t)(kbench_lcg(&random) % 32u);
  }
  for (index = 0u; index < weight_bytes; ++index) {
    ((int8_t *)state->job.weights)[index] =
        (int8_t)(kbench_lcg(&random) % 9u) - 4;
  }
  for (index = 0u; index < problem->output_channels; ++index) {
    const int32_t bias_value = (int32_t)(kbench_lcg(&random) % 9u) - 4;
    /* Written at the same stride the kernel and the oracle will read. */
#if CF_QCONV_DORY_INT32_BIAS
    ((int32_t *)state->job.bias)[index] = bias_value;
#else
    ((int8_t *)state->job.bias)[index] = (int8_t)bias_value;
#endif
  }
  for (index = 0u; index < problem->output_channels; ++index) {
    ((int32_t *)state->job.batch_norm_multiplier)[index] =
        (int32_t)(kbench_lcg(&random) % 4u) + 1;
    ((int32_t *)state->job.batch_norm_bias)[index] =
        (int32_t)(kbench_lcg(&random) % 1025u) - 512;
  }
  memset(state->job.im2col, 0, im2col_bytes);
  {
    cf_qconv_u8s8_u8_job_t oracle = state->job;
    oracle.output = state->expected;
    cf_qconv_u8s8_u8_reference(&oracle);
  }
  state->checksum = kbench_checksum_u8(state->expected, state->output_elements);
  state->team_arg = &state->job;
  state->team_cores = (uint8_t)state->job.cores;
  pi_cluster_task(&state->task, conv_cluster_entry, state);
  state->task.stack_size = KBENCH_CONV_MASTER_STACK_BYTES;
  state->task.slave_stack_size = KBENCH_CONV_SLAVE_STACK_BYTES;
  state->initialized = 1u;
  return KBENCH_OK;
}

static int conv_init_common(void *state_memory, const kbench_case_t *test_case,
                            kbench_runtime_t *runtime, conv_team_t team) {
  return conv_init_common_ex(state_memory, test_case, runtime, team, 0u);
}

static int conv_init_pulp(void *state, const kbench_case_t *test_case,
                          kbench_runtime_t *runtime) {
  return conv_init_common(state, test_case, runtime,
                          cf_qconv_u8s8_u8_pulp_nn_team);
}

static int conv_init_stem(void *state, const kbench_case_t *test_case,
                          kbench_runtime_t *runtime) {
  return conv_init_common_ex(state, test_case, runtime,
                             cf_qconv_u8s8_u8_stem_repack_team, 1u);
}

static int conv_init_stem_direct(void *state, const kbench_case_t *test_case,
                                 kbench_runtime_t *runtime) {
  return conv_init_common_ex(state, test_case, runtime,
                             cf_qconv_u8s8_u8_stem_direct_team, 1u);
}

static int conv_init_matmul4x2(void *state, const kbench_case_t *test_case,
                               kbench_runtime_t *runtime) {
  return conv_init_common(state, test_case, runtime,
                          cf_qconv_u8s8_u8_matmul4x2_team);
}

static int conv_init_flattened(void *state, const kbench_case_t *test_case,
                               kbench_runtime_t *runtime) {
  return conv_init_common(state, test_case, runtime,
                          cf_qconv_u8s8_u8_flattened_team);
}

static int conv_init_flattened3(void *state, const kbench_case_t *test_case,
                                kbench_runtime_t *runtime) {
  return conv_init_common(state, test_case, runtime,
                          cf_qconv_u8s8_u8_flattened3_team);
}

static int conv_init_pointwise_pulp(void *state,
                                    const kbench_case_t *test_case,
                                    kbench_runtime_t *runtime) {
  return conv_init_common(state, test_case, runtime,
                          cf_qconv_u8s8_u8_pointwise_pulp_nn_team);
}

static int conv_init_pointwise_flattened(void *state,
                                         const kbench_case_t *test_case,
                                         kbench_runtime_t *runtime) {
  return conv_init_common(state, test_case, runtime,
                          cf_qconv_u8s8_u8_pointwise_flattened_team);
}

static int conv_begin(void *state_memory, const kbench_case_t *test_case,
                      kbench_runtime_t *runtime, uint32_t iteration) {
  conv_state_t *state = (conv_state_t *)state_memory;
  (void)test_case;
  (void)runtime;
  (void)iteration;
  if (state == NULL || !state->initialized) return KBENCH_ERR_RUNTIME;
  memset(state->job.output, 0xa5, state->output_elements);
  memset(&state->counters, 0, sizeof(state->counters));
  return KBENCH_OK;
}

static int conv_run(void *state_memory, kbench_runtime_t *runtime) {
  conv_state_t *state = (conv_state_t *)state_memory;
  if (state == NULL || !state->initialized || runtime == NULL ||
      runtime->cluster == NULL) {
    return KBENCH_ERR_RUNTIME;
  }
  if (pi_cluster_send_task_to_cl(runtime->cluster, &state->task) != 0) {
    return KBENCH_ERR_RUNTIME;
  }
  return KBENCH_OK;
}

static int conv_check(void *state_memory, const kbench_case_t *test_case,
                      kbench_runtime_t *runtime, kbench_check_t *check) {
  conv_state_t *state = (conv_state_t *)state_memory;
  uint32_t index;
  (void)test_case;
  (void)runtime;
  if (state == NULL || check == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  memset(check, 0, sizeof(*check));
  check->operations = state->macs;
  check->executed_operations = state->executed_macs;
  check->repetitions = state->team_repetitions;
  check->checksum = kbench_checksum_u8(state->job.output, state->output_elements);
  check->expected_checksum = state->checksum;
  kbench_counters_reduce(&state->counters, state->team_cores, check);
  /* The stride the kernel indexes weight rows at, which is the padded one for
   * the stem variant. Reporting the compact K here would say "misaligned" for
   * precisely the kernel that exists to fix the misalignment. */
  check->weight_row_aligned = (uint8_t)((state->column_stride & 3u) == 0u);
  check->exact = 1u;
  for (index = 0u; index < state->output_elements; ++index) {
    if (state->job.output[index] != state->expected[index]) {
      check->exact = 0u;
      return KBENCH_ERR_INCORRECT;
    }
  }
  return KBENCH_OK;
}

static void conv_deinit(void *state_memory, kbench_runtime_t *runtime) {
  (void)runtime;
  if (state_memory != NULL) memset(state_memory, 0, sizeof(conv_state_t));
}

#define KBENCH_CONV_DESCRIPTOR_PLAN(symbol, variant_name, validate_fn, plan_fn,  \
                                    init_fn)                                   \
  const kbench_kernel_t symbol = {                                             \
      .name = "conv2d_u8s8_u8",                                               \
      .variant = variant_name,                                                 \
      .family = KBENCH_PROBLEM_CONV2D,                                         \
      .state_bytes = sizeof(conv_state_t),                                     \
      .state_alignment_bytes = 8u,                                             \
      .validate = validate_fn,                                                 \
      .plan = plan_fn,                                                         \
      .init = init_fn,                                                         \
      .begin = conv_begin,                                                     \
      .run = conv_run,                                                         \
      .check = conv_check,                                                     \
      .deinit = conv_deinit,                                                   \
  }

#define KBENCH_CONV_DESCRIPTOR(symbol, variant_name, validate_fn, init_fn)     \
  KBENCH_CONV_DESCRIPTOR_PLAN(symbol, variant_name, validate_fn, conv_plan,    \
                              init_fn)

KBENCH_CONV_DESCRIPTOR(kbench_conv2d_u8s8_u8_pulp_nn, "pulp_nn",
                       conv_validate_pulp, conv_init_pulp);
KBENCH_CONV_DESCRIPTOR(kbench_conv2d_u8s8_u8_matmul4x2, "matmul4x2",
                       conv_validate_matmul4x2, conv_init_matmul4x2);
KBENCH_CONV_DESCRIPTOR(kbench_conv2d_u8s8_u8_flattened,
                       "flattened_matmul4x2", conv_validate_optimized,
                       conv_init_flattened);
/* Three im2col columns per core and cf_qconv_matmul4x3. Its own plan because
 * the extra column is 50% more scratch; see conv_plan3. */
KBENCH_CONV_DESCRIPTOR_PLAN(kbench_conv2d_u8s8_u8_flattened3,
                            "flattened_matmul4x3", conv_validate_optimized,
                            conv_plan3, conv_init_flattened3);
KBENCH_CONV_DESCRIPTOR(kbench_pointwise_u8s8_u8_pulp_nn, "pointwise_pulp_nn",
                       conv_validate_pointwise_pulp,
                       conv_init_pointwise_pulp);
KBENCH_CONV_DESCRIPTOR(kbench_pointwise_u8s8_u8_flattened,
                       "pointwise_flattened4x2",
                       conv_validate_pointwise_common,
                       conv_init_pointwise_flattened);

/* Padded-K stem variant. Its plan carries the extra repack buffer and the
 * im2col at the padded stride; every other descriptor keeps conv_plan. */
KBENCH_CONV_DESCRIPTOR_PLAN(kbench_conv2d_u8s8_u8_stem_repack, "stem_repack4",
                            conv_validate_optimized, conv_plan_stem,
                            conv_init_stem);

/* Direct single-channel stem. Shares conv_plan_stem's repack buffer, fills it
 * with a different layout, and never touches im2col. */
KBENCH_CONV_DESCRIPTOR_PLAN(kbench_conv2d_u8s8_u8_stem_direct, "stem_direct3x3",
                            conv_validate_stem_direct, conv_plan_stem,
                            conv_init_stem_direct);
