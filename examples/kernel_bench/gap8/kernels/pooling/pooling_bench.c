#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "pmsis.h"

#include "cf_qpooling.h"
#include "kernel_bench/fixture.h"
#include "kernel_bench/kernel.h"

#define KBENCH_POOL_TARGET_VISITS (1u << 18)
#define KBENCH_POOL_MAX_REPETITIONS 8u
#define KBENCH_POOL_MASTER_STACK_BYTES 4096u
#define KBENCH_POOL_SLAVE_STACK_BYTES 3072u
#define KBENCH_POOL_L1_BUDGET_BYTES 32764u

/* Single-invocation team function; the adapter owns the fork. */
typedef void (*pool_team_t)(void *argument);

typedef struct {
  cf_qpooling_job_t job;
  uint8_t *source;
  uint8_t *expected;
  struct pi_cluster_task task;
  kbench_counters_t counters;
  pool_team_t team;
  void *team_arg;
  uint32_t team_repetitions;
  uint8_t team_cores;
  uint32_t input_elements;
  uint32_t output_elements;
  uint32_t visits;
  uint32_t expected_checksum;
  uint8_t initialized;
} pool_state_t;

/*
 * One fork per invocation with the repetitions inside, so the counters bracket
 * the arithmetic only. FC elapsed time around the cluster task remains the
 * dispatch-inclusive number reported alongside it.
 */
static void pool_worker(void *argument) {
  pool_state_t *state = (pool_state_t *)argument;
  uint32_t repetition;

  kbench_perf_begin();
  for (repetition = 0u; repetition < state->team_repetitions; ++repetition) {
    state->team(state->team_arg);
  }
  kbench_perf_end(&state->counters, (uint32_t)pi_core_id());
}

static void pool_cluster_entry(void *argument) {
  pool_state_t *state = (pool_state_t *)argument;
  pi_cl_team_fork(state->team_cores, pool_worker, state);
}

static int pool_validate_base(const kbench_case_t *test_case) {
  const kbench_problem_t *problem;
  uint64_t input_elements;
  uint64_t output_elements;
  if (test_case == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  problem = &test_case->problem;
  input_elements = (uint64_t)problem->height * problem->width *
                   problem->input_channels;
  output_elements =
      (uint64_t)problem->m * problem->n * problem->output_channels;
  if (problem->family != KBENCH_PROBLEM_POOLING || problem->batch != 1u ||
      problem->height == 0u || problem->width == 0u ||
      problem->input_channels == 0u || problem->output_channels == 0u ||
      problem->input_channels != problem->output_channels || problem->m == 0u ||
      problem->n == 0u || problem->height > UINT16_MAX ||
      problem->width > UINT16_MAX || problem->input_channels > UINT16_MAX ||
      problem->m > UINT16_MAX || problem->n > UINT16_MAX ||
      input_elements > UINT32_MAX || output_elements > UINT32_MAX ||
      test_case->element_bits != 8u ||
      test_case->memory_path != KBENCH_MEMORY_L2_L1 ||
      test_case->activation != KBENCH_ACTIVATION_NONE ||
      test_case->misalignment_offset_bytes != 0u ||
      test_case->base_alignment_bytes < 4u || test_case->cores == 0u ||
      test_case->cores > CF_QPOOLING_MAX_CORES ||
      (test_case->cores & (test_case->cores - 1u)) != 0u) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  return KBENCH_OK;
}

static int pool_fixed_dimensions(const kbench_problem_t *problem) {
  uint32_t padded_height;
  uint32_t padded_width;
  uint32_t output_height;
  uint32_t output_width;
  if (problem->kernel_h == 0u || problem->kernel_w == 0u ||
      problem->stride_h == 0u || problem->stride_w == 0u) {
    return -1;
  }
  padded_height = problem->height + problem->padding_top +
                  problem->padding_bottom;
  padded_width =
      problem->width + problem->padding_left + problem->padding_right;
  if (padded_height < problem->kernel_h || padded_width < problem->kernel_w) {
    return -1;
  }
  output_height =
      (padded_height - problem->kernel_h) / problem->stride_h + 1u;
  output_width =
      (padded_width - problem->kernel_w) / problem->stride_w + 1u;
  return output_height == problem->m && output_width == problem->n ? 0 : -1;
}

static int pool_validate_local(const kbench_case_t *test_case) {
  const int status = pool_validate_base(test_case);
  const kbench_problem_t *problem;
  if (status != KBENCH_OK) return status;
  problem = &test_case->problem;
  if (pool_fixed_dimensions(problem) != 0) return KBENCH_ERR_UNSUPPORTED;
  return problem->m == 1u && problem->n == 1u &&
                 problem->kernel_h == problem->height &&
                 problem->kernel_w == problem->width
             ? KBENCH_ERR_UNSUPPORTED
             : KBENCH_OK;
}

static int pool_validate_max(const kbench_case_t *test_case) {
  const int status = pool_validate_local(test_case);
  if (status != KBENCH_OK) return status;
  return test_case->epilogue == KBENCH_EPILOGUE_INT32
             ? KBENCH_OK
             : KBENCH_ERR_UNSUPPORTED;
}

static int pool_validate_average_epilogue(const kbench_case_t *test_case) {
  return test_case->epilogue == KBENCH_EPILOGUE_INT32 ||
                 (test_case->epilogue == KBENCH_EPILOGUE_REQUANT_RELU_U8 &&
                  test_case->output_shift >= 0 &&
                  test_case->output_shift <= 30)
             ? KBENCH_OK
             : KBENCH_ERR_UNSUPPORTED;
}

static int pool_validate_avg(const kbench_case_t *test_case) {
  const int status = pool_validate_local(test_case);
  if (status != KBENCH_OK) return status;
  return pool_validate_average_epilogue(test_case);
}

static int pool_is_power_of_two(uint32_t value) {
  return value != 0u && (value & (value - 1u)) == 0u;
}

static int pool_validate_max_pulp(const kbench_case_t *test_case) {
  const int status = pool_validate_max(test_case);
  uint32_t active_cores;
  if (status != KBENCH_OK) return status;
  active_cores = test_case->problem.height < 8u
                     ? test_case->problem.height
                     : 8u;
  return test_case->cores == 8u && pool_is_power_of_two(active_cores)
             ? KBENCH_OK
             : KBENCH_ERR_UNSUPPORTED;
}

static int pool_validate_avg_pulp(const kbench_case_t *test_case) {
  const int status = pool_validate_avg(test_case);
  uint32_t active_cores;
  const kbench_problem_t *problem;
  if (status != KBENCH_OK) return status;
  problem = &test_case->problem;
  active_cores = problem->m < 8u ? problem->m : 8u;
  return test_case->cores == 8u && pool_is_power_of_two(active_cores) &&
                 problem->padding_top == problem->padding_bottom &&
                 problem->padding_left == problem->padding_right
             ? KBENCH_OK
             : KBENCH_ERR_UNSUPPORTED;
}

static int pool_validate_global(const kbench_case_t *test_case) {
  const int status = pool_validate_base(test_case);
  const kbench_problem_t *problem;
  if (status != KBENCH_OK) return status;
  problem = &test_case->problem;
  if (pool_validate_average_epilogue(test_case) != KBENCH_OK) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  return problem->m == 1u && problem->n == 1u &&
                 problem->kernel_h == problem->height &&
                 problem->kernel_w == problem->width &&
                 problem->stride_h == 1u && problem->stride_w == 1u &&
                 problem->padding_top == 0u &&
                 problem->padding_bottom == 0u &&
                 problem->padding_left == 0u &&
                 problem->padding_right == 0u
             ? KBENCH_OK
             : KBENCH_ERR_UNSUPPORTED;
}

static int pool_validate_global_pulp(const kbench_case_t *test_case) {
  const int status = pool_validate_global(test_case);
  if (status != KBENCH_OK) return status;
  return test_case->cores == 8u ? KBENCH_OK : KBENCH_ERR_UNSUPPORTED;
}

static int pool_validate_adaptive(const kbench_case_t *test_case) {
  const int status = pool_validate_base(test_case);
  if (status != KBENCH_OK) return status;
  if (pool_validate_average_epilogue(test_case) != KBENCH_OK) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  return test_case->problem.kernel_h == 0u &&
                 test_case->problem.kernel_w == 0u
             ? KBENCH_OK
             : KBENCH_ERR_UNSUPPORTED;
}

static int pool_plan(const kbench_case_t *test_case, kbench_memory_plan_t *plan) {
  const size_t alignment = test_case != NULL &&
                                   test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;
  size_t input_elements;
  size_t output_elements;
  size_t l1_bytes;
  size_t l2_bytes;
  if (test_case == NULL || plan == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  if (pool_validate_base(test_case) != KBENCH_OK) return KBENCH_ERR_UNSUPPORTED;
  input_elements = (size_t)test_case->problem.height *
                   test_case->problem.width *
                   test_case->problem.input_channels;
  output_elements = (size_t)test_case->problem.m * test_case->problem.n *
                    test_case->problem.output_channels;
  l1_bytes = input_elements + output_elements + 4u * alignment;
  l2_bytes = input_elements + output_elements + 4u * alignment;
  if (l1_bytes > KBENCH_POOL_L1_BUDGET_BYTES) return KBENCH_ERR_UNSUPPORTED;
  memset(plan, 0, sizeof(*plan));
  plan->persistent_l1_bytes = l1_bytes;
  plan->persistent_l2_bytes = l2_bytes;
  plan->input_alignment_bytes = 4u;
  plan->output_alignment_bytes = 4u;
  return KBENCH_OK;
}

static uint32_t pool_repetitions(uint32_t visits, cf_qpooling_kind_t kind) {
  uint32_t repetitions;
  if (kind == CF_QPOOLING_MAX) return 1u;
  repetitions = (KBENCH_POOL_TARGET_VISITS + visits - 1u) / visits;
  if (repetitions == 0u) repetitions = 1u;
  if (repetitions > KBENCH_POOL_MAX_REPETITIONS) {
    repetitions = KBENCH_POOL_MAX_REPETITIONS;
  }
  return repetitions;
}

static uint32_t pool_adaptive_visits(const kbench_problem_t *problem) {
  uint32_t total = 0u;
  uint32_t output_y;
  for (output_y = 0u; output_y < problem->m; ++output_y) {
    const uint32_t start_y = output_y * problem->height / problem->m;
    const uint32_t end_y = ((output_y + 1u) * problem->height + problem->m - 1u) /
                           problem->m;
    uint32_t output_x;
    for (output_x = 0u; output_x < problem->n; ++output_x) {
      const uint32_t start_x = output_x * problem->width / problem->n;
      const uint32_t end_x =
          ((output_x + 1u) * problem->width + problem->n - 1u) / problem->n;
      total += (end_y - start_y) * (end_x - start_x) *
               problem->input_channels;
    }
  }
  return total;
}

static int pool_init_common(void *state_memory,
                            const kbench_case_t *test_case,
                            kbench_runtime_t *runtime,
                            pool_team_t team,
                            cf_qpooling_kind_t kind) {
  pool_state_t *state = (pool_state_t *)state_memory;
  const kbench_problem_t *problem = &test_case->problem;
  const size_t alignment = test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;
  kbench_arena_t l1;
  kbench_arena_t l2;
  uint32_t random = test_case->seed;
  uint32_t index;
  if (state == NULL || runtime == NULL || runtime->cluster == NULL ||
      runtime->l1_arena == NULL || runtime->l2_arena == NULL || team == NULL) {
    return KBENCH_ERR_INVALID_ARGUMENT;
  }
  memset(state, 0, sizeof(*state));
  l1.cursor = (uintptr_t)runtime->l1_arena;
  l1.end = l1.cursor + runtime->l1_arena_bytes;
  l2.cursor = (uintptr_t)runtime->l2_arena;
  l2.end = l2.cursor + runtime->l2_arena_bytes;
  state->input_elements =
      problem->height * problem->width * problem->input_channels;
  state->output_elements = problem->m * problem->n * problem->output_channels;
  state->job.input = kbench_arena_take(&l1, state->input_elements, alignment);
  state->job.output = kbench_arena_take(&l1, state->output_elements, alignment);
  state->source = kbench_arena_take(&l2, state->input_elements, alignment);
  state->expected = kbench_arena_take(&l2, state->output_elements, alignment);
  if (state->job.input == NULL || state->job.output == NULL ||
      state->source == NULL || state->expected == NULL) {
    return KBENCH_ERR_NO_MEMORY;
  }
  for (index = 0u; index < state->input_elements; ++index) {
    state->source[index] = (uint8_t)(kbench_lcg(&random) >> 24);
  }
  /* Deterministic extrema and a non-divisible average at the beginning. */
  if (state->input_elements > 0u) state->source[0] = 0u;
  if (state->input_elements > 1u) state->source[1] = 255u;
  if (state->input_elements > 2u) state->source[2] = 1u;
  memcpy(state->job.input, state->source, state->input_elements);

  state->job.input_width = (uint16_t)problem->width;
  state->job.input_height = (uint16_t)problem->height;
  state->job.channels = (uint16_t)problem->input_channels;
  state->job.output_width = (uint16_t)problem->n;
  state->job.output_height = (uint16_t)problem->m;
  state->job.kernel_width = problem->kernel_w;
  state->job.kernel_height = problem->kernel_h;
  state->job.stride_width = problem->stride_w;
  state->job.stride_height = problem->stride_h;
  state->job.padding_top = problem->padding_top;
  state->job.padding_bottom = problem->padding_bottom;
  state->job.padding_left = problem->padding_left;
  state->job.padding_right = problem->padding_right;
  state->job.output_multiplier = test_case->output_multiplier;
  state->job.output_shift = (uint8_t)test_case->output_shift;
  state->job.output_add = test_case->output_shift > 0
                              ? (int32_t)1 << (test_case->output_shift - 1)
                              : 0;
  state->job.requantize =
      test_case->epilogue == KBENCH_EPILOGUE_REQUANT_RELU_U8;
  state->job.cores = test_case->cores;
  state->job.count_include_pad = 1u;
  state->job.kind = kind;
  if (kind == CF_QPOOLING_ADAPTIVE_AVERAGE) {
    state->visits = pool_adaptive_visits(problem);
  } else {
    state->visits = state->output_elements * problem->kernel_h *
                    problem->kernel_w;
  }
  state->team_repetitions =
      test_case->warmups == 0u && test_case->runs == 1u
          ? 1u
          : pool_repetitions(state->visits, kind);
  state->team = team;
  {
    cf_qpooling_job_t oracle = state->job;
    oracle.input = state->source;
    oracle.output = state->expected;
    cf_qpooling_reference(&oracle);
  }
  state->expected_checksum =
      kbench_checksum_u8(state->expected, state->output_elements);
  state->team_arg = &state->job;
  state->team_cores = (uint8_t)state->job.cores;
  pi_cluster_task(&state->task, pool_cluster_entry, state);
  state->task.stack_size = KBENCH_POOL_MASTER_STACK_BYTES;
  state->task.slave_stack_size = KBENCH_POOL_SLAVE_STACK_BYTES;
  state->initialized = 1u;
  return KBENCH_OK;
}

#define POOL_INIT(name, entry_fn, kind_value)                                   \
  static int name(void *state, const kbench_case_t *test_case,                  \
                  kbench_runtime_t *runtime) {                                  \
    return pool_init_common(state, test_case, runtime, entry_fn, kind_value);   \
  }

POOL_INIT(pool_init_max_pulp, cf_qpooling_pulp_nn_team, CF_QPOOLING_MAX)
POOL_INIT(pool_init_max_channel4, cf_qpooling_channel4_team, CF_QPOOLING_MAX)
POOL_INIT(pool_init_avg_pulp, cf_qpooling_pulp_nn_team, CF_QPOOLING_AVERAGE)
POOL_INIT(pool_init_avg_channel4, cf_qpooling_channel4_team,
          CF_QPOOLING_AVERAGE)
POOL_INIT(pool_init_global_pulp, cf_qpooling_pulp_nn_team,
          CF_QPOOLING_AVERAGE)
POOL_INIT(pool_init_global_channel, cf_qglobal_avgpool_channel_team,
          CF_QPOOLING_AVERAGE)
POOL_INIT(pool_init_adaptive_scalar, cf_qpooling_spatial_team,
          CF_QPOOLING_ADAPTIVE_AVERAGE)
POOL_INIT(pool_init_adaptive_channel4, cf_qpooling_channel4_team,
          CF_QPOOLING_ADAPTIVE_AVERAGE)

static int pool_begin(void *state_memory, const kbench_case_t *test_case,
                      kbench_runtime_t *runtime, uint32_t iteration) {
  pool_state_t *state = (pool_state_t *)state_memory;
  (void)test_case;
  (void)runtime;
  (void)iteration;
  if (state == NULL || !state->initialized) return KBENCH_ERR_RUNTIME;
  memcpy(state->job.input, state->source, state->input_elements);
  memset(state->job.output, 0xa5, state->output_elements);
  memset(&state->counters, 0, sizeof(state->counters));
  return KBENCH_OK;
}

static int pool_run(void *state_memory, kbench_runtime_t *runtime) {
  pool_state_t *state = (pool_state_t *)state_memory;
  if (state == NULL || !state->initialized || runtime == NULL ||
      runtime->cluster == NULL) {
    return KBENCH_ERR_RUNTIME;
  }
  return pi_cluster_send_task_to_cl(runtime->cluster, &state->task) == 0
             ? KBENCH_OK
             : KBENCH_ERR_RUNTIME;
}

static int pool_check(void *state_memory, const kbench_case_t *test_case,
                      kbench_runtime_t *runtime, kbench_check_t *check) {
  pool_state_t *state = (pool_state_t *)state_memory;
  uint32_t index;
  (void)test_case;
  (void)runtime;
  if (state == NULL || check == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  memset(check, 0, sizeof(*check));
  check->operations = state->visits;
  check->executed_operations = state->visits;
  check->repetitions = state->team_repetitions;
  check->checksum = kbench_checksum_u8(state->job.output, state->output_elements);
  check->expected_checksum = state->expected_checksum;
  kbench_counters_reduce(&state->counters, state->team_cores, check);
  check->weight_row_aligned = (uint8_t)(1u);
  check->exact = 1u;
  for (index = 0u; index < state->output_elements; ++index) {
    if (state->job.output[index] != state->expected[index]) {
      check->exact = 0u;
      return KBENCH_ERR_INCORRECT;
    }
  }
  return KBENCH_OK;
}

static void pool_deinit(void *state_memory, kbench_runtime_t *runtime) {
  (void)runtime;
  if (state_memory != NULL) memset(state_memory, 0, sizeof(pool_state_t));
}

#define KBENCH_POOL_DESCRIPTOR(symbol, kernel_name, variant_name, validate_fn, \
                               init_fn)                                        \
  const kbench_kernel_t symbol = {                                              \
      .name = kernel_name,                                                      \
      .variant = variant_name,                                                  \
      .family = KBENCH_PROBLEM_POOLING,                                        \
      .state_bytes = sizeof(pool_state_t),                                      \
      .state_alignment_bytes = 8u,                                              \
      .validate = validate_fn,                                                  \
      .plan = pool_plan,                                                        \
      .init = init_fn,                                                          \
      .begin = pool_begin,                                                      \
      .run = pool_run,                                                          \
      .check = pool_check,                                                      \
      .deinit = pool_deinit,                                                    \
  }

KBENCH_POOL_DESCRIPTOR(kbench_maxpool2d_u8_pulp_nn, "maxpool2d_u8", "pulp_nn",
                       pool_validate_max_pulp, pool_init_max_pulp);
KBENCH_POOL_DESCRIPTOR(kbench_maxpool2d_u8_channel4, "maxpool2d_u8", "channel4",
                       pool_validate_max, pool_init_max_channel4);
KBENCH_POOL_DESCRIPTOR(kbench_avgpool2d_u8_pulp_nn, "avgpool2d_u8", "pulp_nn",
                       pool_validate_avg_pulp, pool_init_avg_pulp);
KBENCH_POOL_DESCRIPTOR(kbench_avgpool2d_u8_channel4, "avgpool2d_u8", "channel4",
                       pool_validate_avg, pool_init_avg_channel4);
KBENCH_POOL_DESCRIPTOR(kbench_global_avgpool_u8_pulp_nn, "global_avgpool_u8",
                       "pulp_nn", pool_validate_global_pulp,
                       pool_init_global_pulp);
KBENCH_POOL_DESCRIPTOR(kbench_global_avgpool_u8_channel, "global_avgpool_u8",
                       "channel", pool_validate_global,
                       pool_init_global_channel);
KBENCH_POOL_DESCRIPTOR(kbench_adaptive_avgpool_u8_scalar, "adaptive_avgpool_u8",
                       "scalar", pool_validate_adaptive,
                       pool_init_adaptive_scalar);
KBENCH_POOL_DESCRIPTOR(kbench_adaptive_avgpool_u8_channel4,
                       "adaptive_avgpool_u8", "channel4",
                       pool_validate_adaptive, pool_init_adaptive_channel4);
