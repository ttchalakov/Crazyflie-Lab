#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "pmsis.h"

#include "cf_qlinear.h"
#include "kernel_bench/fixture.h"
#include "kernel_bench/kernel.h"

#define KBENCH_LINEAR_TARGET_MACS (1u << 18)
#define KBENCH_LINEAR_MAX_REPETITIONS 64u
/* The instrumented worker adds a perf frame and an indirect call above the
 * kernel, so the stacks match conv/depthwise/matmul. */
#define KBENCH_LINEAR_MASTER_STACK_BYTES 4096u
#define KBENCH_LINEAR_SLAVE_STACK_BYTES 2048u

/* Single-invocation team function; the adapter owns the fork and the timing. */
typedef void (*linear_team_t)(void *argument);

typedef struct {
  cf_qlinear_u8s8_i32_job_t job;
  int32_t *expected;
  struct pi_cluster_task task;
  kbench_counters_t counters;
  linear_team_t team;
  uint32_t repetitions;
  uint32_t checksum;
  uint8_t initialized;
} linear_state_t;

/*
 * The adapter forks once and repeats inside the team so the performance
 * counters bracket the arithmetic only. FC elapsed time around the cluster task
 * remains the secondary, dispatch-inclusive number.
 */
static void linear_worker(void *argument) {
  linear_state_t *state = (linear_state_t *)argument;
  uint32_t repetition;

  kbench_perf_begin();
  for (repetition = 0u; repetition < state->repetitions; ++repetition) {
    state->team(&state->job);
  }
  kbench_perf_end(&state->counters, (uint32_t)pi_core_id());
}

static void linear_cluster_entry(void *argument) {
  linear_state_t *state = (linear_state_t *)argument;
  pi_cl_team_fork(state->job.cores, linear_worker, state);
}

static int linear_validate_common(const kbench_case_t *test_case) {
  const kbench_problem_t *problem;

  if (test_case == NULL) {
    return KBENCH_ERR_INVALID_ARGUMENT;
  }
  problem = &test_case->problem;
  if (problem->family != KBENCH_PROBLEM_LINEAR || problem->batch != 1u ||
      problem->m != 1u || problem->k == 0u || problem->n == 0u ||
      problem->k > UINT16_MAX - 3u || problem->n > UINT16_MAX) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  if (test_case->element_bits != 8u ||
      test_case->epilogue != KBENCH_EPILOGUE_INT32 ||
      (test_case->memory_path != KBENCH_MEMORY_L2_DIRECT &&
       test_case->memory_path != KBENCH_MEMORY_L2_L1) ||
      test_case->misalignment_offset_bytes != 0u ||
      test_case->base_alignment_bytes < 4u) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  if (test_case->cores == 0u || test_case->cores > CF_QLINEAR_MAX_CORES ||
      (test_case->cores & (test_case->cores - 1u)) != 0u) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  return KBENCH_OK;
}

static int linear_validate_pulp(const kbench_case_t *test_case) {
  const int status = linear_validate_common(test_case);
  if (status != KBENCH_OK) {
    return status;
  }
  return test_case->cores == 8u ? KBENCH_OK : KBENCH_ERR_UNSUPPORTED;
}

static int linear_validate_register4(const kbench_case_t *test_case) {
  return linear_validate_common(test_case);
}

static int linear_plan(const kbench_case_t *test_case,
                       kbench_memory_plan_t *plan) {
  const size_t alignment = test_case != NULL &&
                                   test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;
  const size_t packed_k = test_case != NULL
                              ? cf_qlinear_u8s8_i32_padded_input_dim(
                                    (uint16_t)test_case->problem.k)
                              : 0u;
  size_t bytes;

  if (plan == NULL || test_case == NULL) {
    return KBENCH_ERR_INVALID_ARGUMENT;
  }
  if (linear_validate_common(test_case) != KBENCH_OK) {
    return KBENCH_ERR_UNSUPPORTED;
  }

  /* Operands the kernel actually reads and writes. */
  bytes = packed_k;
  bytes += (size_t)test_case->problem.n * packed_k;
  bytes += 2u * (size_t)test_case->problem.n * sizeof(int32_t);
  bytes += 4u * alignment;

  memset(plan, 0, sizeof(*plan));
  if (test_case->memory_path == KBENCH_MEMORY_L2_L1) {
    /* Operands resident in cluster L1; the preparation scratch and the golden
     * stay in L2 so the measured region is the same arithmetic on faster
     * memory. */
    plan->persistent_l1_bytes = bytes;
    plan->persistent_l2_bytes = (size_t)test_case->problem.k +
                                (size_t)test_case->problem.n * sizeof(int32_t) +
                                2u * alignment;
  } else {
    plan->persistent_l2_bytes =
        bytes + (size_t)test_case->problem.k +
        (size_t)test_case->problem.n * sizeof(int32_t) + 2u * alignment;
  }
  plan->input_alignment_bytes = 4u;
  plan->weight_alignment_bytes = 4u;
  plan->output_alignment_bytes = 4u;
  return KBENCH_OK;
}

static uint32_t linear_repetitions(const kbench_case_t *test_case) {
  const uint32_t macs = test_case->problem.n * test_case->problem.k;
  uint32_t repetitions = (KBENCH_LINEAR_TARGET_MACS + macs - 1u) / macs;

  if (repetitions == 0u) repetitions = 1u;
  if (repetitions > KBENCH_LINEAR_MAX_REPETITIONS) {
    repetitions = KBENCH_LINEAR_MAX_REPETITIONS;
  }
  return repetitions;
}

static int linear_init_common(void *state_memory,
                              const kbench_case_t *test_case,
                              kbench_runtime_t *runtime,
                              linear_team_t team) {
  linear_state_t *state = (linear_state_t *)state_memory;
  kbench_arena_t l2;
  kbench_arena_t l1;
  kbench_arena_t *host;
  kbench_arena_t *ops;
  const size_t alignment = test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;
  const uint16_t logical_k = (uint16_t)test_case->problem.k;
  const uint16_t packed_k =
      cf_qlinear_u8s8_i32_padded_input_dim(logical_k);
  uint8_t *prepare_scratch;
  uint32_t random = test_case->seed;
  uint32_t index;

  if (state == NULL || runtime == NULL || runtime->cluster == NULL ||
      runtime->l2_arena == NULL || team == NULL) {
    return KBENCH_ERR_INVALID_ARGUMENT;
  }

  memset(state, 0, sizeof(*state));
  kbench_arena_init(&l2, runtime->l2_arena, runtime->l2_arena_bytes);
  host = &l2;
  /* Operands move to cluster L1 when the case asks for it. Both pointers alias
   * one arena in L2-direct mode so the sub-allocations cannot overlap. */
  if (test_case->memory_path == KBENCH_MEMORY_L2_L1) {
    if (runtime->l1_arena == NULL) return KBENCH_ERR_NO_MEMORY;
    kbench_arena_init(&l1, runtime->l1_arena, runtime->l1_arena_bytes);
    ops = &l1;
  } else {
    ops = &l2;
  }

  prepare_scratch = kbench_arena_take(host, logical_k, alignment);
  state->job.input = kbench_arena_take(ops, packed_k, alignment);
  state->job.weights = kbench_arena_take(
      ops, (size_t)test_case->problem.n * packed_k, alignment);
  state->job.bias = kbench_arena_take(
      ops, (size_t)test_case->problem.n * sizeof(int32_t), alignment);
  state->job.output = kbench_arena_take(
      ops, (size_t)test_case->problem.n * sizeof(int32_t), alignment);
  state->expected = kbench_arena_take(
      host, (size_t)test_case->problem.n * sizeof(int32_t), alignment);
  if (prepare_scratch == NULL || state->job.input == NULL ||
      state->job.weights == NULL ||
      state->job.bias == NULL || state->job.output == NULL ||
      state->expected == NULL) {
    return KBENCH_ERR_NO_MEMORY;
  }

  state->job.input_dim = logical_k;
  state->job.packed_input_dim = packed_k;
  state->job.weight_row_stride_bytes = packed_k;
  state->job.output_dim = (uint16_t)test_case->problem.n;
  state->job.cores = test_case->cores;
  state->repetitions = linear_repetitions(test_case);
  state->team = team;

  for (index = 0; index < logical_k; ++index) {
    prepare_scratch[index] = (uint8_t)(kbench_lcg(&random) >> 24);
  }
  if (cf_qlinear_u8s8_i32_prepare_input((uint8_t *)state->job.input, packed_k,
                                       prepare_scratch, logical_k) != 0) {
    return KBENCH_ERR_RUNTIME;
  }
  for (index = 0; index < test_case->problem.n; ++index) {
    uint32_t input_index;
    int8_t *prepared_row =
        (int8_t *)state->job.weights + (size_t)index * packed_k;

    for (input_index = 0; input_index < logical_k; ++input_index) {
      ((int8_t *)prepare_scratch)[input_index] =
          (int8_t)((int32_t)(kbench_lcg(&random) >> 28) - 8);
    }
    if (cf_qlinear_u8s8_i32_prepare_weights(
            prepared_row, packed_k, (const int8_t *)prepare_scratch, logical_k,
            logical_k, 1u) != 0) {
      return KBENCH_ERR_RUNTIME;
    }
  }
  for (index = 0; index < test_case->problem.n; ++index) {
    ((int32_t *)state->job.bias)[index] =
        (int32_t)(kbench_lcg(&random) & 2047u) - 1024;
  }

  {
    cf_qlinear_u8s8_i32_job_t oracle = state->job;
    oracle.output = state->expected;
    cf_qlinear_u8s8_i32_reference(&oracle);
  }
  state->checksum = kbench_checksum_i32(state->expected, test_case->problem.n);

  pi_cluster_task(&state->task, linear_cluster_entry, state);
  state->task.stack_size = KBENCH_LINEAR_MASTER_STACK_BYTES;
  state->task.slave_stack_size = KBENCH_LINEAR_SLAVE_STACK_BYTES;
  state->initialized = 1u;
  return KBENCH_OK;
}

static int linear_init_pulp(void *state, const kbench_case_t *test_case,
                            kbench_runtime_t *runtime) {
  return linear_init_common(state, test_case, runtime,
                            cf_qlinear_u8s8_i32_pulp_nn_team);
}

static int linear_init_register4(void *state, const kbench_case_t *test_case,
                                 kbench_runtime_t *runtime) {
  return linear_init_common(state, test_case, runtime,
                            cf_qlinear_u8s8_i32_register4_team);
}

static int linear_init_register4_unroll2(void *state,
                                         const kbench_case_t *test_case,
                                         kbench_runtime_t *runtime) {
  return linear_init_common(state, test_case, runtime,
                            cf_qlinear_u8s8_i32_register4_unroll2_team);
}

static int linear_begin(void *state_memory, const kbench_case_t *test_case,
                        kbench_runtime_t *runtime, uint32_t iteration) {
  linear_state_t *state = (linear_state_t *)state_memory;
  (void)runtime;
  (void)iteration;

  if (state == NULL || !state->initialized) {
    return KBENCH_ERR_RUNTIME;
  }
  memset(state->job.output, 0xa5,
         (size_t)test_case->problem.n * sizeof(int32_t));
  memset(&state->counters, 0, sizeof(state->counters));
  return KBENCH_OK;
}

static int linear_run(void *state_memory, kbench_runtime_t *runtime) {
  linear_state_t *state = (linear_state_t *)state_memory;

  if (state == NULL || !state->initialized || runtime == NULL ||
      runtime->cluster == NULL) {
    return KBENCH_ERR_RUNTIME;
  }
  if (pi_cluster_send_task_to_cl(runtime->cluster, &state->task) != 0) {
    return KBENCH_ERR_RUNTIME;
  }
  return KBENCH_OK;
}

static int linear_check(void *state_memory, const kbench_case_t *test_case,
                        kbench_runtime_t *runtime, kbench_check_t *check) {
  linear_state_t *state = (linear_state_t *)state_memory;
  uint32_t index;
  (void)runtime;

  if (state == NULL || check == NULL) {
    return KBENCH_ERR_INVALID_ARGUMENT;
  }
  memset(check, 0, sizeof(*check));
  check->operations =
      (uint64_t)test_case->problem.n * test_case->problem.k;
  check->executed_operations =
      (uint64_t)test_case->problem.n * state->job.packed_input_dim;
  check->repetitions = state->repetitions;
  check->checksum = kbench_checksum_i32(state->job.output, test_case->problem.n);
  check->expected_checksum = state->checksum;
  kbench_counters_reduce(&state->counters, state->job.cores, check);
  check->weight_row_aligned = (uint8_t)((state->job.weight_row_stride_bytes & 3u) == 0u);
  check->exact = 1u;
  for (index = 0; index < test_case->problem.n; ++index) {
    if (state->job.output[index] != state->expected[index]) {
      check->exact = 0u;
      return KBENCH_ERR_INCORRECT;
    }
  }
  return KBENCH_OK;
}

static void linear_deinit(void *state_memory, kbench_runtime_t *runtime) {
  (void)runtime;
  if (state_memory != NULL) {
    memset(state_memory, 0, sizeof(linear_state_t));
  }
}

const kbench_kernel_t kbench_linear_u8s8_i32_pulp_nn = {
    .name = "linear_u8s8_i32",
    .variant = "pulp_nn",
    .family = KBENCH_PROBLEM_LINEAR,
    .state_bytes = sizeof(linear_state_t),
    .state_alignment_bytes = 8u,
    .validate = linear_validate_pulp,
    .plan = linear_plan,
    .init = linear_init_pulp,
    .begin = linear_begin,
    .run = linear_run,
    .check = linear_check,
    .deinit = linear_deinit,
};

const kbench_kernel_t kbench_linear_u8s8_i32_register4 = {
    .name = "linear_u8s8_i32",
    .variant = "register4",
    .family = KBENCH_PROBLEM_LINEAR,
    .state_bytes = sizeof(linear_state_t),
    .state_alignment_bytes = 8u,
    .validate = linear_validate_register4,
    .plan = linear_plan,
    .init = linear_init_register4,
    .begin = linear_begin,
    .run = linear_run,
    .check = linear_check,
    .deinit = linear_deinit,
};

const kbench_kernel_t kbench_linear_u8s8_i32_register4_unroll2 = {
    .name = "linear_u8s8_i32",
    .variant = "register4_unroll2",
    .family = KBENCH_PROBLEM_LINEAR,
    .state_bytes = sizeof(linear_state_t),
    .state_alignment_bytes = 8u,
    .validate = linear_validate_register4,
    .plan = linear_plan,
    .init = linear_init_register4_unroll2,
    .begin = linear_begin,
    .run = linear_run,
    .check = linear_check,
    .deinit = linear_deinit,
};
