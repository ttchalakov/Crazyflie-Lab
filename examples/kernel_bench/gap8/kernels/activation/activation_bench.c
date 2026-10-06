#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "pmsis.h"

#include "activation_luts.h"
#include "cf_qactivation.h"
#include "kernel_bench/fixture.h"
#include "kernel_bench/kernel.h"

#define KBENCH_ACTIVATION_TARGET_ELEMENTS (1u << 16)
#define KBENCH_ACTIVATION_MAX_REPETITIONS 256u
#define KBENCH_ACTIVATION_MASTER_STACK_BYTES 4096u
#define KBENCH_ACTIVATION_SLAVE_STACK_BYTES 2048u
#define KBENCH_ACTIVATION_DORY_L1_BUDGET_BYTES 32764u

/* Single-invocation team function; the adapter owns the fork. */
typedef void (*activation_team_t)(void *argument);

typedef union {
  cf_qactivation_lut_job_t lut;
  cf_qsoftmax_job_t softmax;
} activation_job_t;

typedef struct {
  activation_job_t job;
  uint8_t *expected;
  struct pi_cluster_task task;
  kbench_counters_t counters;
  activation_team_t team;
  void *team_arg;
  uint32_t team_repetitions;
  uint8_t team_cores;
  uint32_t expected_checksum;
  uint32_t elements;
  uint32_t repetitions;
  uint8_t softmax;
  uint8_t initialized;
} activation_state_t;

/*
 * One fork per invocation with the repetitions inside, so the counters bracket
 * the arithmetic only. FC elapsed time around the cluster task remains the
 * dispatch-inclusive number reported alongside it.
 */
static void activation_worker(void *argument) {
  activation_state_t *state = (activation_state_t *)argument;
  uint32_t repetition;

  kbench_perf_begin();
  for (repetition = 0u; repetition < state->team_repetitions; ++repetition) {
    state->team(state->team_arg);
  }
  kbench_perf_end(&state->counters, (uint32_t)pi_core_id());
}

static void activation_cluster_entry(void *argument) {
  activation_state_t *state = (activation_state_t *)argument;
  pi_cl_team_fork(state->team_cores, activation_worker, state);
}

static int activation_map_kind(kbench_activation_t source,
                               cf_qactivation_kind_t *destination) {
  if (destination == NULL) return -1;
  switch (source) {
    case KBENCH_ACTIVATION_SIGMOID:
      *destination = CF_QACTIVATION_SIGMOID;
      return 0;
    case KBENCH_ACTIVATION_TANH:
      *destination = CF_QACTIVATION_TANH;
      return 0;
    case KBENCH_ACTIVATION_LEAKY_RELU:
      *destination = CF_QACTIVATION_LEAKY_RELU;
      return 0;
    case KBENCH_ACTIVATION_HARD_SIGMOID:
      *destination = CF_QACTIVATION_HARD_SIGMOID;
      return 0;
    case KBENCH_ACTIVATION_HARD_SWISH:
      *destination = CF_QACTIVATION_HARD_SWISH;
      return 0;
    case KBENCH_ACTIVATION_SILU:
      *destination = CF_QACTIVATION_SILU;
      return 0;
    case KBENCH_ACTIVATION_GELU:
      *destination = CF_QACTIVATION_GELU;
      return 0;
    default:
      return -1;
  }
}

static int activation_validate_common(const kbench_case_t *test_case) {
  const kbench_problem_t *problem;
  if (test_case == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  problem = &test_case->problem;
  if (problem->family != KBENCH_PROBLEM_ACTIVATION || problem->batch != 1u ||
      problem->m != 1u || problem->n == 0u ||
      problem->n > UINT16_MAX || test_case->element_bits != 8u ||
      test_case->memory_path != KBENCH_MEMORY_L2_L1 ||
      test_case->epilogue != KBENCH_EPILOGUE_INT32 ||
      test_case->misalignment_offset_bytes != 0u ||
      test_case->base_alignment_bytes < 4u || test_case->cores == 0u ||
      test_case->cores > CF_QACTIVATION_MAX_CORES ||
      (test_case->cores & (test_case->cores - 1u)) != 0u) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  return KBENCH_OK;
}

static int activation_validate_lut(const kbench_case_t *test_case) {
  cf_qactivation_kind_t kind;
  const int status = activation_validate_common(test_case);
  if (status != KBENCH_OK) return status;
  return activation_map_kind(test_case->activation, &kind) == 0
             ? KBENCH_OK
             : KBENCH_ERR_UNSUPPORTED;
}

static int activation_validate_softmax(const kbench_case_t *test_case) {
  const int status = activation_validate_common(test_case);
  if (status != KBENCH_OK) return status;
  return test_case->activation == KBENCH_ACTIVATION_SOFTMAX &&
                 test_case->problem.n <= CF_QSOFTMAX_MAX_COLUMNS
             ? KBENCH_OK
             : KBENCH_ERR_UNSUPPORTED;
}

static int activation_plan_lut(const kbench_case_t *test_case,
                               kbench_memory_plan_t *plan) {
  const size_t alignment = test_case != NULL &&
                                   test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;
  size_t l1_bytes;
  if (test_case == NULL || plan == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  if (activation_validate_lut(test_case) != KBENCH_OK) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  l1_bytes = 2u * (size_t)test_case->problem.n +
             CF_QACTIVATION_LUT_ENTRIES + 4u * alignment;
  if (l1_bytes > KBENCH_ACTIVATION_DORY_L1_BUDGET_BYTES) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  memset(plan, 0, sizeof(*plan));
  plan->persistent_l1_bytes = l1_bytes;
  plan->persistent_l2_bytes = (size_t)test_case->problem.n + alignment;
  plan->input_alignment_bytes = 4u;
  plan->weight_alignment_bytes = 4u;
  plan->output_alignment_bytes = 4u;
  return KBENCH_OK;
}

static int activation_plan_softmax(const kbench_case_t *test_case,
                                   kbench_memory_plan_t *plan) {
  const size_t alignment = test_case != NULL &&
                                   test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;
  size_t l1_bytes;
  if (test_case == NULL || plan == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  if (activation_validate_softmax(test_case) != KBENCH_OK) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  l1_bytes = 2u * (size_t)test_case->problem.n +
             kbench_softmax_default_exp_lut_bytes() + 4u * alignment;
  if (l1_bytes > KBENCH_ACTIVATION_DORY_L1_BUDGET_BYTES) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  memset(plan, 0, sizeof(*plan));
  plan->persistent_l1_bytes = l1_bytes;
  plan->persistent_l2_bytes = (size_t)test_case->problem.n + alignment;
  plan->input_alignment_bytes = 4u;
  plan->weight_alignment_bytes = 4u;
  plan->output_alignment_bytes = 4u;
  return KBENCH_OK;
}

static uint32_t activation_repetitions(uint32_t elements, uint32_t maximum) {
  uint32_t repetitions =
      (KBENCH_ACTIVATION_TARGET_ELEMENTS + elements - 1u) / elements;
  if (repetitions == 0u) repetitions = 1u;
  if (repetitions > maximum) repetitions = maximum;
  return repetitions;
}

static void activation_fill_input(uint8_t *input, uint32_t elements,
                                  uint32_t seed) {
  uint32_t index;
  for (index = 0u; index < elements; ++index) {
    input[index] = elements >= 256u ? (uint8_t)index
                                    : (uint8_t)(kbench_lcg(&seed) >> 24);
  }
  if (elements > 0u) input[0] = 0u;
  if (elements > 1u) input[1] = 127u;
  if (elements > 2u) input[2] = 128u;
  if (elements > 3u) input[3] = 255u;
}

static int activation_init_lut_common(void *state_memory,
                                      const kbench_case_t *test_case,
                                      kbench_runtime_t *runtime,
                                      activation_team_t team) {
  activation_state_t *state = (activation_state_t *)state_memory;
  kbench_arena_t l1;
  kbench_arena_t l2;
  cf_qactivation_kind_t kind;
  const uint8_t *default_lut;
  const size_t alignment = test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;

  if (state == NULL || runtime == NULL || runtime->cluster == NULL ||
      runtime->l1_arena == NULL || runtime->l2_arena == NULL ||
      team == NULL || activation_map_kind(test_case->activation, &kind) != 0) {
    return KBENCH_ERR_INVALID_ARGUMENT;
  }
  default_lut = kbench_activation_default_lut(kind);
  if (default_lut == NULL) return KBENCH_ERR_RUNTIME;

  memset(state, 0, sizeof(*state));
  l1.cursor = (uintptr_t)runtime->l1_arena;
  l1.end = l1.cursor + runtime->l1_arena_bytes;
  l2.cursor = (uintptr_t)runtime->l2_arena;
  l2.end = l2.cursor + runtime->l2_arena_bytes;

  state->job.lut.input = kbench_arena_take(
      &l1, (size_t)test_case->problem.n, alignment);
  state->job.lut.output = kbench_arena_take(
      &l1, (size_t)test_case->problem.n, alignment);
  state->job.lut.lut = kbench_arena_take(
      &l1, CF_QACTIVATION_LUT_ENTRIES, alignment);
  state->expected = kbench_arena_take(
      &l2, (size_t)test_case->problem.n, alignment);
  if (state->job.lut.input == NULL || state->job.lut.output == NULL ||
      state->job.lut.lut == NULL || state->expected == NULL) {
    return KBENCH_ERR_NO_MEMORY;
  }

  state->elements = test_case->problem.n;
  state->repetitions = activation_repetitions(
      state->elements, KBENCH_ACTIVATION_MAX_REPETITIONS);
  state->team = team;
  state->job.lut.elements = state->elements;
  state->job.lut.cores = test_case->cores;
  memcpy((uint8_t *)state->job.lut.lut, default_lut,
         CF_QACTIVATION_LUT_ENTRIES);
  activation_fill_input((uint8_t *)state->job.lut.input, state->elements,
                        test_case->seed);

  {
    cf_qactivation_lut_job_t oracle = state->job.lut;
    oracle.output = state->expected;
    cf_qactivation_lut_reference(&oracle);
  }
  state->expected_checksum =
      kbench_checksum_u8(state->expected, state->elements);
  state->team_arg = &state->job.lut;
  state->team_repetitions = state->repetitions;
  state->team_cores = (uint8_t)state->job.lut.cores;
  pi_cluster_task(&state->task, activation_cluster_entry, state);
  state->task.stack_size = KBENCH_ACTIVATION_MASTER_STACK_BYTES;
  state->task.slave_stack_size = KBENCH_ACTIVATION_SLAVE_STACK_BYTES;
  state->initialized = 1u;
  return KBENCH_OK;
}

static int activation_init_lut_scalar(void *state,
                                      const kbench_case_t *test_case,
                                      kbench_runtime_t *runtime) {
  return activation_init_lut_common(
      state, test_case, runtime, cf_qactivation_lut_scalar_team);
}

static int activation_init_lut_packed4(void *state,
                                       const kbench_case_t *test_case,
                                       kbench_runtime_t *runtime) {
  return activation_init_lut_common(
      state, test_case, runtime, cf_qactivation_lut_packed4_team);
}

static int activation_init_softmax_common(
    void *state_memory, const kbench_case_t *test_case, kbench_runtime_t *runtime,
    activation_team_t team) {
  activation_state_t *state = (activation_state_t *)state_memory;
  kbench_arena_t l1;
  kbench_arena_t l2;
  const size_t alignment = test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;

  if (state == NULL || runtime == NULL || runtime->cluster == NULL ||
      runtime->l1_arena == NULL || runtime->l2_arena == NULL ||
      team == NULL) {
    return KBENCH_ERR_INVALID_ARGUMENT;
  }
  memset(state, 0, sizeof(*state));
  l1.cursor = (uintptr_t)runtime->l1_arena;
  l1.end = l1.cursor + runtime->l1_arena_bytes;
  l2.cursor = (uintptr_t)runtime->l2_arena;
  l2.end = l2.cursor + runtime->l2_arena_bytes;

  state->job.softmax.input = kbench_arena_take(
      &l1, (size_t)test_case->problem.n, alignment);
  state->job.softmax.output = kbench_arena_take(
      &l1, (size_t)test_case->problem.n, alignment);
  state->job.softmax.exp_lut = kbench_arena_take(
      &l1, kbench_softmax_default_exp_lut_bytes(), alignment);
  state->expected = kbench_arena_take(
      &l2, (size_t)test_case->problem.n, alignment);
  if (state->job.softmax.input == NULL || state->job.softmax.output == NULL ||
      state->job.softmax.exp_lut == NULL || state->expected == NULL) {
    return KBENCH_ERR_NO_MEMORY;
  }

  state->elements = test_case->problem.n;
  /* One run-to-completion softmax row per submitted task is the deployment
   * contract. The harness repeats tasks for sampling instead of issuing many
   * reduction-barrier generations inside one cluster entry. */
  state->repetitions = 1u;
  state->team = team;
  state->softmax = 1u;
  state->job.softmax.rows = 1u;
  state->job.softmax.columns = (uint16_t)state->elements;
  state->job.softmax.cores = test_case->cores;
  state->job.softmax.input_signed = 1u;
  memcpy((uint16_t *)state->job.softmax.exp_lut,
         kbench_softmax_default_exp_lut(),
         kbench_softmax_default_exp_lut_bytes());
  activation_fill_input((uint8_t *)state->job.softmax.input, state->elements,
                        test_case->seed);
  if (state->elements > 1u) {
    ((uint8_t *)state->job.softmax.input)[0] = 127u;
    ((uint8_t *)state->job.softmax.input)[1] = 127u;
  }

  {
    cf_qsoftmax_job_t oracle = state->job.softmax;
    oracle.output = state->expected;
    cf_qsoftmax_reference(&oracle);
  }
  state->expected_checksum =
      kbench_checksum_u8(state->expected, state->elements);
  state->team_arg = &state->job.softmax;
  state->team_repetitions = state->repetitions;
  state->team_cores = (uint8_t)state->job.softmax.cores;
  pi_cluster_task(&state->task, activation_cluster_entry, state);
  state->task.stack_size = KBENCH_ACTIVATION_MASTER_STACK_BYTES;
  state->task.slave_stack_size = KBENCH_ACTIVATION_SLAVE_STACK_BYTES;
  state->initialized = 1u;
  return KBENCH_OK;
}

static int activation_init_softmax_scalar(void *state,
                                          const kbench_case_t *test_case,
                                          kbench_runtime_t *runtime) {
  return activation_init_softmax_common(
      state, test_case, runtime, cf_qsoftmax_scalar_team);
}

static int activation_begin(void *state_memory,
                            const kbench_case_t *test_case,
                            kbench_runtime_t *runtime, uint32_t iteration) {
  activation_state_t *state = (activation_state_t *)state_memory;
  (void)test_case;
  (void)runtime;
  (void)iteration;
  if (state == NULL || !state->initialized) return KBENCH_ERR_RUNTIME;
  if (state->softmax) {
    memset(state->job.softmax.output, 0xa5, state->elements);
  } else {
    memset(state->job.lut.output, 0xa5, state->elements);
  }
  memset(&state->counters, 0, sizeof(state->counters));
  return KBENCH_OK;
}

static int activation_run(void *state_memory, kbench_runtime_t *runtime) {
  activation_state_t *state = (activation_state_t *)state_memory;
  if (state == NULL || !state->initialized || runtime == NULL ||
      runtime->cluster == NULL) {
    return KBENCH_ERR_RUNTIME;
  }
  if (pi_cluster_send_task_to_cl(runtime->cluster, &state->task) != 0) {
    return KBENCH_ERR_RUNTIME;
  }
  return KBENCH_OK;
}

static int activation_check(void *state_memory,
                            const kbench_case_t *test_case,
                            kbench_runtime_t *runtime, kbench_check_t *check) {
  activation_state_t *state = (activation_state_t *)state_memory;
  const uint8_t *output;
  uint32_t index;
  (void)test_case;
  (void)runtime;
  if (state == NULL || check == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  output = state->softmax ? state->job.softmax.output : state->job.lut.output;
  memset(check, 0, sizeof(*check));
  check->operations = state->elements;
  check->executed_operations = state->elements;
  check->repetitions = state->repetitions;
  check->checksum = kbench_checksum_u8(output, state->elements);
  check->expected_checksum = state->expected_checksum;
  kbench_counters_reduce(&state->counters, state->team_cores, check);
  check->weight_row_aligned = (uint8_t)(1u);
  check->exact = 1u;
  for (index = 0u; index < state->elements; ++index) {
    if (output[index] != state->expected[index]) {
      check->exact = 0u;
      return KBENCH_ERR_INCORRECT;
    }
  }
  if (state->softmax) {
    uint32_t sum = 0u;
    for (index = 0u; index < state->elements; ++index) sum += output[index];
    if (sum != 255u) {
      check->exact = 0u;
      return KBENCH_ERR_INCORRECT;
    }
  }
  return KBENCH_OK;
}

static void activation_deinit(void *state_memory, kbench_runtime_t *runtime) {
  (void)runtime;
  if (state_memory != NULL) {
    memset(state_memory, 0, sizeof(activation_state_t));
  }
}

const kbench_kernel_t kbench_activation_lut_scalar = {
    .name = "activation_lut8",
    .variant = "scalar",
    .family = KBENCH_PROBLEM_ACTIVATION,
    .state_bytes = sizeof(activation_state_t),
    .state_alignment_bytes = 8u,
    .validate = activation_validate_lut,
    .plan = activation_plan_lut,
    .init = activation_init_lut_scalar,
    .begin = activation_begin,
    .run = activation_run,
    .check = activation_check,
    .deinit = activation_deinit,
};

const kbench_kernel_t kbench_activation_lut_packed4 = {
    .name = "activation_lut8",
    .variant = "packed4",
    .family = KBENCH_PROBLEM_ACTIVATION,
    .state_bytes = sizeof(activation_state_t),
    .state_alignment_bytes = 8u,
    .validate = activation_validate_lut,
    .plan = activation_plan_lut,
    .init = activation_init_lut_packed4,
    .begin = activation_begin,
    .run = activation_run,
    .check = activation_check,
    .deinit = activation_deinit,
};

const kbench_kernel_t kbench_softmax_s8_u8_scalar = {
    .name = "softmax_s8_u8",
    .variant = "scalar",
    .family = KBENCH_PROBLEM_ACTIVATION,
    .state_bytes = sizeof(activation_state_t),
    .state_alignment_bytes = 8u,
    .validate = activation_validate_softmax,
    .plan = activation_plan_softmax,
    .init = activation_init_softmax_scalar,
    .begin = activation_begin,
    .run = activation_run,
    .check = activation_check,
    .deinit = activation_deinit,
};
