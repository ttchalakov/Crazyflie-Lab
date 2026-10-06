#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "pmsis.h"

#include "cf_qadd.h"
#include "kernel_bench/fixture.h"
#include "kernel_bench/kernel.h"

#define KBENCH_ADD_TARGET_ELEMENTS (1u << 18)
#define KBENCH_ADD_MAX_REPETITIONS 128u
#define KBENCH_ADD_MASTER_STACK_BYTES 4096u
#define KBENCH_ADD_SLAVE_STACK_BYTES 2048u

/* Single-invocation team function; the adapter owns the fork. */
typedef void (*add_team_t)(void *argument);

typedef struct {
  cf_qadd_u8_job_t job;
  uint8_t *expected;
  struct pi_cluster_task task;
  kbench_counters_t counters;
  add_team_t team;
  void *team_arg;
  uint32_t team_repetitions;
  uint8_t team_cores;
  uint32_t elements;
  uint32_t expected_checksum;
  uint8_t initialized;
} add_state_t;

/*
 * One fork per invocation with the repetitions inside, so the counters bracket
 * the arithmetic only. FC elapsed time around the cluster task remains the
 * dispatch-inclusive number reported alongside it.
 */
static void add_worker(void *argument) {
  add_state_t *state = (add_state_t *)argument;
  uint32_t repetition;

  kbench_perf_begin();
  for (repetition = 0u; repetition < state->team_repetitions; ++repetition) {
    state->team(state->team_arg);
  }
  kbench_perf_end(&state->counters, (uint32_t)pi_core_id());
}

static void add_cluster_entry(void *argument) {
  add_state_t *state = (add_state_t *)argument;
  pi_cl_team_fork(state->team_cores, add_worker, state);
}

static int add_validate_common(const kbench_case_t *test_case) {
  const kbench_problem_t *problem;
  uint64_t elements;
  if (test_case == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  problem = &test_case->problem;
  elements = (uint64_t)problem->height * problem->width *
             problem->input_channels;
  if (problem->family != KBENCH_PROBLEM_ADDITION || problem->batch != 1u ||
      problem->height == 0u || problem->width == 0u ||
      problem->input_channels == 0u || problem->height > UINT16_MAX ||
      problem->width > UINT16_MAX || problem->input_channels > UINT16_MAX ||
      elements > UINT32_MAX || test_case->output_shift < 0 ||
      test_case->output_shift > 31 || test_case->element_bits != 8u ||
      test_case->memory_path != KBENCH_MEMORY_L2_L1 ||
      test_case->misalignment_offset_bytes != 0u ||
      test_case->base_alignment_bytes < 4u || test_case->cores == 0u ||
      test_case->cores > CF_QADD_MAX_CORES ||
      (test_case->cores & (test_case->cores - 1u)) != 0u) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  return KBENCH_OK;
}

static int add_validate_pulp(const kbench_case_t *test_case) {
  const int status = add_validate_common(test_case);
  if (status != KBENCH_OK) return status;
  return test_case->cores == 8u ? KBENCH_OK : KBENCH_ERR_UNSUPPORTED;
}

static int add_plan(const kbench_case_t *test_case, kbench_memory_plan_t *plan) {
  const size_t alignment = test_case != NULL &&
                                   test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;
  size_t elements;
  if (test_case == NULL || plan == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  if (add_validate_common(test_case) != KBENCH_OK) return KBENCH_ERR_UNSUPPORTED;
  elements = (size_t)test_case->problem.height * test_case->problem.width *
             test_case->problem.input_channels;
  memset(plan, 0, sizeof(*plan));
  plan->persistent_l1_bytes = 3u * elements + 3u * alignment;
  plan->persistent_l2_bytes = elements + alignment;
  plan->input_alignment_bytes = 4u;
  plan->output_alignment_bytes = 4u;
  return KBENCH_OK;
}

static uint32_t add_repetitions(uint32_t elements) {
  uint32_t repetitions =
      (KBENCH_ADD_TARGET_ELEMENTS + elements - 1u) / elements;
  if (repetitions == 0u) repetitions = 1u;
  if (repetitions > KBENCH_ADD_MAX_REPETITIONS) {
    repetitions = KBENCH_ADD_MAX_REPETITIONS;
  }
  return repetitions;
}

static int add_init_common(void *state_memory,
                           const kbench_case_t *test_case,
                           kbench_runtime_t *runtime,
                           add_team_t team) {
  add_state_t *state = (add_state_t *)state_memory;
  const kbench_problem_t *problem = &test_case->problem;
  const size_t alignment = test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;
  kbench_arena_t l1;
  kbench_arena_t l2;
  uint32_t random = test_case->seed;
  uint32_t index;

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
  state->elements = problem->height * problem->width *
                    problem->input_channels;
  state->job.input1 = kbench_arena_take(&l1, state->elements, alignment);
  state->job.input2 = kbench_arena_take(&l1, state->elements, alignment);
  state->job.output = kbench_arena_take(&l1, state->elements, alignment);
  state->expected = kbench_arena_take(&l2, state->elements, alignment);
  if (state->job.input1 == NULL || state->job.input2 == NULL ||
      state->job.output == NULL || state->expected == NULL) {
    return KBENCH_ERR_NO_MEMORY;
  }

  state->job.input1_multiplier = test_case->input1_multiplier;
  state->job.input2_multiplier = test_case->input2_multiplier;
  state->job.output_shift = (uint16_t)test_case->output_shift;
  state->job.height = (uint16_t)problem->height;
  state->job.width = (uint16_t)problem->width;
  state->job.channels = (uint16_t)problem->input_channels;
  state->job.cores = test_case->cores;
  state->team_repetitions =
      test_case->warmups == 0u && test_case->runs == 1u
          ? 1u
          : add_repetitions(state->elements);
  state->team = team;

  for (index = 0u; index < state->elements; ++index) {
    ((uint8_t *)state->job.input1)[index] = (uint8_t)kbench_lcg(&random);
    ((uint8_t *)state->job.input2)[index] = (uint8_t)kbench_lcg(&random);
  }
  {
    cf_qadd_u8_job_t oracle = state->job;
    oracle.output = state->expected;
    cf_qadd_u8_reference(&oracle);
  }
  state->expected_checksum = kbench_checksum_u8(state->expected, state->elements);
  state->team_arg = &state->job;
  state->team_cores = (uint8_t)state->job.cores;
  pi_cluster_task(&state->task, add_cluster_entry, state);
  state->task.stack_size = KBENCH_ADD_MASTER_STACK_BYTES;
  state->task.slave_stack_size = KBENCH_ADD_SLAVE_STACK_BYTES;
  state->initialized = 1u;
  return KBENCH_OK;
}

static int add_init_pulp(void *state, const kbench_case_t *test_case,
                         kbench_runtime_t *runtime) {
  return add_init_common(state, test_case, runtime,
                         cf_qadd_u8_pulp_nn_team);
}

static int add_init_packed4(void *state, const kbench_case_t *test_case,
                            kbench_runtime_t *runtime) {
  return add_init_common(state, test_case, runtime, cf_qadd_u8_packed4_team);
}

static int add_begin(void *state_memory, const kbench_case_t *test_case,
                     kbench_runtime_t *runtime, uint32_t iteration) {
  add_state_t *state = (add_state_t *)state_memory;
  (void)test_case;
  (void)runtime;
  (void)iteration;
  if (state == NULL || !state->initialized) return KBENCH_ERR_RUNTIME;
  memset(state->job.output, 0xa5, state->elements);
  memset(&state->counters, 0, sizeof(state->counters));
  return KBENCH_OK;
}

static int add_run(void *state_memory, kbench_runtime_t *runtime) {
  add_state_t *state = (add_state_t *)state_memory;
  if (state == NULL || !state->initialized || runtime == NULL ||
      runtime->cluster == NULL ||
      pi_cluster_send_task_to_cl(runtime->cluster, &state->task) != 0) {
    return KBENCH_ERR_RUNTIME;
  }
  return KBENCH_OK;
}

static int add_check(void *state_memory, const kbench_case_t *test_case,
                     kbench_runtime_t *runtime, kbench_check_t *check) {
  add_state_t *state = (add_state_t *)state_memory;
  uint32_t index;
  (void)test_case;
  (void)runtime;
  if (state == NULL || check == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  memset(check, 0, sizeof(*check));
  check->operations = (uint64_t)state->elements * 4u;
  check->executed_operations = check->operations;
  check->repetitions = state->team_repetitions;
  check->checksum = kbench_checksum_u8(state->job.output, state->elements);
  check->expected_checksum = state->expected_checksum;
  kbench_counters_reduce(&state->counters, state->team_cores, check);
  check->weight_row_aligned = (uint8_t)(1u);
  check->exact = 1u;
  for (index = 0u; index < state->elements; ++index) {
    if (state->job.output[index] != state->expected[index]) {
      check->exact = 0u;
      return KBENCH_ERR_INCORRECT;
    }
  }
  return KBENCH_OK;
}

static void add_deinit(void *state_memory, kbench_runtime_t *runtime) {
  (void)runtime;
  if (state_memory != NULL) memset(state_memory, 0, sizeof(add_state_t));
}

#define KBENCH_ADD_DESCRIPTOR(symbol, variant_name, validate_fn, init_fn)     \
  const kbench_kernel_t symbol = {                                             \
      .name = "add_u8_u8_u8",                                                \
      .variant = variant_name,                                                 \
      .family = KBENCH_PROBLEM_ADDITION,                                       \
      .state_bytes = sizeof(add_state_t),                                      \
      .state_alignment_bytes = 8u,                                             \
      .validate = validate_fn,                                                 \
      .plan = add_plan,                                                        \
      .init = init_fn,                                                         \
      .begin = add_begin,                                                      \
      .run = add_run,                                                          \
      .check = add_check,                                                      \
      .deinit = add_deinit,                                                    \
  }

KBENCH_ADD_DESCRIPTOR(kbench_add_u8_pulp_nn, "pulp_nn", add_validate_pulp,
                      add_init_pulp);
KBENCH_ADD_DESCRIPTOR(kbench_add_u8_packed4, "packed4_flattened",
                      add_validate_common, add_init_packed4);
