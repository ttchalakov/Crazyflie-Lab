#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "pmsis.h"

#include "cf_qbatchnorm.h"
#include "kernel_bench/fixture.h"
#include "kernel_bench/kernel.h"

#define KBENCH_BN_TARGET_ELEMENTS (1u << 16)
#define KBENCH_BN_MAX_REPETITIONS 8u
#define KBENCH_BN_MASTER_STACK_BYTES 4096u
#define KBENCH_BN_SLAVE_STACK_BYTES 2048u
#define KBENCH_BN_L1_BUDGET_BYTES 32764u

/* Single-invocation team function; the adapter owns the fork. */
typedef void (*bn_team_t)(void *argument);

typedef enum {
  BN_FIXED_SCALAR = 0,
  BN_FIXED_CHANNEL4 = 1,
  BN_UPDATE_CHANNEL = 2,
} bn_flavor_t;

typedef union {
  cf_qbatchnorm_fixed_job_t fixed;
  cf_qbatchnorm_update_job_t update;
} bn_job_t;

typedef struct {
  bn_job_t job;
  uint8_t *expected;
  int32_t *gamma_q16;
  int32_t *beta_q16;
  int32_t *initial_mean_q16;
  uint32_t *initial_variance_q16;
  int32_t *expected_mean_q16;
  uint32_t *expected_variance_q16;
  struct pi_cluster_task task;
  kbench_counters_t counters;
  bn_team_t team;
  void *team_arg;
  uint32_t team_repetitions;
  uint8_t team_cores;
  uint32_t elements;
  uint32_t expected_checksum;
  uint32_t repetitions;
  bn_flavor_t flavor;
  uint8_t initialized;
} bn_state_t;

/*
 * One fork per invocation with the repetitions inside, so the counters bracket
 * the arithmetic only. FC elapsed time around the cluster task remains the
 * dispatch-inclusive number reported alongside it.
 */
static void bn_worker(void *argument) {
  bn_state_t *state = (bn_state_t *)argument;
  uint32_t repetition;

  kbench_perf_begin();
  for (repetition = 0u; repetition < state->team_repetitions; ++repetition) {
    state->team(state->team_arg);
  }
  kbench_perf_end(&state->counters, (uint32_t)pi_core_id());
}

static void bn_cluster_entry(void *argument) {
  bn_state_t *state = (bn_state_t *)argument;
  pi_cl_team_fork(state->team_cores, bn_worker, state);
}

static int bn_validate_common(const kbench_case_t *test_case) {
  const kbench_problem_t *problem;
  uint64_t elements;
  if (test_case == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  problem = &test_case->problem;
  elements = (uint64_t)problem->height * problem->width *
             problem->input_channels;
  if (problem->family != KBENCH_PROBLEM_NORMALIZATION ||
      problem->batch != 1u || problem->height == 0u ||
      problem->width == 0u || problem->input_channels == 0u ||
      problem->input_channels != problem->output_channels ||
      problem->height > UINT16_MAX || problem->width > UINT16_MAX ||
      problem->input_channels > UINT16_MAX || elements > UINT32_MAX ||
      test_case->element_bits != 8u ||
      test_case->memory_path != KBENCH_MEMORY_L2_L1 ||
      test_case->epilogue != KBENCH_EPILOGUE_INT32 ||
      test_case->activation != KBENCH_ACTIVATION_NONE ||
      test_case->misalignment_offset_bytes != 0u ||
      test_case->base_alignment_bytes < 4u || test_case->cores == 0u ||
      test_case->cores > CF_QBATCHNORM_MAX_CORES ||
      (test_case->cores & (test_case->cores - 1u)) != 0u) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  return KBENCH_OK;
}

static int bn_validate_update(const kbench_case_t *test_case) {
  const int status = bn_validate_common(test_case);
  if (status != KBENCH_OK) return status;
  return (uint64_t)test_case->problem.height * test_case->problem.width > 1u
             ? KBENCH_OK
             : KBENCH_ERR_UNSUPPORTED;
}

static int bn_validate_channel4(const kbench_case_t *test_case) {
  const int status = bn_validate_common(test_case);
  if (status != KBENCH_OK) return status;
  return (test_case->problem.input_channels & 3u) == 0u
             ? KBENCH_OK
             : KBENCH_ERR_UNSUPPORTED;
}

static int bn_plan(const kbench_case_t *test_case, kbench_memory_plan_t *plan) {
  const size_t alignment = test_case != NULL &&
                                   test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;
  size_t elements;
  size_t channels;
  size_t l1_bytes;
  size_t l2_bytes;
  if (test_case == NULL || plan == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  if (bn_validate_common(test_case) != KBENCH_OK) return KBENCH_ERR_UNSUPPORTED;
  elements = (size_t)test_case->problem.height * test_case->problem.width *
             test_case->problem.input_channels;
  channels = test_case->problem.input_channels;
  l1_bytes = 2u * elements + 4u * channels * sizeof(int32_t) + 8u * alignment;
  l2_bytes = elements + 6u * channels * sizeof(int32_t) + 8u * alignment;
  if (l1_bytes > KBENCH_BN_L1_BUDGET_BYTES) return KBENCH_ERR_UNSUPPORTED;
  memset(plan, 0, sizeof(*plan));
  plan->persistent_l1_bytes = l1_bytes;
  plan->persistent_l2_bytes = l2_bytes;
  plan->input_alignment_bytes = 4u;
  plan->weight_alignment_bytes = 4u;
  plan->output_alignment_bytes = 4u;
  return KBENCH_OK;
}

static uint32_t bn_repetitions(uint32_t elements) {
  uint32_t repetitions = (KBENCH_BN_TARGET_ELEMENTS + elements - 1u) / elements;
  if (repetitions == 0u) repetitions = 1u;
  if (repetitions > KBENCH_BN_MAX_REPETITIONS) {
    repetitions = KBENCH_BN_MAX_REPETITIONS;
  }
  return repetitions;
}

static int bn_init_common(void *state_memory, const kbench_case_t *test_case,
                          kbench_runtime_t *runtime, bn_team_t team,
                          bn_flavor_t flavor) {
  bn_state_t *state = (bn_state_t *)state_memory;
  const kbench_problem_t *problem = &test_case->problem;
  const size_t alignment = test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;
  kbench_arena_t l1;
  kbench_arena_t l2;
  uint8_t *input;
  uint8_t *output;
  int32_t *multiplier;
  int32_t *bias;
  int32_t *running_mean_q16;
  uint32_t *running_variance_q16;
  uint32_t random = test_case->seed;
  uint32_t index;
  uint16_t channel;
  if (state == NULL || runtime == NULL || runtime->cluster == NULL ||
      runtime->l1_arena == NULL || runtime->l2_arena == NULL || team == NULL) {
    return KBENCH_ERR_INVALID_ARGUMENT;
  }
  memset(state, 0, sizeof(*state));
  l1.cursor = (uintptr_t)runtime->l1_arena;
  l1.end = l1.cursor + runtime->l1_arena_bytes;
  l2.cursor = (uintptr_t)runtime->l2_arena;
  l2.end = l2.cursor + runtime->l2_arena_bytes;
  state->elements = problem->height * problem->width * problem->input_channels;

  input = kbench_arena_take(&l1, state->elements, alignment);
  output = kbench_arena_take(&l1, state->elements, alignment);
  multiplier = kbench_arena_take(
      &l1, (size_t)problem->input_channels * sizeof(int32_t), alignment);
  bias = kbench_arena_take(
      &l1, (size_t)problem->input_channels * sizeof(int32_t), alignment);
  state->gamma_q16 = kbench_arena_take(
      &l1, (size_t)problem->input_channels * sizeof(int32_t), alignment);
  state->beta_q16 = kbench_arena_take(
      &l1, (size_t)problem->input_channels * sizeof(int32_t), alignment);

  state->expected = kbench_arena_take(&l2, state->elements, alignment);
  running_mean_q16 = kbench_arena_take(
      &l2, (size_t)problem->input_channels * sizeof(int32_t), alignment);
  running_variance_q16 = kbench_arena_take(
      &l2, (size_t)problem->input_channels * sizeof(uint32_t), alignment);
  state->initial_mean_q16 = kbench_arena_take(
      &l2, (size_t)problem->input_channels * sizeof(int32_t), alignment);
  state->initial_variance_q16 = kbench_arena_take(
      &l2, (size_t)problem->input_channels * sizeof(uint32_t), alignment);
  state->expected_mean_q16 = kbench_arena_take(
      &l2, (size_t)problem->input_channels * sizeof(int32_t), alignment);
  state->expected_variance_q16 = kbench_arena_take(
      &l2, (size_t)problem->input_channels * sizeof(uint32_t), alignment);
  if (input == NULL || output == NULL || multiplier == NULL || bias == NULL ||
      state->gamma_q16 == NULL || state->beta_q16 == NULL ||
      state->expected == NULL || running_mean_q16 == NULL ||
      running_variance_q16 == NULL || state->initial_mean_q16 == NULL ||
      state->initial_variance_q16 == NULL || state->expected_mean_q16 == NULL ||
      state->expected_variance_q16 == NULL) {
    return KBENCH_ERR_NO_MEMORY;
  }

  for (index = 0u; index < state->elements; ++index) {
    input[index] = (uint8_t)(32u + (kbench_lcg(&random) % 192u));
  }
  for (channel = 0u; channel < problem->input_channels; ++channel) {
    state->gamma_q16[channel] =
        (int32_t)(12u + kbench_lcg(&random) % 13u) << 16;
    state->beta_q16[channel] =
        ((int32_t)(kbench_lcg(&random) % 9u) - 4) * 65536;
    state->initial_mean_q16[channel] =
        (int32_t)(96u + kbench_lcg(&random) % 65u) << 16;
    state->initial_variance_q16[channel] =
        (uint32_t)(512u + kbench_lcg(&random) % 1025u) << 16;
  }
  memcpy(running_mean_q16, state->initial_mean_q16,
         (size_t)problem->input_channels * sizeof(int32_t));
  memcpy(running_variance_q16, state->initial_variance_q16,
         (size_t)problem->input_channels * sizeof(uint32_t));

  state->flavor = flavor;
  state->team = team;
  if (flavor == BN_UPDATE_CHANNEL) {
    cf_qbatchnorm_update_job_t oracle;
    state->repetitions = 1u;
    state->job.update.input = input;
    state->job.update.output = output;
    state->job.update.gamma_q16 = state->gamma_q16;
    state->job.update.beta_q16 = state->beta_q16;
    state->job.update.running_mean_q16 = running_mean_q16;
    state->job.update.running_variance_q16 = running_variance_q16;
    state->job.update.spatial_samples = problem->height * problem->width;
    state->job.update.channels = (uint16_t)problem->input_channels;
    state->job.update.epsilon_q16 = 64u;
    state->job.update.momentum_q16 = 6554u;
    state->job.update.output_zero_point = 128u;
    state->job.update.cores = test_case->cores;
    state->job.update.mode = CF_QBATCHNORM_UPDATE_RUNNING;
    memcpy(state->expected_mean_q16, state->initial_mean_q16,
           (size_t)problem->input_channels * sizeof(int32_t));
    memcpy(state->expected_variance_q16, state->initial_variance_q16,
           (size_t)problem->input_channels * sizeof(uint32_t));
    oracle = state->job.update;
    oracle.output = state->expected;
    oracle.running_mean_q16 = state->expected_mean_q16;
    oracle.running_variance_q16 = state->expected_variance_q16;
    cf_qbatchnorm_update_reference(&oracle);
    state->team_arg = &state->job.update;
    state->team_repetitions = state->repetitions;
    state->team_cores = (uint8_t)state->job.update.cores;
    pi_cluster_task(&state->task, bn_cluster_entry, state);
  } else {
    cf_qbatchnorm_update_job_t prepare;
    state->repetitions = bn_repetitions(state->elements);
    memset(&prepare, 0, sizeof(prepare));
    prepare.gamma_q16 = state->gamma_q16;
    prepare.beta_q16 = state->beta_q16;
    prepare.running_mean_q16 = running_mean_q16;
    prepare.running_variance_q16 = running_variance_q16;
    prepare.channels = (uint16_t)problem->input_channels;
    prepare.epsilon_q16 = 64u;
    prepare.output_zero_point = 128u;
    if (cf_qbatchnorm_prepare_fixed(&prepare, multiplier, bias, 16u) != 0) {
      return KBENCH_ERR_RUNTIME;
    }
    state->job.fixed.input = input;
    state->job.fixed.output = output;
    state->job.fixed.multiplier = multiplier;
    state->job.fixed.bias = bias;
    state->job.fixed.elements = state->elements;
    state->job.fixed.channels = (uint16_t)problem->input_channels;
    state->job.fixed.shift = 16u;
    state->job.fixed.cores = test_case->cores;
    {
      cf_qbatchnorm_fixed_job_t oracle = state->job.fixed;
      oracle.output = state->expected;
      cf_qbatchnorm_fixed_reference(&oracle);
    }
    state->team_arg = &state->job.fixed;
    state->team_repetitions = state->repetitions;
    state->team_cores = (uint8_t)state->job.fixed.cores;
    pi_cluster_task(&state->task, bn_cluster_entry, state);
  }
  state->expected_checksum = kbench_checksum_u8(state->expected, state->elements);
  state->task.stack_size = KBENCH_BN_MASTER_STACK_BYTES;
  state->task.slave_stack_size = KBENCH_BN_SLAVE_STACK_BYTES;
  state->initialized = 1u;
  return KBENCH_OK;
}

static int bn_init_fixed_scalar(void *state, const kbench_case_t *test_case,
                                kbench_runtime_t *runtime) {
  return bn_init_common(state, test_case, runtime,
                        cf_qbatchnorm_fixed_scalar_team, BN_FIXED_SCALAR);
}

static int bn_init_fixed_channel4(void *state, const kbench_case_t *test_case,
                                  kbench_runtime_t *runtime) {
  return bn_init_common(state, test_case, runtime,
                        cf_qbatchnorm_fixed_channel4_team,
                        BN_FIXED_CHANNEL4);
}

static int bn_init_update(void *state, const kbench_case_t *test_case,
                          kbench_runtime_t *runtime) {
  return bn_init_common(state, test_case, runtime,
                        cf_qbatchnorm_update_channel_team,
                        BN_UPDATE_CHANNEL);
}

static int bn_begin(void *state_memory, const kbench_case_t *test_case,
                    kbench_runtime_t *runtime, uint32_t iteration) {
  bn_state_t *state = (bn_state_t *)state_memory;
  (void)test_case;
  (void)runtime;
  (void)iteration;
  if (state == NULL || !state->initialized) return KBENCH_ERR_RUNTIME;
  if (state->flavor == BN_UPDATE_CHANNEL) {
    memcpy(state->job.update.running_mean_q16, state->initial_mean_q16,
           (size_t)state->job.update.channels * sizeof(int32_t));
    memcpy(state->job.update.running_variance_q16,
           state->initial_variance_q16,
           (size_t)state->job.update.channels * sizeof(uint32_t));
    memset(state->job.update.output, 0xa5, state->elements);
  } else {
    memset(state->job.fixed.output, 0xa5, state->elements);
  }
  memset(&state->counters, 0, sizeof(state->counters));
  return KBENCH_OK;
}

static int bn_run(void *state_memory, kbench_runtime_t *runtime) {
  bn_state_t *state = (bn_state_t *)state_memory;
  if (state == NULL || !state->initialized || runtime == NULL ||
      runtime->cluster == NULL) {
    return KBENCH_ERR_RUNTIME;
  }
  return pi_cluster_send_task_to_cl(runtime->cluster, &state->task) == 0
             ? KBENCH_OK
             : KBENCH_ERR_RUNTIME;
}

static int bn_check(void *state_memory, const kbench_case_t *test_case,
                    kbench_runtime_t *runtime, kbench_check_t *check) {
  bn_state_t *state = (bn_state_t *)state_memory;
  const uint8_t *output;
  uint32_t index;
  (void)test_case;
  (void)runtime;
  if (state == NULL || check == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  output = state->flavor == BN_UPDATE_CHANNEL ? state->job.update.output
                                              : state->job.fixed.output;
  memset(check, 0, sizeof(*check));
  check->operations = (uint64_t)state->elements *
                      (state->flavor == BN_UPDATE_CHANNEL ? 4u : 2u);
  check->executed_operations = check->operations;
  check->repetitions = state->repetitions;
  check->checksum = kbench_checksum_u8(output, state->elements);
  check->expected_checksum = state->expected_checksum;
  kbench_counters_reduce(&state->counters, state->team_cores, check);
  check->weight_row_aligned = (uint8_t)(1u);
  check->exact = 1u;
  for (index = 0u; index < state->elements; ++index) {
    if (output[index] != state->expected[index]) goto incorrect;
  }
  if (state->flavor == BN_UPDATE_CHANNEL) {
    uint16_t channel;
    for (channel = 0u; channel < state->job.update.channels; ++channel) {
      if (state->job.update.running_mean_q16[channel] !=
              state->expected_mean_q16[channel] ||
          state->job.update.running_variance_q16[channel] !=
              state->expected_variance_q16[channel]) {
        goto incorrect;
      }
    }
  }
  return KBENCH_OK;

incorrect:
  check->exact = 0u;
  return KBENCH_ERR_INCORRECT;
}

static void bn_deinit(void *state_memory, kbench_runtime_t *runtime) {
  (void)runtime;
  if (state_memory != NULL) memset(state_memory, 0, sizeof(bn_state_t));
}

#define KBENCH_BN_DESCRIPTOR(symbol, kernel_name, variant_name, validate_fn,    \
                             init_fn)                                           \
  const kbench_kernel_t symbol = {                                              \
      .name = kernel_name,                                                      \
      .variant = variant_name,                                                  \
      .family = KBENCH_PROBLEM_NORMALIZATION,                                  \
      .state_bytes = sizeof(bn_state_t),                                        \
      .state_alignment_bytes = 8u,                                              \
      .validate = validate_fn,                                                  \
      .plan = bn_plan,                                                          \
      .init = init_fn,                                                          \
      .begin = bn_begin,                                                        \
      .run = bn_run,                                                            \
      .check = bn_check,                                                        \
      .deinit = bn_deinit,                                                      \
  }

KBENCH_BN_DESCRIPTOR(kbench_batchnorm_u8_fixed_scalar, "batchnorm_fixed_u8",
                     "scalar", bn_validate_common, bn_init_fixed_scalar);
KBENCH_BN_DESCRIPTOR(kbench_batchnorm_u8_fixed_channel4, "batchnorm_fixed_u8",
                     "channel4", bn_validate_channel4,
                     bn_init_fixed_channel4);
KBENCH_BN_DESCRIPTOR(kbench_batchnorm_u8_update_channel, "batchnorm_update_u8",
                     "channel", bn_validate_update, bn_init_update);
