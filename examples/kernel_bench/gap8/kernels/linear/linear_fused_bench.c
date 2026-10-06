#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "pmsis.h"

#include "cf_qlinear.h"
#include "kernel_bench/fixture.h"
#include "kernel_bench/kernel.h"

#define KBENCH_FUSED_TARGET_MACS (1u << 18)
#define KBENCH_FUSED_MAX_REPETITIONS 64u
/* The instrumented worker adds a perf frame and an indirect call above the
 * kernel, so these match conv/depthwise/matmul rather than the 2048/1024 the
 * uninstrumented bench used to get away with. */
#define KBENCH_FUSED_MASTER_STACK_BYTES 4096u
#define KBENCH_FUSED_SLAVE_STACK_BYTES 2048u

/* Single-invocation team function; the adapter owns the fork. */
typedef void (*fused_team_t)(void *argument);

typedef struct {
  cf_qlinear_u8s8_u8_job_t job;
  uint8_t *expected;
  struct pi_cluster_task task;
  kbench_counters_t counters;
  fused_team_t team;
  void *team_arg;
  uint32_t team_repetitions;
  uint8_t team_cores;
  uint32_t checksum;
  uint8_t initialized;
} fused_linear_state_t;

/*
 * One fork per invocation with the repetitions inside, so the counters bracket
 * the arithmetic only. FC elapsed time around the cluster task remains the
 * dispatch-inclusive number reported alongside it.
 */
static void fused_worker(void *argument) {
  fused_linear_state_t *state = (fused_linear_state_t *)argument;
  uint32_t repetition;

  kbench_perf_begin();
  for (repetition = 0u; repetition < state->team_repetitions; ++repetition) {
    state->team(state->team_arg);
  }
  kbench_perf_end(&state->counters, (uint32_t)pi_core_id());
}

static void fused_cluster_entry(void *argument) {
  fused_linear_state_t *state = (fused_linear_state_t *)argument;
  pi_cl_team_fork(state->team_cores, fused_worker, state);
}

/* Reserve three deterministic rows for quantization boundary coverage in every
 * timed fixture: negative -> 0, above range -> 255, and an interior value. */
static void fused_seed_quantization_edges(fused_linear_state_t *state) {
  const uint32_t shift = (uint32_t)state->job.output_shift;
  const uint32_t multiplier = state->job.output_multiplier;
  int32_t *bias = (int32_t *)state->job.bias;
  int32_t *batch_norm_multiplier =
      (int32_t *)state->job.batch_norm_multiplier;
  int32_t *batch_norm_bias = (int32_t *)state->job.batch_norm_bias;
  uint32_t row;

  if (state->job.output_dim < 3u || shift > 20u ||
      (state->job.epilogue == CF_QLINEAR_U8_REQUANT_RELU &&
       multiplier == 0u)) {
    return;
  }
  for (row = 0u; row < 3u; ++row) {
    memset((int8_t *)state->job.weights +
               (size_t)row * state->job.weight_row_stride_bytes,
           0, state->job.packed_input_dim);
    batch_norm_multiplier[row] = 1;
    batch_norm_bias[row] = 0;
  }

  bias[0] = -1;
  if (state->job.epilogue == CF_QLINEAR_U8_REQUANT_RELU) {
    bias[1] = (int32_t)(((300u << shift) + multiplier - 1u) / multiplier);
    bias[2] = (int32_t)(((42u << shift) + multiplier - 1u) / multiplier);
  } else {
    bias[1] = (int32_t)(300u << shift);
    bias[2] = (int32_t)(42u << shift);
  }
}

static int fused_map_epilogue(kbench_epilogue_t source,
                              cf_qlinear_u8_epilogue_t *destination) {
  if (destination == NULL) {
    return -1;
  }
  switch (source) {
    case KBENCH_EPILOGUE_SHIFT_CLIP_U8:
      *destination = CF_QLINEAR_U8_SHIFT_CLIP;
      return 0;
    case KBENCH_EPILOGUE_REQUANT_RELU_U8:
      *destination = CF_QLINEAR_U8_REQUANT_RELU;
      return 0;
    case KBENCH_EPILOGUE_BATCHNORM_RELU_U8:
      *destination = CF_QLINEAR_U8_BATCHNORM_RELU;
      return 0;
    default:
      return -1;
  }
}

static int fused_validate_common(const kbench_case_t *test_case) {
  const kbench_problem_t *problem;
  cf_qlinear_u8_epilogue_t epilogue;

  if (test_case == NULL) {
    return KBENCH_ERR_INVALID_ARGUMENT;
  }
  problem = &test_case->problem;
  if (problem->family != KBENCH_PROBLEM_LINEAR || problem->batch != 1u ||
      problem->m != 1u || problem->k == 0u || problem->n == 0u ||
      problem->k > UINT16_MAX - 3u || problem->n > UINT16_MAX ||
      fused_map_epilogue(test_case->epilogue, &epilogue) != 0 ||
      test_case->output_shift < 0 || test_case->output_shift > 31) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  if (test_case->element_bits != 8u ||
      test_case->memory_path != KBENCH_MEMORY_L2_DIRECT ||
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

static int fused_validate_pulp(const kbench_case_t *test_case) {
  const int status = fused_validate_common(test_case);
  if (status != KBENCH_OK) {
    return status;
  }
  return test_case->cores == 8u ? KBENCH_OK : KBENCH_ERR_UNSUPPORTED;
}

static int fused_validate_register4(const kbench_case_t *test_case) {
  return fused_validate_common(test_case);
}

static int fused_plan(const kbench_case_t *test_case,
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
  if (fused_validate_common(test_case) != KBENCH_OK) {
    return KBENCH_ERR_UNSUPPORTED;
  }

  bytes = (size_t)test_case->problem.k; /* Preparation scratch. */
  bytes += packed_k;
  bytes += (size_t)test_case->problem.n * packed_k;
  bytes += 3u * (size_t)test_case->problem.n * sizeof(int32_t);
  bytes += 2u * (size_t)test_case->problem.n * sizeof(uint8_t);
  bytes += 8u * alignment;

  memset(plan, 0, sizeof(*plan));
  plan->persistent_l2_bytes = bytes;
  plan->input_alignment_bytes = 4u;
  plan->weight_alignment_bytes = 4u;
  plan->output_alignment_bytes = 4u;
  return KBENCH_OK;
}

static uint32_t fused_repetitions(const kbench_case_t *test_case) {
  const uint32_t macs = test_case->problem.n * test_case->problem.k;
  uint32_t repetitions = (KBENCH_FUSED_TARGET_MACS + macs - 1u) / macs;

  if (repetitions == 0u) repetitions = 1u;
  if (repetitions > KBENCH_FUSED_MAX_REPETITIONS) {
    repetitions = KBENCH_FUSED_MAX_REPETITIONS;
  }
  return repetitions;
}

static int fused_init_common(void *state_memory,
                             const kbench_case_t *test_case,
                             kbench_runtime_t *runtime,
                             fused_team_t team) {
  fused_linear_state_t *state = (fused_linear_state_t *)state_memory;
  kbench_arena_t arena;
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
  arena.cursor = (uintptr_t)runtime->l2_arena;
  arena.end = arena.cursor + runtime->l2_arena_bytes;

  prepare_scratch = kbench_arena_take(&arena, logical_k, alignment);
  state->job.input = kbench_arena_take(&arena, packed_k, alignment);
  state->job.weights = kbench_arena_take(
      &arena, (size_t)test_case->problem.n * packed_k, alignment);
  state->job.bias = kbench_arena_take(
      &arena, (size_t)test_case->problem.n * sizeof(int32_t), alignment);
  state->job.batch_norm_multiplier = kbench_arena_take(
      &arena, (size_t)test_case->problem.n * sizeof(int32_t), alignment);
  state->job.batch_norm_bias = kbench_arena_take(
      &arena, (size_t)test_case->problem.n * sizeof(int32_t), alignment);
  state->job.output = kbench_arena_take(
      &arena, (size_t)test_case->problem.n, alignment);
  state->expected = kbench_arena_take(
      &arena, (size_t)test_case->problem.n, alignment);
  if (prepare_scratch == NULL || state->job.input == NULL ||
      state->job.weights == NULL || state->job.bias == NULL ||
      state->job.batch_norm_multiplier == NULL ||
      state->job.batch_norm_bias == NULL || state->job.output == NULL ||
      state->expected == NULL) {
    return KBENCH_ERR_NO_MEMORY;
  }

  state->job.input_dim = logical_k;
  state->job.packed_input_dim = packed_k;
  state->job.weight_row_stride_bytes = packed_k;
  state->job.output_dim = (uint16_t)test_case->problem.n;
  state->job.output_multiplier = test_case->output_multiplier;
  state->job.output_shift = test_case->output_shift;
  if (fused_map_epilogue(test_case->epilogue, &state->job.epilogue) != 0) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  state->job.cores = test_case->cores;
  state->team_repetitions = fused_repetitions(test_case);
  state->team = team;

  for (index = 0u; index < logical_k; ++index) {
    prepare_scratch[index] = (uint8_t)(kbench_lcg(&random) >> 24);
  }
  if (cf_qlinear_u8s8_i32_prepare_input((uint8_t *)state->job.input, packed_k,
                                       prepare_scratch, logical_k) != 0) {
    return KBENCH_ERR_RUNTIME;
  }
  for (index = 0u; index < test_case->problem.n; ++index) {
    uint32_t input_index;
    int8_t *prepared_row =
        (int8_t *)state->job.weights + (size_t)index * packed_k;

    for (input_index = 0u; input_index < logical_k; ++input_index) {
      ((int8_t *)prepare_scratch)[input_index] =
          (int8_t)(kbench_lcg(&random) % 15u) - 7;
    }
    if (cf_qlinear_u8s8_i32_prepare_weights(
            prepared_row, packed_k, (const int8_t *)prepare_scratch, logical_k,
            logical_k, 1u) != 0) {
      return KBENCH_ERR_RUNTIME;
    }
  }
  for (index = 0u; index < test_case->problem.n; ++index) {
    ((int32_t *)state->job.bias)[index] =
        (int32_t)(kbench_lcg(&random) % 2049u) - 1024;
    ((int32_t *)state->job.batch_norm_multiplier)[index] =
        (int32_t)(kbench_lcg(&random) % 4u) + 1;
    ((int32_t *)state->job.batch_norm_bias)[index] =
        (int32_t)(kbench_lcg(&random) % 8193u) - 4096;
  }
  fused_seed_quantization_edges(state);

  {
    cf_qlinear_u8s8_u8_job_t oracle = state->job;
    oracle.output = state->expected;
    cf_qlinear_u8s8_u8_reference(&oracle);
  }
  if (test_case->problem.n >= 3u && test_case->output_shift <= 20 &&
      (state->job.epilogue != CF_QLINEAR_U8_REQUANT_RELU ||
       state->job.output_multiplier != 0u) &&
      (state->expected[0] != 0u || state->expected[1] != UINT8_MAX ||
       state->expected[2] != 42u)) {
    return KBENCH_ERR_INCORRECT;
  }
  state->checksum = kbench_checksum_u8(state->expected, test_case->problem.n);

  state->team_arg = &state->job;
  state->team_cores = (uint8_t)state->job.cores;
  pi_cluster_task(&state->task, fused_cluster_entry, state);
  state->task.stack_size = KBENCH_FUSED_MASTER_STACK_BYTES;
  state->task.slave_stack_size = KBENCH_FUSED_SLAVE_STACK_BYTES;
  state->initialized = 1u;
  return KBENCH_OK;
}

static int fused_init_pulp(void *state, const kbench_case_t *test_case,
                           kbench_runtime_t *runtime) {
  return fused_init_common(state, test_case, runtime,
                           cf_qlinear_u8s8_u8_pulp_nn_team);
}

static int fused_init_register4(void *state, const kbench_case_t *test_case,
                                kbench_runtime_t *runtime) {
  return fused_init_common(state, test_case, runtime,
                           cf_qlinear_u8s8_u8_register4_team);
}

static int fused_begin(void *state_memory, const kbench_case_t *test_case,
                       kbench_runtime_t *runtime, uint32_t iteration) {
  fused_linear_state_t *state = (fused_linear_state_t *)state_memory;
  (void)runtime;
  (void)iteration;

  if (state == NULL || !state->initialized) {
    return KBENCH_ERR_RUNTIME;
  }
  memset(state->job.output, 0xa5, (size_t)test_case->problem.n);
  memset(&state->counters, 0, sizeof(state->counters));
  return KBENCH_OK;
}

static int fused_run(void *state_memory, kbench_runtime_t *runtime) {
  fused_linear_state_t *state = (fused_linear_state_t *)state_memory;

  if (state == NULL || !state->initialized || runtime == NULL ||
      runtime->cluster == NULL) {
    return KBENCH_ERR_RUNTIME;
  }
  if (pi_cluster_send_task_to_cl(runtime->cluster, &state->task) != 0) {
    return KBENCH_ERR_RUNTIME;
  }
  return KBENCH_OK;
}

static int fused_check(void *state_memory, const kbench_case_t *test_case,
                       kbench_runtime_t *runtime, kbench_check_t *check) {
  fused_linear_state_t *state = (fused_linear_state_t *)state_memory;
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
  check->repetitions = state->team_repetitions;
  check->checksum = kbench_checksum_u8(state->job.output, test_case->problem.n);
  check->expected_checksum = state->checksum;
  kbench_counters_reduce(&state->counters, state->team_cores, check);
  check->weight_row_aligned = (uint8_t)((state->job.weight_row_stride_bytes & 3u) == 0u);
  check->exact = 1u;
  for (index = 0u; index < test_case->problem.n; ++index) {
    if (state->job.output[index] != state->expected[index]) {
      check->exact = 0u;
      return KBENCH_ERR_INCORRECT;
    }
  }
  return KBENCH_OK;
}

static void fused_deinit(void *state_memory, kbench_runtime_t *runtime) {
  (void)runtime;
  if (state_memory != NULL) {
    memset(state_memory, 0, sizeof(fused_linear_state_t));
  }
}

const kbench_kernel_t kbench_linear_u8s8_u8_pulp_nn = {
    .name = "linear_u8s8_u8",
    .variant = "pulp_nn",
    .family = KBENCH_PROBLEM_LINEAR,
    .state_bytes = sizeof(fused_linear_state_t),
    .state_alignment_bytes = 8u,
    .validate = fused_validate_pulp,
    .plan = fused_plan,
    .init = fused_init_pulp,
    .begin = fused_begin,
    .run = fused_run,
    .check = fused_check,
    .deinit = fused_deinit,
};

const kbench_kernel_t kbench_linear_u8s8_u8_register4 = {
    .name = "linear_u8s8_u8",
    .variant = "register4_fused",
    .family = KBENCH_PROBLEM_LINEAR,
    .state_bytes = sizeof(fused_linear_state_t),
    .state_alignment_bytes = 8u,
    .validate = fused_validate_register4,
    .plan = fused_plan,
    .init = fused_init_register4,
    .begin = fused_begin,
    .run = fused_run,
    .check = fused_check,
    .deinit = fused_deinit,
};
