#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "pmsis.h"

#include "cf_qdepthwise.h"
#include "cf_qdepthwise_internal.h"
#include "kernel_bench/fixture.h"
#include "kernel_bench/kernel.h"

#define KBENCH_DW_TARGET_MACS (1u << 18)
#define KBENCH_DW_MAX_REPETITIONS 32u
#define KBENCH_DW_MASTER_STACK_BYTES 4096u
#define KBENCH_DW_SLAVE_STACK_BYTES 2048u

/* Single-invocation team function; the adapter owns the fork. */
typedef void (*dw_team_t)(void *argument);

typedef struct {
  cf_qdepthwise_u8s8_u8_job_t job;
  uint8_t *expected;
  struct pi_cluster_task task;
  kbench_counters_t counters;
  dw_team_t team;
  void *team_arg;
  uint32_t team_repetitions;
  uint8_t team_cores;
  uint32_t checksum;
  uint32_t output_elements;
  uint32_t macs;
  uint8_t initialized;
} dw_state_t;

/*
 * One fork per invocation with the repetitions inside, so the counters bracket
 * the arithmetic only. FC elapsed time around the cluster task remains the
 * dispatch-inclusive number reported alongside it.
 */
static void dw_worker(void *argument) {
  dw_state_t *state = (dw_state_t *)argument;
  uint32_t repetition;

  kbench_perf_begin();
  for (repetition = 0u; repetition < state->team_repetitions; ++repetition) {
    state->team(state->team_arg);
  }
  kbench_perf_end(&state->counters, (uint32_t)pi_core_id());
}

static void dw_cluster_entry(void *argument) {
  dw_state_t *state = (dw_state_t *)argument;
  pi_cl_team_fork(state->team_cores, dw_worker, state);
}

static int dw_output_dimensions(const kbench_problem_t *problem,
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
  if (padded_height < problem->kernel_h || padded_width < problem->kernel_w) {
    return -1;
  }
  *output_height =
      (padded_height - problem->kernel_h) / problem->stride_h + 1u;
  *output_width =
      (padded_width - problem->kernel_w) / problem->stride_w + 1u;
  return *output_height != 0u && *output_width != 0u ? 0 : -1;
}

static int dw_map_epilogue(kbench_epilogue_t source,
                           cf_qdepthwise_u8_epilogue_t *destination) {
  if (destination == NULL) return -1;
  switch (source) {
    case KBENCH_EPILOGUE_SHIFT_CLIP_U8:
      *destination = CF_QDEPTHWISE_U8_SHIFT_CLIP;
      return 0;
    case KBENCH_EPILOGUE_REQUANT_RELU_U8:
      *destination = CF_QDEPTHWISE_U8_REQUANT_RELU;
      return 0;
    case KBENCH_EPILOGUE_BATCHNORM_RELU_U8:
      *destination = CF_QDEPTHWISE_U8_BATCHNORM_RELU;
      return 0;
    default:
      return -1;
  }
}

static int dw_validate_common(const kbench_case_t *test_case) {
  const kbench_problem_t *problem;
  cf_qdepthwise_u8_epilogue_t epilogue;
  uint32_t output_height;
  uint32_t output_width;
  if (test_case == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  problem = &test_case->problem;
  if (problem->family != KBENCH_PROBLEM_DEPTHWISE_CONV2D ||
      problem->batch != 1u || problem->input_channels == 0u ||
      problem->input_channels != problem->output_channels ||
      problem->kernel_h != 3u || problem->kernel_w != 3u ||
      problem->stride_h == 0u || problem->stride_h > 2u ||
      problem->stride_w == 0u || problem->stride_w > 2u ||
      problem->height > UINT16_MAX || problem->width > UINT16_MAX ||
      problem->input_channels > UINT16_MAX ||
      dw_output_dimensions(problem, &output_height, &output_width) != 0 ||
      output_height > UINT16_MAX || output_width > UINT16_MAX ||
      dw_map_epilogue(test_case->epilogue, &epilogue) != 0 ||
      test_case->output_shift < 0 || test_case->output_shift > 31) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  if (test_case->element_bits != 8u ||
      (test_case->memory_path != KBENCH_MEMORY_L2_DIRECT &&
       test_case->memory_path != KBENCH_MEMORY_L2_L1) ||
      test_case->misalignment_offset_bytes != 0u ||
      test_case->base_alignment_bytes < 4u || test_case->cores == 0u ||
      test_case->cores > CF_QDEPTHWISE_MAX_CORES ||
      (test_case->cores & (test_case->cores - 1u)) != 0u) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  return KBENCH_OK;
}

static int dw_validate_pulp(const kbench_case_t *test_case) {
  const int status = dw_validate_common(test_case);
  if (status != KBENCH_OK) return status;
#if CF_QDEPTHWISE_DORY_INT32_BIAS
  /* The pinned kernel reads a one-byte bias per channel and knows nothing of
   * the define, so with DORY's int32 bias one buffer cannot serve both it and
   * our kernels. The DORY overlay replaces it in that configuration, so it
   * declines the case instead of reporting `incorrect`. */
  return KBENCH_ERR_UNSUPPORTED;
#else
  return test_case->cores == 8u ? KBENCH_OK : KBENCH_ERR_UNSUPPORTED;
#endif
}

static int dw_plan(const kbench_case_t *test_case, kbench_memory_plan_t *plan) {
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
  size_t scratch_bytes;
  size_t weight_buffer_bytes;
  size_t bytes;
  if (test_case == NULL || plan == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  if (dw_validate_common(test_case) != KBENCH_OK) return KBENCH_ERR_UNSUPPORTED;
  problem = &test_case->problem;
  if (dw_output_dimensions(problem, &output_height, &output_width) != 0) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  input_bytes =
      (size_t)problem->height * problem->width * problem->input_channels;
  weight_bytes = (size_t)problem->input_channels * 9u;
  output_bytes =
      (size_t)output_height * output_width * problem->output_channels;
  scratch_bytes = cf_qdepthwise_pulp_nn_scratch_bytes(
      (uint16_t)problem->height, 3u, problem->padding_top,
      problem->padding_bottom, test_case->cores);
  weight_buffer_bytes = 4u * (size_t)problem->output_channels;
  /* Operands the kernel reads and writes, excluding the host-side golden. */
  bytes = input_bytes + weight_bytes + output_bytes + scratch_bytes +
          weight_buffer_bytes +
          (size_t)problem->output_channels * CF_QDEPTHWISE_BIAS_STRIDE +
          2u * (size_t)problem->output_channels * sizeof(int32_t) +
          8u * alignment;

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

static uint32_t dw_repetitions(uint32_t macs) {
  uint32_t repetitions = (KBENCH_DW_TARGET_MACS + macs - 1u) / macs;
  if (repetitions == 0u) repetitions = 1u;
  if (repetitions > KBENCH_DW_MAX_REPETITIONS) {
    repetitions = KBENCH_DW_MAX_REPETITIONS;
  }
  return repetitions;
}

static int dw_init_common(void *state_memory, const kbench_case_t *test_case,
                          kbench_runtime_t *runtime,
                          dw_team_t team) {
  dw_state_t *state = (dw_state_t *)state_memory;
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
  uint32_t scratch_bytes;
  uint32_t weight_buffer_bytes;
  uint32_t index;
  if (state == NULL || runtime == NULL || runtime->cluster == NULL ||
      runtime->l2_arena == NULL || team == NULL ||
      dw_output_dimensions(problem, &output_height, &output_width) != 0) {
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
  weight_bytes = problem->input_channels * 9u;
  state->output_elements =
      output_height * output_width * problem->output_channels;
  scratch_bytes = cf_qdepthwise_pulp_nn_scratch_bytes(
      (uint16_t)problem->height, 3u, problem->padding_top,
      problem->padding_bottom, test_case->cores);
  weight_buffer_bytes = 4u * problem->output_channels;
  state->macs = state->output_elements * 9u;

  state->job.input = kbench_arena_take(ops, input_bytes, alignment);
  state->job.weights = kbench_arena_take(ops, weight_bytes, alignment);
  /* CF_QDEPTHWISE_BIAS_STRIDE, not sizeof(int8_t): with DORY's int32 bias the
   * kernel reads four bytes per channel through cf_qdepthwise_bias_at(), so a
   * one-byte-per-channel buffer would be read three bytes past its end. */
  state->job.bias = kbench_arena_take(
      ops, (size_t)problem->output_channels * CF_QDEPTHWISE_BIAS_STRIDE,
      alignment);
  state->job.batch_norm_multiplier = kbench_arena_take(
      ops, (size_t)problem->output_channels * sizeof(int32_t), alignment);
  state->job.batch_norm_bias = kbench_arena_take(
      ops, (size_t)problem->output_channels * sizeof(int32_t), alignment);
  state->job.output =
      kbench_arena_take(ops, state->output_elements, alignment);
  state->expected =
      kbench_arena_take(host, state->output_elements, alignment);
  state->job.scratch = kbench_arena_take(ops, scratch_bytes, alignment);
  state->job.weight_buffer =
      kbench_arena_take(ops, weight_buffer_bytes, alignment);
  if (state->job.input == NULL || state->job.weights == NULL ||
      state->job.bias == NULL ||
      state->job.batch_norm_multiplier == NULL ||
      state->job.batch_norm_bias == NULL || state->job.output == NULL ||
      state->expected == NULL || state->job.scratch == NULL ||
      state->job.weight_buffer == NULL) {
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
  state->team_repetitions = dw_repetitions(state->macs);
  state->team = team;
  if (dw_map_epilogue(test_case->epilogue, &state->job.epilogue) != 0) {
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
    const int32_t value = (int32_t)(kbench_lcg(&random) % 9u) - 4;
    /* Written at the same stride the kernel and the oracle will read. */
#if CF_QDEPTHWISE_DORY_INT32_BIAS
    ((int32_t *)state->job.bias)[index] = value;
#else
    ((int8_t *)state->job.bias)[index] = (int8_t)value;
#endif
  }
  for (index = 0u; index < problem->output_channels; ++index) {
    ((int32_t *)state->job.batch_norm_multiplier)[index] =
        (int32_t)(kbench_lcg(&random) % 4u) + 1;
    ((int32_t *)state->job.batch_norm_bias)[index] =
        (int32_t)(kbench_lcg(&random) % 1025u) - 512;
  }
  memset(state->job.scratch, 0, scratch_bytes);
  memset(state->job.weight_buffer, 0, weight_buffer_bytes);
  {
    cf_qdepthwise_u8s8_u8_job_t oracle = state->job;
    oracle.output = state->expected;
    cf_qdepthwise_u8s8_u8_reference(&oracle);
  }
  state->checksum = kbench_checksum_u8(state->expected, state->output_elements);
  state->team_arg = &state->job;
  state->team_cores = (uint8_t)state->job.cores;
  pi_cluster_task(&state->task, dw_cluster_entry, state);
  state->task.stack_size = KBENCH_DW_MASTER_STACK_BYTES;
  state->task.slave_stack_size = KBENCH_DW_SLAVE_STACK_BYTES;
  state->initialized = 1u;
  return KBENCH_OK;
}

static int dw_init_pulp(void *state, const kbench_case_t *test_case,
                        kbench_runtime_t *runtime) {
  return dw_init_common(state, test_case, runtime,
                        cf_qdepthwise_u8s8_u8_pulp_nn_team);
}

static int dw_init_direct(void *state, const kbench_case_t *test_case,
                          kbench_runtime_t *runtime) {
  return dw_init_common(state, test_case, runtime,
                        cf_qdepthwise_u8s8_u8_direct3x3_team);
}

static int dw_init_channel4(void *state, const kbench_case_t *test_case,
                            kbench_runtime_t *runtime) {
  return dw_init_common(state, test_case, runtime,
                        cf_qdepthwise_u8s8_u8_channel4_team);
}

/* The packed store fuses four channels of one output pixel into one word, so
 * the channel count must be a multiple of four. Channel tails keep the
 * per-channel kernel. */
static int dw_validate_channel4(const kbench_case_t *test_case) {
  const int status = dw_validate_common(test_case);
  if (status != KBENCH_OK) return status;
  return test_case->problem.output_channels % 4u == 0u ? KBENCH_OK
                                                       : KBENCH_ERR_UNSUPPORTED;
}

static int dw_begin(void *state_memory, const kbench_case_t *test_case,
                    kbench_runtime_t *runtime, uint32_t iteration) {
  dw_state_t *state = (dw_state_t *)state_memory;
  (void)test_case;
  (void)runtime;
  (void)iteration;
  if (state == NULL || !state->initialized) return KBENCH_ERR_RUNTIME;
  memset(state->job.output, 0xa5, state->output_elements);
  memset(&state->counters, 0, sizeof(state->counters));
  return KBENCH_OK;
}

static int dw_run(void *state_memory, kbench_runtime_t *runtime) {
  dw_state_t *state = (dw_state_t *)state_memory;
  if (state == NULL || !state->initialized || runtime == NULL ||
      runtime->cluster == NULL) {
    return KBENCH_ERR_RUNTIME;
  }
  if (pi_cluster_send_task_to_cl(runtime->cluster, &state->task) != 0) {
    return KBENCH_ERR_RUNTIME;
  }
  return KBENCH_OK;
}

static int dw_check(void *state_memory, const kbench_case_t *test_case,
                    kbench_runtime_t *runtime, kbench_check_t *check) {
  dw_state_t *state = (dw_state_t *)state_memory;
  uint32_t index;
  (void)test_case;
  (void)runtime;
  if (state == NULL || check == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  memset(check, 0, sizeof(*check));
  check->operations = state->macs;
  check->executed_operations = state->macs;
  check->repetitions = state->team_repetitions;
  check->checksum = kbench_checksum_u8(state->job.output, state->output_elements);
  check->expected_checksum = state->checksum;
  kbench_counters_reduce(&state->counters, state->team_cores, check);
  check->weight_row_aligned = (uint8_t)(0u);
  check->exact = 1u;
  for (index = 0u; index < state->output_elements; ++index) {
    if (state->job.output[index] != state->expected[index]) {
      check->exact = 0u;
      return KBENCH_ERR_INCORRECT;
    }
  }
  return KBENCH_OK;
}

static void dw_deinit(void *state_memory, kbench_runtime_t *runtime) {
  (void)runtime;
  if (state_memory != NULL) memset(state_memory, 0, sizeof(dw_state_t));
}

#define KBENCH_DW_DESCRIPTOR(symbol, variant_name, validate_fn, init_fn)       \
  const kbench_kernel_t symbol = {                                             \
      .name = "depthwise_conv2d_u8s8_u8",                                    \
      .variant = variant_name,                                                 \
      .family = KBENCH_PROBLEM_DEPTHWISE_CONV2D,                               \
      .state_bytes = sizeof(dw_state_t),                                       \
      .state_alignment_bytes = 8u,                                             \
      .validate = validate_fn,                                                 \
      .plan = dw_plan,                                                         \
      .init = init_fn,                                                         \
      .begin = dw_begin,                                                       \
      .run = dw_run,                                                           \
      .check = dw_check,                                                       \
      .deinit = dw_deinit,                                                     \
  }

KBENCH_DW_DESCRIPTOR(kbench_depthwise_u8s8_u8_pulp_nn, "pulp_nn",
                     dw_validate_pulp, dw_init_pulp);
KBENCH_DW_DESCRIPTOR(kbench_depthwise_u8s8_u8_direct3x3, "direct3x3_2x1",
                     dw_validate_common, dw_init_direct);
KBENCH_DW_DESCRIPTOR(kbench_depthwise_u8s8_u8_channel4, "channel4",
                     dw_validate_channel4, dw_init_channel4);
