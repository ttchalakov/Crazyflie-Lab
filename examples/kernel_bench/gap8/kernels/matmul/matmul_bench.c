#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "pmsis.h"

#include "cf_qconv.h"
#include "kernel_bench/fixture.h"
#include "kernel_bench/kernel.h"

#ifndef KBENCH_MATMUL_TARGET_MACS
#define KBENCH_MATMUL_TARGET_MACS (1u << 16)
#endif
#define KBENCH_MATMUL_MAX_REPETITIONS 32u
#define KBENCH_MATMUL_MASTER_STACK_BYTES 4096u
#define KBENCH_MATMUL_SLAVE_STACK_BYTES 2048u
#define KBENCH_MATMUL_MAX_CORES 8u
#define KBENCH_MATMUL_DORY_L1_BUDGET_BYTES 36700u

typedef enum {
  KBENCH_MATMUL_PULP_NN = 0,
  KBENCH_MATMUL_SEQUENTIAL = 1,
  KBENCH_MATMUL_KROTATE = 2,
  /* Four output pixels per block instead of two. Its shapes carry m=4, so it
   * is a different case and not a drop-in swap -- see matmul_validate_m. */
  KBENCH_MATMUL_BLOCK4X4 = 3,
  /* Three pixels: the block size the register file can actually hold. */
  KBENCH_MATMUL_BLOCK4X3 = 4,
} matmul_variant_t;

typedef struct {
  const int8_t *weights;
  uint8_t *input;
  int32_t *batch_norm_multiplier;
  int32_t *batch_norm_bias;
  uint8_t *output;
  uint8_t *expected;
  struct pi_cluster_task task;
  kbench_counters_t counters;
  uint32_t expected_checksum;
  uint32_t output_elements;
  uint64_t macs;
  uint32_t repetitions;
  uint16_t output_channels;
  uint16_t columns;
  uint16_t output_shift;
  uint16_t output_multiplier;
  uint8_t cores;
  uint8_t flag_relu;
  uint8_t flag_batch_norm;
  uint8_t initialized;
  uint8_t pixels;            /* output pixels per core: 2, 3 or 4 */
  matmul_variant_t variant;
} matmul_state_t;

extern uint8_t *pulp_nn_matmul(
    const int8_t *weights, uint8_t *input, uint16_t output_channels,
    uint16_t columns, uint16_t output_shift, uint16_t output_multiplier,
    int32_t *batch_norm_multiplier, int32_t *batch_norm_bias,
    const int8_t *bias, uint8_t *output, uint8_t *output2, int flag_relu,
    int flag_batch_norm);

static uint8_t matmul_clip_u8(int32_t value) {
  if (value < 0) return 0u;
  if (value > UINT8_MAX) return UINT8_MAX;
  return (uint8_t)value;
}

static uint8_t matmul_quantize(const matmul_state_t *state, uint32_t channel,
                               int32_t accumulator) {
  if (state->flag_batch_norm) {
    return matmul_clip_u8(
        (state->batch_norm_multiplier[channel] * accumulator +
         state->batch_norm_bias[channel]) >>
        state->output_shift);
  }
  if (state->flag_relu) {
    const int32_t multiplier = (int16_t)state->output_multiplier;
    const int16_t result =
        (int16_t)((multiplier * accumulator) >> state->output_shift);
    return matmul_clip_u8(result);
  }
  return matmul_clip_u8(accumulator >> state->output_shift);
}

static uint32_t matmul_repetitions(uint64_t macs) {
  uint32_t repetitions =
      (uint32_t)(((uint64_t)KBENCH_MATMUL_TARGET_MACS + macs - 1u) / macs);
  if (repetitions == 0u) repetitions = 1u;
  if (repetitions > KBENCH_MATMUL_MAX_REPETITIONS) {
    repetitions = KBENCH_MATMUL_MAX_REPETITIONS;
  }
  return repetitions;
}

/*
 * `pixels` is the number of output positions one call produces, and it is a
 * property of the kernel rather than of the case: 4x2 blocks two, 4x3 three,
 * 4x4 four. It is folded into validate rather than checked later because it
 * changes the L1 footprint -- doubling the im2col columns and the output tile
 * is what puts K = 576, N = 32 (matmul4x2-576to32) over the 36,700-byte budget
 * at four pixels, and a shape that does not fit must be declined here rather
 * than truncated in init.
 */
static int matmul_validate_m(const kbench_case_t *test_case, uint32_t pixels) {
  const kbench_problem_t *problem;
  uint64_t required_l1_bytes;
  uint32_t alignment;
  if (test_case == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  problem = &test_case->problem;
  alignment = test_case->base_alignment_bytes > 4u
                  ? test_case->base_alignment_bytes
                  : 4u;
  required_l1_bytes =
      (uint64_t)problem->n * problem->k +
      (uint64_t)test_case->cores * pixels * problem->k +
      (uint64_t)test_case->cores * pixels * problem->n +
      2u * (uint64_t)problem->n * sizeof(int32_t) +
      test_case->misalignment_offset_bytes + 6u * alignment;
  if (problem->family != KBENCH_PROBLEM_MATMUL || problem->batch != 1u ||
      problem->m != pixels || problem->n == 0u || problem->n > UINT16_MAX ||
      problem->k == 0u || problem->k > UINT16_MAX ||
      test_case->element_bits != 8u ||
      test_case->base_alignment_bytes < 4u ||
      (test_case->misalignment_offset_bytes & 3u) != 0u ||
      test_case->misalignment_offset_bytes > 12u ||
      test_case->memory_path != KBENCH_MEMORY_L2_L1 ||
      test_case->cores == 0u ||
      test_case->cores > KBENCH_MATMUL_MAX_CORES ||
      (test_case->cores & (test_case->cores - 1u)) != 0u ||
      test_case->output_shift < 0 || test_case->output_shift > 31 ||
      test_case->epilogue < KBENCH_EPILOGUE_SHIFT_CLIP_U8 ||
      test_case->epilogue > KBENCH_EPILOGUE_BATCHNORM_RELU_U8 ||
      required_l1_bytes > KBENCH_MATMUL_DORY_L1_BUDGET_BYTES) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  return KBENCH_OK;
}

static int matmul_validate(const kbench_case_t *test_case) {
  return matmul_validate_m(test_case, 2u);
}

static int matmul_validate4(const kbench_case_t *test_case) {
  return matmul_validate_m(test_case, 4u);
}

static int matmul_validate3(const kbench_case_t *test_case) {
  return matmul_validate_m(test_case, 3u);
}

static int matmul_plan_m(const kbench_case_t *test_case,
                         kbench_memory_plan_t *plan, uint32_t pixels) {
  const kbench_problem_t *problem;
  const size_t alignment = test_case != NULL &&
                                   test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;
  size_t weights_bytes;
  size_t input_bytes;
  size_t output_bytes;
  size_t quantization_bytes;

  if (test_case == NULL || plan == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  if (matmul_validate_m(test_case, pixels) != KBENCH_OK) {
    return KBENCH_ERR_UNSUPPORTED;
  }
  problem = &test_case->problem;
  weights_bytes = (size_t)problem->n * problem->k;
  input_bytes = (size_t)test_case->cores * pixels * problem->k;
  output_bytes = (size_t)test_case->cores * pixels * problem->n;
  quantization_bytes = 2u * (size_t)problem->n * sizeof(int32_t);

  memset(plan, 0, sizeof(*plan));
  plan->persistent_l1_bytes =
      weights_bytes + input_bytes + output_bytes + quantization_bytes +
      test_case->misalignment_offset_bytes + 6u * alignment;
  plan->persistent_l2_bytes = output_bytes + alignment;
  plan->input_alignment_bytes = 4u;
  plan->weight_alignment_bytes = 4u;
  plan->output_alignment_bytes = 4u;
  return KBENCH_OK;
}

static int matmul_plan(const kbench_case_t *test_case,
                       kbench_memory_plan_t *plan) {
  return matmul_plan_m(test_case, plan, 2u);
}

static int matmul_plan4(const kbench_case_t *test_case,
                        kbench_memory_plan_t *plan) {
  return matmul_plan_m(test_case, plan, 4u);
}

static int matmul_plan3(const kbench_case_t *test_case,
                        kbench_memory_plan_t *plan) {
  return matmul_plan_m(test_case, plan, 3u);
}

static void matmul_reference(matmul_state_t *state) {
  uint32_t core;
  const uint32_t pixels = state->pixels;
  for (core = 0u; core < state->cores; ++core) {
    const uint8_t *core_input = state->input + core * pixels * state->columns;
    uint8_t *core_expected =
        state->expected + core * pixels * state->output_channels;
    uint32_t column;
    for (column = 0u; column < pixels; ++column) {
      uint32_t channel;
      for (channel = 0u; channel < state->output_channels; ++channel) {
        const int8_t *weight =
            state->weights + channel * state->columns;
        const uint8_t *input = core_input + column * state->columns;
        int32_t accumulator = 0;
        uint32_t k;
        for (k = 0u; k < state->columns; ++k) {
          accumulator += (int32_t)input[k] * weight[k];
        }
        core_expected[column * state->output_channels + channel] =
            matmul_quantize(state, channel, accumulator);
      }
    }
  }
}

static void matmul_worker(void *argument) {
  matmul_state_t *state = (matmul_state_t *)argument;
  const uint32_t core = (uint32_t)pi_core_id();
  const uint16_t output_channels = state->output_channels;
  const uint16_t columns = state->columns;
  const uint16_t output_shift = state->output_shift;
  const uint16_t output_multiplier = state->output_multiplier;
  const uint32_t repetitions = state->repetitions;
  const int flag_relu = state->flag_relu;
  const int flag_batch_norm = state->flag_batch_norm;
  const int8_t *weights = state->weights;
  uint8_t *input = state->input + core * state->pixels * columns;
  uint8_t *output = state->output + core * state->pixels * output_channels;
  int32_t *batch_norm_multiplier = state->batch_norm_multiplier;
  int32_t *batch_norm_bias = state->batch_norm_bias;
  uint32_t repetition;

  kbench_perf_begin();
  switch (state->variant) {
    case KBENCH_MATMUL_PULP_NN:
      for (repetition = 0u; repetition < repetitions; ++repetition) {
        pulp_nn_matmul(weights, input, output_channels, columns, output_shift,
                       output_multiplier, batch_norm_multiplier,
                       batch_norm_bias, NULL, output,
                       output + output_channels, flag_relu, flag_batch_norm);
      }
      break;
    case KBENCH_MATMUL_SEQUENTIAL:
      for (repetition = 0u; repetition < repetitions; ++repetition) {
        cf_qconv_matmul4x2(weights, input, output_channels, columns,
                           output_shift, output_multiplier,
                           batch_norm_multiplier, batch_norm_bias, NULL,
                           output, output + output_channels, flag_relu,
                           flag_batch_norm);
      }
      break;
    case KBENCH_MATMUL_KROTATE:
      for (repetition = 0u; repetition < repetitions; ++repetition) {
        cf_qconv_matmul4x2_krotate(
            weights, input, output_channels, columns, output_shift,
            output_multiplier, batch_norm_multiplier, batch_norm_bias, NULL,
            output, output + output_channels, flag_relu, flag_batch_norm);
      }
      break;
    case KBENCH_MATMUL_BLOCK4X3:
      for (repetition = 0u; repetition < repetitions; ++repetition) {
        cf_qconv_matmul4x3(weights, input, output_channels, columns,
                           output_shift, output_multiplier,
                           batch_norm_multiplier, batch_norm_bias, NULL,
                           output, output_channels, flag_relu,
                           flag_batch_norm);
      }
      break;
    case KBENCH_MATMUL_BLOCK4X4:
      for (repetition = 0u; repetition < repetitions; ++repetition) {
        cf_qconv_matmul4x4(weights, input, output_channels, columns,
                           output_shift, output_multiplier,
                           batch_norm_multiplier, batch_norm_bias, NULL,
                           output, output_channels, flag_relu,
                           flag_batch_norm);
      }
      break;
    default:
      break;
  }
  kbench_perf_end(&state->counters, core);
}

static void matmul_cluster_entry(void *argument) {
  matmul_state_t *state = (matmul_state_t *)argument;
  pi_cl_team_fork(state->cores, matmul_worker, state);
}

static int matmul_init_common(void *state_memory,
                              const kbench_case_t *test_case,
                              kbench_runtime_t *runtime,
                              matmul_variant_t variant) {
  matmul_state_t *state = (matmul_state_t *)state_memory;
  const kbench_problem_t *problem = &test_case->problem;
  const size_t alignment = test_case->base_alignment_bytes > 4u
                               ? test_case->base_alignment_bytes
                               : 4u;
  kbench_arena_t l1;
  kbench_arena_t l2;
  uint32_t random = test_case->seed;
  uint32_t weights_bytes;
  uint32_t input_bytes;
  uint32_t output_bytes;
  uint32_t index;

  if (state == NULL || runtime == NULL || runtime->cluster == NULL ||
      runtime->l1_arena == NULL || runtime->l2_arena == NULL) {
    return KBENCH_ERR_INVALID_ARGUMENT;
  }
  memset(state, 0, sizeof(*state));
  l1.cursor = (uintptr_t)runtime->l1_arena;
  l1.end = l1.cursor + runtime->l1_arena_bytes;
  l2.cursor = (uintptr_t)runtime->l2_arena;
  l2.end = l2.cursor + runtime->l2_arena_bytes;
  state->pixels = (uint8_t)problem->m;
  weights_bytes = problem->n * problem->k;
  input_bytes = test_case->cores * state->pixels * problem->k;
  output_bytes = test_case->cores * state->pixels * problem->n;

  l1.cursor = kbench_align_up(l1.cursor, alignment) +
              test_case->misalignment_offset_bytes;
  state->weights = kbench_arena_take(&l1, weights_bytes, 4u);
  state->input = kbench_arena_take(&l1, input_bytes, alignment);
  state->batch_norm_multiplier = kbench_arena_take(
      &l1, (size_t)problem->n * sizeof(int32_t), alignment);
  state->batch_norm_bias = kbench_arena_take(
      &l1, (size_t)problem->n * sizeof(int32_t), alignment);
  state->output = kbench_arena_take(&l1, output_bytes, alignment);
  state->expected = kbench_arena_take(&l2, output_bytes, alignment);
  if (state->weights == NULL || state->input == NULL ||
      state->batch_norm_multiplier == NULL ||
      state->batch_norm_bias == NULL || state->output == NULL ||
      state->expected == NULL) {
    return KBENCH_ERR_NO_MEMORY;
  }

  state->output_elements = output_bytes;
  state->macs = (uint64_t)output_bytes * problem->k;
  state->repetitions = matmul_repetitions(state->macs);
  state->output_channels = (uint16_t)problem->n;
  state->columns = (uint16_t)problem->k;
  state->output_shift = (uint16_t)test_case->output_shift;
  state->output_multiplier = test_case->output_multiplier;
  state->cores = test_case->cores;
  state->flag_relu = test_case->epilogue != KBENCH_EPILOGUE_SHIFT_CLIP_U8;
  state->flag_batch_norm =
      test_case->epilogue == KBENCH_EPILOGUE_BATCHNORM_RELU_U8;
  state->variant = variant;

  for (index = 0u; index < weights_bytes; ++index) {
    ((int8_t *)state->weights)[index] =
        (int8_t)(kbench_lcg(&random) % 9u) - 4;
  }
  for (index = 0u; index < input_bytes; ++index) {
    state->input[index] = (uint8_t)(kbench_lcg(&random) % 32u);
  }
  for (index = 0u; index < problem->n; ++index) {
    state->batch_norm_multiplier[index] =
        (int32_t)(kbench_lcg(&random) % 4u) + 1;
    state->batch_norm_bias[index] =
        (int32_t)(kbench_lcg(&random) % 1025u) - 512;
  }
  matmul_reference(state);
  state->expected_checksum =
      kbench_checksum_u8(state->expected, state->output_elements);
  pi_cluster_task(&state->task, matmul_cluster_entry, state);
  state->task.stack_size = KBENCH_MATMUL_MASTER_STACK_BYTES;
  state->task.slave_stack_size = KBENCH_MATMUL_SLAVE_STACK_BYTES;
  state->initialized = 1u;
  return KBENCH_OK;
}

static int matmul_init_pulp(void *state, const kbench_case_t *test_case,
                            kbench_runtime_t *runtime) {
  return matmul_init_common(state, test_case, runtime, KBENCH_MATMUL_PULP_NN);
}

static int matmul_init_sequential(void *state, const kbench_case_t *test_case,
                                  kbench_runtime_t *runtime) {
  return matmul_init_common(state, test_case, runtime,
                            KBENCH_MATMUL_SEQUENTIAL);
}

static int matmul_init_block4x3(void *state_memory,
                                const kbench_case_t *test_case,
                                kbench_runtime_t *runtime) {
  return matmul_init_common(state_memory, test_case, runtime,
                            KBENCH_MATMUL_BLOCK4X3);
}

static int matmul_init_block4x4(void *state_memory,
                                const kbench_case_t *test_case,
                                kbench_runtime_t *runtime) {
  return matmul_init_common(state_memory, test_case, runtime,
                            KBENCH_MATMUL_BLOCK4X4);
}

static int matmul_init_krotate(void *state, const kbench_case_t *test_case,
                               kbench_runtime_t *runtime) {
  return matmul_init_common(state, test_case, runtime, KBENCH_MATMUL_KROTATE);
}

static int matmul_begin(void *state_memory, const kbench_case_t *test_case,
                        kbench_runtime_t *runtime, uint32_t iteration) {
  matmul_state_t *state = (matmul_state_t *)state_memory;
  (void)test_case;
  (void)runtime;
  (void)iteration;
  if (state == NULL || !state->initialized) return KBENCH_ERR_RUNTIME;
  memset(state->output, 0xa5, state->output_elements);
  memset(&state->counters, 0, sizeof(state->counters));
  return KBENCH_OK;
}

static int matmul_run(void *state_memory, kbench_runtime_t *runtime) {
  matmul_state_t *state = (matmul_state_t *)state_memory;
  if (state == NULL || !state->initialized || runtime == NULL ||
      runtime->cluster == NULL) {
    return KBENCH_ERR_RUNTIME;
  }
  if (pi_cluster_send_task_to_cl(runtime->cluster, &state->task) != 0) {
    return KBENCH_ERR_RUNTIME;
  }
  return KBENCH_OK;
}

static int matmul_check(void *state_memory, const kbench_case_t *test_case,
                        kbench_runtime_t *runtime, kbench_check_t *check) {
  matmul_state_t *state = (matmul_state_t *)state_memory;
  uint32_t index;
  (void)test_case;
  (void)runtime;
  if (state == NULL || check == NULL) return KBENCH_ERR_INVALID_ARGUMENT;
  memset(check, 0, sizeof(*check));
  check->operations = state->macs;
  check->executed_operations = state->macs;
  check->repetitions = state->repetitions;
  check->checksum = kbench_checksum_u8(state->output, state->output_elements);
  check->expected_checksum = state->expected_checksum;
  check->weight_row_aligned = (uint8_t)((state->columns & 3u) == 0u);
  check->exact = 1u;
  kbench_counters_reduce(&state->counters, state->cores, check);
  for (index = 0u; index < state->output_elements; ++index) {
    if (state->output[index] != state->expected[index]) {
      check->exact = 0u;
      return KBENCH_ERR_INCORRECT;
    }
  }
  return KBENCH_OK;
}

static void matmul_deinit(void *state_memory, kbench_runtime_t *runtime) {
  (void)runtime;
  if (state_memory != NULL) memset(state_memory, 0, sizeof(matmul_state_t));
}

#define KBENCH_MATMUL_DESCRIPTOR_M(symbol, kernel_name, variant_name, init_fn,  \
                                   validate_fn, plan_fn)                       \
  const kbench_kernel_t symbol = {                                             \
      .name = kernel_name,                                                     \
      .variant = variant_name,                                                 \
      .family = KBENCH_PROBLEM_MATMUL,                                         \
      .state_bytes = sizeof(matmul_state_t),                                   \
      .state_alignment_bytes = 8u,                                             \
      .validate = validate_fn,                                                 \
      .plan = plan_fn,                                                         \
      .init = init_fn,                                                         \
      .begin = matmul_begin,                                                   \
      .run = matmul_run,                                                       \
      .check = matmul_check,                                                   \
      .deinit = matmul_deinit,                                                 \
  }

#define KBENCH_MATMUL_DESCRIPTOR(symbol, variant_name, init_fn)               \
  KBENCH_MATMUL_DESCRIPTOR_M(symbol, "matmul4x2_u8s8_u8", variant_name,       \
                             init_fn, matmul_validate, matmul_plan)

KBENCH_MATMUL_DESCRIPTOR(kbench_matmul4x2_u8s8_u8_pulp_nn, "pulp_nn",
                         matmul_init_pulp);
KBENCH_MATMUL_DESCRIPTOR(kbench_matmul4x2_u8s8_u8_sequential, "sequential",
                         matmul_init_sequential);
KBENCH_MATMUL_DESCRIPTOR(kbench_matmul4x2_u8s8_u8_krotate, "krotate",
                         matmul_init_krotate);

/* Its own kernel name, not a variant of the 4x2: the operands differ (four
 * im2col columns per core, not two), so a report row for it is not comparable
 * to a 4x2 row cycle-for-cycle -- only per MAC. report.py's speedup columns
 * compare variants within one kernel name, and comparing these two directly
 * there would be wrong. */
KBENCH_MATMUL_DESCRIPTOR_M(kbench_matmul4x4_u8s8_u8_sequential,
                           "matmul4x4_u8s8_u8", "sequential",
                           matmul_init_block4x4, matmul_validate4,
                           matmul_plan4);
KBENCH_MATMUL_DESCRIPTOR_M(kbench_matmul4x3_u8s8_u8_sequential,
                           "matmul4x3_u8s8_u8", "sequential",
                           matmul_init_block4x3, matmul_validate3,
                           matmul_plan3);
