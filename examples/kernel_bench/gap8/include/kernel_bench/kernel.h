#ifndef CF_KERNEL_BENCH_KERNEL_H
#define CF_KERNEL_BENCH_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct pi_device;

#define KBENCH_RESULT_SCHEMA_VERSION 2u

typedef enum {
  KBENCH_OK = 0,
  KBENCH_ERR_INVALID_ARGUMENT = -1,
  KBENCH_ERR_UNSUPPORTED = -2,
  KBENCH_ERR_NO_MEMORY = -3,
  KBENCH_ERR_RUNTIME = -4,
  KBENCH_ERR_INCORRECT = -5,
} kbench_status_t;

typedef enum {
  KBENCH_PROBLEM_LINEAR = 0,
  KBENCH_PROBLEM_MATMUL = 1,
  KBENCH_PROBLEM_CONV2D = 2,
  KBENCH_PROBLEM_CUSTOM = 4,
  KBENCH_PROBLEM_DEPTHWISE_CONV2D = 5,
  KBENCH_PROBLEM_ACTIVATION = 6,
  KBENCH_PROBLEM_NORMALIZATION = 7,
  KBENCH_PROBLEM_POOLING = 8,
  KBENCH_PROBLEM_ADDITION = 9,
} kbench_problem_family_t;

typedef enum {
  KBENCH_MEMORY_L2_DIRECT = 0,
  KBENCH_MEMORY_L2_L1 = 1,
} kbench_memory_path_t;

typedef enum {
  KBENCH_EPILOGUE_INT32 = 0,
  KBENCH_EPILOGUE_SHIFT_CLIP_U8 = 1,
  KBENCH_EPILOGUE_REQUANT_RELU_U8 = 2,
  KBENCH_EPILOGUE_BATCHNORM_RELU_U8 = 3,
  KBENCH_EPILOGUE_ADD_U8 = 4,
} kbench_epilogue_t;

typedef enum {
  KBENCH_ACTIVATION_NONE = 0,
  KBENCH_ACTIVATION_SIGMOID = 1,
  KBENCH_ACTIVATION_TANH = 2,
  KBENCH_ACTIVATION_LEAKY_RELU = 3,
  KBENCH_ACTIVATION_HARD_SIGMOID = 4,
  KBENCH_ACTIVATION_HARD_SWISH = 5,
  KBENCH_ACTIVATION_SILU = 6,
  KBENCH_ACTIVATION_GELU = 7,
  KBENCH_ACTIVATION_SOFTMAX = 8,
} kbench_activation_t;

/*
 * Common dimensions are named instead of hidden in an opaque byte blob so a
 * report remains useful without the source tree. A kernel must document which
 * fields it consumes and reject combinations it does not support.
 */
typedef struct {
  kbench_problem_family_t family;
  uint32_t batch;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t height;
  uint32_t width;
  uint32_t input_channels;
  uint32_t output_channels;
  uint16_t kernel_h;
  uint16_t kernel_w;
  uint16_t stride_h;
  uint16_t stride_w;
  uint16_t padding_top;
  uint16_t padding_bottom;
  uint16_t padding_left;
  uint16_t padding_right;
} kbench_problem_t;

typedef struct {
  uint32_t id;
  const char *name;
  kbench_problem_t problem;
  uint32_t seed;
  uint16_t warmups;
  uint16_t runs;
  uint8_t cores;
  uint8_t element_bits;
  uint8_t base_alignment_bytes;
  uint8_t misalignment_offset_bytes;
  kbench_memory_path_t memory_path;
  kbench_activation_t activation;
  kbench_epilogue_t epilogue;
  uint16_t input1_multiplier;
  uint16_t input2_multiplier;
  uint16_t output_multiplier;
  int8_t output_shift;
} kbench_case_t;

/* No allocation is allowed in plan(). Sizes include padding/double buffers. */
typedef struct {
  size_t persistent_l1_bytes;
  size_t transient_l1_bytes;
  size_t persistent_l2_bytes;
  size_t transient_l2_bytes;
  uint16_t input_alignment_bytes;
  uint16_t weight_alignment_bytes;
  uint16_t output_alignment_bytes;
  uint32_t max_l2_l1_transfer_bytes;
} kbench_memory_plan_t;

/*
 * The harness/scheduler owns this object and the persistent cluster device.
 * Kernels borrow them. They must never open, close, or relock the cluster.
 */
typedef struct {
  struct pi_device *cluster;
  void *l1_arena;
  size_t l1_arena_bytes;
  void *l2_arena;
  size_t l2_arena_bytes;
} kbench_runtime_t;

typedef struct {
  uint32_t checksum;
  uint32_t expected_checksum;
  /* Logical useful work; padding lanes are excluded. */
  uint64_t operations;
  /* Physical SIMD lanes executed, including prepared zero padding. */
  uint64_t executed_operations;
  /* Repeated identical invocations used only to amortize benchmark timing. */
  uint32_t repetitions;
  /* Optional cluster-local counters, accumulated across active cores. */
  uint32_t tcdm_contentions;
  uint32_t load_stalls;
  /* Maximum measured compute cycles on any participating core. */
  uint32_t kernel_cycles;
  /*
   * Zero when consecutive weight rows are not word aligned, which happens
   * whenever the compact row stride is not a multiple of four (a 3x3x7 patch
   * gives 63). GAP8 splits those loads in hardware at roughly double cost, so
   * the condition is legal but must be reported next to the timing rather than
   * assumed away.
   */
  uint8_t weight_row_aligned;
  uint8_t exact;
} kbench_check_t;

/*
 * Lifecycle:
 *
 *   validate -> plan -> init -> [begin -> run -> check]* -> deinit
 *
 * run() is the timed interval: it submits the cluster task and returns when the
 * task has completed.
 */
typedef struct {
  const char *name;
  const char *variant;
  kbench_problem_family_t family;
  size_t state_bytes;
  size_t state_alignment_bytes;

  int (*validate)(const kbench_case_t *test_case);
  int (*plan)(const kbench_case_t *test_case, kbench_memory_plan_t *plan);
  int (*init)(void *state, const kbench_case_t *test_case,
              kbench_runtime_t *runtime);
  int (*begin)(void *state, const kbench_case_t *test_case,
               kbench_runtime_t *runtime, uint32_t iteration);
  int (*run)(void *state, kbench_runtime_t *runtime);
  int (*check)(void *state, const kbench_case_t *test_case,
               kbench_runtime_t *runtime, kbench_check_t *check);
  void (*deinit)(void *state, kbench_runtime_t *runtime);
} kbench_kernel_t;

#ifdef __cplusplus
}
#endif

#endif /* CF_KERNEL_BENCH_KERNEL_H */
