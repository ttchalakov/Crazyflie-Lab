/* Copy into kernels/<kernel>/<kernel>.c; this template is not built by default. */
#include <string.h>

#include "kernel_bench/kernel.h"

typedef struct {
  uint32_t checksum;
} template_state_t;

static int template_validate(const kbench_case_t *test_case) {
  (void)test_case;
  return KBENCH_ERR_UNSUPPORTED;
}

static int template_plan(const kbench_case_t *test_case,
                         kbench_memory_plan_t *plan) {
  (void)test_case;
  if (plan == NULL) {
    return KBENCH_ERR_INVALID_ARGUMENT;
  }
  memset(plan, 0, sizeof(*plan));
  plan->input_alignment_bytes = 4;
  plan->weight_alignment_bytes = 4;
  plan->output_alignment_bytes = 4;
  plan->max_l2_l1_transfer_bytes = 32764;
  return KBENCH_ERR_UNSUPPORTED;
}

static int template_init(void *state, const kbench_case_t *test_case,
                         kbench_runtime_t *runtime) {
  (void)test_case;
  (void)runtime;
  memset(state, 0, sizeof(template_state_t));
  return KBENCH_ERR_UNSUPPORTED;
}

static int template_begin(void *state, const kbench_case_t *test_case,
                          kbench_runtime_t *runtime, uint32_t iteration) {
  (void)test_case;
  (void)runtime;
  (void)iteration;
  ((template_state_t *)state)->checksum = 0;
  return KBENCH_ERR_UNSUPPORTED;
}

static int template_run(void *state, kbench_runtime_t *runtime) {
  (void)state;
  (void)runtime;
  return KBENCH_ERR_RUNTIME;
}

static int template_check(void *state, const kbench_case_t *test_case,
                          kbench_runtime_t *runtime, kbench_check_t *check) {
  (void)state;
  (void)test_case;
  (void)runtime;
  (void)check;
  return KBENCH_ERR_UNSUPPORTED;
}

static void template_deinit(void *state, kbench_runtime_t *runtime) {
  (void)state;
  (void)runtime;
}

const kbench_kernel_t kbench_kernel_template = {
    .name = "replace_me",
    .variant = "template",
    .family = KBENCH_PROBLEM_CUSTOM,
    .state_bytes = sizeof(template_state_t),
    .state_alignment_bytes = 4,
    .validate = template_validate,
    .plan = template_plan,
    .init = template_init,
    .begin = template_begin,
    .run = template_run,
    .check = template_check,
    .deinit = template_deinit,
};
