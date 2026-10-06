#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#include "pmsis.h"

#include "kbench_matrix.h"
#include "kernel_bench/registry.h"

#ifndef KBENCH_FC_FREQ_HZ
#define KBENCH_FC_FREQ_HZ 150000000
#endif
#ifndef KBENCH_CL_FREQ_HZ
#define KBENCH_CL_FREQ_HZ 150000000
#endif
#ifndef KBENCH_BOOT_DELAY_MS
#define KBENCH_BOOT_DELAY_MS 12000
#endif

#ifdef KBENCH_GVSOC
/*
 * GVSOC writes asynchronous warnings through the same stream as semihosted
 * printf. A warning can land inside a KBENCH record and destroy that record even
 * though the guest continues and passes. Emit each out-of-band record twice;
 * report.py deduplicates samples by run and keeps the intact copy.
 */
#define KBENCH_LOG(...)          \
  do {                           \
    printf(__VA_ARGS__);         \
    printf(__VA_ARGS__);         \
  } while (0)
#else
#include "bsp/bsp.h"
#include "cpx.h"
#ifndef KBENCH_CPX_PACE_US
#define KBENCH_CPX_PACE_US 20000
#endif

/*
 * One CPX console message expands into several 30-byte CRTP console packets on
 * the STM32. Emitting the full matrix report without pacing overruns that
 * queue; the Crazyflie then inserts ``<F>`` markers and drops bytes from the
 * machine-readable records. Reporting is outside every timed region, so this
 * pause makes the transport lossless without changing a measurement.
 */
static void kbench_log(const char *format, ...) {
  char line[512];
  va_list args;

  va_start(args, format);
  vsnprintf(line, sizeof(line), format, args);
  va_end(args);
  cpxPrintToConsole(LOG_TO_CRTP, "%s", line);
  pi_time_wait_us(KBENCH_CPX_PACE_US);
}

#define KBENCH_LOG(...) kbench_log(__VA_ARGS__)
#define KBENCH_STACK_WORDS (configMINIMAL_STACK_SIZE * 8)
#endif

static struct pi_device g_cluster;
static uint8_t g_cluster_open;

typedef struct {
  void *raw;
  void *aligned;
  size_t raw_bytes;
} kbench_allocation_t;

static const char *memory_path_name(kbench_memory_path_t path) {
  switch (path) {
    case KBENCH_MEMORY_L2_DIRECT:
      return "l2_direct";
    case KBENCH_MEMORY_L2_L1:
      return "l1";
    default:
      return "invalid";
  }
}

static const char *epilogue_name(kbench_epilogue_t epilogue) {
  switch (epilogue) {
    case KBENCH_EPILOGUE_INT32:
      return "none";
    case KBENCH_EPILOGUE_SHIFT_CLIP_U8:
      return "u8shift";
    case KBENCH_EPILOGUE_REQUANT_RELU_U8:
      return "u8requant";
    case KBENCH_EPILOGUE_BATCHNORM_RELU_U8:
      return "u8bnfold";
    case KBENCH_EPILOGUE_ADD_U8:
      return "add";
    default:
      return "invalid";
  }
}

static const char *activation_name(kbench_activation_t activation) {
  switch (activation) {
    case KBENCH_ACTIVATION_NONE:
      return "none";
    case KBENCH_ACTIVATION_SIGMOID:
      return "sigmoid";
    case KBENCH_ACTIVATION_TANH:
      return "tanh";
    case KBENCH_ACTIVATION_LEAKY_RELU:
      return "leaky_relu";
    case KBENCH_ACTIVATION_HARD_SIGMOID:
      return "hard_sigmoid";
    case KBENCH_ACTIVATION_HARD_SWISH:
      return "hard_swish";
    case KBENCH_ACTIVATION_SILU:
      return "silu";
    case KBENCH_ACTIVATION_GELU:
      return "gelu";
    case KBENCH_ACTIVATION_SOFTMAX:
      return "softmax";
    default:
      return "invalid";
  }
}

static int open_cluster(void) {
  struct pi_cluster_conf config;

  pi_cluster_conf_init(&config);
  config.id = 0;
  pi_open_from_conf(&g_cluster, &config);
  if (pi_cluster_open(&g_cluster) != 0) {
    return -1;
  }
  g_cluster_open = 1;

  /* The cluster power domain must be up before programming its FLL. */
  (void)pi_freq_set(PI_FREQ_DOMAIN_CL, KBENCH_CL_FREQ_HZ);
  /* Log the clock read back, not the one requested: every `cycles=` field is
   * elapsed microseconds scaled by KBENCH_CL_FREQ_HZ, so a set that did not
   * take would otherwise go unnoticed. nn_bench logs the same reading as
   * `cluster_hz`, which is what lets the two benches' cycles be compared. */
  KBENCH_LOG("KBENCH|META|cl_hz_actual=%u\n",
             (unsigned)pi_freq_get(PI_FREQ_DOMAIN_CL));
  return 0;
}

static int allocate_aligned(size_t bytes, size_t alignment,
                            kbench_allocation_t *allocation) {
  uintptr_t address;

  if (allocation == NULL || bytes == 0u || alignment == 0u ||
      (alignment & (alignment - 1u)) != 0u) {
    return -1;
  }
  allocation->raw_bytes = bytes + alignment - 1u;
  allocation->raw = pi_l2_malloc(allocation->raw_bytes);
  if (allocation->raw == NULL) {
    allocation->aligned = NULL;
    return -1;
  }
  address = ((uintptr_t)allocation->raw + alignment - 1u) &
            ~((uintptr_t)alignment - 1u);
  allocation->aligned = (void *)address;
  return 0;
}

static void free_aligned(kbench_allocation_t *allocation) {
  if (allocation != NULL && allocation->raw != NULL) {
    pi_l2_free(allocation->raw, allocation->raw_bytes);
    allocation->raw = NULL;
    allocation->aligned = NULL;
    allocation->raw_bytes = 0u;
  }
}

static int allocate_l1_aligned(struct pi_device *cluster, size_t bytes,
                               size_t alignment,
                               kbench_allocation_t *allocation) {
  uintptr_t address;

  if (cluster == NULL || allocation == NULL || bytes == 0u ||
      alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
    return -1;
  }
  allocation->raw_bytes = bytes + alignment - 1u;
  allocation->raw = pi_cl_l1_malloc(cluster, allocation->raw_bytes);
  if (allocation->raw == NULL) {
    allocation->aligned = NULL;
    return -1;
  }
  address = ((uintptr_t)allocation->raw + alignment - 1u) &
            ~((uintptr_t)alignment - 1u);
  allocation->aligned = (void *)address;
  return 0;
}

static void free_l1_aligned(struct pi_device *cluster,
                            kbench_allocation_t *allocation) {
  if (cluster != NULL && allocation != NULL && allocation->raw != NULL) {
    pi_cl_l1_free(cluster, allocation->raw, allocation->raw_bytes);
    allocation->raw = NULL;
    allocation->aligned = NULL;
    allocation->raw_bytes = 0u;
  }
}

static const char *status_name(int status) {
  switch (status) {
    case KBENCH_OK:
      return "pass";
    case KBENCH_ERR_UNSUPPORTED:
      return "skip";
    case KBENCH_ERR_NO_MEMORY:
      return "no_memory";
    case KBENCH_ERR_INCORRECT:
      return "incorrect";
    default:
      return "failed";
  }
}

static int run_kernel_case(const kbench_kernel_t *kernel,
                           const kbench_case_t *test_case,
                           uint8_t *applicable) {
  kbench_memory_plan_t plan;
  kbench_runtime_t runtime;
  kbench_allocation_t state_allocation = {0};
  kbench_allocation_t l2_allocation = {0};
  kbench_allocation_t l1_allocation = {0};
  kbench_check_t last_check;
  const size_t state_alignment = kernel->state_alignment_bytes != 0u
                                     ? kernel->state_alignment_bytes
                                     : sizeof(uint32_t);
  size_t l2_bytes;
  size_t l1_bytes;
  uint32_t iteration;
  uint32_t sample_index = 0u;
  int status;
  uint8_t initialized = 0u;

  if (applicable != NULL) {
    *applicable = 0u;
  }

  status = kernel->validate(test_case);
  if (status != KBENCH_OK) {
    return KBENCH_OK;
  }

  /* `applicable` is set only at the bottom, once this descriptor has emitted
   * a passing sample: a descriptor that passes validate() and then fails
   * plan(), allocation or init() measured nothing and must not make its case
   * `complete`. */
  memset(&plan, 0, sizeof(plan));
  status = kernel->plan(test_case, &plan);
  if (status != KBENCH_OK) {
    KBENCH_LOG("KBENCH|RESULT|case=%u|kernel=%s|variant=%s|status=%s\n",
               (unsigned)test_case->id, kernel->name, kernel->variant,
               status_name(status));
    return status == KBENCH_ERR_UNSUPPORTED ? KBENCH_OK : status;
  }

  l2_bytes = plan.persistent_l2_bytes + plan.transient_l2_bytes;
  l1_bytes = plan.persistent_l1_bytes + plan.transient_l1_bytes;
  if (allocate_aligned(kernel->state_bytes, state_alignment,
                       &state_allocation) != 0 ||
      allocate_aligned(l2_bytes, test_case->base_alignment_bytes,
                       &l2_allocation) != 0 ||
      (l1_bytes != 0u &&
       allocate_l1_aligned(&g_cluster, l1_bytes,
                           test_case->base_alignment_bytes,
                           &l1_allocation) != 0)) {
    status = KBENCH_ERR_NO_MEMORY;
    goto finished;
  }

  memset(&runtime, 0, sizeof(runtime));
  runtime.cluster = &g_cluster;
  runtime.l2_arena = l2_allocation.aligned;
  runtime.l2_arena_bytes = l2_bytes;
  runtime.l1_arena = l1_allocation.aligned;
  runtime.l1_arena_bytes = l1_bytes;

  status = kernel->init(state_allocation.aligned, test_case, &runtime);
  if (status != KBENCH_OK) {
    goto finished;
  }
  initialized = 1u;
  memset(&last_check, 0, sizeof(last_check));

  for (iteration = 0u;
       iteration < (uint32_t)test_case->warmups + test_case->runs;
       ++iteration) {
    uint32_t started_us;
    uint32_t elapsed_us;
    uint32_t cycles;

    status = kernel->begin(state_allocation.aligned, test_case, &runtime,
                           iteration);
    if (status != KBENCH_OK) {
      break;
    }

    started_us = pi_time_get_us();
    status = kernel->run(state_allocation.aligned, &runtime);
    elapsed_us = pi_time_get_us() - started_us;
    if (status != KBENCH_OK) {
      break;
    }

    status = kernel->check(state_allocation.aligned, test_case, &runtime,
                           &last_check);
    if (status != KBENCH_OK) {
      break;
    }

    if (iteration >= test_case->warmups) {
      cycles = (uint32_t)(((uint64_t)elapsed_us * KBENCH_CL_FREQ_HZ) /
                          1000000u);
      KBENCH_LOG("KBENCH|SAMPLE|case=%u|kernel=%s|variant=%s|run=%u|us=%u|cycles=%u|repetitions=%u|kernel_cycles=%u|tcdm_contentions=%u|load_stalls=%u\n",
                 (unsigned)test_case->id, kernel->name, kernel->variant,
                 (unsigned)sample_index, (unsigned)elapsed_us,
                 (unsigned)cycles, (unsigned)last_check.repetitions,
                 (unsigned)last_check.kernel_cycles,
                 (unsigned)last_check.tcdm_contentions,
                 (unsigned)last_check.load_stalls);
      ++sample_index;
    }
  }

finished:
  /* Applicable means this descriptor measured the case and passed. */
  if (applicable != NULL && status == KBENCH_OK && sample_index != 0u) {
    *applicable = 1u;
  }

  KBENCH_LOG("KBENCH|RESULT|case=%u|kernel=%s|variant=%s|status=%s|exact=%u|checksum=%u|expected=%u\n",
             (unsigned)test_case->id, kernel->name, kernel->variant,
             status_name(status),
             (unsigned)(status == KBENCH_OK ? last_check.exact : 0u),
             (unsigned)(status == KBENCH_OK ? last_check.checksum : 0u),
             (unsigned)(status == KBENCH_OK ? last_check.expected_checksum : 0u));
  KBENCH_LOG("KBENCH|RESULT|case=%u|kernel=%s|variant=%s|operations=%u|executed_operations=%u|l1_bytes=%u|l2_bytes=%u|weight_row_aligned=%u\n",
             (unsigned)test_case->id, kernel->name, kernel->variant,
             (unsigned)(status == KBENCH_OK ? last_check.operations : 0u),
             (unsigned)(status == KBENCH_OK ? last_check.executed_operations : 0u),
             (unsigned)l1_bytes,
             (unsigned)l2_bytes,
             (unsigned)(status == KBENCH_OK ? last_check.weight_row_aligned : 0u));

  if (initialized) {
    kernel->deinit(state_allocation.aligned, &runtime);
  }
  free_aligned(&l2_allocation);
  free_l1_aligned(&g_cluster, &l1_allocation);
  free_aligned(&state_allocation);
  return status;
}

static int kernel_bench_run(void) {
  const kbench_kernel_t *const *kernels;
  size_t kernel_count = 0;
  uint32_t result_count = 0u;
  int overall_status = KBENCH_OK;

  kernels = kbench_registry(&kernel_count);

  KBENCH_LOG("KBENCH|META|schema=%u\n", (unsigned)KBENCH_RESULT_SCHEMA_VERSION);
  KBENCH_LOG("KBENCH|META|chip=gap8\n");
  KBENCH_LOG("KBENCH|META|fc_hz=%u\n", (unsigned)KBENCH_FC_FREQ_HZ);
  KBENCH_LOG("KBENCH|META|cl_hz=%u\n", (unsigned)KBENCH_CL_FREQ_HZ);
  KBENCH_LOG("KBENCH|META|cases=%u\n", (unsigned)KBENCH_CASE_COUNT);
  KBENCH_LOG("KBENCH|META|kernels=%u\n", (unsigned)kernel_count);

  if (open_cluster() != 0) {
    KBENCH_LOG("KBENCH|DONE|status=cluster_open_failed\n");
    KBENCH_LOG("#DONE\n");
    return -1;
  }

  for (size_t i = 0; i < KBENCH_CASE_COUNT; ++i) {
    const kbench_case_t *test_case = &kbench_cases[i];
    uint32_t family_kernel_count = 0u;
    uint32_t applicable_count = 0u;

    KBENCH_LOG("KBENCH|CASE|id=%u|name=%s|batch=%u|cores=%u\n",
               (unsigned)test_case->id, test_case->name,
               (unsigned)test_case->problem.batch,
               (unsigned)test_case->cores);
    KBENCH_LOG("KBENCH|CASE|id=%u|m=%u|n=%u|k=%u|align=%u|offset=%u\n",
               (unsigned)test_case->id, (unsigned)test_case->problem.m,
               (unsigned)test_case->problem.n, (unsigned)test_case->problem.k,
               (unsigned)test_case->base_alignment_bytes,
               (unsigned)test_case->misalignment_offset_bytes);
    KBENCH_LOG("KBENCH|CASE|id=%u|height=%u|width=%u|cin=%u|cout=%u|kh=%u|kw=%u|sh=%u|sw=%u|pt=%u|pb=%u|pl=%u|pr=%u\n",
               (unsigned)test_case->id,
               (unsigned)test_case->problem.height,
               (unsigned)test_case->problem.width,
               (unsigned)test_case->problem.input_channels,
               (unsigned)test_case->problem.output_channels,
               (unsigned)test_case->problem.kernel_h,
               (unsigned)test_case->problem.kernel_w,
               (unsigned)test_case->problem.stride_h,
               (unsigned)test_case->problem.stride_w,
               (unsigned)test_case->problem.padding_top,
               (unsigned)test_case->problem.padding_bottom,
               (unsigned)test_case->problem.padding_left,
               (unsigned)test_case->problem.padding_right);
    KBENCH_LOG("KBENCH|CASE|id=%u|memory=%s|warmups=%u|runs=%u\n",
               (unsigned)test_case->id,
               memory_path_name(test_case->memory_path),
               (unsigned)test_case->warmups, (unsigned)test_case->runs);
    KBENCH_LOG("KBENCH|CASE|id=%u|epilogue=%s|out_mult=%u|out_shift=%d\n",
               (unsigned)test_case->id, epilogue_name(test_case->epilogue),
               (unsigned)test_case->output_multiplier,
               (int)test_case->output_shift);
    if (test_case->problem.family == KBENCH_PROBLEM_ADDITION) {
      KBENCH_LOG("KBENCH|CASE|id=%u|in1_mult=%u|in2_mult=%u\n",
                 (unsigned)test_case->id,
                 (unsigned)test_case->input1_multiplier,
                 (unsigned)test_case->input2_multiplier);
    }
    KBENCH_LOG("KBENCH|CASE|id=%u|activation=%s\n",
               (unsigned)test_case->id,
               activation_name(test_case->activation));

    for (size_t kernel_index = 0; kernel_index < kernel_count;
         ++kernel_index) {
      uint8_t applicable = 0u;
      if (kernels[kernel_index]->family != test_case->problem.family) {
        continue;
      }
      ++family_kernel_count;
      const int status =
          run_kernel_case(kernels[kernel_index], test_case, &applicable);
      applicable_count += applicable;
      result_count += applicable;
      if (status != KBENCH_OK) {
        overall_status = status;
      }
    }
    KBENCH_LOG("KBENCH|CASE_DONE|id=%u|applicable=%u|skipped=%u|status=%s\n",
               (unsigned)test_case->id, (unsigned)applicable_count,
               (unsigned)(family_kernel_count - applicable_count),
               applicable_count != 0u ? "complete" : "unsupported");
    if (applicable_count == 0u && overall_status == KBENCH_OK) {
      overall_status = KBENCH_ERR_UNSUPPORTED;
    }
  }

  KBENCH_LOG("KBENCH|READY|registered_kernels=%u\n", (unsigned)kernel_count);
  KBENCH_LOG("KBENCH|DONE|status=%s|results=%u\n",
             overall_status == KBENCH_OK
               ? "pass"
               : (overall_status == KBENCH_ERR_UNSUPPORTED ? "incomplete"
                                                         : "failed"),
           (unsigned)result_count);
  KBENCH_LOG("#DONE\n");
  return overall_status == KBENCH_OK ? 0 : -1;
}

#ifdef KBENCH_GVSOC

static void simulator_entry(void) {
  int status = kernel_bench_run();

  if (g_cluster_open) {
    pi_cluster_close(&g_cluster);
    g_cluster_open = 0;
  }
  pmsis_exit(status);
}

int main(void) { return pmsis_kickoff((void *)simulator_entry); }

#else

static void benchmark_task(void *argument) {
  (void)argument;

  /* The CRTP console is not buffered. Give the host time to reconnect after a
   * radio flash so the one-shot report is observable. */
  pi_time_wait_us((uint32_t)KBENCH_BOOT_DELAY_MS * 1000u);
  (void)kernel_bench_run();

  KBENCH_LOG("KBENCH|STACK|min_free_bytes=%u\n",
             (unsigned)(uxTaskGetStackHighWaterMark(NULL) * sizeof(StackType_t)));
  vTaskDelete(NULL);
}

static void hardware_entry(void) {
  cpxInit();
  KBENCH_LOG("-- kernel_bench GAP8 benchmark --\n");

  if (xTaskCreate(benchmark_task, "kernel_bench", KBENCH_STACK_WORDS, NULL,
                  tskIDLE_PRIORITY + 1, NULL) != pdPASS) {
    KBENCH_LOG("KBENCH|DONE|status=task_create_failed\n");
    pmsis_exit(-1);
  }
  while (1) {
    pi_yield();
  }
}

int main(void) {
  pi_bsp_init();
  __pi_pmu_voltage_set(PI_PMU_DOMAIN_FC, 1200);
  (void)pi_freq_set(PI_FREQ_DOMAIN_FC, KBENCH_FC_FREQ_HZ);
  return pmsis_kickoff((void *)hardware_entry);
}

#endif
