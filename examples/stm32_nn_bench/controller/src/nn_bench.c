// Benchmark of lib/shared/include/cf_lab/nn.h on the STM32 (motors never armed).
//
// Set the parameter nnbench.run to 1 and the app reports, on the console, for every network and format in
// nnb_generated.c (written by host/bench.py):
//   hash    FNV-1a of the outputs on the test inputs, which the host compares with its own build of the same C;
//   run     cycles of a whole network, once with the scheduler suspended (the compute alone; interrupts still run)
//           and once at the app's priority, where the stabilizer and the other tasks preempt it;
//   layer   cycles of each layer with the scheduler suspended;
//   place   cycles of one layer whose weights were copied from flash to SRAM or to the core-coupled RAM;
//   cmsis   cycles of CMSIS-NN's arm_fully_connected_s8 on an s8 layer's shape (timing only: it reads the weights
//           in their natural order, the generated arrays are permuted for cf_lab's kernels);
//   act     cycles of 128 activations (float SiLU, the int16 table).
// The stabilizer keeps running throughout: the suspended measurements delay it by at most one network.

#include <stdint.h>
#include <string.h>

#include "FreeRTOS.h"
#include "app.h"
#include "debug.h"
#include "param.h"
#include "static_mem.h"
#include "task.h"

#include "arm_nnfunctions.h"
#include "cf_cycles.h"
#include "cf_lab/fpmath.h"
#include "cf_lab/nn.h"
#include "nnb_controller.h"
#include "nnb_generated.h"

#define DEBUG_MODULE "NNB"

#define NNB_EXACT_RUNS 20
#define NNB_TASK_RUNS 200
#define NNB_LAYER_RUNS 10
#define NNB_LINE_GAP_MS 20  // the console drops lines sent faster than the radio carries them

static uint8_t paramRun;
static uint8_t paramLoad;
static uint8_t benchState;  // 0 idle, 1 running, 2 done

NO_DMA_CCM_SAFE_ZERO_INIT static uint8_t work[NNB_WORK_BYTES] __attribute__((aligned(8)));
static float out[NNB_MAX_OUT];
NO_DMA_CCM_SAFE_ZERO_INIT static uint32_t samples[NNB_TASK_RUNS];

static int8_t ramWeights[NNB_RAM_BYTES] __attribute__((aligned(8)));
NO_DMA_CCM_SAFE_ZERO_INIT static int8_t ccmWeights[NNB_CCM_BYTES] __attribute__((aligned(8)));

#define REPORT(...)                    \
  do {                                 \
    DEBUG_PRINT(__VA_ARGS__);          \
    vTaskDelay(M2T(NNB_LINE_GAP_MS));  \
  } while (0)

static uint32_t fnv1a(const void *data, uint32_t n, uint32_t h) {
  const uint8_t *p = (const uint8_t *)data;
  for (uint32_t i = 0; i < n; i++) {
    h = (h ^ p[i]) * 0x01000193u;
  }
  return h;
}

static void sortCycles(uint32_t *v, int n) {
  for (int i = 1; i < n; i++) {
    const uint32_t key = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > key) {
      v[j + 1] = v[j];
      j--;
    }
    v[j + 1] = key;
  }
}

static uint32_t minOf(const uint32_t *v, int n) {
  uint32_t m = v[0];
  for (int i = 1; i < n; i++) {
    m = v[i] < m ? v[i] : m;
  }
  return m;
}

static uint32_t maxOf(const uint32_t *v, int n) {
  uint32_t m = v[0];
  for (int i = 1; i < n; i++) {
    m = v[i] > m ? v[i] : m;
  }
  return m;
}

static void reportClock(void) {
  const uint32_t t0 = cf_cycles();
  vTaskDelay(M2T(100));
  REPORT("NNB clock cycles_per_100ms=%lu\n", (unsigned long)(cf_cycles() - t0));
}

static void reportHash(const nnb_model_t *m) {
  uint32_t h = 0x811C9DC5u;
  for (uint32_t k = 0; k < m->n_test; k++) {
    cf_nn_run(m->model, m->test_x + (size_t)k * m->model->in, out, work);
    h = fnv1a(out, m->model->out * sizeof(float), h);
  }
  REPORT("NNB hash net=%s fmt=%s h=%08lx out0=%f\n", m->net, m->fmt, (unsigned long)h, (double)out[0]);
}

static void reportRuns(const nnb_model_t *m) {
  for (int r = 0; r < NNB_EXACT_RUNS; r++) {
    vTaskDelay(M2T(3));
    vTaskSuspendAll();
    const uint32_t t0 = cf_cycles();
    cf_nn_run(m->model, m->test_x, out, work);
    samples[r] = cf_cycles() - t0;
    xTaskResumeAll();
  }
  REPORT("NNB run net=%s fmt=%s mode=suspended min=%lu max=%lu\n", m->net, m->fmt,
         (unsigned long)minOf(samples, NNB_EXACT_RUNS), (unsigned long)maxOf(samples, NNB_EXACT_RUNS));
  for (int r = 0; r < NNB_TASK_RUNS; r++) {
    vTaskDelay(M2T(2));
    const uint32_t t0 = cf_cycles();
    cf_nn_run(m->model, m->test_x, out, work);
    samples[r] = cf_cycles() - t0;
  }
  sortCycles(samples, NNB_TASK_RUNS);
  REPORT("NNB run net=%s fmt=%s mode=task min=%lu med=%lu p99=%lu max=%lu\n", m->net, m->fmt,
         (unsigned long)samples[0], (unsigned long)samples[NNB_TASK_RUNS / 2],
         (unsigned long)samples[NNB_TASK_RUNS * 99 / 100], (unsigned long)samples[NNB_TASK_RUNS - 1]);
}

// One layer of a network on placeholder activations (the cycle counts do not depend on the values), the scheduler
// suspended; weights replaces the layer's weights when not NULL. Returns the fastest of NNB_LAYER_RUNS.
static uint32_t timeLayer(const cf_nn_net_t *net, int l, const void *weights) {
  uint8_t *a = work;
  uint8_t *b = work + NNB_WORK_BYTES / 2;
  uint32_t best = UINT32_MAX;
  for (int r = 0; r < NNB_LAYER_RUNS; r++) {
    vTaskDelay(M2T(2));
    uint32_t t0 = 0, t1 = 0;
    switch (net->format) {
      case CF_NN_S8: {
        cf_nn_s8_layer_t layer = ((const cf_nn_s8_layer_t *)net->layers)[l];
        if (weights != NULL) {
          layer.w = (const int8_t *)weights;
        }
        int16_t *scratch = (int16_t *)(void *)(b + NNB_WORK_BYTES / 4);
        memset(a, 3, layer.in_pad);
        vTaskSuspendAll();
        t0 = cf_cycles();
        cf_nn_dense_s8(&layer, (const int8_t *)a, (int8_t *)b, scratch);
        t1 = cf_cycles();
        xTaskResumeAll();
        break;
      }
      case CF_NN_S16S8: {
        cf_nn_s16s8_layer_t layer = ((const cf_nn_s16s8_layer_t *)net->layers)[l];
        if (weights != NULL) {
          layer.w = (const int8_t *)weights;
        }
        for (uint32_t i = 0; i < layer.in_pad; i++) {
          ((int16_t *)(void *)a)[i] = 300;
        }
        vTaskSuspendAll();
        t0 = cf_cycles();
        cf_nn_dense_s16s8(&layer, (const int16_t *)(void *)a, (int16_t *)(void *)b);
        t1 = cf_cycles();
        xTaskResumeAll();
        break;
      }
      case CF_NN_S16: {
        cf_nn_s16_layer_t layer = ((const cf_nn_s16_layer_t *)net->layers)[l];
        if (weights != NULL) {
          layer.w = (const int16_t *)weights;
        }
        for (uint32_t i = 0; i < layer.in_pad; i++) {
          ((int16_t *)(void *)a)[i] = 300;
        }
        vTaskSuspendAll();
        t0 = cf_cycles();
        cf_nn_dense_s16(&layer, (const int16_t *)(void *)a, (int16_t *)(void *)b);
        t1 = cf_cycles();
        xTaskResumeAll();
        break;
      }
      default: {
        cf_nn_f32_layer_t layer = ((const cf_nn_f32_layer_t *)net->layers)[l];
        if (weights != NULL) {
          layer.w = (const float *)weights;
        }
        for (uint32_t i = 0; i < layer.in_pad; i++) {
          ((float *)(void *)a)[i] = 0.25f;
        }
        vTaskSuspendAll();
        t0 = cf_cycles();
        if (net->format == CF_NN_F32_FMA) {
          cf_nn_dense_f32_fma(&layer, (const float *)(void *)a, (float *)(void *)b);
        } else if (net->format == CF_NN_F32_X4) {
          cf_nn_dense_f32_x4(&layer, (const float *)(void *)a, (float *)(void *)b);
        } else {
          cf_nn_dense_f32(&layer, (const float *)(void *)a, (float *)(void *)b);
        }
        t1 = cf_cycles();
        xTaskResumeAll();
        break;
      }
    }
    best = (t1 - t0) < best ? (t1 - t0) : best;
  }
  return best;
}

static void layerShape(const cf_nn_net_t *net, int l, uint32_t *inPad, uint32_t *n, uint32_t *weightBytes) {
  const uint32_t elem = net->format == CF_NN_S8 || net->format == CF_NN_S16S8 ? 1u
                        : net->format == CF_NN_S16                           ? 2u
                                                                             : 4u;
  switch (net->format) {
    case CF_NN_S8: {
      const cf_nn_s8_layer_t *layer = &((const cf_nn_s8_layer_t *)net->layers)[l];
      *inPad = layer->in_pad;
      *n = layer->out;
      break;
    }
    case CF_NN_S16S8: {
      const cf_nn_s16s8_layer_t *layer = &((const cf_nn_s16s8_layer_t *)net->layers)[l];
      *inPad = layer->in_pad;
      *n = layer->out;
      break;
    }
    case CF_NN_S16: {
      const cf_nn_s16_layer_t *layer = &((const cf_nn_s16_layer_t *)net->layers)[l];
      *inPad = layer->in_pad;
      *n = layer->out;
      break;
    }
    default: {
      const cf_nn_f32_layer_t *layer = &((const cf_nn_f32_layer_t *)net->layers)[l];
      *inPad = layer->in_pad;
      *n = layer->out;
      break;
    }
  }
  // The stored rows: whole blocks of four (two for s16) in every layout but the row-major CF_NN_F32 (nn.h).
  const uint32_t block = net->format == CF_NN_F32 ? 1u : net->format == CF_NN_S16 ? 2u : 4u;
  *weightBytes = *inPad * ((*n + block - 1u) / block * block) * elem;
}

static const void *layerWeights(const cf_nn_net_t *net, int l) {
  switch (net->format) {
    case CF_NN_S8:
      return ((const cf_nn_s8_layer_t *)net->layers)[l].w;
    case CF_NN_S16S8:
      return ((const cf_nn_s16s8_layer_t *)net->layers)[l].w;
    case CF_NN_S16:
      return ((const cf_nn_s16_layer_t *)net->layers)[l].w;
    default:
      return ((const cf_nn_f32_layer_t *)net->layers)[l].w;
  }
}

static void reportLayers(const nnb_model_t *m) {
  for (int l = 0; l < m->model->n_layers; l++) {
    uint32_t inPad, n, bytes;
    layerShape(m->model, l, &inPad, &n, &bytes);
    REPORT("NNB layer net=%s fmt=%s l=%d in=%lu out=%lu cyc=%lu\n", m->net, m->fmt, l, (unsigned long)inPad,
           (unsigned long)n, (unsigned long)timeLayer(m->model, l, NULL));
  }
}

static void reportPlacement(const nnb_place_t *p) {
  const nnb_model_t *m = &nnb_models[p->model];
  uint32_t inPad, n, bytes;
  layerShape(m->model, p->layer, &inPad, &n, &bytes);
  const void *flash = layerWeights(m->model, p->layer);
  REPORT("NNB place net=%s fmt=%s l=%d mem=flash bytes=%lu cyc=%lu\n", m->net, m->fmt, p->layer,
         (unsigned long)bytes, (unsigned long)timeLayer(m->model, p->layer, NULL));
  if (bytes <= sizeof(ramWeights)) {
    memcpy(ramWeights, flash, bytes);
    REPORT("NNB place net=%s fmt=%s l=%d mem=sram bytes=%lu cyc=%lu\n", m->net, m->fmt, p->layer,
           (unsigned long)bytes, (unsigned long)timeLayer(m->model, p->layer, ramWeights));
  }
  if (bytes <= sizeof(ccmWeights)) {
    memcpy(ccmWeights, flash, bytes);
    REPORT("NNB place net=%s fmt=%s l=%d mem=ccm bytes=%lu cyc=%lu\n", m->net, m->fmt, p->layer,
           (unsigned long)bytes, (unsigned long)timeLayer(m->model, p->layer, ccmWeights));
  }
}

static void reportCmsis(void) {
#if NNB_CMSIS_MODEL >= 0
  const nnb_model_t *m = &nnb_models[NNB_CMSIS_MODEL];
  const cf_nn_s8_layer_t *layer = &((const cf_nn_s8_layer_t *)m->model->layers)[NNB_CMSIS_LAYER];
  int8_t *a = (int8_t *)work;
  int8_t *b = (int8_t *)work + NNB_WORK_BYTES / 2;
  memset(a, 3, layer->in_pad);
  uint32_t best = UINT32_MAX;
  for (int r = 0; r < NNB_LAYER_RUNS; r++) {
    vTaskDelay(M2T(2));
    vTaskSuspendAll();
    const uint32_t t0 = cf_cycles();
    arm_fully_connected_s8(a, layer->w, layer->in_pad, layer->out, 1, layer->in_offset, 0, layer->mult[0],
                           layer->shift[0], layer->out_offset, layer->b, b, -128, 127, NULL);
    const uint32_t t1 = cf_cycles();
    xTaskResumeAll();
    best = (t1 - t0) < best ? (t1 - t0) : best;
  }
  REPORT("NNB cmsis net=%s l=%d in=%lu out=%lu cyc=%lu\n", m->net, NNB_CMSIS_LAYER, (unsigned long)layer->in_pad,
         (unsigned long)layer->out, (unsigned long)best);
#endif
}

static void reportActivations(void) {
  float *x = (float *)(void *)work;
  for (int i = 0; i < 128; i++) {
    x[i] = -4.0f + 0.0625f * (float)i;
  }
  uint32_t best = UINT32_MAX;
  for (int r = 0; r < NNB_LAYER_RUNS; r++) {
    vTaskSuspendAll();
    const uint32_t t0 = cf_cycles();
    for (int i = 0; i < 128; i++) {
      out[i % NNB_MAX_OUT] = cf_siluf(x[i]);
    }
    const uint32_t t1 = cf_cycles();
    xTaskResumeAll();
    best = (t1 - t0) < best ? (t1 - t0) : best;
    vTaskDelay(M2T(2));
  }
  REPORT("NNB act type=silu_f32 n=128 cyc=%lu\n", (unsigned long)best);
#if NNB_LUT16_MODEL >= 0
  const cf_nn_s16s8_layer_t *layer = &((const cf_nn_s16s8_layer_t *)nnb_models[NNB_LUT16_MODEL].model->layers)[0];
  volatile int32_t sink = 0;
  best = UINT32_MAX;
  for (int r = 0; r < NNB_LAYER_RUNS; r++) {
    vTaskSuspendAll();
    const uint32_t t0 = cf_cycles();
    for (int i = 0; i < 128; i++) {
      sink += cf_nn_lut16(layer->lut, i * 500 - 32000);
    }
    const uint32_t t1 = cf_cycles();
    xTaskResumeAll();
    best = (t1 - t0) < best ? (t1 - t0) : best;
    vTaskDelay(M2T(2));
  }
  REPORT("NNB act type=lut16 n=128 cyc=%lu\n", (unsigned long)best);
#endif
}

// ---- load test: the networks as a periodic task beside the stabilizer ----
//
// Parameter nnbench.load = 1. For every load set (networks run back to back per period, as one coder-controller
// iteration would), task priority and rate, the load task runs for NNB_LOAD_MS while nnb_controller.c times the
// stabilizer loop around the stock controller; first once without load. Reported in microseconds: the load's
// execution (start to end, preemption included), how late it started against its ideal schedule, the periods it
// missed, the stabilizer's longest period and late ticks, the controller's execution time, and the IMU-to-output
// latency.

#define NNB_LOAD_MS 4000
#define NNB_LOAD_STACK 384

static const uint8_t loadPriorities[] = {1, 2, 3, 4};
static const uint16_t loadRates[] = {20, 50, 100, 200};

static StaticTask_t loadTcb;
static StackType_t loadStack[NNB_LOAD_STACK];
static TaskHandle_t loadTask;
static const nnb_load_set_t *volatile loadSet;
static volatile uint16_t loadRate;
static volatile uint8_t loadOn;
static cf_loop_stats_t loadExec;
static int32_t loadLateMin, loadLateMax;
static uint32_t loadMisses;

static void loadTaskMain(void *arg) {
  (void)arg;
  while (1) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    const TickType_t periodTicks = M2T(1000u / loadRate);
    const uint32_t period = 1000000u / loadRate * CF_CYCLES_PER_US;
    TickType_t wake = xTaskGetTickCount();
    vTaskDelayUntil(&wake, 1);
    const uint32_t base = cf_cycles();
    uint32_t n = 0;
    while (loadOn) {
      vTaskDelayUntil(&wake, periodTicks);
      n++;
      const uint32_t start = cf_cycles();
      const int32_t late = (int32_t)(start - (base + n * period));
      loadLateMin = late < loadLateMin ? late : loadLateMin;
      loadLateMax = late > loadLateMax ? late : loadLateMax;
      const nnb_load_set_t *set = loadSet;
      for (uint32_t m = 0; m < set->n; m++) {
        const nnb_model_t *model = &nnb_models[set->models[m]];
        cf_nn_run(model->model, model->test_x, out, work);
      }
      const uint32_t exec = cf_cycles() - start;
      cf_loop_span(&loadExec, exec);
      loadMisses += exec > period;
    }
  }
}

static void reportLoad(const char *name, uint8_t prio, uint16_t rate) {
  nnb_inner_stats_t inner;
  nnbInnerSnapshot(&inner);
  const uint32_t us = CF_CYCLES_PER_US;
  const uint32_t ctrlMean = inner.tick.spans ? (uint32_t)(inner.tick.spanSum / inner.tick.spans) : 0u;
  const uint32_t latMean = inner.latency.spans ? (uint32_t)(inner.latency.spanSum / inner.latency.spans) : 0u;
  REPORT("NNB load set=%s prio=%d rate=%d n=%lu exec_med=%lu exec_p99=%lu exec_max=%lu late=%ld miss=%lu "
         "ticks=%lu tick_min=%lu tick_max=%lu late10=%lu late50=%lu late100=%lu ctrl_mean=%lu ctrl_max=%lu "
         "lat_mean=%lu lat_p99=%lu lat_max=%lu\n",
         name, prio, rate, (unsigned long)loadExec.spans, (unsigned long)(cf_loop_span_quantile(&loadExec, 0.5f) / us),
         (unsigned long)(cf_loop_span_quantile(&loadExec, 0.99f) / us), (unsigned long)(loadExec.spanMax / us),
         (long)(loadExec.spans ? (loadLateMax - loadLateMin) / (int32_t)us : 0), (unsigned long)loadMisses,
         (unsigned long)inner.tick.ticks, (unsigned long)(inner.tick.periodMin / us),
         (unsigned long)(inner.tick.periodMax / us), (unsigned long)inner.tick.late10, (unsigned long)inner.tick.late50,
         (unsigned long)inner.tick.late100, (unsigned long)(ctrlMean / us), (unsigned long)(inner.tick.spanMax / us),
         (unsigned long)(latMean / us), (unsigned long)(cf_loop_span_quantile(&inner.latency, 0.99f) / us),
         (unsigned long)(inner.latency.spanMax / us));
}

static void runLoadTest(void) {
  REPORT("NNB load-begin sets=%d seconds=%d\n", NNB_N_LOAD_SETS, NNB_LOAD_MS / 1000);
  cf_loop_stats_reset(&loadExec, 0, 50u * CF_CYCLES_PER_US);
  nnbInnerReset();
  vTaskDelay(M2T(NNB_LOAD_MS));
  reportLoad("none", 0, 0);
  for (int s = 0; s < NNB_N_LOAD_SETS; s++) {
    for (uint32_t p = 0; p < sizeof(loadPriorities); p++) {
      for (uint32_t r = 0; r < sizeof(loadRates) / sizeof(loadRates[0]); r++) {
        loadSet = &nnb_load_sets[s];
        loadRate = loadRates[r];
        cf_loop_stats_reset(&loadExec, 1000000u / loadRate * CF_CYCLES_PER_US, 50u * CF_CYCLES_PER_US);
        loadLateMin = INT32_MAX;
        loadLateMax = INT32_MIN;
        loadMisses = 0;
        vTaskPrioritySet(loadTask, loadPriorities[p]);
        nnbInnerReset();
        loadOn = 1;
        xTaskNotifyGive(loadTask);
        vTaskDelay(M2T(NNB_LOAD_MS));
        loadOn = 0;
        vTaskDelay(M2T(100));  // the last iteration finishes
        reportLoad(nnb_load_sets[s].name, loadPriorities[p], loadRate);
      }
    }
  }
  REPORT("NNB load-done\n");
}

static void runBench(void) {
  REPORT("NNB begin models=%d\n", NNB_N_MODELS);
  reportClock();
  for (int i = 0; i < NNB_N_MODELS; i++) {
    reportHash(&nnb_models[i]);
  }
  for (int i = 0; i < NNB_N_MODELS; i++) {
    reportRuns(&nnb_models[i]);
    reportLayers(&nnb_models[i]);
  }
  for (int i = 0; i < NNB_N_PLACES; i++) {
    reportPlacement(&nnb_places[i]);
  }
  reportCmsis();
  reportActivations();
  REPORT("NNB done\n");
}

void appMain(void) {
  cf_cycles_init();
  loadTask = xTaskCreateStatic(loadTaskMain, "NNBLOAD", NNB_LOAD_STACK, NULL, 1, loadStack, &loadTcb);
  while (1) {
    vTaskDelay(M2T(100));
    if (paramRun) {
      paramRun = 0;
      benchState = 1;
      runBench();
      benchState = 2;
    }
    if (paramLoad) {
      paramLoad = 0;
      benchState = 1;
      runLoadTest();
      benchState = 2;
    }
  }
}

PARAM_GROUP_START(nnbench)
PARAM_ADD(PARAM_UINT8, run, &paramRun)
PARAM_ADD(PARAM_UINT8, load, &paramLoad)
PARAM_ADD(PARAM_UINT8, inner, &nnbInnerIndi)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, state, &benchState)
PARAM_GROUP_STOP(nnbench)
