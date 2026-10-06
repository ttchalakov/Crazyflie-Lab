// scheduler_bench: the sequential and the pipelined scheduler of lib/gap8/sched,
// run back to back on one boot of one image, so they share every byte of layout.
//
// The network is the generated sched_net.h (host/sched_net.py). A state network
// gets synthetic states at STATE_RATE_HZ; an image network runs on the camera's
// frames. Each phase prints its measurements, prefixed SB|, to the radio
// console.
#include <stdint.h>
#include <string.h>

#include "FreeRTOS.h"
#include "task.h"

#include "pmsis.h"
#include "bsp/bsp.h"

#include "cpx.h"
#include "com.h"
#include "cpx_config.h"
#include "sched.h"
#include "sched_build.h"
#include "sched_net.h"

#ifndef PHASE_MS
#define PHASE_MS 20000u
#endif
#ifndef ROUNDS
#define ROUNDS 1
#endif
#ifndef STATE_RATE_HZ
#define STATE_RATE_HZ 100u
#endif
// Seconds after setup for the host to join the access point and attach.
#ifndef SETTLE_MS
#define SETTLE_MS 8000u
#endif

static volatile uint32_t outputSum;
static volatile uint32_t outputs;

static void onOutput(const sched_output_t *output, void *context) {
  (void)context;
  const uint8_t *bytes = (const uint8_t *)output->data;
  uint32_t sum = 0;
  for (uint32_t i = 0; i < output->bytes; i++) sum += bytes[i];
  outputSum += sum;
  outputs++;
}

// Reports go to the radio console: over WiFi they would be UDP packets on a
// link a raw stream saturates, and they were lost there. CPX queues them behind
// the frames instead.
static CPXConsoleTarget_t target(void) { return LOG_TO_CRTP; }

// Synthetic states, as a full-state policy would get them from the STM32.
static volatile int generating;
static void stateTask(void *argument) {
  (void)argument;
  static uint8_t state[SCHED_NET_STATE_BYTES > 0 ? SCHED_NET_STATE_BYTES : 1];
  uint32_t x = 0x12345678u;
  TickType_t wake = xTaskGetTickCount();
  const TickType_t period = pdMS_TO_TICKS(1000u / STATE_RATE_HZ) ? pdMS_TO_TICKS(1000u / STATE_RATE_HZ) : 1;
  while (1) {
    vTaskDelayUntil(&wake, period);
    if (!generating) continue;
    for (uint32_t i = 0; i < sizeof(state); i++) {
      x ^= x << 13;
      x ^= x >> 17;
      x ^= x << 5;
      state[i] = (uint8_t)x;
    }
    (void)sched_post_state(state, SCHED_NET_STATE_BYTES, 0u);
  }
}

static void report(const char *kind, uint32_t round, const sched_stats_t *st) {
  const CPXConsoleTarget_t to = target();
  const uint32_t ms = st->elapsed_us / 1000u ? st->elapsed_us / 1000u : 1u;
  cpxPrintToConsole(to, "SB|phase k=%s r=%lu ms=%lu fr=%lu sent=%lu skip=%lu inf=%lu\n", kind,
                    (unsigned long)round, (unsigned long)ms, (unsigned long)st->frames,
                    (unsigned long)st->frames_sent, (unsigned long)st->frames_not_sent,
                    (unsigned long)st->inferences);
  cpxPrintToConsole(to, "SB|lat k=%s p50=%lu p90=%lu p99=%lu max=%lu\n", kind,
                    (unsigned long)sched_hist_percentile(&st->latency, 50),
                    (unsigned long)sched_hist_percentile(&st->latency, 90),
                    (unsigned long)sched_hist_percentile(&st->latency, 99),
                    (unsigned long)st->latency.max_us);
  cpxPrintToConsole(to, "SB|inf k=%s p50=%lu max=%lu per50=%lu per99=%lu enc50=%lu encmax=%lu\n",
                    kind, (unsigned long)sched_hist_percentile(&st->inference, 50),
                    (unsigned long)st->inference.max_us,
                    (unsigned long)sched_hist_percentile(&st->period, 50),
                    (unsigned long)sched_hist_percentile(&st->period, 99),
                    (unsigned long)sched_hist_percentile(&st->encode, 50),
                    (unsigned long)st->encode.max_us);
  cpxPrintToConsole(to, "SB|wait k=%s p50=%lu p90=%lu p99=%lu max=%lu send50=%lu sendmax=%lu\n",
                    kind, (unsigned long)sched_hist_percentile(&st->wait, 50),
                    (unsigned long)sched_hist_percentile(&st->wait, 90),
                    (unsigned long)sched_hist_percentile(&st->wait, 99),
                    (unsigned long)st->wait.max_us,
                    (unsigned long)sched_hist_percentile(&st->send, 50),
                    (unsigned long)st->send.max_us);
  // clpm: cluster busy, per mille.
  cpxPrintToConsole(to, "SB|res k=%s clpm=%lu st=%lu rep=%lu yld=%lu drop=%lu encfail=%lu clerr=%lu vf=%lu/%lu\n",
                    kind, (unsigned long)(st->cluster_busy_us / ms),
                    (unsigned long)st->states_posted, (unsigned long)st->states_replaced,
                    (unsigned long)st->encode_yields, (unsigned long)st->camera_drops,
                    (unsigned long)st->encode_failures, (unsigned long)st->cluster_errors,
                    (unsigned long)st->verify_failures,
                    (unsigned long)st->verify_runs);
}

// Backward steps of the SDK's microsecond clock and of the scheduler's over
// CLOCK_CHECK_MS of back-to-back reads (sched_time_us() exists because the
// SDK's steps back a tick when the SysTick timer wraps mid-read).
#define CLOCK_CHECK_MS 500u
static void clockCheck(void) {
  uint32_t sdkBack = 0, schedBack = 0, reads = 0;
  uint32_t sdkLast = pi_time_get_us(), schedLast = sched_time_us();
  const uint32_t end = schedLast + CLOCK_CHECK_MS * 1000u;
  while ((int32_t)(schedLast - end) < 0) {
    const uint32_t sdk = pi_time_get_us();
    const uint32_t now = sched_time_us();
    if ((int32_t)(sdk - sdkLast) < 0) sdkBack++;
    if ((int32_t)(now - schedLast) < 0) schedBack++;
    sdkLast = sdk;
    schedLast = now;
    reads++;
  }
  cpxPrintToConsole(LOG_TO_CRTP, "SB|clock reads=%lu sdk_back=%lu sched_back=%lu\n",
                    (unsigned long)reads, (unsigned long)sdkBack, (unsigned long)schedBack);
}

static void benchTask(void *argument) {
  (void)argument;
  sched_config_t config;
  sched_config_init(&config);
  config.network = &SCHED_NETWORK;
  config.on_output = onOutput;

  const int status = sched_open(&config);
  if (status != SCHED_OK) {
    while (1) {
      const char *what;
      const int detail = sched_error_detail(&what);
      cpxPrintToConsole(LOG_TO_CRTP, "SB|fail open=%d at=%s detail=%d\n", status, what, detail);
      vTaskDelay(pdMS_TO_TICKS(1000));
    }
  }
  uint32_t l2Total, l2Largest;
  sched_l2_free(&l2Total, &l2Largest);
  cpxPrintToConsole(LOG_TO_CRTP, "SB|open net=%s input=%d arena=%lu rate=%u\n", SCHED_NETWORK.name,
                    (int)SCHED_NETWORK.input, (unsigned long)SCHED_NETWORK.arena_bytes,
                    (unsigned)STATE_RATE_HZ);
  cpxPrintToConsole(LOG_TO_CRTP, "SB|mem l2_free=%lu l2_largest=%lu txq=%u\n",
                    (unsigned long)l2Total, (unsigned long)l2Largest, (unsigned)TXQ_SIZE);
  // Where the L2 heap went: the camera buffers, the JPEG encoder's buffers, the
  // CPX queues (send, urgent 8, receive 5; com.c), DORY's arena.
  camera_stream_info_t camera;
  (void)camera_stream_get_info(sched_camera(), &camera);
  uint32_t jpegBytes = 0;
#if SCHED_STREAM_FORMAT == SCHED_FORMAT_JPEG
  jpeg_pipeline_memory_plan_t plan;
  if (jpeg_pipeline_get_plan(sched_jpeg(), &plan) == JPEG_PIPELINE_OK) jpegBytes = plan.total_l2_bytes;
#endif
  cpxPrintToConsole(LOG_TO_CRTP, "SB|l2 cam=%lu jpeg=%lu cpx=%lu arena=%lu free=%lu\n",
                    (unsigned long)(camera.capture_buffer_count * camera.frame_size_bytes),
                    (unsigned long)jpegBytes,
                    (unsigned long)((TXQ_SIZE + TXQ_URGENT_SIZE + RXQ_SIZE) * sizeof(packet_t)),
                    (unsigned long)SCHED_NETWORK.arena_bytes, (unsigned long)l2Total);
  clockCheck();
  if (SCHED_NETWORK.input == SCHED_INPUT_STATE) {
    xTaskCreate(stateTask, "state", configMINIMAL_STACK_SIZE * 2, NULL, tskIDLE_PRIORITY + 1, NULL);
  }
  vTaskDelay(pdMS_TO_TICKS(SETTLE_MS));

  static sched_stats_t st;
  for (uint32_t round = 0; round < ROUNDS; round++) {
    static const sched_kind_t kinds[2] = {SCHED_SEQUENTIAL, SCHED_PIPELINED};
    static const char *names[2] = {"seq", "pipe"};
    for (int i = 0; i < 2; i++) {
      if (sched_start(kinds[i]) != SCHED_OK) {
        cpxPrintToConsole(target(), "SB|fail start=%s\n", names[i]);
        continue;
      }
      generating = 1;
      vTaskDelay(pdMS_TO_TICKS(PHASE_MS));
      generating = 0;
      (void)sched_get_stats(&st, true);
      (void)sched_stop();
      report(names[i], round, &st);
    }
  }
  while (1) {
    cpxPrintToConsole(target(), "SB|done outputs=%lu sum=%lu\n", (unsigned long)outputs,
                      (unsigned long)outputSum);
    vTaskDelay(pdMS_TO_TICKS(2000));
  }
}

static void start(void) {
  if (xTaskCreate(benchTask, "bench", configMINIMAL_STACK_SIZE * 6, NULL, tskIDLE_PRIORITY + 1,
                  NULL) != pdPASS) {
    pmsis_exit(-1);
  }
  // No spinning: an idle FC answers HyperBus interrupts sooner.
  vTaskDelete(NULL);
}

int main(void) {
  pi_bsp_init();
  __pi_pmu_voltage_set(PI_PMU_DOMAIN_FC, 1200);
  pi_freq_set(PI_FREQ_DOMAIN_FC, SCHED_FC_HZ);
  return pmsis_kickoff((void *)start);
}
