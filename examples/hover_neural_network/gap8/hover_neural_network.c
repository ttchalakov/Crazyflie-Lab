// Neural hover on the AI-deck: answer the STM32's 100 Hz state requests with
// the NEMO/DORY policy, and stream QVGA JPEG video beside it.
//
// lib/gap8/sched does the scheduling (see its README): one task owns the
// cluster and serves a posted state ahead of the video, between the slices of a
// JPEG encode, so a request waits for at most one slice. This file is the CPX
// plumbing and the boundary between the STM32's floats and the integer network.
// The camera and video settings are in the Makefile.

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "FreeRTOS.h"
#include "task.h"

#include "bsp/bsp.h"
#include "pmsis.h"

#include "cpx.h"
#include "hover_nn_protocol.h"
#include "policy_network.h"
#include "sched.h"
#include "sched_build.h"

// Deployment ABI between the trained policy and this app.
//
// DORY's network is an integer graph: it eats a uint8[11] and returns an
// int32[4], neither of which is a physical quantity. These two functions are the
// whole boundary between that and a drone.
//
// Every constant below is a DESIGN constant, derived from the top of
// model/hover_policy.py -- MASS_KG, THRUST_DELTA_FRACTION, MAX_BODY_RATE_RAD_S,
// STATE_LIMITS -- and byte-identical on every retrain. Change one of those,
// change the matching line here, rebuild. Retraining alone never invalidates it.
//
// The one value that DOES change with the weights is NEMO's output quantum. The
// export drops it beside its ONNX as build/dory/policy/nemo/output_epsilon.txt
// and the Makefile passes it in as -D, so it cannot be pasted wrong and cannot
// go stale.
#ifndef HNN_POLICY_OUTPUT_EPSILON
#error "HNN_POLICY_OUTPUT_EPSILON undefined -- build via the GAP8 Makefile after a NEMO/DORY deploy"
#endif

#define STATE_COUNT CF_LAB_HNN_STATE_FLOATS
#define ACTION_COUNT CF_LAB_HNN_CONTROL_FLOATS

// STATE_LIMITS in model/hover_policy.py. The last five channels are components
// of unit vectors, so 1.0 is exact rather than generous.
//     ex    ey    ez    vx    vy    vz   upx   upy   upz   sin   cos
static const float stateLimits[STATE_COUNT] = {
  0.6f, 0.6f, 0.5f, 1.2f, 1.2f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

// ACTION_OFFSET / ACTION_SCALE: hover thrust MASS_KG * g, and a +/-50% thrust
// band with the +/-4, 4, 2 rad/s body-rate limits.
static const float actionOffset[ACTION_COUNT] = {0.50798188f, 0.0f, 0.0f, 0.0f};
static const float actionScale[ACTION_COUNT] = {0.25399094f, 4.0f, 4.0f, 2.0f};

// The same mapping as _encode_state in hover_policy.py, rounded to uint8. If
// these two ever disagree the network sees a differently-scaled world than it
// trained in, and nothing anywhere reports it. False for a state that is not
// finite.
static bool encodeState(const float state[STATE_COUNT], uint8_t encoded[STATE_COUNT]) {
  for (size_t i = 0; i < STATE_COUNT; ++i) {
    if (!isfinite(state[i])) return false;
    float normalized = state[i] / stateLimits[i];
    if (normalized < -1.0f) normalized = -1.0f;
    if (normalized > 1.0f) normalized = 1.0f;
    encoded[i] = (uint8_t)(0.5f * (normalized + 1.0f) * 255.0f + 0.5f);
  }
  return true;
}

// int32 accumulator -> [thrust N, wx, wy, wz rad/s].
static void decodeControl(const int32_t raw[ACTION_COUNT], float control[ACTION_COUNT]) {
  for (size_t i = 0; i < ACTION_COUNT; ++i) {
    float normalized = (float)raw[i] * HNN_POLICY_OUTPUT_EPSILON;
    if (normalized < -1.0f) normalized = -1.0f;
    if (normalized > 1.0f) normalized = 1.0f;
    control[i] = actionOffset[i] + actionScale[i] * normalized;
  }
}

// The DORY network, for lib/gap8/sched. The PULP.GAP8_L2 target: the weights
// are resident L2 arrays, and the input is read in place, so the arena holds
// two consecutive layer outputs: 128 bytes for 11-64-64-4. 1 KB covers any
// two-layer MLP whose weights fit L2 (up to ~450 wide).
static const sched_network_t policyNetwork = {
    .name = "hover_policy",
    .input = SCHED_INPUT_STATE,
    .input_bytes = STATE_COUNT,
    .output_bytes = ACTION_COUNT * sizeof(int32_t),
    .arena_bytes = 1024u,
    .three_level = 0,
    .initial_dir = 1,
    .cluster_entry = policy_network_run_cluster,
    .cluster_stack = 3500,       // DORY's own, from policy_network.c
    .cluster_slave_stack = 3400,
};

#define REPLY_TIMEOUT_MS 5u

// Called by the scheduler after each inference: decode and reply. The urgent
// CPX class drains the reply ahead of the video's packets.
static void onOutput(const sched_output_t *output, void *context) {
  (void)context;
  static CPXPacket_t packet;
  cfLabHoverNnControl_t reply = {
      .version = CF_LAB_HNN_PROTOCOL_VERSION,
      .msgType = CF_LAB_HNN_MSG_CONTROL,
      .stateTimestampMs = output->tag,
  };
  decodeControl((const int32_t *)output->data, reply.control);
  cpxInitRoute(CPX_T_GAP8, CPX_T_STM32, CPX_F_APP, &packet.route);
  memcpy(packet.data, &reply, sizeof(reply));
  packet.dataLength = sizeof(reply);
  (void)cpxSendPacketUrgent(&packet, pdMS_TO_TICKS(REPLY_TIMEOUT_MS));
}

// Posts each valid request to the scheduler, tagged with its timestamp so the
// reply can echo it. A newer request replaces one still waiting.
static void stateTask(void *argument) {
  (void)argument;
  static CPXPacket_t packet;
  while (1) {
    cpxReceivePacketBlocking(CPX_F_APP, &packet);
    cfLabHoverNnState_t request;
    if (packet.route.source != CPX_T_STM32 || packet.dataLength != sizeof(request)) continue;
    memcpy(&request, packet.data, sizeof(request));
    if (request.version != CF_LAB_HNN_PROTOCOL_VERSION || request.msgType != CF_LAB_HNN_MSG_STATE) {
      continue;
    }
    // Ties the video's frame times to the STM32 clock (see lib/gap8/sched).
    sched_clock_anchor(request.stateTimestampMs);
    uint8_t input[STATE_COUNT];
    if (!encodeState(request.state, input)) continue;  // no reply: the STM32 keeps the PID
    (void)sched_post_state(input, sizeof(input), request.stateTimestampMs);
  }
}

static void start(void) {
  sched_config_t config;
  sched_config_init(&config);
  config.network = &policyNetwork;
  config.on_output = onOutput;
  int status = sched_open(&config);  // also starts CPX, after the camera
  if (status == SCHED_OK) {
    cpxEnableFunction(CPX_F_APP);
    status = sched_start(config.kind);
  }
  // At the CPX tasks' priority, so a request reaches the scheduler at once.
  if (status == SCHED_OK && xTaskCreate(stateTask, "hnn_state", configMINIMAL_STACK_SIZE * 2,
                                        NULL, tskIDLE_PRIORITY + 2, NULL) != pdPASS) {
    status = SCHED_ERR_RTOS;
  }
  if (status == SCHED_OK) {
    cpxPrintToConsole(LOG_TO_CRTP, "hover_nn: ready\n");
  } else {
    const char *what;
    const int detail = sched_error_detail(&what);
    cpxPrintToConsole(LOG_TO_CRTP, "hover_nn: setup failed (%d, %s %d)\n", status, what, detail);
  }
  vTaskSuspend(NULL);
}

int main(void) {
  pi_bsp_init();
  __pi_pmu_voltage_set(PI_PMU_DOMAIN_FC, 1200);
  pi_freq_set(PI_FREQ_DOMAIN_FC, SCHED_FC_HZ);
  return pmsis_kickoff((void *)start);
}
