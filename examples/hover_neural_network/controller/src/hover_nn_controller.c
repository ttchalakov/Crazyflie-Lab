// Neural CTBR hover controller for a Crazyflie 2.1 Brushless.
//
// This file is a communication layer and nothing else. Two directions:
//
//   1. Every 10 ms, send the 11-float yaw-invariant observation to the AI-deck.
//   2. When a fresh [thrust N, wx, wy, wz rad/s] comes back and hnn.enable=1,
//      fly it. Otherwise the stock PID flies the real setpoint, unchanged.
//
// The policy commands a setpoint, not motor torques, so the vehicle's own rate
// loop tracks it: the CTBR command becomes a rate setpoint and the firmware's
// tuned attitude-rate PID does the rest at its normal 500 Hz. That is the whole
// point of a 100 Hz neural controller over a link with milliseconds of dead
// time, and it means this file adds no gains of its own.

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "app.h"
#include "controller.h"
#include "controller_pid.h"
#include "cpx.h"
#include "cpx_internal_router.h"
#include "FreeRTOS.h"
#include "log.h"
#include "param.h"
#include "platform_defaults.h"  // THRUST_MAX, N per motor
#include "task.h"

#include "hover_nn_protocol.h"

#define HNN_PERIOD_TICKS M2T(10)    // 100 Hz
#define HNN_MAX_AGE_MS 30u          // hold a command through two missed replies

// Back-off when nothing is answering. A deck that is absent, rebooting, or
// being flashed must not be streamed at 100 Hz: the GAP8's own radio flash
// travels over this same STM32->deck link, and a steady 100 Hz of state packets
// starves it -- the write stalls at 0% and the deck is left in its bootloader.
// So after HNN_LINK_IDLE_MS of silence we drop to a slow probe and stay there
// until replies resume, at which point the next reply restores full rate on its
// own. This is not a handshake: nothing is negotiated and no readiness is
// announced, it is only a rate limit on talking to something that is not there.
//
// 300 ms, not seconds: a flash of a deck that is currently replying starts with
// the link at full rate, and every millisecond before the back-off engages is
// written into the image. 300 ms costs nothing operationally -- the policy
// already refuses any command older than HNN_MAX_AGE_MS, so 30 consecutive
// missed replies is a dead link by any measure.
#define HNN_LINK_IDLE_MS 300u
#define HNN_PROBE_DIVIDER 100u      // 100 Hz / 100 = 1 Hz while silent

// Must match ACTION_SCALE / MAX_BODY_RATE_RAD_S in model/hover_policy.py: the
// last check before a corrupted float reaches the motors.
#define HNN_MAX_THRUST_N 0.8f
#define HNN_MAX_RATE_XY_RAD_S 4.0f
#define HNN_MAX_RATE_Z_RAD_S 2.0f

#define RAD_TO_DEG (180.0f / 3.14159265358979f)

// Controller step -> app task.
static float observation[CF_LAB_HNN_STATE_FLOATS];
static uint32_t observationMs;
static bool observationValid;

// CPX callback -> controller step.
static float command[CF_LAB_HNN_CONTROL_FLOATS];
static uint32_t commandMs;
static bool commandValid;

static uint8_t paramEnable;   // 0 at boot: the PID flies until asked otherwise
static uint8_t paramActive;   // 1 while the network is actually flying
static uint32_t rxCount;

static CPXPacket_t txPacket;

static float clampf(float value, float low, float high) {
  return value < low ? low : (value > high ? high : value);
}

// ctbr.observe() in C. Position error and velocity rotated into the body frame,
// attitude as world-up-in-body, heading as (sin, cos) against a zero reference.
// This must agree with the Python or the network sees a different world than it
// trained in, so it is written from the same rotation matrix, in the same order.
static void buildObservation(const state_t *state, const point_t *reference,
                             float out[CF_LAB_HNN_STATE_FLOATS]) {
  const float w = state->attitudeQuaternion.w;
  const float x = state->attitudeQuaternion.x;
  const float y = state->attitudeQuaternion.y;
  const float z = state->attitudeQuaternion.z;

  // Body-to-world rotation, exactly ctbr.quat_to_rotation().
  const float r00 = 1.0f - 2.0f * (y * y + z * z);
  const float r01 = 2.0f * (x * y - w * z);
  const float r02 = 2.0f * (x * z + w * y);
  const float r10 = 2.0f * (x * y + w * z);
  const float r11 = 1.0f - 2.0f * (x * x + z * z);
  const float r12 = 2.0f * (y * z - w * x);
  const float r20 = 2.0f * (x * z - w * y);
  const float r21 = 2.0f * (y * z + w * x);
  const float r22 = 1.0f - 2.0f * (x * x + y * y);

  const float dx = state->position.x - reference->x;
  const float dy = state->position.y - reference->y;
  const float dz = state->position.z - reference->z;

  // World -> body is the transpose, so each body component reads down a column.
  out[0] = r00 * dx + r10 * dy + r20 * dz;
  out[1] = r01 * dx + r11 * dy + r21 * dz;
  out[2] = r02 * dx + r12 * dy + r22 * dz;

  out[3] = r00 * state->velocity.x + r10 * state->velocity.y + r20 * state->velocity.z;
  out[4] = r01 * state->velocity.x + r11 * state->velocity.y + r21 * state->velocity.z;
  out[5] = r02 * state->velocity.x + r12 * state->velocity.y + r22 * state->velocity.z;

  // World up expressed in the body frame: the third row of the same matrix.
  out[6] = r20;
  out[7] = r21;
  out[8] = r22;

  // Heading error against yaw_ref = 0, so sin_err = -sin_yaw and cos_err = cos_yaw.
  const float sinRaw = 2.0f * (w * z + x * y);
  const float cosRaw = 1.0f - 2.0f * (y * y + z * z);
  float norm = sqrtf(sinRaw * sinRaw + cosRaw * cosRaw);
  if (norm < 1.0e-9f) {
    norm = 1.0e-9f;
  }
  out[9] = -sinRaw / norm;
  out[10] = cosRaw / norm;
}

static void onCpxPacket(const CPXPacket_t *packet) {
  if (packet == NULL || packet->route.source != CPX_T_GAP8 ||
      packet->route.function != CPX_F_APP ||
      packet->dataLength != sizeof(cfLabHoverNnControl_t)) {
    return;
  }

  cfLabHoverNnControl_t reply;
  memcpy(&reply, packet->data, sizeof(reply));
  if (reply.version != CF_LAB_HNN_PROTOCOL_VERSION ||
      reply.msgType != CF_LAB_HNN_MSG_CONTROL) {
    return;
  }
  for (size_t i = 0; i < CF_LAB_HNN_CONTROL_FLOATS; ++i) {
    if (!isfinite(reply.control[i])) {
      return;
    }
  }

  taskENTER_CRITICAL();
  command[CF_LAB_HNN_CONTROL_THRUST] =
    clampf(reply.control[CF_LAB_HNN_CONTROL_THRUST], 0.0f, HNN_MAX_THRUST_N);
  command[CF_LAB_HNN_CONTROL_RATE_X] =
    clampf(reply.control[CF_LAB_HNN_CONTROL_RATE_X], -HNN_MAX_RATE_XY_RAD_S, HNN_MAX_RATE_XY_RAD_S);
  command[CF_LAB_HNN_CONTROL_RATE_Y] =
    clampf(reply.control[CF_LAB_HNN_CONTROL_RATE_Y], -HNN_MAX_RATE_XY_RAD_S, HNN_MAX_RATE_XY_RAD_S);
  command[CF_LAB_HNN_CONTROL_RATE_Z] =
    clampf(reply.control[CF_LAB_HNN_CONTROL_RATE_Z], -HNN_MAX_RATE_Z_RAD_S, HNN_MAX_RATE_Z_RAD_S);
  commandMs = reply.stateTimestampMs;
  commandValid = true;
  rxCount++;
  taskEXIT_CRITICAL();
}

void appMain(void) {
  cpxRegisterAppMessageHandler(onCpxPacket);

  TickType_t lastWake = xTaskGetTickCount();
  uint32_t ticks = 0;
  while (1) {
    vTaskDelayUntil(&lastWake, HNN_PERIOD_TICKS);
    ticks++;

    bool answering;
    taskENTER_CRITICAL();
    answering = commandValid &&
      (uint32_t)(T2M(xTaskGetTickCount()) - commandMs) <= HNN_LINK_IDLE_MS;
    taskEXIT_CRITICAL();
    if (!answering && (ticks % HNN_PROBE_DIVIDER) != 0u) {
      continue;
    }

    cfLabHoverNnState_t request = {
      .version = CF_LAB_HNN_PROTOCOL_VERSION,
      .msgType = CF_LAB_HNN_MSG_STATE,
    };
    bool valid;
    taskENTER_CRITICAL();
    valid = observationValid;
    request.stateTimestampMs = observationMs;
    memcpy(request.state, observation, sizeof(request.state));
    taskEXIT_CRITICAL();
    if (!valid) {
      continue;
    }

    cpxInitRoute(CPX_T_STM32, CPX_T_GAP8, CPX_F_APP, &txPacket.route);
    txPacket.route.lastPacket = true;  // the STM32 cpxInitRoute leaves this unset
    memcpy(txPacket.data, &request, sizeof(request));
    txPacket.dataLength = sizeof(request);
    cpxSendPacketBlockingTimeout(&txPacket, 0);
  }
}

void controllerOutOfTreeInit(void) {
  controllerPidInit();
}

bool controllerOutOfTreeTest(void) {
  return controllerPidTest();
}

void controllerOutOfTree(
  control_t *control,
  const setpoint_t *setpoint,
  const sensorData_t *sensors,
  const state_t *state,
  const stabilizerStep_t stabilizerStep
) {
  // Direction 1: publish what the network should see. The hover target is the
  // setpoint the host is already commanding, so handover carries no step.
  float latest[CF_LAB_HNN_STATE_FLOATS];
  buildObservation(state, &setpoint->position, latest);

  float latestCommand[CF_LAB_HNN_CONTROL_FLOATS];
  bool fresh;
  taskENTER_CRITICAL();
  memcpy(observation, latest, sizeof(observation));
  observationMs = T2M(xTaskGetTickCount());
  observationValid = true;
  // A stale command is not a slightly old hover, it is a body-rate setpoint the
  // vehicle will keep tracking into a wall. Age it or do not fly it.
  fresh = commandValid && (uint32_t)(observationMs - commandMs) <= HNN_MAX_AGE_MS;
  memcpy(latestCommand, command, sizeof(latestCommand));
  taskEXIT_CRITICAL();

  // Direction 2: fly it, or hand the real setpoint back to the PID untouched.
  if (!paramEnable || !fresh) {
    paramActive = 0;
    controllerPid(control, setpoint, sensors, state, stabilizerStep);
    return;
  }
  paramActive = 1;

  // Collective Newtons -> the per-motor units the legacy mixer sums, using the
  // platform's own thrust constant rather than a number invented here.
  // Firmware pitch is negated relative to the FLU frame the policy is trained in.
  setpoint_t ctbr = {
    .mode = { .x = modeDisable, .y = modeDisable, .z = modeDisable,
              .roll = modeVelocity, .pitch = modeVelocity, .yaw = modeVelocity },
    .thrust = 0.25f * latestCommand[CF_LAB_HNN_CONTROL_THRUST] / THRUST_MAX * 65535.0f,
    .attitudeRate = {
      .roll = latestCommand[CF_LAB_HNN_CONTROL_RATE_X] * RAD_TO_DEG,
      .pitch = -latestCommand[CF_LAB_HNN_CONTROL_RATE_Y] * RAD_TO_DEG,
      .yaw = latestCommand[CF_LAB_HNN_CONTROL_RATE_Z] * RAD_TO_DEG,
    },
  };
  controllerPid(control, &ctbr, sensors, state, stabilizerStep);
}

PARAM_GROUP_START(hnn)
PARAM_ADD(PARAM_UINT8, enable, &paramEnable)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, active, &paramActive)
PARAM_GROUP_STOP(hnn)

LOG_GROUP_START(hnn)
LOG_ADD(LOG_UINT8, active, &paramActive)
LOG_ADD(LOG_UINT32, rxCount, &rxCount)
LOG_ADD(LOG_FLOAT, thrustN, &command[CF_LAB_HNN_CONTROL_THRUST])
LOG_ADD(LOG_FLOAT, wx, &command[CF_LAB_HNN_CONTROL_RATE_X])
LOG_ADD(LOG_FLOAT, wy, &command[CF_LAB_HNN_CONTROL_RATE_Y])
LOG_ADD(LOG_FLOAT, wz, &command[CF_LAB_HNN_CONTROL_RATE_Z])
LOG_GROUP_STOP(hnn)
