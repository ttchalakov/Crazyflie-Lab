#include <math.h>
#include <stdint.h>

#include "app.h"
#define DEBUG_MODULE "TBENCH"
#include "debug.h"
#include "FreeRTOS.h"
#include "task.h"
#include "log.h"
#include "cpx_bridge_stm32.h"
#include "gap8_sample.h"

// Synthetic telemetry signals with closed-form ground truth, updated at 1 kHz.
// Every logged sample can be verified on the host from `tick` alone:
//   phase  = frac(tick / 1000)          (1 Hz period)
//   sin1   = sin(2*pi*phase),  cos1 = cos(2*pi*phase)
//   saw    = phase
//   ramp   = tick / 1000                (seconds since app start)
//   square = +1 for phase < 0.5 else -1
//   tri    = 4*phase-1 rising, 3-4*phase falling
// `blk` in the second log group echoes `tick` so the host can check that both
// blocks sample the same timeline.

#define PI2_F 6.28318530718f

static uint32_t tick;
static float sin1;
static float cos1;
static float saw;
static uint32_t blk;
static float ramp;
static float square;
static float tri;

static cf_cpx_bridge_t bridge;
static gap8_sample_t gap8;

static void onGap8Packet(const CPXPacket_t *packet) {
  gap8_sample_t sample;
  size_t size;
  if (!cf_cpx_bridge_read(&bridge, packet, &sample, sizeof(sample), &size) ||
      size != sizeof(sample)) return;
  taskENTER_CRITICAL();
  gap8 = sample;
  taskEXIT_CRITICAL();
}

// The logger calls every field in a block with the same timestamp, under its
// log lock. Latch once per timestamp so a CPX update between field reads cannot
// mix samples. Only log callbacks access this snapshot; the receiver writes gap8.
static gap8_sample_t gap8Snapshot;
static uint32_t gap8SnapshotTimestamp;
static bool gap8SnapshotValid;

static void latchGap8Snapshot(uint32_t timestamp) {
  if (!gap8SnapshotValid || timestamp != gap8SnapshotTimestamp) {
    taskENTER_CRITICAL();
    gap8Snapshot = gap8;
    taskEXIT_CRITICAL();
    gap8SnapshotTimestamp = timestamp;
    gap8SnapshotValid = true;
  }
}

static uint32_t readGap8Tick(uint32_t timestamp, void *data) {
  latchGap8Snapshot(timestamp);
  return *(uint32_t *)data;
}

static float readGap8Float(uint32_t timestamp, void *data) {
  latchGap8Snapshot(timestamp);
  return *(float *)data;
}

static logByFunction_t gap8TickLog = {
  .acquireUInt32 = readGap8Tick, .data = &gap8Snapshot.tick,
};
static logByFunction_t gap8SinLog = {
  .aquireFloat = readGap8Float, .data = &gap8Snapshot.sin1,
};
static logByFunction_t gap8CosLog = {
  .aquireFloat = readGap8Float, .data = &gap8Snapshot.cos1,
};
static logByFunction_t gap8SawLog = {
  .aquireFloat = readGap8Float, .data = &gap8Snapshot.saw,
};
static logByFunction_t gap8RampLog = {
  .aquireFloat = readGap8Float, .data = &gap8Snapshot.ramp,
};
static logByFunction_t gap8SquareLog = {
  .aquireFloat = readGap8Float, .data = &gap8Snapshot.square,
};
static logByFunction_t gap8TriLog = {
  .aquireFloat = readGap8Float, .data = &gap8Snapshot.tri,
};

void appMain(void) {
  cf_cpx_bridge_stm32_init(&bridge, CPX_T_GAP8);
  cpxRegisterAppMessageHandler(onGap8Packet);
  DEBUG_PRINT("telemetry bench: synthetic signals at 1 kHz (groups tba/tbb)\n");
  TickType_t last = xTaskGetTickCount();
  while (1) {
    vTaskDelayUntil(&last, M2T(1));
    tick++;
    float t = tick * 0.001f;
    float phase = t - (uint32_t)t;
    sin1 = sinf(PI2_F * phase);
    cos1 = cosf(PI2_F * phase);
    saw = phase;
    ramp = t;
    square = (phase < 0.5f) ? 1.0f : -1.0f;
    tri = (phase < 0.5f) ? (4.0f * phase - 1.0f) : (3.0f - 4.0f * phase);
    blk = tick;
  }
}

LOG_GROUP_START(tba)
LOG_ADD(LOG_UINT32, tick, &tick)
LOG_ADD(LOG_FLOAT, sin1, &sin1)
LOG_ADD(LOG_FLOAT, cos1, &cos1)
LOG_ADD(LOG_FLOAT, saw, &saw)
LOG_GROUP_STOP(tba)

LOG_GROUP_START(tbb)
LOG_ADD(LOG_UINT32, blk, &blk)
LOG_ADD(LOG_FLOAT, ramp, &ramp)
LOG_ADD(LOG_FLOAT, square, &square)
LOG_ADD(LOG_FLOAT, tri, &tri)
LOG_GROUP_STOP(tbb)

// GAP8-generated values relayed through the STM32 log framework.
LOG_GROUP_START(tga)
LOG_ADD_BY_FUNCTION(LOG_UINT32, tick, &gap8TickLog)
LOG_ADD_BY_FUNCTION(LOG_FLOAT, sin1, &gap8SinLog)
LOG_ADD_BY_FUNCTION(LOG_FLOAT, cos1, &gap8CosLog)
LOG_ADD_BY_FUNCTION(LOG_FLOAT, saw, &gap8SawLog)
LOG_GROUP_STOP(tga)

LOG_GROUP_START(tgb)
LOG_ADD_BY_FUNCTION(LOG_UINT32, blk, &gap8TickLog)
LOG_ADD_BY_FUNCTION(LOG_FLOAT, ramp, &gap8RampLog)
LOG_ADD_BY_FUNCTION(LOG_FLOAT, square, &gap8SquareLog)
LOG_ADD_BY_FUNCTION(LOG_FLOAT, tri, &gap8TriLog)
LOG_GROUP_STOP(tgb)
