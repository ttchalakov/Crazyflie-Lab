// Out-of-tree controller of the network bench: the stock INDI (or PID, parameter nnbench.inner) unchanged, with the
// stabilizer loop timed around it. Per tick it records the period since the previous tick, the controller's
// execution time, and the time from the IMU's data-ready interrupt to the controller's output. The motors stay off:
// the bench never arms, and the stabilizer stops the motors whatever a disarmed controller outputs.

#include <stdint.h>

#include "FreeRTOS.h"
#include "controller.h"
#include "controller_indi.h"
#include "controller_pid.h"
#include "task.h"
#include "usec_time.h"

#include "cf_cycles.h"
#include "nnb_controller.h"

uint8_t nnbInnerIndi = 1;  // 1 INDI, 0 PID (parameter nnbench.inner)

static nnb_inner_stats_t stats;

void nnbInnerReset(void) {
  taskENTER_CRITICAL();
  cf_loop_stats_reset(&stats.tick, 1000u * CF_CYCLES_PER_US, 5u * CF_CYCLES_PER_US);
  cf_loop_stats_reset(&stats.latency, 1000u * CF_CYCLES_PER_US, 25u * CF_CYCLES_PER_US);
  taskEXIT_CRITICAL();
}

void nnbInnerSnapshot(nnb_inner_stats_t *out) {
  taskENTER_CRITICAL();
  *out = stats;
  taskEXIT_CRITICAL();
}

void controllerOutOfTreeInit(void) {
  controllerINDIInit();
  controllerPidInit();
  nnbInnerReset();
}

bool controllerOutOfTreeTest(void) {
  return controllerINDITest() && controllerPidTest();
}

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors,
                         const state_t *state, const stabilizerStep_t stabilizerStep) {
  const uint32_t t0 = cf_cycles();
  cf_loop_tick(&stats.tick, t0);
  if (nnbInnerIndi) {
    controllerINDI(control, setpoint, sensors, state, stabilizerStep);
  } else {
    controllerPid(control, setpoint, sensors, state, stabilizerStep);
  }
  cf_loop_span(&stats.tick, cf_cycles() - t0);
  // IMU interrupt to controller output, in microseconds, kept in the cycle-based struct as us x CF_CYCLES_PER_US.
  const uint32_t us = (uint32_t)(usecTimestamp() - sensors->interruptTimestamp);
  cf_loop_span(&stats.latency, us * CF_CYCLES_PER_US);
}
