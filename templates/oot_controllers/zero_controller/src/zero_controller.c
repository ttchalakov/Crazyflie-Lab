#include <stdbool.h>
#include <stdint.h>

#include "controller.h"

void controllerOutOfTreeInit(void) {}

bool controllerOutOfTreeTest(void) {
  return true;
}

void controllerOutOfTree(
  control_t *control,
  const setpoint_t *setpoint,
  const sensorData_t *sensors,
  const state_t *state,
  const uint32_t tick
) {
  (void)setpoint;
  (void)sensors;
  (void)state;
  (void)tick;

  control->controlMode = controlModeLegacy;
  control->roll = 0;
  control->pitch = 0;
  control->yaw = 0;
  control->thrust = 0.0f;
}
