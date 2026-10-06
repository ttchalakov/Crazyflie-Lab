// Out-of-tree controller for mocap positioning demo.
//
// This controller NEVER spins the propellers. In this demo the
// drone is never armed. Each tick it reads the state
// estimate (which the host feeds from motion capture over extpose)
// and reports whether the drone is inside the
// configured safety box, via the `box.inside` log variable.
//
// The host computes the identical decision from the mocap it
// is already streaming and lights the front LEDs through the firmware's
// `led.bitmask` parameter: GREEN inside the box, RED outside. The LED is driven
// host-side because `led.bitmask` is the firmware's sanctioned LED override and
// drives the GPIOs reliably, whereas a raw ledSet() from here is gated by the
// LED sequencer.
//
// Frame: the host re-zeroes the mocap origin onto the drone's starting pose on
// the floor (OriginZeroer), so the estimator's (x, y, z) is in the same frame
// as the box below: origin at the start point, +z up. The box is an
// axis-aligned rectangle defined by a center and full side widths. The
// host pushes the configured values into the `box.*` parameters on connect; the
// defaults here are a fallback.

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#include "app.h"
#include "controller.h"
#define DEBUG_MODULE "BOXVIZ"
#include "debug.h"
#include "FreeRTOS.h"
#include "log.h"
#include "param.h"
#include "task.h"

// Safety box in the host-zeroed mocap frame: center + full side widths, metres.
// The host overrides them through the `box.*` params.
static float boxCenterX = 0.0f;
static float boxCenterY = 0.0f;
static float boxCenterZ = 1.5f;
static float boxSizeX = 2.0f;
static float boxSizeY = 1.0f;
static float boxSizeZ = 1.0f;

// Telemetry the host can read back
static uint8_t insideBox = 0;
static float lastPositionX = 0.0f;
static float lastPositionY = 0.0f;
static float lastPositionZ = 0.0f;

static bool positionInsideBox(const state_t *state) {
  return fabsf(state->position.x - boxCenterX) <= boxSizeX / 2.0f
      && fabsf(state->position.y - boxCenterY) <= boxSizeY / 2.0f
      && fabsf(state->position.z - boxCenterZ) <= boxSizeZ / 2.0f;
}

// appMain() is the OPTIONAL app-layer entry point. It exists only because this
// examples "app-config" sets CONFIG_APP_ENABLE=y: at startup the firmware spawns an app task
// that calls appMain() once (here, just to print the load message). The
// out-of-tree controller does NOT depend on it — controllerOutOfTree() is driven
// by the stabilizer loop, independent of the app layer (appInit() is gated by
// #ifdef CONFIG_APP_ENABLE in system.c). Disable CONFIG_APP_ENABLE in app-config
// and this whole function can be deleted; the box check still runs.
// After appMain() returns the app task idles itself.
void appMain(void) {
  DEBUG_PRINT("Mocap positioning controller loaded: motors stay off, box.inside reports the box check.\n");
}

void controllerOutOfTreeInit(void) {
  insideBox = 0;
  lastPositionX = 0.0f;
  lastPositionY = 0.0f;
  lastPositionZ = 0.0f;
}

// Required by crazyflie controller interface
bool controllerOutOfTreeTest(void) {
  return true;
}

void controllerOutOfTree(
  control_t *control,
  const setpoint_t *setpoint,
  const sensorData_t *sensors,
  const state_t *state,
  const stabilizerStep_t stabilizerStep
) {
  (void)setpoint;
  (void)sensors;
  (void)stabilizerStep;

  lastPositionX = state->position.x;
  lastPositionY = state->position.y;
  lastPositionZ = state->position.z;
  insideBox = positionInsideBox(state) ? 1 : 0;

  control->controlMode = controlModeLegacy;
  control->roll = 0;
  control->pitch = 0;
  control->yaw = 0;
  control->thrust = 0.0f;
}

PARAM_GROUP_START(box)
PARAM_ADD(PARAM_FLOAT, centerX, &boxCenterX)
PARAM_ADD(PARAM_FLOAT, centerY, &boxCenterY)
PARAM_ADD(PARAM_FLOAT, centerZ, &boxCenterZ)
PARAM_ADD(PARAM_FLOAT, sizeX, &boxSizeX)
PARAM_ADD(PARAM_FLOAT, sizeY, &boxSizeY)
PARAM_ADD(PARAM_FLOAT, sizeZ, &boxSizeZ)
PARAM_GROUP_STOP(box)

LOG_GROUP_START(box)
LOG_ADD(LOG_UINT8, inside, &insideBox)
LOG_ADD(LOG_FLOAT, posX, &lastPositionX)
LOG_ADD(LOG_FLOAT, posY, &lastPositionY)
LOG_ADD(LOG_FLOAT, posZ, &lastPositionZ)
LOG_GROUP_STOP(box)
