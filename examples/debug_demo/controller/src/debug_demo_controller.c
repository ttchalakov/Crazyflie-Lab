#include <stdint.h>

#include "app.h"
#define DEBUG_MODULE "DBGDEMO"
#include "debug.h"
#include "FreeRTOS.h"
#include "task.h"

// Minimal app: print one line per second over the firmware console (DEBUG_PRINT
// -> CRTP), which the host tags [STM32].
void appMain(void) {
  uint32_t n = 0;
  while (1) {
    DEBUG_PRINT("controller tick %lu\n", (unsigned long)n++);
    vTaskDelay(M2T(1000));  // 1 Hz
  }
}
