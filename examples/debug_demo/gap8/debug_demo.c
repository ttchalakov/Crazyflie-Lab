#include <stdint.h>

#include "bsp/bsp.h"
#include "cpx.h"
#include "pmsis.h"

// Minimal GAP8 app: bring up CPX and print one line per second to the console
// (LOG_TO_CRTP). The STM32 forwards it over the radio, where the host tags it
// [GAP8].
void start_example(void) {
  pi_bsp_init();
  cpxInit();
  cpxPrintToConsole(LOG_TO_CRTP, "debug_demo GAP8 app up\n");

  uint32_t n = 0;
  while (1) {
    cpxPrintToConsole(LOG_TO_CRTP, "gap8 tick %lu\n", (unsigned long)n++);
    vTaskDelay(1000);  // ~1 Hz
  }
}

int main(void) {
  return pmsis_kickoff((void *)start_example);
}
