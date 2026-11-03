#ifndef CF_LAB_CYCLES_H
#define CF_LAB_CYCLES_H

// The Cortex-M4 cycle counter (DWT CYCCNT): one count per CPU cycle, 168 per microsecond on the Crazyflie, wrapping
// every 25.6 s. Unsigned subtraction of two readings is correct across one wrap. The firmware does not use the DWT,
// and it counts without a debugger attached once cf_cycles_init() has enabled it.

#include <stdint.h>

#define CF_CYCLES_PER_US 168u

#define CF_CYCLES_DEMCR (*(volatile uint32_t *)0xE000EDFCu)
#define CF_CYCLES_DWT_CTRL (*(volatile uint32_t *)0xE0001000u)
#define CF_CYCLES_DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)

static inline void cf_cycles_init(void) {
  CF_CYCLES_DEMCR |= 1u << 24;  // TRCENA: enable the DWT
  CF_CYCLES_DWT_CTRL |= 1u;     // CYCCNTENA
}

static inline uint32_t cf_cycles(void) {
  return CF_CYCLES_DWT_CYCCNT;
}

#endif
