# `timing`: cycle counter and loop statistics

Header-only, for STM32 apps (`ccflags-y += -I$(CF_LAB)/lib/stm32/timing/inc`).

- `cf_cycles.h`: the Cortex-M4 cycle counter (DWT `CYCCNT`, 168 per microsecond), enabled by `cf_cycles_init()`;
  the firmware does not use the DWT, and it counts without a debugger attached.
- `cf_loop_stats.h`: the period of a periodic loop against its nominal period (late ticks above 10, 50 and 100 %)
  and a histogram of whatever each tick measures. `examples/stm32_nn_bench` wraps the stock INDI with it to time the
  stabilizer loop.
