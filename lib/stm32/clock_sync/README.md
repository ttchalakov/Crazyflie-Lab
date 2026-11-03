# `clock_sync`: the STM32's clock on the host's

The STM32 side of `cf_lab.timing.clock_sync`: answers clock pings on one CRTP port (channel 0) with its microsecond
clock (`usecTimestamp`) at reception and just before replying.

```make
# src/Kbuild of the app
include $(CF_LAB)/lib/stm32/clock_sync/clock_sync.mk
```

```c
#include "cf_clock_sync.h"
cfClockSyncInit(CF_CLOCK_SYNC_PORT_DEFAULT);  // once, from appMain
```

```python
from cf_lab.timing import fit_clock, measure_clock
clock, before = measure_clock(cf)       # ~1 s of pings through an open cflib link
...                                      # the run whose drone timestamps need host times
_, after = measure_clock(cf)
clock = fit_clock(before + after)        # drift needs a baseline as long as the run
host_time = clock.to_host_us(drone_us)   # +- clock.uncertainty_us(drone_us)
```

Every drone event happened between the latest send that could reach it and the earliest reception that could follow
it; the map returns the middle and the half-width, which bounds the error of any one-way latency whatever the link's
asymmetry. Over a Crazyradio PA the shortest round trip is about 2.3 ms, so one-way latencies are known to about
+-1.1 ms. The drone's microsecond clock ran 418 ppm fast against this computer's monotonic clock, so maps need the
drift term.
