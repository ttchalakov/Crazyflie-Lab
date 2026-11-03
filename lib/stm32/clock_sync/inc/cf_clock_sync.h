#ifndef CF_LAB_CLOCK_SYNC_H
#define CF_LAB_CLOCK_SYNC_H

// Answers the host's clock pings on one CRTP port (channel 0) so cf_lab.timing.clock_sync can map the STM32's
// microsecond clock (usecTimestamp) onto the host's. A ping carries 6 bytes (sequence u16, host time u32); the reply
// echoes them and adds the STM32 time at reception and just before sending (u32 microseconds each). The reply is
// sent from the CRTP receive task, so it leaves on the next radio poll.

#include <stdint.h>

#define CF_CLOCK_SYNC_PORT_DEFAULT 0x0B

void cfClockSyncInit(uint8_t port);

#endif
