#include "cf_clock_sync.h"

#include <string.h>

#include "crtp.h"
#include "usec_time.h"

static uint8_t syncPort = CF_CLOCK_SYNC_PORT_DEFAULT;

static void onPing(CRTPPacket *p) {
  const uint32_t rx = (uint32_t)usecTimestamp();
  if (p->channel != 0 || p->size < 6) {
    return;
  }
  CRTPPacket reply = {.size = 14};
  reply.header = CRTP_HEADER(syncPort, 0);
  memcpy(reply.data, p->data, 6);
  memcpy(reply.data + 6, &rx, 4);
  const uint32_t tx = (uint32_t)usecTimestamp();
  memcpy(reply.data + 10, &tx, 4);
  crtpSendPacket(&reply);
}

void cfClockSyncInit(uint8_t port) {
  syncPort = port;
  crtpRegisterPortCB(port, onPing);
}
