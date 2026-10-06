#include "cpx_bridge_gap8.h"
bool cf_cpx_bridge_platform_send(const CPXPacket_t *packet, uint32_t timeout) {
  return cpxSendPacket(packet, timeout);
}
bool cf_cpx_bridge_gap8_init(cf_cpx_bridge_t *b, CPXTarget_t peer) {
  return cf_cpx_bridge_init(b, CPX_T_GAP8, peer);
}
bool cf_cpx_bridge_gap8_receive(const cf_cpx_bridge_t *b,
                                void *output, size_t capacity, size_t *size) {
  if (!b || !b->initialized || b->local != CPX_T_GAP8 ||
      !output || !size || capacity == 0) return false;
  CPXPacket_t packet;
  cpxReceivePacketBlocking(CPX_F_APP, &packet);
  return cf_cpx_bridge_read(b, &packet, output, capacity, size);
}
