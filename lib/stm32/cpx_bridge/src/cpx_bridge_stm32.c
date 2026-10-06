#include "cpx_bridge_stm32.h"
#include "cpx_internal_router.h"
bool cf_cpx_bridge_platform_send(const CPXPacket_t *packet, uint32_t timeout) {
  return cpxSendPacketBlockingTimeout(packet, timeout);
}
bool cf_cpx_bridge_stm32_init(cf_cpx_bridge_t *b, CPXTarget_t peer) {
  return cf_cpx_bridge_init(b, CPX_T_STM32, peer);
}
