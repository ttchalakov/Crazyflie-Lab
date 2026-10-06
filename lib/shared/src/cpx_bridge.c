#include "cf_lab/cpx_bridge.h"
#include <string.h>

/* Supplied by the selected target adapter. */
bool cf_cpx_bridge_platform_send(const CPXPacket_t *packet, uint32_t timeout);

bool cf_cpx_bridge_init(cf_cpx_bridge_t *b, CPXTarget_t local, CPXTarget_t peer) {
  if (!b) return false;
  memset(b, 0, sizeof(*b));
  if ((local != CPX_T_STM32 && local != CPX_T_GAP8) ||
      peer < CPX_T_STM32 || peer > CPX_T_GAP8 || peer == local) return false;
  b->local = local;
  b->peer = peer;
  b->initialized = true;
  return true;
}

bool cf_cpx_bridge_send(const cf_cpx_bridge_t *b, const void *data,
                        size_t size, uint32_t timeout) {
  if (!b || !b->initialized || !data || size == 0 ||
      size > CF_CPX_BRIDGE_MAX_PAYLOAD) return false;
  CPXPacket_t packet = {0};
  cpxInitRoute(b->local, b->peer, CPX_F_APP, &packet.route);
  /* A bridge message is always one whole packet; the STM32 firmware's
   * cpxInitRoute leaves lastPacket unset and the reader requires it. */
  packet.route.lastPacket = true;
  packet.dataLength = (uint16_t)size;
  memcpy(packet.data, data, size);
  return cf_cpx_bridge_platform_send(&packet, timeout);
}

bool cf_cpx_bridge_read(const cf_cpx_bridge_t *b, const CPXPacket_t *p,
                        void *output, size_t capacity, size_t *size) {
  if (!b || !b->initialized || !p || !output || !size ||
      p->route.source != b->peer || p->route.destination != b->local ||
      p->route.function != CPX_F_APP || p->route.version != CPX_VERSION ||
      !p->route.lastPacket || p->dataLength == 0 ||
      p->dataLength > CF_CPX_BRIDGE_MAX_PAYLOAD || p->dataLength > capacity) return false;
  memcpy(output, p->data, p->dataLength);
  *size = p->dataLength;
  return true;
}
