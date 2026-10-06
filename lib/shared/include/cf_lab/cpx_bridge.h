#ifndef CF_LAB_CPX_BRIDGE_H
#define CF_LAB_CPX_BRIDGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "cpx.h"

/* Single CPX_F_APP packet, bounded by the STM32 link's wire MTU.
 * Payload bytes are passed through unchanged. No fragmentation is supplied. */
#define CF_CPX_BRIDGE_MAX_PAYLOAD 98u

/* Caller-owned routing configuration only. Initialize before concurrent use
 * and leave unchanged afterwards. The bridge stores no received data. */
typedef struct {
  CPXTarget_t local;
  CPXTarget_t peer;
  bool initialized;
} cf_cpx_bridge_t;

/* Prefer the target-specific init wrapper. Does not initialize CPX or register
 * a receiver. An invalid configuration leaves the instance disabled. */
bool cf_cpx_bridge_init(cf_cpx_bridge_t *bridge, CPXTarget_t local, CPXTarget_t peer);
/* Queue bytes. timeout is passed through in native FreeRTOS ticks; zero does
 * not wait for queue space. False means invalid input or queue timeout.
 * Task-only; avoid blocking sends from a controller loop or router callback. */
bool cf_cpx_bridge_send(const cf_cpx_bridge_t *bridge, const void *data,
                        size_t size, uint32_t timeout);
/* Check route, CPX version, complete-packet flag and payload bounds, then copy
 * into caller-owned output. size receives the actual payload length. Both output
 * and size are required and remain untouched on failure. Application packet
 * decoding/validation, storage, synchronization and freshness are caller-owned. */
bool cf_cpx_bridge_read(const cf_cpx_bridge_t *bridge, const CPXPacket_t *packet,
                        void *output, size_t capacity, size_t *size);
#endif
