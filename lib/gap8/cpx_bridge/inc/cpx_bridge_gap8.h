#ifndef CF_CPX_BRIDGE_GAP8_H
#define CF_CPX_BRIDGE_GAP8_H
#include "cf_lab/cpx_bridge.h"
/* CPX must be initialized and CPX_F_APP enabled once by the application. */
bool cf_cpx_bridge_gap8_init(cf_cpx_bridge_t *bridge, CPXTarget_t peer);
/* Consume one packet, blocking for arrival, then copy its bytes with read().
 * False for rejected/oversized packets; outputs remain untouched. One receiver
 * task should own CPX_F_APP. For shared dispatch, receive once yourself and use
 * cf_cpx_bridge_read. Invalid arguments return without consuming a packet. */
bool cf_cpx_bridge_gap8_receive(const cf_cpx_bridge_t *bridge,
                                void *output, size_t capacity, size_t *size);
#endif
