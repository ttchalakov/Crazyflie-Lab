#ifndef CF_GAP8_CPX_WIFI_LOG_H
#define CF_GAP8_CPX_WIFI_LOG_H

#include "cpx_wifi_session.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Optional context for cpx_wifi_log_event_callback(). Pass NULL to log to the
 * Crazyflie radio console (LOG_TO_CRTP). An invalid target also falls back to
 * LOG_TO_CRTP, so a zero-initialized context is safe.
 *
 * The caller owns this object and must keep it alive while the session can
 * invoke its event callback.
 */
typedef struct {
  CPXConsoleTarget_t target;
} cpx_wifi_log_context_t;

void cpx_wifi_log_context_init(cpx_wifi_log_context_t *context);

/*
 * Signature-compatible with cpx_wifi_session_config_t::event_callback.
 * It prints these console lines:
 *
 *   WiFi up (a.b.c.d)
 *   WiFi client status: 0|1
 *
 * It keeps no state, performs no allocation, and reuses the session's
 * callback-scoped RX packet with cpxSendPacket(..., 0). A line is
 * intentionally dropped when the CPX queue is busy so the Wi-Fi status
 * receiver can continue draining events.
 */
void cpx_wifi_log_event_callback(const cpx_wifi_session_event_t *event,
                                 void *context);

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_CPX_WIFI_LOG_H */
