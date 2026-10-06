/* pmsis.h remaps snprintf onto the SDK's freestanding printf; newlib's stdio is
 * not linked into the image. Including it first is what resolves snprintf. */
#include "pmsis.h"

#include "cpx_wifi_log.h"

#include <stdio.h>

static CPXConsoleTarget_t log_target(const void *context) {
  if (context != NULL) {
    const CPXConsoleTarget_t target =
        ((const cpx_wifi_log_context_t *)context)->target;
    if (target == LOG_TO_CRTP || target == LOG_TO_WIFI) {
      return target;
    }
  }
  return LOG_TO_CRTP;
}

void cpx_wifi_log_context_init(cpx_wifi_log_context_t *context) {
  if (context != NULL) {
    context->target = LOG_TO_CRTP;
  }
}

void cpx_wifi_log_event_callback(const cpx_wifi_session_event_t *event,
                                 void *context) {
  CPXConsoleTarget_t target;
  CPXPacket_t *packet;
  int length = -1;

  if (event == NULL || event->callback_packet == NULL) {
    return;
  }
  target = log_target(context);
  packet = event->callback_packet;

  switch (event->type) {
    case CPX_WIFI_SESSION_EVENT_WIFI_UP:
      length = snprintf((char *)packet->data, sizeof(packet->data),
                        "WiFi up (%u.%u.%u.%u)\n",
                        event->status.ip[0], event->status.ip[1],
                        event->status.ip[2], event->status.ip[3]);
      break;

    case CPX_WIFI_SESSION_EVENT_CLIENT_CONNECTED:
      length = snprintf((char *)packet->data, sizeof(packet->data),
                        "WiFi client status: 1\n");
      break;

    case CPX_WIFI_SESSION_EVENT_CLIENT_DISCONNECTED:
      length = snprintf((char *)packet->data, sizeof(packet->data),
                        "WiFi client status: 0\n");
      break;

    default:
      break;
  }

  if (length <= 0) {
    return;
  }
  if (length >= (int)sizeof(packet->data)) {
    length = (int)sizeof(packet->data) - 1;
    packet->data[length] = '\0';
  }
  cpxInitRoute(CPX_T_GAP8, (CPXTarget_t)target, CPX_F_CONSOLE,
               &packet->route);
  packet->dataLength = (uint16_t)(length + 1);
  (void)cpxSendPacket(packet, 0);
}
