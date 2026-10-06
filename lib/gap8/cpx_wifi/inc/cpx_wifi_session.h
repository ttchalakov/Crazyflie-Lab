#ifndef CF_GAP8_CPX_WIFI_SESSION_H
#define CF_GAP8_CPX_WIFI_SESSION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cpx.h"
#include "cpx_wifi_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CPX_WIFI_SESSION_DEFAULT_SETTLE_MS
#define CPX_WIFI_SESSION_DEFAULT_SETTLE_MS 150u
#endif

typedef enum {
  CPX_WIFI_SESSION_OK = 0,
  CPX_WIFI_SESSION_ERR_INVALID_ARGUMENT = -1,
  CPX_WIFI_SESSION_ERR_INVALID_STATE = -2,
  CPX_WIFI_SESSION_ERR_BUSY = -3,
  CPX_WIFI_SESSION_ERR_RTOS = -4,
  CPX_WIFI_SESSION_ERR_SEND = -5,
} cpx_wifi_session_result_t;

typedef struct {
  bool wifi_up;
  bool client_connected;
  bool client_ready;
  uint8_t ip[4];
  uint32_t client_connected_at_ms;
  uint32_t event_sequence;
  cpx_wifi_transport_t transport;
} cpx_wifi_session_status_t;

typedef enum {
  CPX_WIFI_SESSION_EVENT_WIFI_UP = 0,
  CPX_WIFI_SESSION_EVENT_CLIENT_CONNECTED = 1,
  CPX_WIFI_SESSION_EVENT_CLIENT_DISCONNECTED = 2,
} cpx_wifi_session_event_type_t;

typedef struct {
  cpx_wifi_session_event_type_t type;
  cpx_wifi_session_status_t status;

  /*
   * Callback-scoped RX packet owned by the session. A short adapter may reuse
   * it for a nonblocking response/log after calling cpxInitRoute(). The next
   * receive reclaims it as soon as the callback returns, so do not retain the
   * pointer or call connect() from the callback.
   */
  CPXPacket_t *callback_packet;
} cpx_wifi_session_event_t;

/*
 * Runs synchronously in the Wi-Fi control receiver task after state is
 * published. The callback must not block and must not call connect(). The RX
 * packet is not touched again until the callback returns.
 */
typedef void (*cpx_wifi_session_event_callback_t)(
    const cpx_wifi_session_event_t *event,
    void *context);

typedef struct {
  /* At most CPX_WIFI_SSID_MAX_LENGTH (32) bytes. */
  const char *ssid;
  /* NULL/empty selects an open network; a secured AP key needs >= 8 bytes. */
  const char *key;
  cpx_wifi_transport_t transport;
  cpx_wifi_mode_t mode;
  /* Access point channel, 1..13; CPX_WIFI_CHANNEL_DEFAULT keeps the ESP32's. */
  uint8_t channel;
  /*
   * The deck's name on the network: its mDNS hostname (<hostname>.local) and a
   * readable TXT "name", each at most CPX_WIFI_NAME_MAX_LENGTH bytes. A
   * NULL/empty hostname keeps the ESP32's aideck-XXXXXX.
   */
  const char *hostname;
  const char *name;
  uint32_t client_settle_ms;

  /* Zero selects CPX_WIFI_SESSION_RECEIVER_STACK_WORDS / _PRIORITY. */
  uint32_t receiver_task_stack_words;
  uint32_t receiver_task_priority;

  cpx_wifi_session_event_callback_t event_callback;
  void *event_callback_context;
} cpx_wifi_session_config_t;

/* The receiver task's defaults. */
#define CPX_WIFI_SESSION_RECEIVER_STACK_WORDS (configMINIMAL_STACK_SIZE * 2u)
#define CPX_WIFI_SESSION_RECEIVER_PRIORITY (tskIDLE_PRIORITY + 1u)

/*
 * Caller-owned, opaque storage. It may be static/global or embedded in another
 * object, but must not be copied after cpx_wifi_session_init(). The two CPX
 * packets live here rather than on the small receiver-task stack.
 */
#define CPX_WIFI_SESSION_OBJECT_BYTES (2u * sizeof(CPXPacket_t) + 256u)
typedef union {
  uint64_t _alignment;
  uint8_t _storage[CPX_WIFI_SESSION_OBJECT_BYTES];
} cpx_wifi_session_t;

int cpx_wifi_session_config_init(cpx_wifi_session_config_t *config);

/* Validate and copy only: no CPX call, allocation, or task creation occurs. */
int cpx_wifi_session_init(cpx_wifi_session_t *session,
                          const cpx_wifi_session_config_t *config);

/*
 * The application must call cpxInit() first. open() claims the process-wide
 * CPX_F_WIFI_CTRL receiver, enables its queue, and creates one receiver task.
 * No application task may also receive from CPX_F_WIFI_CTRL.
 * CPX has no queue/task teardown API, so a successfully opened session
 * intentionally remains open for the life of the application.
 */
int cpx_wifi_session_open(cpx_wifi_session_t *session);

/* Send SSID, optional key, transport, the access point channel and the names
 * when set, then connect, all as blocking CPX sends. */
int cpx_wifi_session_connect(cpx_wifi_session_t *session);

int cpx_wifi_session_get_status(const cpx_wifi_session_t *session,
                                cpx_wifi_session_status_t *status);
bool cpx_wifi_session_client_connected(const cpx_wifi_session_t *session);
bool cpx_wifi_session_client_ready(const cpx_wifi_session_t *session);

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_CPX_WIFI_SESSION_H */
