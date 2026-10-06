#include "pmsis.h"

#include <stddef.h>
#include <string.h>

#include "cpx_wifi_session.h"

#define CPX_WIFI_SESSION_MAGIC 0x43505746u /* CPWF */

typedef enum {
  CPX_WIFI_STATE_UNINITIALIZED = 0,
  CPX_WIFI_STATE_INITIALIZED = 1,
  CPX_WIFI_STATE_OPENING = 2,
  CPX_WIFI_STATE_OPEN = 3,
  CPX_WIFI_STATE_CONNECTING = 4,
  CPX_WIFI_STATE_CONNECT_REQUESTED = 5,
} cpx_wifi_state_t;

typedef struct {
  uint32_t magic;
  uintptr_t self;
  cpx_wifi_state_t state;

  char ssid[CPX_WIFI_SSID_MAX_LENGTH + 1u];
  char key[CPX_WIFI_KEY_MAX_LENGTH + 1u];
  cpx_wifi_transport_t transport;
  cpx_wifi_mode_t mode;
  uint8_t channel;
  /* SET_NAME's payload, "<hostname>\0<name>\0"; names_length 0 sends none. */
  char names[2u * (CPX_WIFI_NAME_MAX_LENGTH + 1u)];
  uint8_t names_length;
  TickType_t settle_ticks;
  uint32_t receiver_task_stack_words;
  uint32_t receiver_task_priority;
  cpx_wifi_session_event_callback_t event_callback;
  void *event_callback_context;

  TaskHandle_t receiver_task;
  CPXPacket_t rx_packet;
  CPXPacket_t tx_packet;

  TickType_t client_connected_at_ticks;
  uint32_t client_connected_at_ms;
  uint32_t event_sequence;
  uint8_t ip[4];
  uint8_t wifi_up;
  uint8_t client_connected;
} cpx_wifi_session_impl_t;

typedef char cpx_wifi_session_object_must_fit[
    sizeof(cpx_wifi_session_impl_t) <= CPX_WIFI_SESSION_OBJECT_BYTES ? 1 : -1];

/* CPX provides one receive queue per function, so it can have one owner. */
static cpx_wifi_session_impl_t *wifi_ctrl_owner;

static cpx_wifi_session_impl_t *session_impl(cpx_wifi_session_t *session) {
  return (cpx_wifi_session_impl_t *)(void *)session->_storage;
}

static const cpx_wifi_session_impl_t *session_const_impl(
    const cpx_wifi_session_t *session) {
  return (const cpx_wifi_session_impl_t *)(const void *)session->_storage;
}

static bool session_is_initialized(const cpx_wifi_session_impl_t *session) {
  return session != NULL && session->magic == CPX_WIFI_SESSION_MAGIC &&
         session->self == (uintptr_t)session &&
         session->state >= CPX_WIFI_STATE_INITIALIZED &&
         session->state <= CPX_WIFI_STATE_CONNECT_REQUESTED;
}

static bool storage_looks_active(const cpx_wifi_session_t *storage) {
  const cpx_wifi_session_impl_t *address = session_const_impl(storage);
  const uint32_t expected_magic = CPX_WIFI_SESSION_MAGIC;
  const uintptr_t expected_self = (uintptr_t)address;
  const uint8_t *bytes = storage->_storage;

  /* Reading object representation bytes is safe for never-initialized storage. */
  if (memcmp(bytes + offsetof(cpx_wifi_session_impl_t, magic),
             &expected_magic, sizeof(expected_magic)) != 0 ||
      memcmp(bytes + offsetof(cpx_wifi_session_impl_t, self),
             &expected_self, sizeof(expected_self)) != 0) {
    return false;
  }
  for (int value = CPX_WIFI_STATE_OPENING;
       value <= CPX_WIFI_STATE_CONNECT_REQUESTED; value++) {
    const cpx_wifi_state_t expected_state = (cpx_wifi_state_t)value;
    if (memcmp(bytes + offsetof(cpx_wifi_session_impl_t, state),
               &expected_state, sizeof(expected_state)) == 0) {
      return true;
    }
  }
  return false;
}

static TickType_t ms_to_ticks(uint32_t milliseconds) {
  TickType_t ticks = pdMS_TO_TICKS(milliseconds);
  return milliseconds != 0u && ticks == 0u ? 1u : ticks;
}

static uint32_t ticks_to_ms(TickType_t ticks) {
  return (uint32_t)ticks * (uint32_t)portTICK_PERIOD_MS;
}

static bool valid_transport(cpx_wifi_transport_t transport) {
  return transport == CPX_WIFI_TRANSPORT_TCP ||
         transport == CPX_WIFI_TRANSPORT_UDP;
}

static bool valid_mode(cpx_wifi_mode_t mode) {
  return mode == CPX_WIFI_MODE_STATION ||
         mode == CPX_WIFI_MODE_ACCESS_POINT;
}

static bool valid_key_for_mode(const char *key, cpx_wifi_mode_t mode) {
  if (key == NULL || key[0] == '\0') {
    return true;
  }
  /* ESP-IDF rejects a secured soft-AP passphrase shorter than eight bytes. */
  return mode != CPX_WIFI_MODE_ACCESS_POINT || strlen(key) >= 8u;
}

static bool valid_string(const char *value, size_t max_length, bool allow_empty) {
  size_t length;

  if (value == NULL) {
    return allow_empty;
  }
  /* Bounded scan by hand: the SDK's freestanding string.h has no memchr. */
  for (length = 0u; length <= max_length; length++) {
    if (value[length] == '\0') {
      return allow_empty || length != 0u;
    }
  }
  return false;
}

static bool ready_at(const cpx_wifi_session_impl_t *session,
                     TickType_t now) {
  return session->client_connected != 0u &&
         (TickType_t)(now - session->client_connected_at_ticks) >=
             session->settle_ticks;
}

static void copy_status_locked(const cpx_wifi_session_impl_t *session,
                               TickType_t now,
                               cpx_wifi_session_status_t *status) {
  status->wifi_up = session->wifi_up != 0u;
  status->client_connected = session->client_connected != 0u;
  status->client_ready = ready_at(session, now);
  memcpy(status->ip, session->ip, sizeof(status->ip));
  status->client_connected_at_ms = session->client_connected_at_ms;
  status->event_sequence = session->event_sequence;
  status->transport = session->transport;
}

static void publish_event(cpx_wifi_session_impl_t *session,
                          cpx_wifi_session_event_type_t type) {
  cpx_wifi_session_event_callback_t callback;
  void *callback_context;
  cpx_wifi_session_event_t event;
  const TickType_t now = xTaskGetTickCount();

  taskENTER_CRITICAL();
  copy_status_locked(session, now, &event.status);
  callback = session->event_callback;
  callback_context = session->event_callback_context;
  taskEXIT_CRITICAL();

  event.type = type;
  /*
   * The callback runs synchronously before receiver_task() asks CPX for the
   * next packet, so the RX packet is safe callback-scoped scratch. Keep the
   * TX packet private to send_control(): sharing it with a logger would race
   * a caller connecting or retrying and could replace the Wi-Fi route.
   */
  event.callback_packet = &session->rx_packet;
  if (callback != NULL) {
    callback(&event, callback_context);
  }
}

static void process_status_packet(cpx_wifi_session_impl_t *session,
                                  const CPXPacket_t *packet) {
  cpx_wifi_session_event_type_t event_type;
  bool have_event = false;

  if (packet->dataLength < 1u) {
    return;
  }

  switch (packet->data[0]) {
    case CPX_WIFI_CTRL_STATUS_WIFI_CONNECTED:
      if (packet->dataLength < 5u) {
        return;
      }
      taskENTER_CRITICAL();
      memcpy(session->ip, &packet->data[1], sizeof(session->ip));
      session->wifi_up = 1u;
      session->event_sequence++;
      taskEXIT_CRITICAL();
      event_type = CPX_WIFI_SESSION_EVENT_WIFI_UP;
      have_event = true;
      break;

    case CPX_WIFI_CTRL_STATUS_CLIENT_CONNECTED:
      if (packet->dataLength < 2u) {
        return;
      }
      if (packet->data[1] != 0u) {
        taskENTER_CRITICAL();
        if (session->client_connected == 0u) {
          const TickType_t connected_at = xTaskGetTickCount();
          /* Publish the epoch before the flag: readers check the flag first. */
          session->client_connected_at_ticks = connected_at;
          session->client_connected_at_ms = ticks_to_ms(connected_at);
          session->client_connected = 1u;
          session->event_sequence++;
          have_event = true;
        }
        taskEXIT_CRITICAL();
        event_type = CPX_WIFI_SESSION_EVENT_CLIENT_CONNECTED;
      } else {
        taskENTER_CRITICAL();
        if (session->client_connected != 0u) {
          session->client_connected = 0u;
          session->client_connected_at_ticks = 0u;
          session->client_connected_at_ms = 0u;
          session->event_sequence++;
          have_event = true;
        }
        taskEXIT_CRITICAL();
        event_type = CPX_WIFI_SESSION_EVENT_CLIENT_DISCONNECTED;
      }
      break;

    default:
      break;
  }

  if (have_event) {
    publish_event(session, event_type);
  }
}

static void receiver_task(void *argument) {
  cpx_wifi_session_impl_t *session =
      (cpx_wifi_session_impl_t *)argument;

  while (1) {
    cpxReceivePacketBlocking(CPX_F_WIFI_CTRL, &session->rx_packet);
    process_status_packet(session, &session->rx_packet);
  }
}

static bool send_control(cpx_wifi_session_impl_t *session,
                         cpx_wifi_ctrl_command_t command,
                         const uint8_t *data,
                         size_t data_length) {
  CPXPacket_t *packet = &session->tx_packet;

  if (data_length + 1u > sizeof(packet->data)) {
    return false;
  }
  /*
   * Treat every command as a complete packet. Besides making retries robust,
   * this prevents any accidental scratch use from redirecting a later Wi-Fi
   * control command to another CPX function.
   */
  cpxInitRoute(CPX_T_GAP8, CPX_T_ESP32, CPX_F_WIFI_CTRL, &packet->route);
  packet->data[0] = (uint8_t)command;
  if (data_length != 0u) {
    memcpy(&packet->data[1], data, data_length);
  }
  packet->dataLength = (uint16_t)(data_length + 1u);
  return cpxSendPacket(packet, (uint32_t)portMAX_DELAY);
}

int cpx_wifi_session_config_init(cpx_wifi_session_config_t *config) {
  if (config == NULL) {
    return CPX_WIFI_SESSION_ERR_INVALID_ARGUMENT;
  }
  memset(config, 0, sizeof(*config));
  config->transport = CPX_WIFI_TRANSPORT_TCP;
  config->mode = CPX_WIFI_MODE_ACCESS_POINT;
  config->client_settle_ms = CPX_WIFI_SESSION_DEFAULT_SETTLE_MS;
  return CPX_WIFI_SESSION_OK;
}

int cpx_wifi_session_init(cpx_wifi_session_t *storage,
                          const cpx_wifi_session_config_t *config) {
  cpx_wifi_session_impl_t *session;
  const char *key;
  const char *name;
  size_t hostname_size;
  size_t name_size;

  if (storage == NULL || config == NULL ||
      !valid_string(config->ssid, CPX_WIFI_SSID_MAX_LENGTH, false) ||
      !valid_string(config->key, CPX_WIFI_KEY_MAX_LENGTH, true) ||
      !valid_string(config->hostname, CPX_WIFI_NAME_MAX_LENGTH, true) ||
      !valid_string(config->name, CPX_WIFI_NAME_MAX_LENGTH, true) ||
      !valid_transport(config->transport) || !valid_mode(config->mode) ||
      config->channel > CPX_WIFI_CHANNEL_MAX ||
      !valid_key_for_mode(config->key, config->mode)) {
    return CPX_WIFI_SESSION_ERR_INVALID_ARGUMENT;
  }
  if (storage_looks_active(storage)) {
    return CPX_WIFI_SESSION_ERR_BUSY;
  }

  memset(storage, 0, sizeof(*storage));
  session = session_impl(storage);
  session->magic = CPX_WIFI_SESSION_MAGIC;
  session->self = (uintptr_t)session;
  session->state = CPX_WIFI_STATE_INITIALIZED;
  memcpy(session->ssid, config->ssid, strlen(config->ssid) + 1u);
  key = config->key != NULL ? config->key : "";
  memcpy(session->key, key, strlen(key) + 1u);
  session->transport = config->transport;
  session->mode = config->mode;
  session->channel = config->channel;
  if (config->hostname != NULL && config->hostname[0] != '\0') {
    name = config->name != NULL ? config->name : "";
    hostname_size = strlen(config->hostname) + 1u;
    name_size = strlen(name) + 1u;
    memcpy(session->names, config->hostname, hostname_size);
    memcpy(session->names + hostname_size, name, name_size);
    session->names_length = (uint8_t)(hostname_size + name_size);
  }
  session->settle_ticks = ms_to_ticks(config->client_settle_ms);
  session->receiver_task_stack_words =
      config->receiver_task_stack_words != 0u
          ? config->receiver_task_stack_words
          : CPX_WIFI_SESSION_RECEIVER_STACK_WORDS;
  session->receiver_task_priority =
      config->receiver_task_priority != 0u
          ? config->receiver_task_priority
          : CPX_WIFI_SESSION_RECEIVER_PRIORITY;
  session->event_callback = config->event_callback;
  session->event_callback_context = config->event_callback_context;
  return CPX_WIFI_SESSION_OK;
}

int cpx_wifi_session_open(cpx_wifi_session_t *storage) {
  cpx_wifi_session_impl_t *session;
  int task_result;

  if (storage == NULL) {
    return CPX_WIFI_SESSION_ERR_INVALID_ARGUMENT;
  }
  session = session_impl(storage);
  if (!session_is_initialized(session) ||
      session->state != CPX_WIFI_STATE_INITIALIZED) {
    return CPX_WIFI_SESSION_ERR_INVALID_STATE;
  }

  taskENTER_CRITICAL();
  if (wifi_ctrl_owner != NULL) {
    taskEXIT_CRITICAL();
    return CPX_WIFI_SESSION_ERR_BUSY;
  }
  wifi_ctrl_owner = session;
  session->state = CPX_WIFI_STATE_OPENING;
  taskEXIT_CRITICAL();

  /* cpxInit() is a documented caller precondition. */
  cpxEnableFunction(CPX_F_WIFI_CTRL);
  task_result = xTaskCreate(receiver_task, "cpx_wifi",
                            session->receiver_task_stack_words, session,
                            session->receiver_task_priority,
                            &session->receiver_task);
  if (task_result != pdPASS) {
    taskENTER_CRITICAL();
    session->state = CPX_WIFI_STATE_INITIALIZED;
    wifi_ctrl_owner = NULL;
    taskEXIT_CRITICAL();
    return CPX_WIFI_SESSION_ERR_RTOS;
  }
  taskENTER_CRITICAL();
  session->state = CPX_WIFI_STATE_OPEN;
  taskEXIT_CRITICAL();
  return CPX_WIFI_SESSION_OK;
}

int cpx_wifi_session_connect(cpx_wifi_session_t *storage) {
  cpx_wifi_session_impl_t *session;
  size_t ssid_length;
  size_t key_length;
  uint8_t value;
  bool sent;

  if (storage == NULL) {
    return CPX_WIFI_SESSION_ERR_INVALID_ARGUMENT;
  }
  session = session_impl(storage);
  if (!session_is_initialized(session)) {
    return CPX_WIFI_SESSION_ERR_INVALID_STATE;
  }
  ssid_length = strlen(session->ssid);

  taskENTER_CRITICAL();
  if (session->state != CPX_WIFI_STATE_OPEN) {
    taskEXIT_CRITICAL();
    return CPX_WIFI_SESSION_ERR_INVALID_STATE;
  }
  session->state = CPX_WIFI_STATE_CONNECTING;
  taskEXIT_CRITICAL();

  sent = send_control(session, CPX_WIFI_CTRL_SET_SSID,
                      (const uint8_t *)session->ssid, ssid_length);
  key_length = strlen(session->key);
  if (sent && key_length != 0u) {
    sent = send_control(session, CPX_WIFI_CTRL_SET_KEY,
                        (const uint8_t *)session->key, key_length);
  }
  value = (uint8_t)session->transport;
  if (sent) {
    sent = send_control(session, CPX_WIFI_CTRL_SET_TRANSPORT, &value, 1u);
  }
  if (sent && session->mode == CPX_WIFI_MODE_ACCESS_POINT &&
      session->channel != CPX_WIFI_CHANNEL_DEFAULT) {
    sent = send_control(session, CPX_WIFI_CTRL_SET_CHANNEL, &session->channel, 1u);
  }
  if (sent && session->names_length != 0u) {
    sent = send_control(session, CPX_WIFI_CTRL_SET_NAME,
                        (const uint8_t *)session->names, session->names_length);
  }
  value = (uint8_t)session->mode;
  if (sent) {
    sent = send_control(session, CPX_WIFI_CTRL_WIFI_CONNECT, &value, 1u);
  }

  taskENTER_CRITICAL();
  session->state = sent ? CPX_WIFI_STATE_CONNECT_REQUESTED
                        : CPX_WIFI_STATE_OPEN;
  taskEXIT_CRITICAL();
  return sent ? CPX_WIFI_SESSION_OK : CPX_WIFI_SESSION_ERR_SEND;
}

int cpx_wifi_session_get_status(const cpx_wifi_session_t *storage,
                                cpx_wifi_session_status_t *status) {
  const cpx_wifi_session_impl_t *session;
  TickType_t now;

  if (storage == NULL || status == NULL) {
    return CPX_WIFI_SESSION_ERR_INVALID_ARGUMENT;
  }
  session = session_const_impl(storage);
  if (!session_is_initialized(session)) {
    return CPX_WIFI_SESSION_ERR_INVALID_STATE;
  }

  now = xTaskGetTickCount();
  taskENTER_CRITICAL();
  copy_status_locked(session, now, status);
  taskEXIT_CRITICAL();
  return CPX_WIFI_SESSION_OK;
}

bool cpx_wifi_session_client_connected(const cpx_wifi_session_t *session) {
  cpx_wifi_session_status_t status;
  return cpx_wifi_session_get_status(session, &status) ==
             CPX_WIFI_SESSION_OK &&
         status.client_connected;
}

bool cpx_wifi_session_client_ready(const cpx_wifi_session_t *session) {
  cpx_wifi_session_status_t status;
  return cpx_wifi_session_get_status(session, &status) ==
             CPX_WIFI_SESSION_OK &&
         status.client_ready;
}
