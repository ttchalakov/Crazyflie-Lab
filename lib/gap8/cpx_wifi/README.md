# GAP8 CPX Wi-Fi session

`cpx_wifi` owns the nontrivial GAP8 side of the ESP32 Wi-Fi control plane. It is
separate from image framing (`imgstream`) and from the camera, so the same session
can carry images, inference results, telemetry, or another CPX application.

```c
cpx_wifi_session_config_t config;
cpx_wifi_session_t session;

cpx_wifi_session_config_init(&config);
config.ssid = "my-deck";
config.key = NULL; /* open AP; at most 64 bytes, and an AP key at least 8 */
config.transport = CPX_WIFI_TRANSPORT_TCP; /* or UDP */
config.mode = CPX_WIFI_MODE_ACCESS_POINT;
config.channel = 1; /* access point channel 1..13; 0 keeps the ESP32's default, 6 */
config.hostname = "cf-80-2m-e7e7e7e7e7"; /* mDNS name; NULL keeps aideck-XXXXXX */
config.name = "80/2M/E7E7E7E7E7";        /* its _cpx._tcp TXT "name" */
cpx_wifi_session_init(&session, &config);  /* copy/validate only */

cpxInit();
cpx_wifi_session_open(&session);           /* receiver queue/task */
cpx_wifi_session_connect(&session);        /* SSID/key/transport/channel/name/connect */

if (cpx_wifi_session_client_ready(&session)) {
  /* encode/admit/send */
}
```

The caller must initialize CPX first. `open()` creates the one receiver for
`CPX_F_WIFI_CTRL`; CPX has no function-queue teardown, so an opened session
lasts for the application lifetime. It does not initialize
CPX, send images, or stop a camera when a client disconnects.

The session consumes the ESP32's explicit client connect/disconnect status for
either transport; it does not invent a local UDP inactivity timeout. The readiness
accessor also applies the configured post-connect settle period with tick-wrap-safe
arithmetic. RX/TX `CPXPacket_t` storage is held in the caller-owned object
instead of the receiver task's small stack, and no event-path allocation
occurs. Control sends have exclusive use of the TX packet and initialize its
Wi-Fi route for every command. A synchronous event callback may borrow the RX
packet until it returns; the receiver then immediately reclaims it.

Logging is separately selectable. Set the session callback to
`cpx_wifi_log_event_callback` and include `cpx_wifi_log.mk` to print the
session's status lines on the radio console, or provide an application callback.
The adapter reuses the session's callback-scoped RX packet and calls
`cpxSendPacket(..., 0)`, dropping a diagnostic line instead of stalling Wi-Fi
client-state processing when transport is full.

```make
include $(CF_LAB)/lib/gap8/cpx/cpx.mk
include $(CF_LAB)/lib/gap8/cpx_wifi/cpx_wifi.mk
# Optional:
include $(CF_LAB)/lib/gap8/cpx_wifi/cpx_wifi_log.mk
```
