#ifndef CF_GAP8_CPX_WIFI_PROTOCOL_H
#define CF_GAP8_CPX_WIFI_PROTOCOL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * GAP8 <-> ESP32 Wi-Fi control protocol carried by CPX_F_WIFI_CTRL.
 * These numeric values are part of the wire ABI and must remain synchronized
 * with external/aideck-esp-firmware/main/wifi.c.
 */
typedef enum {
  CPX_WIFI_CTRL_SET_SSID = 0x10,
  CPX_WIFI_CTRL_SET_KEY = 0x11,
  /* Access point channel, 1..13. */
  CPX_WIFI_CTRL_SET_CHANNEL = 0x12,
  /* The deck's mDNS hostname and TXT name, "<hostname>\0<name>\0". */
  CPX_WIFI_CTRL_SET_NAME = 0x13,

  CPX_WIFI_CTRL_WIFI_CONNECT = 0x20,
  CPX_WIFI_CTRL_SET_TRANSPORT = 0x21,

  CPX_WIFI_CTRL_STATUS_WIFI_CONNECTED = 0x31,
  CPX_WIFI_CTRL_STATUS_CLIENT_CONNECTED = 0x32,
} cpx_wifi_ctrl_command_t;

typedef enum {
  CPX_WIFI_TRANSPORT_TCP = 0,
  CPX_WIFI_TRANSPORT_UDP = 1,
} cpx_wifi_transport_t;

typedef enum {
  CPX_WIFI_MODE_STATION = 0,
  CPX_WIFI_MODE_ACCESS_POINT = 1,
} cpx_wifi_mode_t;

/* The 2.4 GHz channels an access point may use; 0 leaves the ESP32's default (6). */
#define CPX_WIFI_CHANNEL_DEFAULT 0u
#define CPX_WIFI_CHANNEL_MAX 13u

/* An 802.11 SSID is at most 32 bytes; the ESP32 and cf_lab reject a longer one. */
#define CPX_WIFI_SSID_MAX_LENGTH 32u

/* The longest key: a WPA2 key is an 8-63 character passphrase or 64 hex digits,
 * and the ESP32 takes up to 64. */
#define CPX_WIFI_KEY_MAX_LENGTH 64u

/* The longest hostname or name; cf_lab's (cf-80-2m-e7e7e7e7e7) are 22 at most. */
#define CPX_WIFI_NAME_MAX_LENGTH 23u

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_CPX_WIFI_PROTOCOL_H */
