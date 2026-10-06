# cpx_wifi.mk -- ESP32 Wi-Fi control-plane session over CPX for GAP8.
# Include CPX first; this module deliberately does not initialize or own CPX:
#
#     include $(CF_LAB)/lib/gap8/cpx/cpx.mk
#     include $(CF_LAB)/lib/gap8/cpx_wifi/cpx_wifi.mk
#
# Image framing and transmission remain in lib/gap8/imgstream.

ifndef CF_CPX_WIFI_MK_INCLUDED
CF_CPX_WIFI_MK_INCLUDED := 1

CF_CPX_WIFI_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

APP_SRCS += $(CF_CPX_WIFI_DIR)/src/cpx_wifi_session.c
APP_INC  += $(CF_CPX_WIFI_DIR)/inc

endif # CF_CPX_WIFI_MK_INCLUDED
