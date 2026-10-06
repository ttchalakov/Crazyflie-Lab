# cpx_wifi_log.mk -- optional console logging for CPX Wi-Fi events.
# Include after CPX and cpx_wifi:
#
#     include $(CF_LAB)/lib/gap8/cpx/cpx.mk
#     include $(CF_LAB)/lib/gap8/cpx_wifi/cpx_wifi.mk
#     include $(CF_LAB)/lib/gap8/cpx_wifi/cpx_wifi_log.mk
#
# The adapter is stateless and does not add another task or allocation.

CF_CPX_WIFI_LOG_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

APP_SRCS += $(CF_CPX_WIFI_LOG_DIR)/src/cpx_wifi_log.c
APP_INC  += $(CF_CPX_WIFI_LOG_DIR)/inc
