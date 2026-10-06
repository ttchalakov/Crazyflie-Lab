io=uart
PMSIS_OS=freertos
APP = wifi_bench
APP_SRCS += wifi_bench.c
CF_LAB ?= $(abspath $(CURDIR)/../../..)
APP_CFLAGS += -O2 -g $(WIFI_DEFINES)
include $(CF_LAB)/lib/gap8/cpx/cpx.mk
include $(CF_LAB)/lib/gap8/cpx_wifi/cpx_wifi.mk
include $(RULES_DIR)/pmsis_rules.mk
