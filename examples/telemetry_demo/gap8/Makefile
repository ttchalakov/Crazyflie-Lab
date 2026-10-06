io=uart
PMSIS_OS=freertos

APP = telemetry_demo
APP_SRCS += telemetry_demo.c
CF_LAB ?= $(abspath $(CURDIR)/../../..)
include $(CF_LAB)/lib/gap8/cpx_bridge/cpx_bridge.mk
APP_CFLAGS += -O2 -g

include $(RULES_DIR)/pmsis_rules.mk
