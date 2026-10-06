io=uart
PMSIS_OS=freertos

APP = telemetry_bench
APP_SRCS += telemetry_bench.c
APP_INC += ../include
CF_LAB ?= $(abspath $(CURDIR)/../../..)
include $(CF_LAB)/lib/gap8/cpx_bridge/cpx_bridge.mk
APP_CFLAGS += -O2 -g
APP_LDFLAGS += -lm

include $(RULES_DIR)/pmsis_rules.mk
