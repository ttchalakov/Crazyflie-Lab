io=uart
PMSIS_OS=freertos

APP = debug_demo
APP_SRCS += debug_demo.c

# CPX transport, from the shared C library.
CF_LAB ?= $(abspath $(CURDIR)/../../..)
include $(CF_LAB)/lib/gap8/cpx/cpx.mk
APP_INC += .

APP_CFLAGS += -O3 -g


include $(RULES_DIR)/pmsis_rules.mk
