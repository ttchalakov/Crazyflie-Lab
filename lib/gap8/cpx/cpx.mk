# cpx.mk -- CPX transport for GAP8. Include this from an app's Makefile:
#
#     CF_LAB ?= $(abspath $(CURDIR)/../cf-lab)
#     include $(CF_LAB)/lib/gap8/cpx/cpx.mk
#
# and override any knob (see inc/cpx_config.h) from the app, e.g.
#
#     APP_CFLAGS += -DTXQ_SIZE=160
#
# Self-locating: it works out its own directory from MAKEFILE_LIST, so callers do no
# path arithmetic beyond pointing CF_LAB at the cf-lab checkout.
#
# This resolves because the GAP8 Docker build mounts the cf-lab checkout, not
# just the app directory (cf-lab build gap8). See lib/README.md.

ifndef CF_CPX_MK_INCLUDED
CF_CPX_MK_INCLUDED := 1

CF_CPX_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

APP_SRCS += $(CF_CPX_DIR)/src/com.c
APP_SRCS += $(CF_CPX_DIR)/src/cpx.c

APP_INC += $(CF_CPX_DIR)/inc
APP_INC += $(CF_CPX_DIR)/../../shared/include

# CPX needs FreeRTOS timers for its transport task.
APP_CFLAGS += -DconfigUSE_TIMERS=1 -DINCLUDE_xTimerPendFunctionCall=1

endif # CF_CPX_MK_INCLUDED
