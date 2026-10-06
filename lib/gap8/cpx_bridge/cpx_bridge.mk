ifndef CF_CPX_BRIDGE_MK_INCLUDED
CF_CPX_BRIDGE_MK_INCLUDED := 1
CF_CPX_BRIDGE_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
include $(CF_CPX_BRIDGE_DIR)/../cpx/cpx.mk
APP_INC += $(CF_CPX_BRIDGE_DIR)/inc
APP_SRCS += $(CF_CPX_BRIDGE_DIR)/../../shared/src/cpx_bridge.c
APP_SRCS += $(CF_CPX_BRIDGE_DIR)/src/cpx_bridge_gap8.c
endif
