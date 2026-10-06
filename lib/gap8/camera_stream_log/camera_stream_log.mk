# camera_stream_log.mk -- optional camera/transport status line over CPX.
#
# Dependencies are explicit and intentionally not included recursively. Include
# these fragments first in the application, in this order:
#
#   include $(CF_LAB)/lib/gap8/cpx/cpx.mk
#   include $(CF_LAB)/lib/gap8/camera_stream/camera_stream.mk
#   include $(CF_LAB)/lib/gap8/camera_stream_log/camera_stream_log.mk

ifndef CF_CAMERA_STREAM_LOG_INCLUDED
CF_CAMERA_STREAM_LOG_INCLUDED := 1

CF_CAMERA_STREAM_LOG_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

APP_SRCS += $(CF_CAMERA_STREAM_LOG_DIR)/src/camera_stream_log.c
APP_INC  += $(CF_CAMERA_STREAM_LOG_DIR)/inc

endif
