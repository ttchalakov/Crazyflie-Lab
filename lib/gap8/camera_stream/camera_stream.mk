# camera_stream.mk -- transport-independent continuous Himax capture for GAP8.
#
#     CF_LAB ?= $(abspath $(CURDIR)/../cf-lab)
#     include $(CF_LAB)/lib/gap8/camera_stream/camera_stream.mk
#
# The library owns Himax, raw L2 capture buffers, one event group/mutex, and a
# small ownership/watchdog worker. JPEG, resizing, CPX, WiFi and wire framing
# remain application concerns.

ifndef CF_CAMERA_STREAM_MK_INCLUDED
CF_CAMERA_STREAM_MK_INCLUDED := 1

CF_CAMERA_STREAM_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

APP_SRCS += $(CF_CAMERA_STREAM_DIR)/src/camera_stream.c
APP_INC  += $(CF_CAMERA_STREAM_DIR)/inc

endif # CF_CAMERA_STREAM_MK_INCLUDED
