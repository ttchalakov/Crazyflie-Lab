# camera_pipeline.mk -- output size and frame ownership above camera_stream.
#
#     include $(CF_LAB)/lib/gap8/camera_stream/camera_stream.mk
#     include $(CF_LAB)/lib/gap8/camera_pipeline/camera_pipeline.mk
#
# Brings the one resize (downsample.h, also usable on its own). JPEG, transport,
# logging, and Wi-Fi remain separately selectable components.

ifndef CF_CAMERA_PIPELINE_MK_INCLUDED
CF_CAMERA_PIPELINE_MK_INCLUDED := 1

CF_CAMERA_PIPELINE_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

APP_SRCS += $(CF_CAMERA_PIPELINE_DIR)/src/camera_pipeline.c
APP_SRCS += $(CF_CAMERA_PIPELINE_DIR)/src/downsample.c
APP_INC  += $(CF_CAMERA_PIPELINE_DIR)/inc

endif # CF_CAMERA_PIPELINE_MK_INCLUDED
