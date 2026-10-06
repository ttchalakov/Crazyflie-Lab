# camera_preview.mk -- plug-and-play QVGA60 JPEG camera preview facade.

ifndef CF_CAMERA_PREVIEW_MK_INCLUDED
CF_CAMERA_PREVIEW_MK_INCLUDED := 1

# The facade uses FreeRTOS tasks, critical sections, and the FreeRTOS-backed
# camera/CPX implementations. Select it automatically unless the application
# explicitly requested an incompatible PMSIS OS.
PMSIS_OS ?= freertos
ifneq ($(strip $(PMSIS_OS)),freertos)
$(error camera_preview requires PMSIS_OS=freertos)
endif

CF_CAMERA_PREVIEW_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
CF_CAMERA_PREVIEW_LIB_ROOT := $(abspath $(CF_CAMERA_PREVIEW_DIR)/../..)

# Each component guards itself, so an application may include them too.
include $(CF_CAMERA_PREVIEW_LIB_ROOT)/gap8/cpx/cpx.mk
include $(CF_CAMERA_PREVIEW_LIB_ROOT)/gap8/camera_stream/camera_stream.mk
include $(CF_CAMERA_PREVIEW_LIB_ROOT)/gap8/cpx_wifi/cpx_wifi.mk
include $(CF_CAMERA_PREVIEW_LIB_ROOT)/gap8/jpeg_pipeline/jpeg_pipeline.mk
include $(CF_CAMERA_PREVIEW_LIB_ROOT)/gap8/imgstream/imgstream.mk

APP_SRCS += $(CF_CAMERA_PREVIEW_DIR)/src/camera_preview.c
APP_INC += $(CF_CAMERA_PREVIEW_DIR)/inc

# The preview transport queue, in packets. TXQ_SIZE sizes the whole
# process's CPX queue, so set it (make TXQ_SIZE=...) for every CPX producer in
# the firmware, not only the preview.
TXQ_SIZE ?= 8
APP_CFLAGS += -DTXQ_SIZE=$(TXQ_SIZE)

# The preset can specialize camera_stream to its one-buffer QVGA60 policy. Turn
# this off when the same binary also creates other camera_stream configurations.
CAMERA_PREVIEW_SPECIALIZE_CAMERA ?= 1
ifeq ($(CAMERA_PREVIEW_SPECIALIZE_CAMERA),1)
APP_CFLAGS += -DCAMERA_STREAM_MAX_CAPTURE_BUFFERS=1
APP_CFLAGS += -DCAMERA_STREAM_SPECIALIZE_SENSOR_MODE=1
APP_CFLAGS += -DCAMERA_STREAM_SPECIALIZE_POLICY=0
endif

endif # CF_CAMERA_PREVIEW_MK_INCLUDED
