# sched.mk -- lib/gap8/sched and the libraries under it.
#
# Include from an app Makefile after setting DORY_NET_DIR and including
# lib/gap8/dory/dory.mk. Make variables (all optional):
#
#   SENSOR=QVGA_60     QVGA_30 QVGA_60 QQVGA_BIN2_30 QQVGA_BIN2_60 QQVGA_BIN2_120
#   FORMAT=JPEG        JPEG (cluster) or RAW, for the video stream
#   QUALITY=50         JPEG quality
#   BUFFERS=           camera buffers, 2 or 3 (default: 2 at QVGA, else 3)
#   STREAM=1           0 sends no video
#   SCHED=PIPELINED    PIPELINED, SEQUENTIAL, or BOTH (a bench comparing them)
#   SCHED_METRICS=0 SCHED_VERIFY=0 SCHED_LOG=0   instrumentation, 1 to compile in
#   WIFI_DEFINES=...   -DWIFI_SSID=... etc., as cf_lab passes them
#   TXQ_SIZE           CPX send-queue slots (~1 KB of L2 each); defaults to what
#                      one frame of the chosen sensor and format needs

ifndef CF_SCHED_MK_INCLUDED
CF_SCHED_MK_INCLUDED := 1

CF_SCHED_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
CF_GAP8_LIB := $(CF_SCHED_DIR)/..

include $(CF_GAP8_LIB)/cpx/cpx.mk
include $(CF_GAP8_LIB)/cpx_wifi/cpx_wifi.mk
include $(CF_GAP8_LIB)/camera_stream/camera_stream.mk
include $(CF_GAP8_LIB)/jpeg_pipeline/jpeg_pipeline.mk
include $(CF_GAP8_LIB)/imgstream/imgstream.mk
# The SDK's memcpy copies a byte at a time; every frame and packet copy is FC time.
include $(CF_GAP8_LIB)/fastmem/fastmem.mk

APP_SRCS += $(CF_SCHED_DIR)/src/sched.c
APP_INC  += $(CF_SCHED_DIR)/inc

sched_upper = $(shell printf '%s' '$1' | tr a-z A-Z)
sched_choose = $(if $($1),$(if $(filter $(call sched_upper,$($1)),$2),-D$3=$4$(call sched_upper,$($1)),$(error $1=$($1) is not one of: $2)))
sched_number = $(if $($1),-D$2=$($1))

APP_CFLAGS += $(call sched_choose,SENSOR,QVGA_30 QVGA_60 QQVGA_BIN2_30 QQVGA_BIN2_60 QQVGA_BIN2_120,SCHED_SENSOR_MODE,CAMERA_STREAM_SENSOR_)
APP_CFLAGS += $(call sched_choose,FORMAT,RAW JPEG,SCHED_STREAM_FORMAT,SCHED_FORMAT_)
APP_CFLAGS += $(call sched_number,QUALITY,SCHED_JPEG_QUALITY)
APP_CFLAGS += $(call sched_number,BUFFERS,SCHED_CAMERA_BUFFERS)
APP_CFLAGS += $(call sched_number,STREAM,SCHED_STREAM)
APP_CFLAGS += $(call sched_number,SCHED_METRICS,SCHED_METRICS)
APP_CFLAGS += $(call sched_number,SCHED_VERIFY,SCHED_VERIFY)
APP_CFLAGS += $(call sched_number,SCHED_LOG,SCHED_LOG)
APP_CFLAGS += $(WIFI_DEFINES)

SCHED ?= PIPELINED
ifeq ($(call sched_upper,$(SCHED)),PIPELINED)
APP_CFLAGS += -DSCHED_BUILD_PIPELINED=1 -DSCHED_BUILD_SEQUENTIAL=0
else ifeq ($(call sched_upper,$(SCHED)),SEQUENTIAL)
APP_CFLAGS += -DSCHED_BUILD_PIPELINED=0 -DSCHED_BUILD_SEQUENTIAL=1
else ifeq ($(call sched_upper,$(SCHED)),BOTH)
APP_CFLAGS += -DSCHED_BUILD_PIPELINED=1 -DSCHED_BUILD_SEQUENTIAL=1
else
$(error SCHED=$(SCHED) is not one of: PIPELINED SEQUENTIAL BOTH)
endif

# The CPX send queue holds a whole frame, so a frame is copied into it at once
# and its camera buffer goes back. Packets carry 1020 bytes: raw QVGA is 76 of
# them, raw QQVGA 19; a JPEG at most its output budget (a third of the raw
# frame: 26 at QVGA, 8 at QQVGA). Plus the header packet and room for console
# lines.
ifndef TXQ_SIZE
ifeq ($(call sched_upper,$(FORMAT)),RAW)
TXQ_SIZE := $(if $(findstring QQVGA,$(call sched_upper,$(SENSOR))),24,80)
else
TXQ_SIZE := $(if $(findstring QQVGA,$(call sched_upper,$(SENSOR))),12,30)
endif
endif
APP_CFLAGS += -DTXQ_SIZE=$(TXQ_SIZE)

endif # CF_SCHED_MK_INCLUDED
