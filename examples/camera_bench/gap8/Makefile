# camera_bench -- camera_stream's pipeline, instrumented for the bench.
#
# The bench runner (host/bench.py) builds this for each case in host/bench.yaml.
# It takes these of camera_stream's camera make variables, documented in
# examples/camera_stream/gap8/Makefile: SENSOR, SIZE, FIT, FORMAT, JPEG, QUALITY,
# POLICY, BUFFERS, FPS and FLIP. Window and exposure stay at the camera library's
# defaults. Plus two of its own:
#
#   LIFECYCLE      0 1    run the camera start/stop/resume/close self-test instead of streaming
#   TRACE          0 1    a trace line for each of the first frames
#
# WiFi is the bench's access point from bench.yaml, passed as WIFI_DEFINES.

io=uart
PMSIS_OS=freertos

APP = camera_bench
APP_SRCS += camera_bench.c
APP_INC += .
APP_CFLAGS += -O3 -g

CF_LAB ?= $(abspath $(CURDIR)/../../..)
include $(CF_LAB)/lib/gap8/cpx/cpx.mk
include $(CF_LAB)/lib/gap8/cpx_wifi/cpx_wifi.mk
include $(CF_LAB)/lib/gap8/cpx_wifi/cpx_wifi_log.mk
include $(CF_LAB)/lib/gap8/camera_stream/camera_stream.mk
include $(CF_LAB)/lib/gap8/camera_pipeline/camera_pipeline.mk
include $(CF_LAB)/lib/gap8/jpeg_pipeline/jpeg_pipeline.mk
include $(CF_LAB)/lib/gap8/imgstream/imgstream.mk
include $(CF_LAB)/lib/gap8/camera_stream_log/camera_stream_log.mk

# $(call choose,VAR,CHOICES,MACRO,PREFIX) -> -DMACRO=PREFIX<VALUE> when VAR is set.
upper = $(shell printf '%s' '$1' | tr a-z A-Z)
choose = $(if $($1),$(if $(filter $(call upper,$($1)),$2),-D$3=$4$(call upper,$($1)),$(error $1=$($1) is not one of: $2)))
number = $(if $($1),-D$2=$($1))

APP_CFLAGS += $(call choose,SENSOR,QVGA_30 QVGA_60 QQVGA_BIN2_30 QQVGA_BIN2_60 QQVGA_BIN2_120,SENSOR_MODE,SENSOR_MODE_)
APP_CFLAGS += $(call choose,FIT,SQUASH CROP,OUT_FIT,FIT_)
APP_CFLAGS += $(call choose,FORMAT,RAW JPEG,STREAM_FORMAT,FORMAT_)
APP_CFLAGS += $(call choose,JPEG,FC CLUSTER,JPEG_ENCODER,JPEG_ENCODER_)
APP_CFLAGS += $(call number,QUALITY,JPEG_QUALITY)
APP_CFLAGS += $(call choose,POLICY,PREVIEW EXACT,CAPTURE_POLICY,CAPTURE_POLICY_)
APP_CFLAGS += $(call choose,BUFFERS,2 3,EXACT_BUFFERS,)
ifneq ($(BUFFERS),)
ifneq ($(call upper,$(POLICY)),EXACT)
$(error BUFFERS is for POLICY=EXACT; preview always has one buffer)
endif
endif
APP_CFLAGS += $(call number,FPS,TARGET_FPS)
APP_CFLAGS += $(call number,FLIP,CAMERA_STREAM_DEFAULT_ORIENTATION)
APP_CFLAGS += $(call choose,LIFECYCLE,0 1,BENCH_LIFECYCLE,)
APP_CFLAGS += $(call choose,TRACE,0 1,BENCH_TRACE,)
ifdef SIZE
size_words := $(subst x, ,$(subst X, ,$(SIZE)))
$(if $(filter 2,$(words $(size_words))),,$(error SIZE=$(SIZE) is not WxH, e.g. 96x96))
APP_CFLAGS += -DOUT_WIDTH=$(word 1,$(size_words)) -DOUT_HEIGHT=$(word 2,$(size_words))
endif
# Task stack headroom for the status line.
APP_CFLAGS += -DINCLUDE_uxTaskGetStackHighWaterMark=1

APP_CFLAGS += $(WIFI_DEFINES)

# Every source, the libraries included, sees this app's settings, so the
# libraries are sized for exactly this build (see camera_bench_config.h).
APP_CFLAGS += -include camera_bench_config.h

include $(RULES_DIR)/pmsis_rules.mk
