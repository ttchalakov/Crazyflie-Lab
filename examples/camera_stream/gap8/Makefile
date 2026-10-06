# camera_stream -- the AI-deck camera as a WiFi video server.
#
# Camera settings are make variables, passed through cf-lab:
#
#   cf-lab flash gap8 --example camera_stream --make-args "SENSOR=QVGA_60 FORMAT=JPEG JPEG=CLUSTER"
#
# Each one becomes a -D flag: the stream settings are named beside their default
# in camera_stream_config.h, and FLIP and the exposure settings set
# lib/gap8/camera_stream's own defaults (CAMERA_STREAM_DEFAULT_* in
# camera_stream.h). Leave a variable out to keep its default. Values are not
# case-sensitive.
#
#   SENSOR         QVGA_30 QVGA_60 QQVGA_BIN2_30 QQVGA_BIN2_60 QQVGA_BIN2_120
#   SIZE           output WxH, e.g. 96x96 (default: the capture size)
#   FIT            SQUASH CROP                     how SIZE is reached
#   FORMAT         RAW JPEG
#   JPEG           FC CLUSTER                      JPEG encoder
#   QUALITY        1..100                          JPEG quality
#   POLICY         PREVIEW EXACT
#   BUFFERS        2 3                             EXACT camera buffers (default 2)
#   FPS            frame rate cap, below the sensor's
#   FLIP           0..3                            orientation bits
#   WINDOW         0..84                           first sensor row shown (42 = centred)
#   EXPOSURE       AUTO MANUAL
#   AE_TARGET      0..255                          auto exposure brightness
#   AE_MAX_EXPOSURE_US   microseconds              auto: longest exposure (0 = frame)
#   AE_MAX_ANALOG_GAIN   1 2 4 8                   auto: highest analog gain
#   AE_MAX_DIGITAL_GAIN  256..1023                 auto: highest digital gain, 256 = 1x
#   EXPOSURE_US    microseconds                    manual exposure time
#   ANALOG_GAIN    1 2 4 8                         manual analog gain
#   DIGITAL_GAIN   1..1023                         manual, 256 = 1x
#   FLICKER        OFF 50_HZ 60_HZ                 auto exposure only
#   LOG            0 1                             status line every second
#
# WiFi comes from the drone config: `cf-lab build gap8` and `cf-lab flash gap8`
# pass the camera: section of --config as WIFI_DEFINES.

io=uart
PMSIS_OS=freertos

APP = camera_stream
APP_SRCS += camera_stream.c
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

# $(call choose,VAR,CHOICES,MACRO,PREFIX[,SUFFIX]) -> -DMACRO=PREFIX<VALUE>SUFFIX when VAR is set.
upper = $(shell printf '%s' '$1' | tr a-z A-Z)
choose = $(if $($1),$(if $(filter $(call upper,$($1)),$2),-D$3=$4$(call upper,$($1))$5,$(error $1=$($1) is not one of: $2)))
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
APP_CFLAGS += $(call number,WINDOW,WINDOW_TOP)
APP_CFLAGS += $(call number,FLIP,CAMERA_STREAM_DEFAULT_ORIENTATION)
APP_CFLAGS += $(call choose,EXPOSURE,AUTO MANUAL,CAMERA_STREAM_DEFAULT_EXPOSURE_MODE,CAMERA_STREAM_EXPOSURE_)
APP_CFLAGS += $(call number,AE_TARGET,CAMERA_STREAM_DEFAULT_AE_TARGET_MEAN)
APP_CFLAGS += $(call number,AE_MAX_EXPOSURE_US,CAMERA_STREAM_DEFAULT_AE_MAX_EXPOSURE_US)
APP_CFLAGS += $(call choose,AE_MAX_ANALOG_GAIN,1 2 4 8,CAMERA_STREAM_DEFAULT_AE_MAX_ANALOG_GAIN,CAMERA_STREAM_ANALOG_GAIN_,X)
APP_CFLAGS += $(call number,AE_MAX_DIGITAL_GAIN,CAMERA_STREAM_DEFAULT_AE_MAX_DIGITAL_GAIN)
APP_CFLAGS += $(call number,EXPOSURE_US,CAMERA_STREAM_DEFAULT_MANUAL_EXPOSURE_US)
APP_CFLAGS += $(call choose,ANALOG_GAIN,1 2 4 8,CAMERA_STREAM_DEFAULT_MANUAL_ANALOG_GAIN,CAMERA_STREAM_ANALOG_GAIN_,X)
APP_CFLAGS += $(call number,DIGITAL_GAIN,CAMERA_STREAM_DEFAULT_MANUAL_DIGITAL_GAIN)
# The library spells FLICKER=OFF as CAMERA_STREAM_FLICKER_DISABLED.
APP_CFLAGS += $(subst FLICKER_OFF,FLICKER_DISABLED,$(call choose,FLICKER,OFF 50_HZ 60_HZ,CAMERA_STREAM_DEFAULT_FLICKER_MODE,CAMERA_STREAM_FLICKER_))
APP_CFLAGS += $(call choose,LOG,0 1,CAMERA_STREAM_LOG,)
ifdef SIZE
size_words := $(subst x, ,$(subst X, ,$(SIZE)))
$(if $(filter 2,$(words $(size_words))),,$(error SIZE=$(SIZE) is not WxH, e.g. 96x96))
APP_CFLAGS += -DOUT_WIDTH=$(word 1,$(size_words)) -DOUT_HEIGHT=$(word 2,$(size_words))
endif
ifeq ($(LOG),1)
include $(CF_LAB)/lib/gap8/camera_stream_log/camera_stream_log.mk
endif

APP_CFLAGS += $(WIFI_DEFINES)

# Every source, the libraries included, sees this app's settings, so the
# libraries are sized for exactly this build (see camera_stream_config.h).
APP_CFLAGS += -include camera_stream_config.h

include $(RULES_DIR)/pmsis_rules.mk
