io=uart
PMSIS_OS=freertos

APP = hover_neural_network
APP_SRCS += hover_neural_network.c
APP_INC += . ../include
APP_CFLAGS += -O3 -g

CF_LAB ?= $(abspath $(CURDIR)/../../..)

POLICY_GENERATED_DIR := policy/generated
POLICY_EPSILON_FILE := ../build/dory/policy/nemo/output_epsilon.txt

# generated/ is a deploy artifact. Keep the failure close to the cause instead
# of allowing the compiler to fail later with a misleading missing symbol.
ifeq ($(wildcard $(POLICY_GENERATED_DIR)/inc/policy_network.h),)
ifneq ($(strip $(MAKECMDGOALS)),clean)
$(error Missing $(POLICY_GENERATED_DIR). Generate the NEMO/DORY policy first; see policy/README.md)
endif
endif
# NEMO's output quantum: the one deployment value the trained weights determine,
# rather than a constant in model/hover_policy.py. The export writes it beside its
# ONNX and the build reads it here, so it cannot be pasted wrong or go stale.
ifeq ($(wildcard $(POLICY_EPSILON_FILE)),)
ifneq ($(strip $(MAKECMDGOALS)),clean)
$(error Missing $(POLICY_EPSILON_FILE). Run the NEMO/DORY deploy first; see policy/README.md)
endif
endif
APP_CFLAGS += -DHNN_POLICY_OUTPUT_EPSILON=$(strip $(shell cat $(POLICY_EPSILON_FILE)))f

# The generated policy's sources, include path and the flags a DORY network needs
# inside a FreeRTOS app. dory.mk would otherwise apply DORY's -O2 to every source.
DORY_NET_DIR := $(POLICY_GENERATED_DIR)
DORY_OPT :=
include $(CF_LAB)/lib/gap8/dory/dory.mk
# The app is built for the 2-level target only: resident L2 weights, no flash.
ifneq ($(strip $(DORY_NET_HEX)),)
$(error $(DORY_NET_DIR) holds a 3-level network; deploy with --dory-target PULP.GAP8_L2)
endif

# Camera and video: QVGA at 60 fps, JPEG-encoded on the cluster.
SENSOR = QVGA_60
FORMAT = JPEG
# Anti-flicker for the mocap cameras' IR strobes, which fire at the mocap frame
# rate: the auto exposure steps in 8.33 ms, a whole number of strobe periods at
# 120, 240 or 360 Hz. Use CAMERA_STREAM_FLICKER_50_HZ for a 100 or 200 Hz system.
APP_CFLAGS += -DCAMERA_STREAM_DEFAULT_FLICKER_MODE=CAMERA_STREAM_FLICKER_60_HZ
# The scheduler, CPX, WiFi (WIFI_DEFINES, from the drone config's camera:
# section via cf-lab), the camera, the JPEG encoder and fastmem.
include $(CF_LAB)/lib/gap8/sched/sched.mk

include $(RULES_DIR)/pmsis_rules.mk
