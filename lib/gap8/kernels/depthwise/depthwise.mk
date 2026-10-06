# DORY-compatible batch-1 uint8 x int8 depthwise kernels for GAP8.

ifndef CF_QDEPTHWISE_MK_INCLUDED
CF_QDEPTHWISE_MK_INCLUDED := 1

CF_QDEPTHWISE_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
CF_QDEPTHWISE_REPO_ROOT := $(abspath $(CF_QDEPTHWISE_DIR)/../../../..)
CF_QDEPTHWISE_PULP_NN ?= $(CF_QDEPTHWISE_REPO_ROOT)/external/dory/dory/Hardware_targets/PULP/Backend_Kernels/pulp-nn/32bit
CF_QDEPTHWISE_REPLACE_PULP_NN ?= 0

APP_SRCS += $(CF_QDEPTHWISE_DIR)/src/cf_qdepthwise_reference.c
APP_SRCS += $(CF_QDEPTHWISE_DIR)/src/cf_qdepthwise_pulp_nn.c
APP_SRCS += $(CF_QDEPTHWISE_DIR)/src/cf_qdepthwise_direct3x3.c
APP_SRCS += $(CF_QDEPTHWISE_DIR)/src/cf_qdepthwise_channel4.c
APP_SRCS += $(CF_QDEPTHWISE_DIR)/src/cf_qdepthwise_vendor.c

ifeq ($(CF_QDEPTHWISE_REPLACE_PULP_NN),1)
APP_CFLAGS += -DCF_QDEPTHWISE_REPLACE_PULP_NN=1
else
APP_SRCS += $(CF_QDEPTHWISE_PULP_NN)/src/pulp_nn_depthwise_generic.c
endif

ifndef CF_PULP_NN_UTILS_SRC_INCLUDED
CF_PULP_NN_UTILS_SRC_INCLUDED := 1
APP_SRCS += $(CF_QDEPTHWISE_PULP_NN)/src/pulp_nn_utils.c
endif

APP_INC += $(CF_QDEPTHWISE_DIR)/inc
# cf_dory_profile.h: compiled out unless CF_DORY_PROFILE=1.
APP_INC += $(abspath $(CF_QDEPTHWISE_DIR)/../../dory)
# src/ carries cf_qdepthwise_internal.h, which a fixture must include to size
# and fill the bias at the stride the kernel reads it back at.
APP_INC += $(CF_QDEPTHWISE_DIR)/src
APP_INC += $(CF_QDEPTHWISE_PULP_NN)/include
APP_INC += $(CF_QDEPTHWISE_PULP_NN)/src

# NUM_CORES sizes every PULP-NN kernel's work split. One knob for the whole app
# (dory.mk and the kernel fragments share it), passed to the compiler once.
NUM_CORES ?= 8
ifndef CF_NUM_CORES_DEFINED
CF_NUM_CORES_DEFINED := 1
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
endif

endif # CF_QDEPTHWISE_MK_INCLUDED
