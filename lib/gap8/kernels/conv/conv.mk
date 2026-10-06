# DORY-compatible batch-1 uint8 x int8 convolution kernels for GAP8.

ifndef CF_QCONV_MK_INCLUDED
CF_QCONV_MK_INCLUDED := 1

CF_QCONV_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
CF_QCONV_REPO_ROOT := $(abspath $(CF_QCONV_DIR)/../../../..)
CF_QCONV_PULP_NN ?= $(CF_QCONV_REPO_ROOT)/external/dory/dory/Hardware_targets/PULP/Backend_Kernels/pulp-nn/32bit
CF_QCONV_REPLACE_PULP_NN ?= 0

APP_SRCS += $(CF_QCONV_DIR)/src/cf_qconv_reference.c
APP_SRCS += $(CF_QCONV_DIR)/src/cf_qconv_pulp_nn.c
APP_SRCS += $(CF_QCONV_DIR)/src/cf_qconv_matmul4x2.c
APP_SRCS += $(CF_QCONV_DIR)/src/cf_qconv_matmul4x2_acc.c
APP_SRCS += $(CF_QCONV_DIR)/src/cf_qconv_matmul4x4.c
APP_SRCS += $(CF_QCONV_DIR)/src/cf_qconv_matmul4x3.c
APP_SRCS += $(CF_QCONV_DIR)/src/cf_qconv_matmul_conv.c
APP_SRCS += $(CF_QCONV_DIR)/src/cf_qconv_pointwise.c
ifeq ($(CF_QCONV_REPLACE_PULP_NN),1)
APP_CFLAGS += -DCF_QCONV_REPLACE_PULP_NN=1
endif
APP_SRCS += $(CF_QCONV_DIR)/src/cf_qconv_flattened.c
APP_SRCS += $(CF_QCONV_DIR)/src/cf_qconv_stem.c
APP_SRCS += $(CF_QCONV_DIR)/src/cf_qconv_stem_direct.c

ifeq ($(CF_QCONV_REPLACE_PULP_NN),0)
APP_SRCS += $(CF_QCONV_PULP_NN)/src/pulp_nn_conv_Ho_parallel.c
APP_SRCS += $(CF_QCONV_PULP_NN)/src/pulp_nn_matmul.c
APP_SRCS += $(CF_QCONV_PULP_NN)/src/pulp_nn_pointwise_HoWo_parallel.c
endif
ifndef CF_PULP_NN_UTILS_SRC_INCLUDED
CF_PULP_NN_UTILS_SRC_INCLUDED := 1
APP_SRCS += $(CF_QCONV_PULP_NN)/src/pulp_nn_utils.c
endif

APP_INC += $(CF_QCONV_DIR)/inc
# cf_dory_profile.h: compiled out unless CF_DORY_PROFILE=1.
APP_INC += $(abspath $(CF_QCONV_DIR)/../../dory)
# src/ carries cf_qconv_internal.h, which a fixture must include to size and
# fill the bias at the stride the kernel reads it back at.
APP_INC += $(CF_QCONV_DIR)/src
APP_INC += $(CF_QCONV_PULP_NN)/include
APP_INC += $(CF_QCONV_PULP_NN)/src

# NUM_CORES sizes every PULP-NN kernel's work split. One knob for the whole app
# (dory.mk and the kernel fragments share it), passed to the compiler once.
NUM_CORES ?= 8
ifndef CF_NUM_CORES_DEFINED
CF_NUM_CORES_DEFINED := 1
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
endif

endif # CF_QCONV_MK_INCLUDED
