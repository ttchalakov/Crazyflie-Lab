# Batch-1 uint8 x int8 -> int32 or fused uint8 inference projection for GAP8.

ifndef CF_QLINEAR_MK_INCLUDED
CF_QLINEAR_MK_INCLUDED := 1

CF_QLINEAR_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
CF_QLINEAR_REPO_ROOT := $(abspath $(CF_QLINEAR_DIR)/../../../..)
CF_QLINEAR_PULP_NN ?= $(CF_QLINEAR_REPO_ROOT)/external/dory/dory/Hardware_targets/PULP/Backend_Kernels/pulp-nn/32bit
CF_QLINEAR_REPLACE_PULP_NN ?= 0

APP_SRCS += $(CF_QLINEAR_DIR)/src/cf_qlinear_reference.c
APP_SRCS += $(CF_QLINEAR_DIR)/src/cf_qlinear_prepare.c
APP_SRCS += $(CF_QLINEAR_DIR)/src/cf_qlinear_pulp_nn.c
APP_SRCS += $(CF_QLINEAR_DIR)/src/cf_qlinear_register4.c
APP_SRCS += $(CF_QLINEAR_DIR)/src/cf_qlinear_fused_reference.c
APP_SRCS += $(CF_QLINEAR_DIR)/src/cf_qlinear_fused_pulp_nn.c
APP_SRCS += $(CF_QLINEAR_DIR)/src/cf_qlinear_fused_register4.c
APP_SRCS += $(CF_QLINEAR_DIR)/src/cf_qlinear_dory.c
APP_SRCS += $(CF_QLINEAR_DIR)/src/cf_qlinear_vendor.c

# Link the exact pinned baseline from the submodule; do not copy vendor code into
# lib/. The local adapter above supplies the stable cf_qlinear interface.
ifeq ($(CF_QLINEAR_REPLACE_PULP_NN),1)
APP_CFLAGS += -DCF_QLINEAR_REPLACE_PULP_NN=1
else
APP_SRCS += $(CF_QLINEAR_PULP_NN)/src/pulp_nn_linear_out_32.c
APP_SRCS += $(CF_QLINEAR_PULP_NN)/src/pulp_nn_linear.c
endif
ifndef CF_PULP_NN_UTILS_SRC_INCLUDED
CF_PULP_NN_UTILS_SRC_INCLUDED := 1
APP_SRCS += $(CF_QLINEAR_PULP_NN)/src/pulp_nn_utils.c
endif

APP_INC += $(CF_QLINEAR_DIR)/inc
# cf_dory_profile.h: compiled out unless CF_DORY_PROFILE=1.
APP_INC += $(abspath $(CF_QLINEAR_DIR)/../../dory)
APP_INC += $(CF_QLINEAR_PULP_NN)/include
APP_INC += $(CF_QLINEAR_PULP_NN)/src

# NUM_CORES sizes every PULP-NN kernel's work split. One knob for the whole app
# (dory.mk and the kernel fragments share it), passed to the compiler once.
NUM_CORES ?= 8
ifndef CF_NUM_CORES_DEFINED
CF_NUM_CORES_DEFINED := 1
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
endif

endif # CF_QLINEAR_MK_INCLUDED
