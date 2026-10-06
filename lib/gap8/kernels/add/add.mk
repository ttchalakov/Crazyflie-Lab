# DORY-compatible uint8 residual addition for GAP8.

ifndef CF_QADD_MK_INCLUDED
CF_QADD_MK_INCLUDED := 1

CF_QADD_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
CF_QADD_REPO_ROOT := $(abspath $(CF_QADD_DIR)/../../../..)
CF_QADD_PULP_NN ?= $(CF_QADD_REPO_ROOT)/external/dory/dory/Hardware_targets/PULP/Backend_Kernels/pulp-nn/32bit
CF_QADD_REPLACE_PULP_NN ?= 0

APP_SRCS += $(CF_QADD_DIR)/src/cf_qadd_reference.c
APP_SRCS += $(CF_QADD_DIR)/src/cf_qadd_pulp_nn.c
APP_SRCS += $(CF_QADD_DIR)/src/cf_qadd_packed4.c
APP_SRCS += $(CF_QADD_DIR)/src/cf_qadd_vendor.c

ifeq ($(CF_QADD_REPLACE_PULP_NN),1)
APP_CFLAGS += -DCF_QADD_REPLACE_PULP_NN=1
else
APP_SRCS += $(CF_QADD_PULP_NN)/src/pulp_nn_add.c
endif

ifndef CF_PULP_NN_UTILS_SRC_INCLUDED
CF_PULP_NN_UTILS_SRC_INCLUDED := 1
APP_SRCS += $(CF_QADD_PULP_NN)/src/pulp_nn_utils.c
endif

APP_INC += $(CF_QADD_DIR)/inc
# cf_dory_profile.h: compiled out unless CF_DORY_PROFILE=1.
APP_INC += $(abspath $(CF_QADD_DIR)/../../dory)
APP_INC += $(CF_QADD_PULP_NN)/src
APP_INC += $(CF_QADD_PULP_NN)/include

# NUM_CORES sizes every PULP-NN kernel's work split. One knob for the whole app
# (dory.mk and the kernel fragments share it), passed to the compiler once.
NUM_CORES ?= 8
ifndef CF_NUM_CORES_DEFINED
CF_NUM_CORES_DEFINED := 1
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
endif

endif # CF_QADD_MK_INCLUDED
