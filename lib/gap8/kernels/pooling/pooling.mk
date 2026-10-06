# DORY-compatible compact-HWC uint8 pooling kernels and pinned baselines.

ifndef CF_QPOOLING_MK_INCLUDED
CF_QPOOLING_MK_INCLUDED := 1

CF_QPOOLING_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
CF_QPOOLING_REPO_ROOT := $(abspath $(CF_QPOOLING_DIR)/../../../..)
CF_QPOOLING_PULP_NN ?= $(CF_QPOOLING_REPO_ROOT)/external/dory/dory/Hardware_targets/PULP/Backend_Kernels/pulp-nn/32bit

APP_SRCS += $(CF_QPOOLING_DIR)/src/cf_qpooling_reference.c
APP_SRCS += $(CF_QPOOLING_DIR)/src/cf_qpooling_direct.c
APP_SRCS += $(CF_QPOOLING_DIR)/src/cf_qpooling_pulp_nn.c
APP_SRCS += $(CF_QPOOLING_PULP_NN)/src/pulp_nn_avgpool.c
APP_SRCS += $(CF_QPOOLING_PULP_NN)/src/pulp_nn_maxpool.c

ifndef CF_PULP_NN_UTILS_SRC_INCLUDED
CF_PULP_NN_UTILS_SRC_INCLUDED := 1
APP_SRCS += $(CF_QPOOLING_PULP_NN)/src/pulp_nn_utils.c
endif

APP_INC += $(CF_QPOOLING_DIR)/inc
APP_INC += $(CF_QPOOLING_DIR)/src
APP_INC += $(CF_QPOOLING_PULP_NN)/include
APP_INC += $(CF_QPOOLING_PULP_NN)/src

endif # CF_QPOOLING_MK_INCLUDED
