# DORY-shaped forward-only uint8 BatchNorm kernels for GAP8.

ifndef CF_QBATCHNORM_MK_INCLUDED
CF_QBATCHNORM_MK_INCLUDED := 1

CF_QBATCHNORM_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

APP_SRCS += $(CF_QBATCHNORM_DIR)/src/cf_qbatchnorm_reference.c
APP_SRCS += $(CF_QBATCHNORM_DIR)/src/cf_qbatchnorm_fixed.c
APP_SRCS += $(CF_QBATCHNORM_DIR)/src/cf_qbatchnorm_update.c
APP_INC += $(CF_QBATCHNORM_DIR)/inc
APP_INC += $(CF_QBATCHNORM_DIR)/src

endif # CF_QBATCHNORM_MK_INCLUDED
