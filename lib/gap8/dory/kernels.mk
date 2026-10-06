# Exact-symbol arithmetic overlay for a DORY-generated GAP8 application.
#
# Include this after dory.mk. DORY still owns every generated layer wrapper,
# L1 offset, DMA transfer, double buffer, and L2/L3 allocation. The lib_kernels
# variant changes only the PULP-NN symbols linked into those wrappers.

ifndef CF_DORY_KERNELS_MK_INCLUDED
CF_DORY_KERNELS_MK_INCLUDED := 1

CF_DORY_KERNELS_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
CF_DORY_LIB_DIR := $(abspath $(CF_DORY_KERNELS_DIR)/../..)

DORY_KERNEL_VARIANT ?= pulp_nn

ifeq ($(DORY_KERNEL_VARIANT),lib_kernels)
# DORY copies the complete pinned PULP-NN source directory into every generated
# app. Remove only symbols supplied by the overlay; unrelated pooling kernels,
# the depthwise variants the overlay does not implement, and DORY's generated
# orchestration remain byte-for-byte unchanged.
APP_SRCS := $(filter-out \
  %/pulp_nn_linear.c \
  %/pulp_nn_linear_out_32.c \
  %/pulp_nn_conv_Ho_parallel.c \
  %/pulp_nn_matmul.c \
  %/pulp_nn_pointwise_HoWo_parallel.c \
  %/pulp_nn_depthwise_generic.c \
  %/pulp_nn_add.c,$(APP_SRCS))

# Reuse DORY's copied utility implementation instead of linking a second copy
# from the pinned submodule through each arithmetic make fragment.
CF_PULP_NN_UTILS_SRC_INCLUDED := 1
CF_QLINEAR_REPLACE_PULP_NN := 1
CF_QCONV_REPLACE_PULP_NN := 1
CF_QADD_REPLACE_PULP_NN := 1
# Depthwise: only pulp_nn_depthwise_generic is replaced, which DORY's template
# calls for every 8-bit depthwise layer of four weights or more (every 3x3 one);
# pulp_nn_depthwise_generic_less_4_weights and the mixed-precision variants stay
# PULP-NN's and are still linked from DORY's copy.
CF_QDEPTHWISE_REPLACE_PULP_NN := 1
APP_CFLAGS += -DCF_QCONV_DORY_INT32_BIAS=1
APP_CFLAGS += -DCF_QDEPTHWISE_DORY_INT32_BIAS=1
include $(CF_DORY_LIB_DIR)/gap8/kernels/linear/linear.mk
include $(CF_DORY_LIB_DIR)/gap8/kernels/conv/conv.mk
include $(CF_DORY_LIB_DIR)/gap8/kernels/add/add.mk
include $(CF_DORY_LIB_DIR)/gap8/kernels/depthwise/depthwise.mk
else ifneq ($(DORY_KERNEL_VARIANT),pulp_nn)
$(error DORY_KERNEL_VARIANT must be pulp_nn or lib_kernels)
endif

# Compute-versus-orchestration decomposition. Off by default; build the network
# with CF_DORY_PROFILE=1 to have every overlaid kernel accumulate the cluster
# cycles it actually spends, against DORY's own per-layer total. See
# cf_dory_profile.h for why this exists.
CF_DORY_PROFILE ?= 0
ifeq ($(CF_DORY_PROFILE),1)
APP_SRCS += $(CF_DORY_KERNELS_DIR)/cf_dory_profile.c
APP_CFLAGS += -DCF_DORY_PROFILE=1
endif
APP_INC += $(CF_DORY_KERNELS_DIR)

endif # CF_DORY_KERNELS_MK_INCLUDED
