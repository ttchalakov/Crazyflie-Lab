# kernel_bench -- GAP8 kernel development and profiling harness.
#
#   io=host  -> GVSOC, semihosted stdout, one matrix pass, then #DONE.
#   io=uart  -> AI-deck, CPX/CRTP console after a host-attach delay.
#
# Add an experiment adapter under kernels/<name>/, register its descriptor in
# src/registry.c, and optionally add extra sources without editing this file:
#
#   make KBENCH_KERNEL_SRCS="kernels/foo/foo.c" \
#        KBENCH_KERNEL_INC="kernels/foo" ...

APP = kernel_bench

KBENCH_TARGET ?= gvsoc
ifeq ($(KBENCH_TARGET),gvsoc)
io ?= host
platform ?= gvsoc
else ifeq ($(KBENCH_TARGET),aideck)
io ?= uart
else
$(error KBENCH_TARGET must be gvsoc or aideck)
endif
PMSIS_OS = freertos

APP_SRCS = src/main.c src/registry.c src/fixture.c kernels/linear/linear_bench.c
APP_SRCS += kernels/linear/linear_fused_bench.c
APP_SRCS += kernels/conv/conv_bench.c
APP_SRCS += kernels/depthwise/depthwise_bench.c
APP_SRCS += kernels/matmul/matmul_bench.c
APP_SRCS += kernels/activation/activation_bench.c
APP_SRCS += kernels/activation/activation_luts.c
APP_SRCS += kernels/batchnorm/batchnorm_bench.c
APP_SRCS += kernels/pooling/pooling_bench.c
APP_SRCS += kernels/add/add_bench.c
APP_SRCS += $(KBENCH_KERNEL_SRCS)

KBENCH_MATRIX_DIR ?= generated

# KBENCH_DORY_INT32_BIAS=1 builds the convolution and depthwise kernels as
# lib/gap8/dory/kernels.mk does, reading DORY's int32 bias per output channel
# instead of an int8 one, so the bench can test the code DORY ships.
KBENCH_DORY_INT32_BIAS ?= 0
ifeq ($(KBENCH_DORY_INT32_BIAS),1)
APP_CFLAGS += -DCF_QCONV_DORY_INT32_BIAS=1
APP_CFLAGS += -DCF_QDEPTHWISE_DORY_INT32_BIAS=1
endif

APP_INC += include $(KBENCH_MATRIX_DIR) $(KBENCH_KERNEL_INC)
APP_CFLAGS += -O2 -g -Wall -Wextra
APP_CFLAGS += -DKBENCH_FC_FREQ_HZ=$(KBENCH_FC_FREQ_HZ)
APP_CFLAGS += -DKBENCH_CL_FREQ_HZ=$(KBENCH_CL_FREQ_HZ)
APP_CFLAGS += -DKBENCH_BOOT_DELAY_MS=$(KBENCH_BOOT_DELAY_MS)

KBENCH_FC_FREQ_HZ ?= 150000000
KBENCH_CL_FREQ_HZ ?= 150000000
KBENCH_BOOT_DELAY_MS ?= 12000

CF_LAB ?= $(abspath $(CURDIR)/../../..)
include $(CF_LAB)/lib/gap8/kernels/linear/linear.mk
include $(CF_LAB)/lib/gap8/kernels/conv/conv.mk
include $(CF_LAB)/lib/gap8/kernels/depthwise/depthwise.mk
include $(CF_LAB)/lib/gap8/kernels/activation/activation.mk
include $(CF_LAB)/lib/gap8/kernels/batchnorm/batchnorm.mk
include $(CF_LAB)/lib/gap8/kernels/pooling/pooling.mk
include $(CF_LAB)/lib/gap8/kernels/add/add.mk

ifeq ($(io),host)
APP_CFLAGS += -DKBENCH_GVSOC=1
else
# GAP8 stdout is not connected on the AI-deck. CPX carries reports through the
# ESP32 and STM32 to the Crazyflie console on hardware.
include $(CF_LAB)/lib/gap8/cpx/cpx.mk
APP_CFLAGS += -DTXQ_SIZE=16
APP_CFLAGS += -DINCLUDE_uxTaskGetStackHighWaterMark=1
endif

include $(RULES_DIR)/pmsis_rules.mk
