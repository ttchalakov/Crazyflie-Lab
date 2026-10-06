# nn_bench -- one generated network on the GAP8 (host/point.py and host/run.py build it).
#
#   make NNBENCH_NET_DIR=<point>/gap8_app NNBENCH_GEN_DIR=<point>/gen \
#        NNBENCH_RUNNER=pulp_nn clean all run platform=gvsoc io=host
#
#   io=host   GVSOC: printf is semihosted to the simulator's stdout.
#   io=uart   the deck: printf goes to UART pins nothing reads, so the report
#             travels over CPX to the radio console.
#
#   NNBENCH_RUNNER  pulp_nn      DORY with the PULP-NN kernels it ships
#                   lib_kernels  DORY with lib/gap8/kernels in their place
#                   fused        a fused executor, fused/$(NNBENCH_FUSED).c (host/fused.py)
#
# Do not name variables COMMON, CFLAGS, ... -- the GAP SDK rules own those.

APP = nn_bench
APP_SRCS = nn_bench.c
io ?= host
PMSIS_OS = freertos

NNBENCH_RUNS ?= 6
NNBENCH_RUNNER ?= pulp_nn
ifndef NNBENCH_NET_DIR
$(error set NNBENCH_NET_DIR to the DORY-generated app directory)
endif
ifndef NNBENCH_GEN_DIR
$(error set NNBENCH_GEN_DIR to the directory holding nn_bench_net.h and nn_bench_golden.h)
endif

CF_LAB ?= $(abspath $(CURDIR)/../../..)
ifeq ($(NNBENCH_RUNNER),fused)
APP_SRCS += fused/$(NNBENCH_FUSED).c
APP_INC += fused
APP_CFLAGS += -O2 -DNNBENCH_FUSED=1
include $(CF_LAB)/lib/gap8/kernels/conv/conv.mk
# An MLP on l3 streams DORY's own weight files from HyperRAM, loaded with
# DORY's mem.c; nothing else of DORY's app is built.
ifneq ($(wildcard $(NNBENCH_NET_DIR)/src/mem.c),)
APP_SRCS += $(NNBENCH_NET_DIR)/src/mem.c
APP_INC += $(NNBENCH_NET_DIR)/inc
APP_CFLAGS += -DUSE_HYPERFLASH -DUSE_HYPERRAM
DORY_NET_HEX := $(wildcard $(NNBENCH_NET_DIR)/hex/*_weights.hex)
endif
else
DORY_NET_DIR := $(NNBENCH_NET_DIR)
include $(CF_LAB)/lib/gap8/dory/dory.mk
# Only the arithmetic symbols differ between pulp_nn and lib_kernels: DORY's
# generated tiling and DMA are the same in both, so the comparison is controlled.
DORY_KERNEL_VARIANT := $(NNBENCH_RUNNER)
include $(CF_LAB)/lib/gap8/dory/kernels.mk
endif

APP_INC += . $(NNBENCH_GEN_DIR)
APP_CFLAGS += -g -w -DNNBENCH_RUNS=$(NNBENCH_RUNS) -DNNBENCH_RUNNER='"$(NNBENCH_RUNNER)"'
APP_LDFLAGS += -lm

ifeq ($(io),host)
APP_CFLAGS += -DNNBENCH_CPX=0
else
include $(CF_LAB)/lib/gap8/cpx/cpx.mk
APP_CFLAGS += -DNNBENCH_CPX=1 -DTXQ_SIZE=16
endif

# On l3 the weights come from a readfs at boot. The image starts at 0x40000 on
# the deck (the GAP8 bootloader owns the first 256 KB), at 0 in GVSOC.
ifneq ($(strip $(DORY_NET_HEX)),)
READFS_FILES := $(DORY_NET_HEX)
APP_CFLAGS += -DFS_READ_FS -DFS_PARTITIONTABLE_OFFSET=$(if $(filter host,$(io)),0,0x40000)
endif

include $(RULES_DIR)/pmsis_rules.mk
