# dory.mk -- build glue for a DORY-generated network inside a FreeRTOS app.
#
# DORY generates a standalone PULP-OS application: its own main.c, its own
# Makefile, and sources that assume the GAP SDK's bare-metal headers. Dropping
# that into a FreeRTOS app that also owns a camera, a JPEG encoder and a radio
# needs a handful of specific adjustments. They live here, so no app has to
# rediscover them one compile error at a time.
#
# Include this from an app Makefile, having set DORY_NET_DIR to the generated
# app directory (the one holding inc/, src/, hex/, vars.mk):
#
#     CF_LAB ?= $(abspath $(CURDIR)/../cf-lab)
#     DORY_NET_DIR := policy/generated
#     include $(CF_LAB)/lib/gap8/dory/dory.mk
#
# It contributes DORY's sources (minus its main.c), its include path, the shims,
# and the flags below -- but deliberately NOT the readfs wiring, because whether
# the weights live in flash depends on which DORY target generated the network:
#
#     PULP.GAP8      3-level: weights are .hex blobs read from the readfs at boot.
#                    The app must add  READFS_FILES := $(DORY_NET_HEX)  and call
#                    mem_init() FROM A SPAWNED TASK (see below).
#     PULP.GAP8_L2   2-level: weights are C arrays in inc/<prefix>weights.h. No
#                    flash, no readfs, no mem_init at all.
#
# ---------------------------------------------------------------------------
# The mount trap
#
# mem_init() returns -1 when pi_fs_mount() cannot read the partition table. A
# flash image that ends in a zero-length area (gapy sometimes writes one) leaves
# the HyperBus read address unprogrammed for every later flash read. cf-lab's
# flasher drops such areas, so flash through it (repair_gap8_image() in
# src/cf_lab/aideck/flash.py), and call mem_init() from a spawned task, as DORY's
# own main.c does. See examples/hyperbus_bench/README.md.

ifndef CF_DORY_MK_INCLUDED
CF_DORY_MK_INCLUDED := 1

CF_DORY_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

ifndef DORY_NET_DIR
$(error dory.mk: set DORY_NET_DIR to the DORY-generated app directory before including)
endif

# DORY ships its own main(); the host app has one already.
DORY_NET_SRCS := $(filter-out %/main.c,$(wildcard $(DORY_NET_DIR)/src/*.c))
DORY_NET_HEX  := $(wildcard $(DORY_NET_DIR)/hex/*.hex)

ifeq ($(strip $(DORY_NET_SRCS)),)
ifneq ($(strip $(MAKECMDGOALS)),clean)
$(error dory.mk: no sources under $(DORY_NET_DIR)/src -- generate the network first)
endif
endif

APP_SRCS += $(DORY_NET_SRCS)
APP_INC  += $(DORY_NET_DIR)/inc
APP_INC  += $(CF_DORY_DIR)/inc
# DORY's generated code includes archi/ headers directly.
APP_INC  += $(GAP_SDK_HOME)/rtos/pulp/gap_archi/include

# Force-included rather than #include'd, so DORY's generated tree stays exactly
# as DORY produced it -- no overlay to re-derive every time DORY is updated.
# Supplies the archi register map and the event-unit intrinsic that dory_dma.c
# and mchan.h expect to already be in scope under PULP-OS. See inc/ for why.
APP_CFLAGS += -include $(CF_DORY_DIR)/inc/dory_freertos_compat.h

# NUM_CORES is read by every pulp-nn kernel to size its per-core work chunk. It
# is not optional and there is no default in the kernels. The kernel fragments
# that kernels.mk includes share this knob and its guard, so the flag is passed
# once.
NUM_CORES ?= 8
ifndef CF_NUM_CORES_DEFINED
CF_NUM_CORES_DEFINED := 1
APP_CFLAGS += -DNUM_CORES=$(NUM_CORES)
endif
APP_CFLAGS += -DGAP_SDK=1

# DORY's generated Makefile sets these from its own HW description; a host app
# that ignores that Makefile has to supply them.
APP_CFLAGS += -DUSE_HYPERFLASH -DUSE_HYPERRAM

# Make every DMA transfer blocking. DORY's tiling emits async transfers followed
# by a barrier, but that barrier is per-core (see the core-0 routing below), so
# the async path is not safe to rely on here.
#
# Blocking every transfer gives up the transfer/compute overlap DORY's paper
# relies on.
#
# DORY_BLOCK_DMA=0 builds the overlapped path so that cost can be quantified in
# GVSOC. It is an EXPERIMENT ONLY and must not be used for a deck build: the
# per-core barrier issue below is invisible in simulation, so a GVSOC pass under
# =0 proves the speed available, not that the transfers are safe on silicon.
DORY_BLOCK_DMA ?= 1
ifeq ($(DORY_BLOCK_DMA),1)
APP_CFLAGS += -DALWAYS_BLOCK_DMA_TRANSFERS
endif

# Route every MCHAN transfer through core 0.
#
# GAP8's MCHAN counter allocator is documented as: "This allocates a counter and
# activates it for all the next transfers until another one is allocated" --
# activation is a property of the DMA, not of the calling core. DORY's default
# path has all NUM_CORES cores call dory_dma_allocate() at layer entry, so the
# counter a transfer is actually accounted against belongs to whichever core
# allocated last, while every core's dory_dma_barrier() waits on its own
# counter. Seven of eight waits therefore inspect a counter with nothing pending
# and return immediately.
#
# GVSOC does not reproduce this: its MCHAN model keeps current_counter per demux
# channel (models/pulp/mchan/mchan_v6_impl.cpp), so a core-0 push is accounted to
# core 0's own counter and the barrier is correct in simulation. That is why this
# only ever shows up on the deck.
DORY_SINGLE_CORE_DMA ?= 1
ifeq ($(DORY_SINGLE_CORE_DMA),1)
APP_CFLAGS += -DSINGLE_CORE_DMA
endif

# Largest byte count one MCHAN command can express. dory_dma.c defaults to 32764
# because the GAP8's length field is 15 bits wide, not the 16 the SDK's archi
# header advertises -- a bigger transfer is truncated to size & 0x7FFF and
# reported complete. Override only to reproduce the unchunked behaviour: setting
# it to 65535 makes the chunking a no-op for any realistic tile. See Limits in
# examples/nn_bench/README.md.
ifdef DORY_MCHAN_MAX_TRANSFER
APP_CFLAGS += -DMCHAN_MAX_TRANSFER_SIZE=$(DORY_MCHAN_MAX_TRANSFER)
endif

# L1_DATA / L2_DATA are GAP-SDK (PULP-OS) section attributes that place a symbol
# in a specific memory. The FreeRTOS SDK defines neither, and DORY's generated
# code uses both -- L2_DATA on the resident weight and input arrays the 2-level
# target emits, L1_DATA on the SINGLE_CORE_DMA path's shared channel. Defining
# them empty is not a workaround but the correct answer here: an ordinary global
# in this build already lands in L2, and the SINGLE_CORE_DMA channel only needs
# to be shared, not fast. Without these the 2-level target does not compile at
# all outside DORY's own Makefile.
APP_CFLAGS += -DL1_DATA= -DL2_DATA=

# -O2 -fno-indirect-inlining is what DORY's own Makefile uses: as fast as -O3 on
# these kernels and materially smaller.
DORY_OPT ?= -O2 -fno-indirect-inlining
APP_CFLAGS += $(DORY_OPT)

endif # CF_DORY_MK_INCLUDED
