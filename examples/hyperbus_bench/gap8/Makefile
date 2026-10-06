io=uart
PMSIS_OS=freertos

APP = hyperbus_bench
APP_SRCS += hyperbus_bench.c

CF_LAB ?= $(abspath $(CURDIR)/../../..)
include $(CF_LAB)/lib/gap8/cpx/cpx.mk

APP_INC += .
APP_INC += $(GAP_SDK_HOME)/rtos/pulp/gap_archi/include

# FC clock (the HyperBus clock follows the SoC clock). 150 MHz is nn_bench's.
HB_FC_MHZ ?= 150
APP_CFLAGS += -DHB_FC_MHZ=$(HB_FC_MHZ)

# The L3 weight load (hyperbus_bench.c, test 4); host/bench.py sets these and
# writes readfs/hb_load.bin. HB_LOAD_KB=0 leaves the test out.
HB_LOAD_KB ?= 0
HB_LOAD_SEED ?= 0x12345678
HB_LOAD_ITERS ?= 5
HB_PLAN ?= 128:burst:150
APP_CFLAGS += -DHB_LOAD_KB=$(HB_LOAD_KB) -DHB_LOAD_SEED=$(HB_LOAD_SEED)u
APP_CFLAGS += -DHB_LOAD_ITERS=$(HB_LOAD_ITERS) '-DHB_PLAN="$(HB_PLAN)"'
# The tiled layer (test 5): cl_mhz:spi steps over the loaded file, HB_TILE_ITERS
# passes each. Empty leaves the test out.
HB_TILE_PLAN ?=
HB_TILE_ITERS ?= 20
APP_CFLAGS += '-DHB_TILE_PLAN="$(HB_TILE_PLAN)"' -DHB_TILE_ITERS=$(HB_TILE_ITERS)

APP_CFLAGS += -O2 -g
# The Crazyflie radio flasher writes the GAP8 image to HyperFlash at 0x40000, so
# the readfs partition table lives there.
APP_CFLAGS += -DFS_PARTITIONTABLE_OFFSET=0x40000
APP_CFLAGS += -DNUM_CORES=8 -DGAP_SDK=1
APP_CFLAGS += -DUSE_HYPERFLASH -DUSE_HYPERRAM
APP_CFLAGS += -DALWAYS_BLOCK_DMA_TRANSFERS

APP_LDFLAGS += -lm

# The readfs: hb_probe.txt for the mount test (the host checks its contents) and,
# with the load test, hb_load.bin.
READFS_FILES := readfs/hb_probe.txt
ifneq ($(HB_LOAD_KB),0)
READFS_FILES += readfs/hb_load.bin
endif

include $(RULES_DIR)/pmsis_rules.mk
