# jpeg_cluster_bench: the grayscale JPEG encoder (lib/gap8/jpeg) held to the FC
# path's bytes. host/bench.py builds it both ways:
#   io=uart (default)  the AI-deck; reports over the CPX radio console, forever
#   io=host            GVSOC; one round, then a hex dump of the JPEG and #DONE
#
# BENCH_DEFINES adds -D flags without replacing the ones below (a command-line
# APP_CFLAGS would): the encoded window (-DIMG_W=316 -DIMG_H=236) and the
# encoder's switches in lib/gap8/jpeg/jpeg.mk.
#
# Do not name variables COMMON, CFLAGS, ...: the GAP SDK rules own those.

APP = jpeg_cluster_bench

io ?= uart
PMSIS_OS = freertos

APP_SRCS = main.c realimg.c

CF_LAB ?= $(abspath $(CURDIR)/../../..)
include $(CF_LAB)/lib/gap8/jpeg/jpeg.mk

APP_CFLAGS += -O3 -g
APP_CFLAGS += $(BENCH_DEFINES)

ifeq ($(io),host)
APP_CFLAGS += -DJPEG_CLUSTER_BENCH_SIM=1
else
# The GAP8's printf goes to UART pins nothing listens to; the console travels
# over CPX (-> ESP32 -> STM32 -> radio). GVSOC has no ESP32, so CPX is left out
# of the simulator build.
include $(CF_LAB)/lib/gap8/cpx/cpx.mk
APP_CFLAGS += -DTXQ_SIZE=8
# The app reports its stack high-water mark (#STACK) every round.
APP_CFLAGS += -DINCLUDE_uxTaskGetStackHighWaterMark=1
endif

include $(RULES_DIR)/pmsis_rules.mk
