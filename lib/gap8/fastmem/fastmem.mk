# fastmem.mk -- word-at-a-time memcpy/memset in place of the SDK's byte loops,
# for every caller in the image (see src/fastmem.c). Include from an app Makefile:
#
#     include $(CF_LAB)/lib/gap8/fastmem/fastmem.mk

ifndef CF_FASTMEM_MK_INCLUDED
CF_FASTMEM_MK_INCLUDED := 1

CF_FASTMEM_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

APP_SRCS += $(CF_FASTMEM_DIR)/src/fastmem.c
APP_LDFLAGS += -Wl,--wrap=memcpy -Wl,--wrap=memset

endif # CF_FASTMEM_MK_INCLUDED
