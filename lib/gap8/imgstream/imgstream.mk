# imgstream.mk -- image streaming over CPX for GAP8. Include from an app Makefile:
#
#     CF_LAB ?= $(abspath $(CURDIR)/../cf-lab)
#     include $(CF_LAB)/lib/gap8/cpx/cpx.mk        # imgstream rides on CPX
#     include $(CF_LAB)/lib/gap8/imgstream/imgstream.mk
#
# See inc/imgstream.h. The wire format is lib/shared/include/cf_lab/img_protocol.h,
# and the host decoder is cf_lab.imgstream -- one definition each.

ifndef CF_IMGSTREAM_MK_INCLUDED
CF_IMGSTREAM_MK_INCLUDED := 1

CF_IMGSTREAM_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

APP_SRCS += $(CF_IMGSTREAM_DIR)/src/imgstream.c
APP_INC  += $(CF_IMGSTREAM_DIR)/inc
APP_INC  += $(CF_IMGSTREAM_DIR)/../../shared/include

endif # CF_IMGSTREAM_MK_INCLUDED
