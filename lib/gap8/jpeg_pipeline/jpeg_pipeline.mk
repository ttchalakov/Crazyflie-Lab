# jpeg_pipeline.mk -- caller-owned JPEG lifecycle and L2 planning for GAP8.
#
#     CF_LAB ?= $(abspath $(CURDIR)/../cf-lab)
#     include $(CF_LAB)/lib/gap8/jpeg_pipeline/jpeg_pipeline.mk
#
# The wrapper depends on jpeg_enc, so this includes jpeg.mk (which guards itself
# against a second include).

ifndef CF_JPEG_PIPELINE_MK_INCLUDED
CF_JPEG_PIPELINE_MK_INCLUDED := 1

CF_JPEG_PIPELINE_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

include $(CF_JPEG_PIPELINE_DIR)/../jpeg/jpeg.mk

APP_SRCS += $(CF_JPEG_PIPELINE_DIR)/src/jpeg_pipeline.c
APP_INC  += $(CF_JPEG_PIPELINE_DIR)/inc

endif # CF_JPEG_PIPELINE_MK_INCLUDED
