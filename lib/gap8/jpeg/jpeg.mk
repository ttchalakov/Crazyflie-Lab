# jpeg.mk -- grayscale JPEG encoder for GAP8. Include this from an app's Makefile:
#
#     CF_LAB ?= $(abspath $(CURDIR)/../cf-lab)
#     include $(CF_LAB)/lib/gap8/jpeg/jpeg.mk
#
# Then use the one public entry point, jpeg_enc.h:
#
#     jpeg_cfg_t cfg = { .path = JPEG_PATH_CLUSTER, .cores = 8, .quality = 50,
#                        .restart_interval = 0, .l1_policy = JPEG_CL_L1_RELEASE_ALL };
#     jpeg_enc_init(&enc, &cfg, w, h, stride, &cluster, segs, seg_cap);
#     size_t n = jpeg_enc_frame(&enc, img, out, cap);
#
# Everything else here is an implementation detail. See examples/jpeg_cluster_bench/README.md.
#
# The FC and cluster paths emit BYTE-IDENTICAL JPEGs; the choice is a runtime knob
# because it is a question about what else wants the cluster (e.g. a DORY network),
# not about the encoder.
#
# Ablation switches (all default on, all bit-exact -- see jpeg_cl.c):
#     -DJPEG_CL_FAST_GATHER=0     -DJPEG_CL_EARLY_EOB=0     -DJPEG_CL_STAGE_ON_CLUSTER=0

ifndef CF_JPEG_MK_INCLUDED
CF_JPEG_MK_INCLUDED := 1

CF_JPEG_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

APP_SRCS += $(CF_JPEG_DIR)/src/jpeg_gray.c
APP_SRCS += $(CF_JPEG_DIR)/src/jpeg_dct.c
APP_SRCS += $(CF_JPEG_DIR)/src/jpeg_cl.c
APP_SRCS += $(CF_JPEG_DIR)/src/jpeg_enc.c

APP_INC += $(CF_JPEG_DIR)/inc

endif # CF_JPEG_MK_INCLUDED
