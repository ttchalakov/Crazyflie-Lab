# Include from the application's src/Kbuild. Objects stay in the app build tree.
#
# The kernels build with -ffp-contract=off so that they round exactly as the host build does (cf_lab.nn.host), and at
# -O2 rather than the firmware's -Os, since they are the hot loop.
CF_NN_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
CF_NN_SHARED := $(CF_NN_DIR)/../../shared
ccflags-y += -I$(CF_NN_SHARED)/include
obj-y += cf_nn.o cf_fpmath.o
CFLAGS_cf_nn.o += -O2 -ffp-contract=off
CFLAGS_cf_fpmath.o += -O2 -ffp-contract=off
$(obj)/cf_nn.o: $(CF_NN_SHARED)/src/nn.c FORCE
	$(call if_changed_rule,cc_o_c)
$(obj)/cf_fpmath.o: $(CF_NN_SHARED)/src/fpmath.c FORCE
	$(call if_changed_rule,cc_o_c)
