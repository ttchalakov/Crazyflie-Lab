# Include from the application's src/Kbuild. Objects stay in the app build tree.
CF_CPX_BRIDGE_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
ccflags-y += -I$(CF_CPX_BRIDGE_DIR)/inc -I$(CF_CPX_BRIDGE_DIR)/../../shared/include
obj-y += cf_cpx_bridge.o cf_cpx_bridge_stm32.o
$(obj)/cf_cpx_bridge.o: $(CF_CPX_BRIDGE_DIR)/../../shared/src/cpx_bridge.c FORCE
	$(call if_changed_rule,cc_o_c)
$(obj)/cf_cpx_bridge_stm32.o: $(CF_CPX_BRIDGE_DIR)/src/cpx_bridge_stm32.c FORCE
	$(call if_changed_rule,cc_o_c)
