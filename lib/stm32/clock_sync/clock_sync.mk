# Include from the application's src/Kbuild; call cfClockSyncInit(port) once from the app.
CF_CLOCK_SYNC_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
ccflags-y += -I$(CF_CLOCK_SYNC_DIR)/inc
obj-y += cf_clock_sync.o
$(obj)/cf_clock_sync.o: $(CF_CLOCK_SYNC_DIR)/src/cf_clock_sync.c FORCE
	$(call if_changed_rule,cc_o_c)
