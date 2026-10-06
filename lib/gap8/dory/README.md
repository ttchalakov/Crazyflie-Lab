# DORY GAP8 integration

`dory.mk` builds a DORY-generated network inside a FreeRTOS app: set
`DORY_NET_DIR` to the generated app directory and include it. It adds DORY's
sources (without its `main.c`), the include paths, a force-included shim for the
SDK headers DORY expects, and the flags DORY's own Makefile would set; its
comments explain each, including why every MCHAN transfer is blocking and goes
through core 0. A 3-level (`PULP.GAP8`) network also needs
`READFS_FILES := $(DORY_NET_HEX)` and `mem_init()` called from a spawned task.

`kernels.mk` is an optional arithmetic-only overlay for a generated DORY
application; include it after `dory.mk` and select one of:

```make
DORY_KERNEL_VARIANT=pulp_nn      # copied, pinned PULP-NN sources (the default)
DORY_KERNEL_VARIANT=lib_kernels  # lib/gap8/kernels exact-symbol replacements
```

`lib_kernels` filters only copied sources whose public symbols are replaced:
linear, pointwise/dense-convolution matmul paths, 8-bit depthwise, and residual
addition. The generated layer wrappers, L1 offsets, directional allocator, DMA
descriptors, barriers, and L2/L3 plan remain unchanged. The replacement entry
points run inside DORY's existing cluster team and do not allocate, fork, or
start DMA.

`examples/nn_bench` generates each network once and builds that exact generated
application with both variants (its `pulp_nn` and `lib_kernels` runners), with
both of DORY's memory models (`l2`, weights linked into L2, and `l3`, weights in
HyperRAM).

`cf_dory_profile.h` adds optional cycle counters to the overlaid kernels, to
split a layer's time between its kernel and DORY's orchestration. They compile
to nothing unless `CF_DORY_PROFILE=1`; the harness calls `cf_dory_prof_reset()`
before the network runs and `cf_dory_prof_report()` after it, as nn_bench does.
DORY's own per-layer timing stays DORY's option.
