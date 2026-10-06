# GAP8 DORY-compatible quantized residual addition

This library preserves the exact pinned `pulp_nn_add` contract:

```text
output = clip_u8((input1 * multiplier1 + input2 * multiplier2) >> shift)
```

The upstream baseline walks one byte at a time and partitions only output
height. `cf_qadd_u8_packed4_team` instead flattens the complete HWC tile across
the active cores, processes eight elements per loop iteration with packed
loads/stores, and handles the final one to three bytes separately. Arbitrary
H/W/C shapes and unequal multipliers are supported. Output may alias either
input because each packed word is fully loaded before it is stored.

`cf_qadd_u8_packed4_team` checks that all three tiles are word aligned before
casting them to `v4u` and drops to a scalar per-element traversal otherwise,
matching the activation and pooling kernels. `cf_qadd_vendor.c` keeps a
privately renamed copy of the pinned kernel as the exact-symbol replacement's
fallback.

The team entry allocates nothing, launches no DMA, forks no nested team, and
does not own the cluster. DORY continues to stage both inputs and the output in
its generated L1 layout. Defining `CF_QADD_REPLACE_PULP_NN=1` exports the exact
vendor symbol; DORY templates and the tiler remain unchanged.
