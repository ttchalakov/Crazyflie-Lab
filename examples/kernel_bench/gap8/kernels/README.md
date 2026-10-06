# Kernel adapters

One directory per family. Each adapter exposes `const kbench_kernel_t`
descriptors, one per variant, with the family set so a descriptor is only
offered its own cases. To add one, copy `template/kernel_template.c`, rename the
descriptor, add its sources (the library's `.mk` fragment, or `KBENCH_KERNEL_SRCS`
for a quick experiment), and register it in `gap8/src/registry.c`.

- **The arithmetic lives in `lib/gap8/kernels/<family>/`**, with its scalar oracle;
  an adapter holds only the operand setup and the lifecycle. Upstream PULP-NN
  sources are linked from `external/dory`, never copied. Do not copy generated
  DORY apps here.
- **Name variants so reports tell them apart**: the unmodified PULP-NN baseline,
  and each candidate. The scalar oracle runs inside `check()`; activation,
  batchnorm and pooling also register a timed `scalar` variant.
- **Use `include/kernel_bench/fixture.h`** for arena allocation, the seeded
  generator, FNV-1a checksums and the performance counters. Fork the team once per
  invocation and repeat the library's `*_team()` call between `kbench_perf_begin()`
  and `kbench_perf_end()`, so the counters bracket the arithmetic and never the fork.

`host/sweep.py` names each case from its fields, so names are unique by
construction; a name over 72 characters is an error, never shortened. The runner
emits each result as two records that the host report merges by case, kernel and
variant, so console buffering cannot truncate one.
