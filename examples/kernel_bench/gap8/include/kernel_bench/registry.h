#ifndef CF_KERNEL_BENCH_REGISTRY_H
#define CF_KERNEL_BENCH_REGISTRY_H

#include <stddef.h>

#include "kernel_bench/kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The returned descriptor pointers have static lifetime. */
const kbench_kernel_t *const *kbench_registry(size_t *count);

#ifdef __cplusplus
}
#endif

#endif /* CF_KERNEL_BENCH_REGISTRY_H */
