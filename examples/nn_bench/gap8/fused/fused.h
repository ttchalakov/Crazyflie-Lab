/* A fused executor as nn_bench.c drives it (host/fused.py says which runs which network). */
#ifndef NNBENCH_FUSED_H
#define NNBENCH_FUSED_H

#include <stddef.h>
#include <stdint.h>

#define FUSED_CORES 8

/* Allocates the executor's memory, and on l3 loads its weights.
 * Called once on the FC; returns 0 or a negative error. */
int fused_init(void);

/* Where nn_bench.c stages the golden input before each run. */
uint8_t *fused_input(void);

/* One inference, on cluster core 0; forks the team. */
void fused_run(void);

/* nn_bench.c: a layer's output is complete. From core 0, inside the team. */
void nnbench_layer(int id, const void *out, size_t bytes);

#endif
