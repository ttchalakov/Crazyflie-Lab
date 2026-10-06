#ifndef CF_KERNEL_BENCH_ACTIVATION_LUTS_H
#define CF_KERNEL_BENCH_ACTIVATION_LUTS_H

#include <stddef.h>
#include <stdint.h>

#include "cf_qactivation.h"

/*
 * Default tables the activation adapter benchmarks with:
 *   input: signed int8 Q4.4
 *   sigmoid/hard-sigmoid output: uint8 Q0.8 (scale 1/256)
 *   tanh output: signed int8 Q1.7 (scale 1/128)
 *   remaining outputs: signed int8 Q4.4
 *   default LeakyReLU negative slope: 1/8
 *
 * They are fixtures, not library tables: production code generates a table for
 * the model's recorded quantization (lib/gap8/kernels/activation/tools/
 * generate_lut.py) and stages its 256 bytes as a caller-owned DORY constant
 * tile.
 */
const uint8_t *kbench_activation_default_lut(cf_qactivation_kind_t kind);

/* round(exp(-d / 16) * 32768) for d in [0,255]: the Q4.4 softmax exponent. */
const uint16_t *kbench_softmax_default_exp_lut(void);
size_t kbench_softmax_default_exp_lut_bytes(void);

#endif /* CF_KERNEL_BENCH_ACTIVATION_LUTS_H */
