# `nn`: small MLPs on the STM32, with the host's bits

Dense layers and multilayer perceptrons for an STM32 out-of-tree app, in float32 and in three integer formats, from
one portable source (`lib/shared/include/cf_lab/nn.h`, `lib/shared/src/nn.c`) that also builds on the host
(`cf_lab.nn.host`). Built with the flags below, the STM32 and the host compute identical output bits; the bench
checks this for every format.

| Format | Activations | Weights | Notes |
|---|---|---|---|
| `f32` | float32 | float32 | row-major, one output at a time |
| `f32x4` | float32 | float32 | rows interleaved by four; same bits as `f32` |
| `f32fma` | float32 | float32 | rows interleaved by four, `fmaf` |
| `s8` | int8, zero points | int8 per channel | the TFLite / CMSIS-NN arithmetic |
| `s16s8` | int16 symmetric | int8 per channel | int32 accumulation |
| `s16` | int16 symmetric | int16 per channel | int64 accumulation |

Activations after integer layers are tables (256 entries for int8, 513 interpolated knots for int16), so any
activation costs the same as ReLU. On a Cortex-M4 the integer kernels use `SXTB16`/`SMLAD`/`SMLALD`. The weights
are stored so that the kernels read them strictly sequentially: one pointer, whole words, consecutive flash lines
(the layouts are in `nn.h`).

## Use

```make
# src/Kbuild of the app
include $(CF_LAB)/lib/stm32/nn/nn.mk
```

```python
from cf_lab.nn import mlp_from_sequential, quantize, write_c
nets = {"policy_s16s8": quantize(mlp_from_sequential(model.net), calibration_rows, "s16s8")}
write_c(nets, Path("src/nets.c"), Path("src/nets.h"))
```

```c
#include "nets.h"
static uint8_t work[2048] __attribute__((aligned(8)));  // at least cf_nn_work_bytes(&policy_s16s8)
cf_nn_run(&policy_s16s8, x, y, work);
```

`nn.mk` builds the kernels at `-O2 -ffp-contract=off`. Code whose float results the host must reproduce needs
`-ffp-contract=off` too (GCC otherwise fuses `a * b + c` on the Cortex-M4F), never `-ffast-math`, and the math of
`cf_lab/fpmath.h` instead of libm.

## Measured on the Crazyflie 2.1 Brushless (168 MHz)

`examples/stm32_nn_bench`, a 34-128-128-4 SiLU network, weights in flash, scheduler suspended:

| Format | µs | KB |
|---|---|---|
| `f32` | 1838 | 85 |
| `f32x4` | 1075 | 85 |
| `s8` | 499 | 25 |
| `s16s8` | 473 | 26 |
| `s16` | 615 | 48 |

The same 128x128 int8 layer takes 321 µs here and 364 µs with the firmware's vendored CMSIS-NN
`arm_fully_connected_s8`. Weights in the core-coupled RAM are 10-15 % faster than in flash, SRAM in between.
