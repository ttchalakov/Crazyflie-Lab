# `lib/`: shared C for the GAP8 and STM32

The C that more than one example needs, and the C peer of `src/` (the Python
package, `cf_lab`). Third-party firmware lives in [`external/`](../external/README.md);
`lib/` holds only code this repository writes. Each component has its own README.

```text
lib/
  shared/include/cf_lab/     portable headers: host, GAP8, STM32, ESP32
    cpx_bridge.h             the application-packet bridge API
    img_protocol.h           the image wire format the GAP8 and host share
  shared/src/cpx_bridge.c    the bridge core
  stm32/cpx_bridge/          STM32 bridge adapter and Kbuild fragment
  gap8/
    cpx/                     CPX transport (GAP8 <-> ESP32 over SPI)
    cpx_bridge/              GAP8 bridge adapter
    cpx_wifi/                the ESP32 WiFi control session, and an optional logger
    fastmem/                 word-at-a-time memcpy/memset in place of the SDK's byte loops
    imgstream/               image framing and CPX segmentation
    camera_stream/           Himax continuous capture, buffer leases, watchdog
    camera_pipeline/         frames at a set size, resized on demand; the one resize
    camera_stream_log/       optional camera and transport status line
    camera_preview/          one-call QVGA60 JPEG preview over WiFi
    jpeg/                    grayscale JPEG encoder, on the FC or the cluster
    jpeg_pipeline/           JPEG buffers and cluster lifecycle
    sched/                   one DORY policy beside the camera and the video stream
    dory/                    running a DORY-generated network inside a FreeRTOS app
    kernels/                 quantized NN kernels in DORY's layouts (linear, conv,
                             depthwise, activation, batchnorm, pooling, add)
```

## Using it

`CF_LAB` is the path to the cf-lab checkout (usually a `cf-lab/` git submodule of
your project; the examples default it to the repo root). Everything else is found
from it.

**GAP8**: include each component's Make fragment; it appends `APP_SRCS` and
`APP_INC`:

```make
CF_LAB ?= $(abspath $(CURDIR)/../cf-lab)
include $(CF_LAB)/lib/gap8/cpx/cpx.mk
include $(CF_LAB)/lib/gap8/camera_stream/camera_stream.mk

APP_CFLAGS += -DTXQ_SIZE=160        # the #ifndef knobs in inc/cpx_config.h
```

**STM32** (out-of-tree controller): the app Makefile exports `CF_LAB` so
`src/Kbuild` can use it through the firmware's sub-make:

```make
export CF_LAB ?= $(abspath $(CURDIR)/../cf-lab)
CRAZYFLIE_BASE ?= $(CF_LAB)/external/crazyflie-firmware
```

For protocol headers alone, in `src/Kbuild`:

```make
ccflags-y += -I$(CF_LAB)/lib/shared/include
```
```c
#include "cf_lab/img_protocol.h"
```

The CPX application bridge shares its core under `shared/src/`, with GAP8 and
STM32 adapters under their `cpx_bridge/` directories. [The bridge
guide](shared/CPX_BRIDGE.md) covers application-defined payloads, receive
dispatch, timeout-controlled sends and build integration; validation, response
storage, synchronization and freshness belong to the application.

### The GAP8 camera: three levels

1. **Helpers**, for full control: `camera_stream` (capture and buffer leases),
   `ds_resize_gray()` from `camera_pipeline/inc/downsample.h` (the one resize),
   the `jpeg` encoder, `imgstream` framing and `cpx_wifi`.
2. **Structured**: `camera_pipeline` hands out frames at a configured size to
   whatever task or scheduler calls `camera_pipeline_acquire()`, resizing each
   one there and nothing else; `jpeg_pipeline` owns the JPEG buffers and,
   optionally, the cluster; `camera_stream_log` adds a status line.
   `examples/camera_stream` is built this way.
3. **Plug and play**: `camera_preview` starts a QVGA 60 FPS JPEG preview over
   WiFi with one call, and assumes nothing else uses the cluster.

The camera driver owns no JPEG, CPX, WiFi, host framing or cluster: each of
those is its own component, and the status-line and WiFi log adapters are
separate `.mk` fragments, so firmware that does not want them leaves their code
and strings out.

### The GAP8 kernels

| Kernel | What it provides |
| --- | --- |
| `linear` | batch-1 uint8 activations × int8 weights + int32 bias, with int32 output or a fused DORY uint8 epilogue |
| `conv` | DORY-layout convolution, its matmul microkernel, and 1x1 pointwise |
| `depthwise` | a direct 3x3 depthwise micro-kernel |
| `activation` | sigmoid, tanh, LeakyReLU, hard-sigmoid, hard-swish, SiLU and GELU from a 256-byte table, and softmax |
| `batchnorm` | fixed affine normalization, and a forward-only mode with per-frame statistics |
| `pooling` | max, average, global and adaptive average pooling on HWC uint8 |
| `add` | DORY's uint8 residual add, flattened across the cores |

Each runs inside a team the caller has already forked, on caller-owned buffers:
none owns the cluster, allocates memory or starts a DMA. `lib/gap8/dory/kernels.mk`
links `linear`, `conv`, `depthwise` and `add` in place of PULP-NN's symbols in a
DORY-generated app, with no template change (nn_bench's `lib_kernels` runner).
[kernel_bench](../examples/kernel_bench/README.md) checks and measures them.

## Rules

1. **The GAP8 Docker build mounts the cf-lab checkout, not just the app.**
   `cf-lab build gap8` mounts the checkout and the app's project at their own host
   paths and builds in the app directory, so `lib/` is always there. If a build
   seems to need a source copied into an example, fix the mount instead: copies
   drift apart, and a fix lands in only one of them.
2. **A library never includes an app-named header.** The app pushes its
   configuration in:

   ```make
   APP_CFLAGS += -DTXQ_SIZE=160                 # one knob
   APP_CFLAGS += -include my_app_config.h       # a whole app config header
   ```

   and every knob in the library is `#ifndef`-guarded with a default.

## Adding something

Put it under `lib/<target>/<name>/`, give it a `<name>.mk` for GAP8 (self-locating
through `MAKEFILE_LIST`, so callers do no path arithmetic), and keep its tunables
`#ifndef`-guarded in a `<name>_config.h`.
