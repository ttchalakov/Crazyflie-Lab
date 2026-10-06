# Application-packet bridge (STM32 and GAP8)

The bridge sends and receives application-defined bytes over `CPX_F_APP`.
It supplies routing, payload bounds checks, and a common send/read interface.
The send timeout passes directly to the underlying CPX API in FreeRTOS ticks.
No payload header or serialization format is imposed.

Applications own packet definitions, validation, response storage,
synchronization, freshness, sequence numbers, and request/reply policy. The
bridge instance contains only routing configuration. Its shared core has no
RTOS storage, clock, or locking dependency; target adapters select the CPX send
API and GAP8's blocking receive mechanism.

## STM32

Include the build fragment from your controller's `src/Kbuild` (adjust the path):

```make
include $(CF_LAB)/lib/stm32/cpx_bridge/cpx_bridge.mk
obj-y += my_controller.o
```

Initialize before the receiver or sending task accesses the instance. Firmware
owns CPX initialization. Register your own app callback:

```c
#include "cpx_bridge_stm32.h"

static cf_cpx_bridge_t bridge;

static void on_packet(const CPXPacket_t *packet) {
  my_reply_t reply;
  size_t size;
  if (!cf_cpx_bridge_read(&bridge, packet, &reply, sizeof(reply), &size) ||
      size != sizeof(reply) || !validate_reply(&reply)) return;
  /* Consume the reply, or store it with application-owned synchronization. */
}

/* Startup: */
if (!cf_cpx_bridge_stm32_init(&bridge, CPX_T_GAP8)) return;
cpxRegisterAppMessageHandler(on_packet);

/* Application task: zero timeout means no wait for queue space. */
if (!cf_cpx_bridge_send(&bridge, &request, sizeof(request), 0)) {
  /* Application decides whether to drop or retry. */
}
```

The firmware has one global app-message callback. The library does not replace
it. Dispatch to multiple consumers there if needed. Avoid blocking sends from
that callback or from a controller loop.

## GAP8

```make
PMSIS_OS=freertos
CF_LAB ?= $(abspath $(CURDIR)/../cf-lab)
include $(CF_LAB)/lib/gap8/cpx_bridge/cpx_bridge.mk
```

The fragment includes CPX transport and public headers. After board initialization:

```c
#include "cpx_bridge_gap8.h"

static cf_cpx_bridge_t bridge;
if (!cf_cpx_bridge_gap8_init(&bridge, CPX_T_STM32)) return;
cpxInit();
cpxEnableFunction(CPX_F_APP);
for (;;) {
  my_request_t request;
  size_t size;
  if (!cf_cpx_bridge_gap8_receive(&bridge, &request, sizeof(request), &size) ||
      size != sizeof(request) || !validate_request(&request)) continue;
  my_reply_t reply = process_request(&request);
  if (!cf_cpx_bridge_send(&bridge, &reply, sizeof(reply), 20)) {
    /* Application decides whether to drop or retry. */
  }
}
```

Initialize CPX and enable its app queue once. One task should consume that queue.
Receive consumes one packet, waits for its arrival, and returns false for a
rejected packet. For several message types, receive into a byte buffer large
enough for all types, then decode in the application. For shared route dispatch,
receive once with `cpxReceivePacketBlocking` and use `cf_cpx_bridge_read`.

## Contract and limits

- Payloads are 1–98 bytes, the STM32 link's single-packet wire limit. No
  fragmentation/reassembly is supplied. Receive checks the peer/local route,
  CPX function and version, complete-packet flag, and payload bounds.
- Read/receive copy bytes and return their actual length. Output buffers and
  length remain untouched on failure. Oversized received packets are rejected,
  never truncated. The application must check length before accessing fields.
- Define an agreed wire format in a header shared by both applications. Use
  explicit serialization or packed fixed-width fields with agreed byte order;
  the bridge does not serialize structs or convert endianness.
- A successful send means queued, not acknowledged by the peer. Its timeout is
  a queue wait, not a reply deadline. Zero is a nonblocking queue attempt.
- Sending and GAP8 receiving each use a native `CPXPacket_t` on the caller's
  stack (about 1 KiB on GAP8). No allocation, task creation, or packet retention
  is performed by the bridge.
- Initialize routing before concurrent use and leave it unchanged afterwards.
  Use send/receive from tasks, not interrupts. Callers own their payload buffers
  and any synchronization needed to share results with other tasks.
