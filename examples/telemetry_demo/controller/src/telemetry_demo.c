#include <stdint.h>
#include "app.h"
#include "cpx_bridge_stm32.h"
#include "FreeRTOS.h"
#include "task.h"
#include "log.h"

static cf_cpx_bridge_t bridge;
static uint32_t stm32Tick;
static uint32_t gap8Tick;
static uint32_t gap8Received;

static void onPacket(const CPXPacket_t *packet)
{
    uint32_t tick;
    size_t size;
    if (!cf_cpx_bridge_read(&bridge, packet, &tick, sizeof(tick), &size) ||
        size != sizeof(tick)) return;
    gap8Tick = tick;
    gap8Received++;
}

void appMain(void)
{
    cf_cpx_bridge_stm32_init(&bridge, CPX_T_GAP8);
    cpxRegisterAppMessageHandler(onPacket);
    TickType_t last = xTaskGetTickCount();
    while (1) {
        vTaskDelayUntil(&last, M2T(100));
        stm32Tick++;
    }
}

LOG_GROUP_START(demoStm)
LOG_ADD(LOG_UINT32, tick, &stm32Tick)
LOG_GROUP_STOP(demoStm)

LOG_GROUP_START(demoGap)
LOG_ADD(LOG_UINT32, tick, &gap8Tick)
LOG_ADD(LOG_UINT32, received, &gap8Received)
LOG_GROUP_STOP(demoGap)
