#include <stdint.h>
#include "bsp/bsp.h"
#include "pmsis.h"
#include "FreeRTOS.h"
#include "task.h"
#include "cpx_bridge_gap8.h"

static cf_cpx_bridge_t bridge;

static void run(void)
{
    pi_bsp_init();
    cpxInit();
    cf_cpx_bridge_gap8_init(&bridge, CPX_T_STM32);
    uint32_t tick = 0;
    TickType_t last = xTaskGetTickCount();
    while (1) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(100));
        tick++;
        /* The demo payload is one little-endian uint32_t. Skip if TX is full. */
        cf_cpx_bridge_send(&bridge, &tick, sizeof(tick), 0);
    }
}

int main(void)
{
    return pmsis_kickoff((void *)run);
}
