#include <math.h>
#include "bsp/bsp.h"
#include "pmsis.h"
#include "FreeRTOS.h"
#include "task.h"
#include "cpx_bridge_gap8.h"
#include "gap8_sample.h"

static cf_cpx_bridge_t bridge;

static void run(void)
{
    pi_bsp_init();
    cpxInit();
    cf_cpx_bridge_gap8_init(&bridge, CPX_T_STM32);
    gap8_sample_t sample = {0};
    TickType_t last = xTaskGetTickCount();
    while (1) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(1));
        sample.tick++;
        float t = sample.tick * 0.001f;
        float phase = t - (uint32_t)t;
        sample.sin1 = sinf(6.28318530718f * phase);
        sample.cos1 = cosf(6.28318530718f * phase);
        sample.saw = phase;
        sample.ramp = t;
        sample.square = phase < 0.5f ? 1.0f : -1.0f;
        sample.tri = phase < 0.5f ? 4.0f * phase - 1.0f : 3.0f - 4.0f * phase;
        /* Generate at 1 kHz, publish at 100 Hz. Skip if the queue is full. */
        if (sample.tick % 10 == 0) {
            cf_cpx_bridge_send(&bridge, &sample, sizeof(sample), 0);
        }
    }
}

int main(void)
{
    return pmsis_kickoff((void *)run);
}
