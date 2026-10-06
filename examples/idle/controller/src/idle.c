#include "app.h"
#include "FreeRTOS.h"
#include "task.h"

void appMain(void)
{
    while (1) {
        vTaskDelay(M2T(1000));
    }
}
