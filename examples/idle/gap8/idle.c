// Park the AI-deck GAP8 in a low-power idle so it stops eating battery.
//
// This app deliberately does nothing: it never opens the Himax camera and
// never starts the cluster (the two big AI-deck power draws), drops the
// fabric-controller clock to its minimum, and then sleeps forever. Flash this
// when you want the AI deck attached but quiet (the NINA WiFi is already off
// via the firmware's CONFIG_DECK_AI_WIFI_NO_SETUP). Reflash a real GAP8 app
// over the radio whenever you need the deck back.

#include "pmsis.h"

static void idle_task(void)
{
    // Lowest practical fabric-controller frequency; the camera/cluster stay off.
    pi_freq_set(PI_FREQ_DOMAIN_FC, 50000000);

    while (1)
    {
        // Yielding sleep so the FC clock-gates between ticks.
        pi_time_wait_us(1000 * 1000);
    }
}

int main(void)
{
    return pmsis_kickoff((void *)idle_task);
}
