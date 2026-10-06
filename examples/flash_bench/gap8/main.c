/* Identifiable, non-flight fixtures for the production GAP8 flashing bench.
 * The larger fixture checks every readfs byte after boot, not just its header.
 * Repeating short records let a late console attach verify the running image.
 */
#include "pmsis.h"
#include "bsp/bsp.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "cpx.h"
#include "FreeRTOS.h"
#include "task.h"

#if !FB_SILENT
static struct pi_device flash, fs;
static uint8_t buffer[1024];

static int check_payload(uint32_t *bytes, uint32_t *hash)
{
    struct pi_hyperflash_conf flash_conf;
    pi_hyperflash_conf_init(&flash_conf);
    pi_open_from_conf(&flash, &flash_conf);
    if (pi_flash_open(&flash)) return 1;

    struct pi_readfs_conf fs_conf;
    pi_readfs_conf_init(&fs_conf);
    fs_conf.fs.flash = &flash;
    pi_open_from_conf(&fs, &fs_conf);
    if (pi_fs_mount(&fs)) {
        pi_flash_close(&flash);
        return 2;
    }
    pi_fs_file_t *file = pi_fs_open(&fs, "payload.bin", 0);
    int error = 0;
    if (!file) error = 3;
    else {
        if (file->size != FB_PAYLOAD_BYTES) error = 4;
        while (!error && *bytes < FB_PAYLOAD_BYTES) {
            unsigned want = FB_PAYLOAD_BYTES - *bytes;
            if (want > sizeof(buffer)) want = sizeof(buffer);
            /* Never pi_fs_read(): it races the PMSIS event kernel (hyperbus_bench README). */
            int got = pi_fs_direct_read(file, buffer, want);
            if (got != (int)want) { error = 5; break; }
            for (int i = 0; i < got; i++) {
                *hash = (*hash ^ buffer[i]) * 16777619u;
                if (buffer[i] != (uint8_t)((*bytes + i) * 7u + FB_ID)) error = 6;
            }
            *bytes += got;
        }
        pi_fs_close(file);
    }
    pi_fs_unmount(&fs);
    pi_flash_close(&flash);
    return error;
}

static void report_task(void *arg)
{
    (void)arg;
    uint32_t bytes = 0, hash = 2166136261u, seq = 0;
    int error = check_payload(&bytes, &hash);
    while (1) {
        cpxPrintToConsole(LOG_TO_CRTP,
            "FBENCH|id=%u|seq=%lu|bytes=%lu|hash=%08lx|error=%d\n",
            (unsigned)FB_ID, (unsigned long)seq++, (unsigned long)bytes,
            (unsigned long)hash, error);
        vTaskDelay(pdMS_TO_TICKS(FB_INTERVAL_MS));
    }
}
#endif

static void start(void)
{
    pi_bsp_init();
    pi_freq_set(PI_FREQ_DOMAIN_FC, 100000000);
#if !FB_SILENT
    cpxInit();
    if (xTaskCreate(report_task, "flash_bench", configMINIMAL_STACK_SIZE * 8,
                    NULL, tskIDLE_PRIORITY + 1, NULL) != pdPASS) {
        while (1) {
            cpxPrintToConsole(LOG_TO_CRTP, "FBENCH|fatal=task_create\n");
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }
#endif
    while (1) vTaskDelay(pdMS_TO_TICKS(1000));
}

int main(void)
{
    return pmsis_kickoff((void *)start);
}
