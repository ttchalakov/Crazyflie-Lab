/*
 * main.c -- GAP8 harness for the grayscale JPEG encoder bench.
 *
 * Drives the public API (jpeg_enc.h) and holds it to one invariant:
 *
 *   THE FC PATH AND THE CLUSTER PATH EMIT BYTE-IDENTICAL JPEGS.
 *
 * Everything else here is measurement. Both paths run every round and are memcmp'd
 * against each other on the device, and an FNV-1a hash is printed so the host and
 * the simulator can be held to the same bytes. An optimization that changes the
 * output is a bug, not a speedup -- so the slow path is kept, deliberately, as the
 * oracle.
 *
 * Also swept: the L1 release policy, which is the DORY handoff question. The encoder
 * and a DORY net share the cluster serially, so the encoder must be able to give L1
 * back between frames; what that costs is measured, not assumed.
 *
 * Two targets from one source:
 *
 *   io=host -> GVSOC. One round; hex-dumps the JPEG over semihosted stdout so the
 *              host can byte-compare and decode it.
 *   io=uart -> the AI-deck. Loops forever, reporting each round over CPX: a
 *              cluster encoder that wedges the fabric only shows it by stopping.
 */

#include "pmsis.h"

#include "jpeg_enc.h"
#include "realimg.h"

#ifdef JPEG_CLUSTER_BENCH_SIM
/* Semihosted stdout; the simulator's console is the host's. */
#define BENCH_LOG(...) printf(__VA_ARGS__)
#else
#include "bsp/bsp.h"
#include "cpx.h"
/* On the AI-deck the GAP8's printf goes to physical UART pins nothing is listening
 * to. The only route back to the host is CPX -> ESP32 -> STM32 -> CRTP console. */
#define BENCH_LOG(...) cpxPrintToConsole(LOG_TO_CRTP, __VA_ARGS__)
#endif

/*
 * The encoded window: the whole real frame by default, or a sub-rectangle of it
 * (BENCH_DEFINES='-DIMG_W=316 -DIMG_H=236'; host/bench.yaml's cases). The stride
 * stays 320, which also exercises stride != width; a size that is not a multiple
 * of 8 exercises the edge blocks.
 */
#ifndef IMG_W
#define IMG_W 320
#endif
#ifndef IMG_H
#define IMG_H 240
#endif
#define IMG_STRIDE REAL_IMG_W

#define QUALITY 50
#define NCORES 8

/*
 * Buffer sizing, from two independent things -- never from a synthetic image:
 *
 *   - jpeg_enc_out_cap() / _seg_cap(): the RIGOROUS ceiling, derived from the quant
 *     and Huffman tables. Clear it and overflow is structurally impossible.
 *   - real captured frames + adversarial patterns (frames/, the bench's corpus check).
 *     A real QVGA frame is ~3.6 kB; the worst adversarial pattern (salt-and-pepper:
 *     max amplitude AND spatially uncorrelated) is 40.9 kB, worst segment 5.1 kB.
 *
 * At QVGA the ceiling is ~315 kB, unaffordable next to a camera buffer and DORY, so
 * QVGA is budget-sized and overflow is COUNTED. At 96x96 the ceiling is 38 kB, so
 * the app reports "overflow IMPOSSIBLE" and there is no failure path left.
 */
#define OUT_CAP (64 * 1024)
#define SEG_CAP (8 * 1024)

#ifdef JPEG_CLUSTER_BENCH_SIM
#define ROUNDS 1
#define ITERATIONS 1
#else
#define ROUNDS 0 /* forever */
#define ITERATIONS 20
#define FC_FREQ 150000000
#define CL_FREQ 150000000

/*
 * The benchmark task's stack, in WORDS (FreeRTOS counts stack depth in StackType_t,
 * which is 4 bytes here). configMINIMAL_STACK_SIZE is 128 words; a 2048-byte stack
 * would be smaller than one jpeg_gray_t (2068 B), and overflowing it kills the chip
 * silently. The app prints the stack's high-water mark (#STACK) every round.
 */
#define BENCH_STACK_WORDS (configMINIMAL_STACK_SIZE * 8) /* 4096 B */

#endif

/* L2. Static because the encoder struct alone (jpeg_enc_t) is over 3 kB, which does
 * not belong on a FreeRTOS task stack. */
static uint8_t l2_img[REAL_IMG_W * REAL_IMG_H];
static uint8_t l2_fc[OUT_CAP]; /* FC path result -- the reference */
static uint8_t l2_cl[OUT_CAP]; /* cluster path result */
static uint8_t l2_segs[NCORES * SEG_CAP];
static jpeg_enc_t l2_enc;

static uint32_t fnv1a(const uint8_t *buf, size_t len)
{
    uint32_t h = 2166136261u;

    for (size_t i = 0; i < len; i++) {
        h ^= buf[i];
        h *= 16777619u;
    }
    return h;
}

#ifdef JPEG_CLUSTER_BENCH_SIM
static void dump_jpeg(const uint8_t *buf, size_t len)
{
    printf("#JPEG-BEGIN %u\n", (unsigned)len);
    for (size_t i = 0; i < len; i++) {
        printf("%02x", buf[i]);
        if ((i & 31) == 31)
            printf("\n");
    }
    if (len & 31)
        printf("\n");
    printf("#JPEG-END\n");
}
#endif

/* Time ITERATIONS frames on `path`; returns us/frame and the last encoded size. */
static uint32_t time_path(jpeg_enc_t *j, jpeg_path_e path, uint8_t *out, size_t *len)
{
    uint32_t t0;

    jpeg_enc_set_path(j, path);
    t0 = pi_time_get_us();
    for (int i = 0; i < ITERATIONS; i++)
        *len = jpeg_enc_frame(j, l2_img, out, OUT_CAP);
    return (pi_time_get_us() - t0) / ITERATIONS;
}

static int bench_main(void)
{
    jpeg_enc_t *j = &l2_enc;
    struct pi_device cluster;
    struct pi_cluster_conf cl_conf;

    BENCH_LOG("== jpeg_cluster_bench: %dx%d gray, q%d, %d cores ==\n",
              IMG_W, IMG_H, QUALITY, NCORES);

    memcpy(l2_img, REAL_IMG_320x240, sizeof(l2_img));

    pi_cluster_conf_init(&cl_conf);
    pi_open_from_conf(&cluster, &cl_conf);
    if (pi_cluster_open(&cluster)) {
        BENCH_LOG("FAIL: cluster open\n");
        pmsis_exit(-1);
    }
#ifndef JPEG_CLUSTER_BENCH_SIM
    /* Only now is the cluster power domain up, so only now can its FLL be
     * programmed. Doing this from main(), before the open, hangs the chip at boot
     * with no output at all. */
    pi_freq_set(PI_FREQ_DOMAIN_CL, CL_FREQ);
#endif

    jpeg_cfg_t cfg = {
        .path = JPEG_PATH_CLUSTER,
        .cores = NCORES,
        .quality = QUALITY,
        .restart_interval = 0, /* auto: one segment per core */
        .l1_policy = JPEG_CL_L1_RELEASE_ALL,
    };

    if (jpeg_enc_init(j, &cfg, IMG_W, IMG_H, IMG_STRIDE, &cluster,
                      l2_segs, SEG_CAP) != 0) {
        BENCH_LOG("FAIL: jpeg_enc_init\n");
        pmsis_exit(-1);
    }

    /* State the buffer situation out loud instead of hoping. */
    size_t need_out = jpeg_enc_out_cap(IMG_W, IMG_H, QUALITY, NCORES);
    size_t need_seg = jpeg_enc_seg_cap(IMG_W, IMG_H, QUALITY, NCORES);
    int provable = (OUT_CAP >= need_out) && (SEG_CAP >= need_seg);

    BENCH_LOG("#BUF out %u/%u B seg %u/%u B -> overflow %s\n",
              (unsigned)OUT_CAP, (unsigned)need_out,
              (unsigned)SEG_CAP, (unsigned)need_seg,
              provable ? "IMPOSSIBLE (buffers clear the rigorous ceiling)"
                     : "possible: budget-sized, detected and counted");
    BENCH_LOG("#L1 footprint %u B of 65536\n", jpeg_cl_l1_footprint(&j->enc, NCORES));

    static const struct {
        jpeg_cl_l1_policy_e policy;
        const char *name;
    } POLICIES[] = {
        {JPEG_CL_L1_KEEP_ALL, "keep_all"},
        {JPEG_CL_L1_KEEP_TABLES, "keep_tables"},
        {JPEG_CL_L1_RELEASE_ALL, "release_all"},
    };

    uint32_t round = 0;
    const uint32_t rounds = ROUNDS; /* 0: forever */
    uint32_t frames = 0;
    uint32_t mismatches = 0;
    size_t fc_len = 0;

    do {
        size_t cl_len = 0;

        /* ---- the invariant: two paths, same bytes ---- */
        uint32_t fc_us = time_path(j, JPEG_PATH_FC, l2_fc, &fc_len);
        uint32_t fc_hash = fnv1a(l2_fc, fc_len);

        j->cfg.l1_policy = JPEG_CL_L1_RELEASE_ALL;
        uint32_t cl_us = time_path(j, JPEG_PATH_CLUSTER, l2_cl, &cl_len);
        uint32_t cl_hash = fnv1a(l2_cl, cl_len);

        if (fc_len == 0 || fc_len != cl_len || memcmp(l2_fc, l2_cl, fc_len) != 0)
            mismatches++;

        frames += 2 * ITERATIONS;

        /* ---- the DORY handoff: what does giving L1 back actually cost? ---- */
        uint32_t pol_us[3];
        uint32_t pol_l1[3];

        for (int p = 0; p < 3; p++) {
            size_t n = 0;

            j->cfg.l1_policy = POLICIES[p].policy;
            pol_us[p] = time_path(j, JPEG_PATH_CLUSTER, l2_cl, &n);
            pol_l1[p] = jpeg_cl_l1_held(&j->l1);
            if (n != fc_len || memcmp(l2_cl, l2_fc, n) != 0)
                mismatches++;
            frames += ITERATIONS;
        }
        j->cfg.l1_policy = JPEG_CL_L1_RELEASE_ALL;

#ifdef JPEG_CLUSTER_BENCH_SIM
        BENCH_LOG("#PATH fc      %8u us  bytes %5u  hash %08x\n",
                  fc_us, (unsigned)fc_len, fc_hash);
        BENCH_LOG("#PATH cluster %8u us  bytes %5u  hash %08x\n",
                  cl_us, (unsigned)cl_len, cl_hash);
        for (int p = 0; p < 3; p++)
            BENCH_LOG("#POL %-12s %8u us | L1 held %5u B | free for DORY %5u B\n",
                      POLICIES[p].name, pol_us[p], pol_l1[p], 65536 - pol_l1[p]);
        BENCH_LOG("#STAT identical %d\n", mismatches == 0 ? 1 : 0);
        BENCH_LOG("#STAT bytes %u\n", (unsigned)cl_len);
        BENCH_LOG("#STAT hash %08x\n", cl_hash);
        BENCH_LOG("#STAT provable %d\n", provable);
        BENCH_LOG("#STAT overflows %u\n", j->overflows);
        BENCH_LOG("#STAT peak_seg %u\n", j->peak_seg_bytes);
#else
        BENCH_LOG("#ROUND %u frames %u | fc %u us | cluster %u us (%u.%02ux) | "
                  "hash %08x/%08x | identical %s | overflows %u | peak_seg %u/%u\n",
                  round, frames, fc_us, cl_us,
                  cl_us ? fc_us / cl_us : 0,
                  cl_us ? (fc_us * 100 / cl_us) % 100 : 0,
                  fc_hash, cl_hash, mismatches == 0 ? "YES" : "NO",
                  j->overflows, j->peak_seg_bytes, (unsigned)SEG_CAP);
        BENCH_LOG("#POL keep_all %u us (L1 %u) | keep_tables %u us (L1 %u) | "
                  "release_all %u us (L1 %u -> DORY gets %u B)\n",
                  pol_us[0], pol_l1[0], pol_us[1], pol_l1[1],
                  pol_us[2], pol_l1[2], 65536 - pol_l1[2]);
        /*
         * Stack headroom: uxTaskGetStackHighWaterMark returns the smallest number of
         * WORDS this task's stack has ever had free.
         */
        {
            uint32_t free_words = (uint32_t)uxTaskGetStackHighWaterMark(NULL);

            BENCH_LOG("#STACK bench task: %u B total, %u B min free, %u B peak used\n",
                      (unsigned)(BENCH_STACK_WORDS * 4),
                      (unsigned)(free_words * 4),
                      (unsigned)((BENCH_STACK_WORDS - free_words) * 4));
        }
        pi_time_wait_us(1000000);
#endif
        round++;
    } while (rounds == 0 || round < rounds);

    jpeg_enc_deinit(j);
    pi_cluster_close(&cluster);

    if (mismatches) {
        BENCH_LOG("FAIL: the FC and cluster paths disagree (%u)\n", mismatches);
        BENCH_LOG("#DONE\n");
        pmsis_exit(-1);
    }

#ifdef JPEG_CLUSTER_BENCH_SIM
    dump_jpeg(l2_cl, fc_len);
#endif
    BENCH_LOG("#DONE\n");

    /* Without this the FreeRTOS entry task just restarts and GVSOC never stops --
     * the app re-encodes the frame forever. */
    pmsis_exit(0);
    return 0;
}

#ifndef JPEG_CLUSTER_BENCH_SIM

static void bench_task(void *arg)
{
    (void)arg;
    bench_main();
}

/*
 * The benchmark cannot run on the main task. CPX ships console lines from its own
 * FreeRTOS task, so a main task sitting in a tight encode loop starves it and
 * nothing ever reaches the host -- the ESP32 just reports "rxp 0". camera_stream has
 * the same shape: start the workers, then yield forever.
 */
static void start_tasks(void)
{
    cpxInit();
    cpxPrintToConsole(LOG_TO_CRTP, "-- jpeg_cluster_bench GAP8 app --\n");

    if (xTaskCreate(bench_task, "bench", BENCH_STACK_WORDS, NULL,
                    tskIDLE_PRIORITY + 1, NULL) != pdPASS) {
        cpxPrintToConsole(LOG_TO_CRTP, "FAIL: bench task creation failed\n");
        pmsis_exit(-1);
    }

    while (1)
        pi_yield();
}

int main(void)
{
    pi_bsp_init();
    __pi_pmu_voltage_set(PI_PMU_DOMAIN_FC, 1200);
    /* 150 MHz, not 200: the FC clock camera_stream runs beside the cluster JPEG
     * encoder.
     *
     * The cluster FLL is deliberately NOT touched here: its power domain is still
     * off, and programming it before pi_cluster_open() hangs the chip at boot. */
    pi_freq_set(PI_FREQ_DOMAIN_FC, FC_FREQ);

    return pmsis_kickoff((void *)start_tasks);
}

#else /* GVSOC: no CPX, nothing to starve -- run it straight. */

int main(void)
{
    return pmsis_kickoff((void *)bench_main);
}

#endif
