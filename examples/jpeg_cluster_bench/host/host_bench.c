/*
 * host_bench.c -- the encoder (lib/gap8/jpeg) on the host, the reference the GAP8 is
 * held to. host/bench.py builds and runs it for each window.
 *
 *   1. checks the reciprocal quantizer against division for every int16 coefficient
 *      and table entry,
 *   2. writes the encoded window of the real frame as a PGM (the ground truth for PSNR),
 *   3. encodes it serially, and as N restart-aligned ranges merged with memcpy (the
 *      host model of the cluster), and asserts the two are byte-identical.
 *
 * The device's JPEG must then equal the serial one byte for byte; its hash is the
 * FNV-1a the device prints.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jpeg_gray.h"
#include "realimg.h"

/* gap8/main.c's encoder. */
#define QUALITY 50
#define CORES 8

/* Same hash the device prints. On hardware we can't afford to ship 6 kB of hex
 * over the CRTP console, so the hash is how we prove the drone produced the same
 * bytes as the host and the simulator. */
static uint32_t fnv1a(const uint8_t *buf, size_t len)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        h ^= buf[i];
        h *= 16777619u;
    }
    return h;
}

static int write_file(const char *path, const uint8_t *buf, size_t len)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return -1;
    }
    fwrite(buf, 1, len, f);
    fclose(f);
    return 0;
}

int main(int argc, char **argv)
{
    int w = 320, h = 240;
    const int stride = REAL_IMG_W; /* a window of the real frame, as on the device */
    const char *pgm_path = "test.pgm";
    const char *serial_path = "host_serial.jpg";
    const char *merged_path = "host_merged.jpg";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--width") && i + 1 < argc) w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--height") && i + 1 < argc) h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pgm") && i + 1 < argc) pgm_path = argv[++i];
        else if (!strcmp(argv[i], "--serial") && i + 1 < argc) serial_path = argv[++i];
        else if (!strcmp(argv[i], "--merged") && i + 1 < argc) merged_path = argv[++i];
        else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (w < 1 || h < 1 || w > REAL_IMG_W || h > REAL_IMG_H) {
        fprintf(stderr, "the window must fit the %dx%d frame\n", REAL_IMG_W, REAL_IMG_H);
        return 2;
    }
    /* Exactly one restart segment per core. Perfectly balanced AND fewer markers
     * than one-per-MCU-row, so it is both faster and smaller. The device uses
     * jpeg_gray_auto_restart() for the same reason; the two must agree or the
     * byte-identical check is comparing different configurations. */
    const int restart = jpeg_gray_auto_restart(w, h, CORES);

    /* ---- exhaustively verify the reciprocal quantizer ----
     *
     * The optimized cluster kernel replaces 64 integer divides per block with a
     * multiply and a shift. That is only allowed to be a speedup, never a change
     * in output, so check it against the division for every quant-table entry and
     * every coefficient an int16 can hold. 64 * 65536 comparisons, instant. */
    {
        jpeg_gray_t q;
        long checked = 0;

        jpeg_gray_init(&q, w, h, w, QUALITY, restart);
        for (int i = 0; i < 64; i++) {
            int d = (int)q.qtab[i] << 3;
            for (int c = -32768; c <= 32767; c++) {
                int want = c < 0 ? -((-c + (d >> 1)) / d) : (c + (d >> 1)) / d;
                int got = jpeg_quant_recip(&q, i, c);
                if (want != got) {
                    fprintf(stderr,
                            "FAIL: reciprocal quant differs at qtab[%d]=%u coef=%d: "
                            "div=%d recip=%d\n", i, q.qtab[i], c, want, got);
                    return 1;
                }
                checked++;
            }
        }
        printf("recip quant: OK, exact over %ld coefficient/table pairs\n", checked);
    }

    /* ---- the window, and its PGM ---- */
    const uint8_t *img = REAL_IMG_320x240;
    {
        FILE *f = fopen(pgm_path, "wb");
        if (!f) { perror(pgm_path); return 1; }
        fprintf(f, "P5\n%d %d\n255\n", w, h);
        for (int y = 0; y < h; y++)
            fwrite(img + (size_t)y * stride, 1, (size_t)w, f);
        fclose(f);
    }

    jpeg_gray_t enc;
    jpeg_gray_init(&enc, w, h, stride, QUALITY, restart);

    size_t cap = (size_t)w * h * 2 + 8192;

    /* ---- serial ---- */
    uint8_t *ser = malloc(cap);
    size_t ser_len = jpeg_gray_encode(&enc, img, ser, cap);
    if (!ser_len) {
        fprintf(stderr, "serial encode overflowed\n");
        return 1;
    }
    write_file(serial_path, ser, ser_len);

    /* ---- N-way split + memcpy merge (host model of the cluster) ---- */
    uint32_t starts[CORES + 1];
    jpeg_gray_partition(&enc, CORES, starts);

    /* Per-range scratch buffers, exactly as each core will own one. */
    uint8_t *seg[CORES];
    size_t seg_len[CORES];
    size_t seg_cap = cap / CORES + 4096;
    for (int c = 0; c < CORES; c++) {
        seg[c] = malloc(seg_cap);
        seg_len[c] = jpeg_gray_encode_range(&enc, img, starts[c], starts[c + 1],
                                            seg[c], seg_cap);
        if (starts[c + 1] > starts[c] && seg_len[c] == 0) {
            fprintf(stderr, "range %d overflowed\n", c);
            return 1;
        }
    }

    uint8_t *mrg = malloc(cap);
    size_t n = jpeg_gray_header(&enc, mrg, cap);
    for (int c = 0; c < CORES; c++) {
        memcpy(mrg + n, seg[c], seg_len[c]);
        n += seg_len[c];
    }
    n += jpeg_gray_footer(mrg + n, cap - n);
    write_file(merged_path, mrg, n);

    /* ---- the invariant ---- */
    int identical = (n == ser_len) && !memcmp(mrg, ser, n);

    printf("image      : %dx%d, %u MCUs, quality %d, restart %d\n",
           w, h, enc.mcu_count, QUALITY, restart);
    printf("serial     : %s (%zu bytes) hash %08x\n",
           serial_path, ser_len, fnv1a(ser, ser_len));
    printf("merged x%-2d : %s (%zu bytes)  [", CORES, merged_path, n);
    for (int c = 0; c < CORES; c++)
        printf("%s%zu", c ? " " : "", seg_len[c]);
    printf("]\n");
    printf("ranges     : [");
    for (int c = 0; c < CORES; c++)
        printf("%s%u-%u", c ? " " : "", starts[c], starts[c + 1]);
    printf("]\n");
    printf("%s: merged bitstream is %sbyte-identical to serial\n",
           identical ? "OK  " : "FAIL", identical ? "" : "NOT ");

    return identical ? 0 : 1;
}
