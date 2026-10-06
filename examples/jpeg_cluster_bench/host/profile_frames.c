/*
 * profile_frames.c -- run real captured frames through the encoder and report what
 * they actually cost, so the device buffers can be sized from evidence instead of
 * from a synthetic gradient.
 *
 * Answers three questions:
 *   1. How big is an encoded frame, really? (test images are far too kind)
 *   2. How big can it get? (the adversarial patterns are the practical ceiling)
 *   3. How big is the largest *per-core segment*? That, not the total, is what
 *      sizes SEG_CAP -- one core overflowing loses the whole frame.
 *
 * Also checks the invariant on every frame: the N-way merged bitstream must be
 * byte-identical to the serial one.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jpeg_gray.h"

/* gap8/main.c's encoder: the corpus is evidence for that app's buffers. */
#define QUALITY 50
#define CORES 8
#define CAP (512 * 1024)

static uint8_t *read_pgm(const char *path, int *w, int *h)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;

    char magic[3] = {0};
    int maxval = 0;
    if (fscanf(f, "%2s %d %d %d", magic, w, h, &maxval) != 4 ||
        strcmp(magic, "P5") != 0 || maxval != 255) {
        fclose(f);
        return NULL;
    }
    fgetc(f); /* the single whitespace byte before the pixel data */

    uint8_t *img = malloc((size_t)(*w) * (*h));
    if (img && fread(img, 1, (size_t)(*w) * (*h), f) != (size_t)(*w) * (*h)) {
        free(img);
        img = NULL;
    }
    fclose(f);
    return img;
}

/* Encode as CORES restart-aligned ranges and merge; assert it matches the serial run. */
static size_t encode_split(jpeg_gray_t *e, const uint8_t *img,
                           uint8_t *out, uint32_t *seg_max)
{
    static uint8_t seg[CORES][CAP];
    uint32_t starts[CORES + 1];
    size_t n;

    jpeg_gray_partition(e, CORES, starts);
    n = jpeg_gray_header(e, out, CAP);
    *seg_max = 0;

    for (int c = 0; c < CORES; c++) {
        size_t len = jpeg_gray_encode_range(e, img, starts[c], starts[c + 1],
                                            seg[c], CAP);
        if (starts[c + 1] > starts[c] && len == 0)
            return 0;
        if (len > *seg_max)
            *seg_max = (uint32_t)len;
        memcpy(out + n, seg[c], len);
        n += len;
    }
    n += jpeg_gray_footer(out + n, CAP - n);
    return n;
}

/* Usage: profile_frames <frame.pgm>... (host/bench.py passes frames/). */
int main(int argc, char **argv)
{
    static uint8_t serial[CAP], merged[CAP];

    printf("%-26s %8s %8s %9s %9s %9s %7s\n",
           "frame", "pixels", "bytes", "bits/px", "seg_max", "ratio", "match");
    printf("--------------------------------------------------------------"
           "---------------------\n");

    size_t worst_bytes = 0, worst_seg = 0;
    const char *worst_name = "";
    jpeg_gray_t last_e;

    for (int i = 1; i < argc; i++) {
        int w, h;
        uint8_t *img = read_pgm(argv[i], &w, &h);
        if (!img) {
            fprintf(stderr, "cannot read %s\n", argv[i]);
            return 1;
        }

        jpeg_gray_t e;
        int mcus = ((w + 7) / 8) * ((h + 7) / 8);
        /* Auto: exactly one restart segment per core -- perfectly balanced, and
         * fewer markers than one-segment-per-MCU-row. */
        int r = jpeg_gray_auto_restart(w, h, CORES);

        jpeg_gray_init(&e, w, h, w, QUALITY, r);

        size_t sn = jpeg_gray_encode(&e, img, serial, CAP);
        uint32_t seg_max = 0;
        size_t mn = encode_split(&e, img, merged, &seg_max);

        int match = (sn > 0 && sn == mn && memcmp(serial, merged, sn) == 0);
        size_t px = (size_t)w * h;

        const char *base = strrchr(argv[i], '/');
        base = base ? base + 1 : argv[i];

        if (i == 1)
            printf("# %dx%d, q%d, %d cores, restart %d (%d MCUs)\n",
                   w, h, QUALITY, CORES, r, mcus);
        printf("%-26s %8zu %8zu %9.3f %9u %8.1fx %7s\n",
               base, px, sn, (double)sn * 8.0 / (double)px, seg_max,
               (double)px / (double)sn, match ? "OK" : "MISMATCH");

        last_e = e;

        if (sn > worst_bytes) {
            worst_bytes = sn;
            worst_name = base;
        }
        if (seg_max > worst_seg)
            worst_seg = seg_max;
        if (!match)
            return fprintf(stderr, "FAIL: merged != serial on %s\n", base), 1;

        free(img);
    }

    /* The rigorous ceiling (jpeg_gray.h), from the actual quant + Huffman tables
     * rather than from whichever adversarial image we happened to think of. It
     * tells us whether "cannot overflow" is affordable, or whether we have to
     * handle overflow gracefully instead. */
    if (argc > 1) {
        printf("\n--- rigorous ceiling (int16 coefficients, all-AC-nonzero, q%d) ---\n",
               QUALITY);
        printf("frame ceiling    : %zu B  (header and EOI included)\n",
               jpeg_gray_max_bytes(&last_e));
        printf("per-core ceiling : %zu B  <- SEG_CAP for provable no-overflow\n",
               jpeg_gray_max_segment_bytes(&last_e, CORES));
    }

    printf("\nworst frame : %s at %zu bytes\n", worst_name, worst_bytes);
    printf("worst seg   : %zu bytes  <- SEG_CAP must exceed this\n", worst_seg);
    return 0;
}
