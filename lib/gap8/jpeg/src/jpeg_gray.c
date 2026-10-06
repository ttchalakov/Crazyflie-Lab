/*
 * jpeg_gray.c -- baseline grayscale JPEG encoder core.
 *
 * Compiled unchanged into both the host reference build and the GAP8 app, so
 * "the simulator produced the same bytes as the host" is a meaningful test of
 * the port rather than a test of two different encoders.
 *
 * See jpeg_gray.h for the restart-interval argument that makes MCU ranges
 * independent.
 */

#include <string.h>
#include "jpeg_gray.h"

/* ------------------------------------------------------------------ tables */

/* Annex K.1 luminance quantization table (the quality-50 baseline). */
static const uint8_t QUANT_LUMA[64] = {
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99};

/* ZIGZAG[k] = natural (row-major) index of the k-th coefficient in zig-zag order. */
static const uint8_t ZIGZAG[64] = {
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63};

/* Annex K.3 standard luma Huffman specs. BITS[len] = number of codes of that
 * length; index 0 unused so lengths stay 1-based. */
static const uint8_t DC_BITS[17] = {0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
static const uint8_t DC_VALS[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

static const uint8_t AC_BITS[17] = {0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d};
static const uint8_t AC_VALS[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa};

#define ZRL 0xF0
#define EOB 0x00

/* The Annex-K table scaled for `quality`. 64 bytes -- the only quality-dependent
 * state in the whole encoder, which is what lets the size bounds below be computed
 * without constructing a (2 kB) jpeg_gray_t. */
static void build_qtab(uint8_t qtab[64], int quality)
{
    if (quality < 1)
        quality = 1;
    if (quality > 100)
        quality = 100;

    int scale = quality < 50 ? 5000 / quality : 200 - 2 * quality;

    for (int i = 0; i < 64; i++) {
        int v = ((int)QUANT_LUMA[i] * scale + 50) / 100;

        if (v < 1)
            v = 1;
        if (v > 255)
            v = 255;
        qtab[i] = (uint8_t)v;
    }
}

/*
 * Code length for one symbol, read straight from the canonical Huffman spec.
 *
 * The size bounds need a handful of code lengths, not the whole 256-entry LUT that
 * build_huff() produces -- and the Huffman tables are the *standard* Annex-K ones,
 * identical for every encoder instance regardless of quality. So the bounds can be
 * computed from the spec directly and never need an encoder to exist.
 */
static int huff_len(const uint8_t bits[17], const uint8_t *vals, int sym)
{
    int k = 0;

    for (int len = 1; len <= 16; len++)
        for (int i = 0; i < bits[len]; i++, k++)
            if (vals[k] == sym)
                return len;
    return 0;
}

static void build_huff(const uint8_t bits[17], const uint8_t *vals, jpeg_huff_t *h)
{
    uint16_t code = 0;
    int k = 0;

    memset(h, 0, sizeof(*h));
    for (int len = 1; len <= 16; len++) {
        for (int i = 0; i < bits[len]; i++) {
            uint8_t sym = vals[k++];
            h->code[sym] = code;
            h->size[sym] = (uint8_t)len;
            code++;
        }
        code <<= 1;
    }
}

/* ------------------------------------------------------------- bit writer */

typedef struct {
    uint8_t *buf;
    size_t cap;
    size_t len;
    uint32_t acc;  /* bit accumulator, MSB-first */
    int nbits;
    int overflow;
} bitw_t;

static void bw_init(bitw_t *b, uint8_t *buf, size_t cap)
{
    b->buf = buf;
    b->cap = cap;
    b->len = 0;
    b->acc = 0;
    b->nbits = 0;
    b->overflow = 0;
}

static inline void bw_byte(bitw_t *b, uint8_t v)
{
    if (b->len < b->cap)
        b->buf[b->len++] = v;
    else
        b->overflow = 1;
}

/* Append the low `n` bits of `v`, MSB-first, stuffing a 0x00 after any 0xFF. */
static inline void bw_bits(bitw_t *b, uint32_t v, int n)
{
    if (n == 0)
        return;
    b->acc = (b->acc << n) | (v & ((1u << n) - 1));
    b->nbits += n;
    while (b->nbits >= 8) {
        b->nbits -= 8;
        uint8_t byte = (uint8_t)(b->acc >> b->nbits);
        bw_byte(b, byte);
        if (byte == 0xFF)
            bw_byte(b, 0x00);
    }
}

/* Pad to a byte boundary with 1-bits. After this the stream is byte-aligned,
 * which is what lets a segment be memcpy'd into place. */
static void bw_align(bitw_t *b)
{
    if (b->nbits > 0) {
        int pad = 8 - b->nbits;
        uint8_t byte = (uint8_t)(((b->acc << pad) | ((1u << pad) - 1)) & 0xFF);
        bw_byte(b, byte);
        if (byte == 0xFF)
            bw_byte(b, 0x00);
    }
    b->acc = 0;
    b->nbits = 0;
}

static void bw_u16(bitw_t *b, uint16_t v)
{
    bw_byte(b, (uint8_t)(v >> 8));
    bw_byte(b, (uint8_t)(v & 0xFF));
}

/* ------------------------------------------------------------ block encode */

/* Magnitude category of v, and the amplitude bits JPEG stores for it. */
static inline int coef_size(int v)
{
    int a = v < 0 ? -v : v;
    int s = 0;
    while (a) {
        a >>= 1;
        s++;
    }
    return s;
}

static inline uint32_t coef_bits(int v, int size)
{
    /* Negatives are stored as (v-1) truncated to `size` bits. */
    return (uint32_t)(v < 0 ? (v - 1) : v) & ((1u << size) - 1);
}

/* Gather one 8x8 block, replicating the edge pixel when the image is not a
 * multiple of 8, and level-shift by -128. */
static void gather_block(const uint8_t *img, const jpeg_gray_t *e,
                         int bx, int by, int16_t out[64])
{
    for (int i = 0; i < 8; i++) {
        int y = by + i;
        if (y >= e->height)
            y = e->height - 1;
        const uint8_t *row = img + (size_t)y * e->stride;
        for (int j = 0; j < 8; j++) {
            int x = bx + j;
            if (x >= e->width)
                x = e->width - 1;
            out[i * 8 + j] = (int16_t)row[x] - 128;
        }
    }
}

/*
 * DCT -> quantize -> zig-zag RLE -> Huffman, for one block.
 *
 * The DCT output is scaled by 8, so the divisor is qtab[i] << 3. Rounding is
 * done on the magnitude to keep it symmetric about zero (the C division
 * truncates toward zero).
 */
static void encode_block(bitw_t *b, const jpeg_gray_t *e, int16_t px[64], int *pred_dc)
{
    int q[64];

    jpeg_fdct_8x8(px);

    for (int i = 0; i < 64; i++) {
        int32_t coef = px[i];
        int32_t div = (int32_t)e->qtab[i] << 3;
        int32_t half = div >> 1;
        q[i] = coef < 0 ? -(int)((-coef + half) / div) : (int)((coef + half) / div);
    }

    /* DC: coded as a difference from the previous block's DC. The predictor is
     * reset to 0 at every restart marker, which is what decouples segments. */
    int diff = q[0] - *pred_dc;
    *pred_dc = q[0];
    int size = coef_size(diff);
    bw_bits(b, e->dc.code[size], e->dc.size[size]);
    bw_bits(b, coef_bits(diff, size), size);

    /* AC: zig-zag scan, run-length of zeros + magnitude category. */
    int run = 0;
    for (int k = 1; k < 64; k++) {
        int coef = q[ZIGZAG[k]];
        if (coef == 0) {
            run++;
            continue;
        }
        while (run > 15) {
            bw_bits(b, e->ac.code[ZRL], e->ac.size[ZRL]);
            run -= 16;
        }
        size = coef_size(coef);
        int rs = (run << 4) | size;
        bw_bits(b, e->ac.code[rs], e->ac.size[rs]);
        bw_bits(b, coef_bits(coef, size), size);
        run = 0;
    }
    if (run > 0)
        bw_bits(b, e->ac.code[EOB], e->ac.size[EOB]);
}

/* --------------------------------------------------------------- public API */

void jpeg_gray_init(jpeg_gray_t *e, int width, int height, int stride,
                    int quality, int restart_interval)
{
    memset(e, 0, sizeof(*e));
    e->width = (uint16_t)width;
    e->height = (uint16_t)height;
    e->stride = (uint16_t)(stride > 0 ? stride : width);
    e->mcus_x = (uint16_t)((width + 7) / 8);
    e->mcus_y = (uint16_t)((height + 7) / 8);
    e->mcu_count = (uint32_t)e->mcus_x * e->mcus_y;
    e->restart_interval = (uint16_t)restart_interval;

    build_qtab(e->qtab, quality);

    build_huff(DC_BITS, DC_VALS, &e->dc);
    build_huff(AC_BITS, AC_VALS, &e->ac);

    /*
     * Reciprocal tables. For divisor d we want floor(a/d) == (a * m) >> s.
     * Writing m = ceil(2^s / d) and e = m*d - 2^s (so 0 <= e < d), the identity
     * holds for every a with a*e < 2^s. The largest a we can ever see is
     * 32767 + d/2, so picking the smallest s with 2^s > a_max * d makes it exact
     * for the entire int16 range -- no bound assumptions about DCT output.
     */
    for (int i = 0; i < 64; i++) {
        uint32_t d = (uint32_t)e->qtab[i] << 3;
        uint32_t a_max = 32767u + (d >> 1);
        uint32_t s = 0;

        while (((uint64_t)1 << s) <= (uint64_t)a_max * d)
            s++;

        e->qshift[i] = (uint8_t)s;
        e->qrecip[i] = (uint32_t)((((uint64_t)1 << s) + d - 1) / d);
        e->qhalf[i] = (uint16_t)(d >> 1);
    }
}

size_t jpeg_gray_header(const jpeg_gray_t *e, uint8_t *out, size_t cap)
{
    bitw_t b;
    bw_init(&b, out, cap);

    /* SOI */
    bw_byte(&b, 0xFF);
    bw_byte(&b, 0xD8);

    /* DQT: one 8-bit table, id 0, stored in zig-zag order. */
    bw_byte(&b, 0xFF);
    bw_byte(&b, 0xDB);
    bw_u16(&b, 2 + 1 + 64);
    bw_byte(&b, 0x00);
    for (int i = 0; i < 64; i++)
        bw_byte(&b, e->qtab[ZIGZAG[i]]);

    /* SOF0: baseline, one component, 1x1 sampling. */
    bw_byte(&b, 0xFF);
    bw_byte(&b, 0xC0);
    bw_u16(&b, 8 + 3 * 1);
    bw_byte(&b, 8);
    bw_u16(&b, e->height);
    bw_u16(&b, e->width);
    bw_byte(&b, 1);
    bw_byte(&b, 1);
    bw_byte(&b, 0x11);
    bw_byte(&b, 0x00);

    /* DHT: DC table 0. */
    bw_byte(&b, 0xFF);
    bw_byte(&b, 0xC4);
    {
        int nv = 0;
        for (int l = 1; l <= 16; l++)
            nv += DC_BITS[l];
        bw_u16(&b, (uint16_t)(2 + 1 + 16 + nv));
        bw_byte(&b, 0x00); /* class 0 (DC), id 0 */
        for (int l = 1; l <= 16; l++)
            bw_byte(&b, DC_BITS[l]);
        for (int i = 0; i < nv; i++)
            bw_byte(&b, DC_VALS[i]);
    }

    /* DHT: AC table 0. */
    bw_byte(&b, 0xFF);
    bw_byte(&b, 0xC4);
    {
        int nv = 0;
        for (int l = 1; l <= 16; l++)
            nv += AC_BITS[l];
        bw_u16(&b, (uint16_t)(2 + 1 + 16 + nv));
        bw_byte(&b, 0x10); /* class 1 (AC), id 0 */
        for (int l = 1; l <= 16; l++)
            bw_byte(&b, AC_BITS[l]);
        for (int i = 0; i < nv; i++)
            bw_byte(&b, AC_VALS[i]);
    }

    /* DRI */
    if (e->restart_interval > 0) {
        bw_byte(&b, 0xFF);
        bw_byte(&b, 0xDD);
        bw_u16(&b, 4);
        bw_u16(&b, e->restart_interval);
    }

    /* SOS */
    bw_byte(&b, 0xFF);
    bw_byte(&b, 0xDA);
    bw_u16(&b, 6 + 2 * 1);
    bw_byte(&b, 1);
    bw_byte(&b, 1);
    bw_byte(&b, 0x00); /* DC table 0, AC table 0 */
    bw_byte(&b, 0x00); /* Ss */
    bw_byte(&b, 0x3F); /* Se */
    bw_byte(&b, 0x00); /* Ah/Al */

    return b.overflow ? 0 : b.len;
}

size_t jpeg_gray_footer(uint8_t *out, size_t cap)
{
    if (cap < 2)
        return 0;
    out[0] = 0xFF;
    out[1] = 0xD9;
    return 2;
}

size_t jpeg_gray_encode_range(const jpeg_gray_t *e, const uint8_t *img,
                              uint32_t mcu_start, uint32_t mcu_end,
                              uint8_t *out, size_t cap)
{
    bitw_t b;
    int pred_dc = 0;
    int16_t px[64];

    bw_init(&b, out, cap);

    for (uint32_t m = mcu_start; m < mcu_end; m++) {
        /* Restart boundary. For m == mcu_start this is the segment's *leading*
         * marker, and the align is a no-op on an empty stream -- which is
         * precisely why the merge can be a plain memcpy. */
        if (e->restart_interval > 0 && m > 0 && (m % e->restart_interval) == 0) {
            bw_align(&b);
            bw_byte(&b, 0xFF);
            bw_byte(&b, (uint8_t)(0xD0 + (((m / e->restart_interval) - 1) & 7)));
            pred_dc = 0;
        }
        gather_block(img, e, (int)(m % e->mcus_x) * 8, (int)(m / e->mcus_x) * 8, px);
        encode_block(&b, e, px, &pred_dc);
    }
    bw_align(&b);

    return b.overflow ? 0 : b.len;
}

size_t jpeg_gray_encode(const jpeg_gray_t *e, const uint8_t *img,
                        uint8_t *out, size_t cap)
{
    size_t n = jpeg_gray_header(e, out, cap);
    if (n == 0)
        return 0;

    size_t body = jpeg_gray_encode_range(e, img, 0, e->mcu_count, out + n, cap - n);
    if (body == 0)
        return 0;
    n += body;

    size_t foot = jpeg_gray_footer(out + n, cap - n);
    if (foot == 0)
        return 0;
    return n + foot;
}

void jpeg_gray_partition(const jpeg_gray_t *e, int n, uint32_t *starts)
{
    uint32_t r = e->restart_interval ? e->restart_interval : e->mcu_count;
    uint32_t nseg = (e->mcu_count + r - 1) / r; /* independent segments available */

    for (int i = 0; i <= n; i++) {
        uint32_t seg = ((uint64_t)nseg * (uint32_t)i) / (uint32_t)n;
        uint32_t m = seg * r;
        starts[i] = m > e->mcu_count ? e->mcu_count : m;
    }
    starts[n] = e->mcu_count;
}

/* --------------------------------------------------------- worst-case sizing */

/*
 * Worst-case *bytes* for one 8x8 block, from the real tables.
 *
 * |coef| <= 32767 because the DCT writes an int16_t. That caps the quantized value
 * at coefficient i at (32767 + d/2)/d for d = qtab[i]*8, which caps its magnitude
 * category, which caps its Huffman code length. The most expensive block therefore
 * has every AC coefficient nonzero with a zero run -- coding a run of zeros (ZRL,
 * or EOB) is strictly cheaper than coding a value, so nothing beats it.
 */
static size_t worst_block_bytes(const uint8_t qtab[64])
{
    int d0 = (int)qtab[0] << 3;
    int dc_maxq = (32767 + (d0 >> 1)) / d0;
    /* A DC *difference* spans twice the coefficient range. */
    int dc_sz = coef_size(2 * dc_maxq);

    unsigned long bits = (unsigned long)huff_len(DC_BITS, DC_VALS, dc_sz) +
                         (unsigned long)dc_sz;

    for (int i = 1; i < 64; i++) {
        int d = (int)qtab[i] << 3;
        int maxq = (32767 + (d >> 1)) / d;
        int sz = coef_size(maxq);

        if (sz > 10)
            sz = 10; /* baseline caps the AC magnitude category at 10 */
        bits += (unsigned long)huff_len(AC_BITS, AC_VALS, (0 << 4) | sz) +
                (unsigned long)sz;
    }
    return (size_t)((bits + 7) / 8);
}

/* Shared arithmetic, so the "I have an encoder" and "I only have dimensions" entry
 * points below cannot drift apart. */
static size_t seg_bound(const uint8_t qtab[64], uint32_t mcu_count,
                        uint32_t restart, int cores)
{
    uint32_t r = restart ? restart : mcu_count;
    uint32_t nseg = (mcu_count + r - 1) / r;
    /* The largest range any single core can be handed by jpeg_gray_partition. */
    uint32_t seg_per_core = (nseg + (uint32_t)cores - 1) / (uint32_t)cores;
    uint32_t mcus = seg_per_core * r;

    if (mcus > mcu_count)
        mcus = mcu_count;

    /* x2 for 0xFF stuffing (every emitted byte could in principle need a 0x00 after
     * it), plus this segment's own RSTn marker and byte-alignment padding. */
    return 2 * worst_block_bytes(qtab) * (size_t)mcus + 16;
}

static size_t frame_bound(const uint8_t qtab[64], uint32_t mcu_count, uint32_t restart)
{
    uint32_t r = restart ? restart : mcu_count;
    uint32_t nseg = (mcu_count + r - 1) / r;

    /* header (SOI+DQT+SOF0+2xDHT+DRI+SOS is comfortably under 700 B) + body + one
     * marker/padding allowance per segment + EOI. */
    return 700 + 2 * worst_block_bytes(qtab) * (size_t)mcu_count +
           3 * (size_t)nseg + 2;
}

size_t jpeg_gray_max_segment_bytes(const jpeg_gray_t *e, int cores)
{
    return seg_bound(e->qtab, e->mcu_count, e->restart_interval, cores);
}

size_t jpeg_gray_max_bytes(const jpeg_gray_t *e)
{
    return frame_bound(e->qtab, e->mcu_count, e->restart_interval);
}

/*
 * The same bounds from dimensions alone -- no encoder needed, and therefore no 2 kB
 * struct anywhere near a stack.
 *
 * This exists because the obvious implementation (build a jpeg_gray_t, ask it) is a
 * trap on this chip: sizeof(jpeg_gray_t) is 2068 bytes, and a FreeRTOS task here gets
 * a 2048-byte stack. A single local instance does not merely risk overflowing the
 * stack, it is LARGER THAN THE WHOLE STACK -- the chip dies instantly, silently, with
 * no output at all.
 *
 * Only 64 bytes of quant table are needed; the Huffman tables are the standard
 * Annex-K ones and never vary.
 */
size_t jpeg_gray_max_segment_bytes_for(int width, int height, int quality, int cores)
{
    uint8_t qtab[64];
    uint32_t mcus = (uint32_t)((width + 7) / 8) * (uint32_t)((height + 7) / 8);

    build_qtab(qtab, quality);
    return seg_bound(qtab, mcus,
                     (uint32_t)jpeg_gray_auto_restart(width, height, cores), cores);
}

size_t jpeg_gray_max_bytes_for(int width, int height, int quality, int cores)
{
    uint8_t qtab[64];
    uint32_t mcus = (uint32_t)((width + 7) / 8) * (uint32_t)((height + 7) / 8);

    build_qtab(qtab, quality);
    return frame_bound(qtab, mcus,
                       (uint32_t)jpeg_gray_auto_restart(width, height, cores));
}
