/*
 * jpeg_gray.h -- baseline JPEG encoder purpose-built for the AI-deck's
 * grayscale Himax camera.
 *
 * Single component, 8-bit, 1x1 sampling, so one MCU is exactly one 8x8 block.
 * There is no color path and no subsampling machinery: the camera gives us Y
 * and only Y. Integer-only arithmetic (GAP8 has no FPU).
 *
 * The one idea the whole design rests on: with a restart interval, every
 * segment of `restart_interval` MCUs is *self-contained* -- its DC predictor
 * starts at 0 and its entropy stream is byte-aligned. So an MCU range that
 * begins and ends on a restart boundary can be encoded by any core, into any
 * buffer, with no knowledge of the other ranges, and the segments concatenate
 * with plain memcpy.
 *
 * That makes the serial and parallel encoders produce *byte-identical* output,
 * which is what we test against.
 *
 * This header is shared verbatim by the host build and the GAP8 build.
 */
#ifndef JPEG_GRAY_H
#define JPEG_GRAY_H

#include <stdint.h>
#include <stddef.h>

/* Canonical Huffman code/length, indexed by symbol value. */
typedef struct {
    uint16_t code[256];
    uint8_t size[256];
} jpeg_huff_t;

typedef struct {
    uint16_t width;
    uint16_t height;
    uint16_t stride;           /* bytes per source row; >= width */
    uint16_t mcus_x;           /* blocks across = ceil(width/8) */
    uint16_t mcus_y;           /* blocks down  = ceil(height/8) */
    uint32_t mcu_count;        /* mcus_x * mcus_y */
    uint16_t restart_interval; /* MCUs between RSTn markers; 0 = none */
    uint8_t qtab[64];          /* quant table, natural (row-major) order */
    jpeg_huff_t dc;
    jpeg_huff_t ac;

    /*
     * Reciprocal-multiply quantization, used by the optimized cluster kernel.
     * The reference kernel divides; a divide is ~35 cycles on the RI5CY core and
     * there are 64 of them per block, so this is the single biggest win available.
     *
     * For divisor d = qtab[i] << 3:  |coef| / d  ==  (a * qrecip[i]) >> qshift[i]
     * where a = |coef| + d/2. The shift is chosen per coefficient so this is
     * *exact* for every int16 coefficient, not an approximation -- so the
     * optimized encoder still emits the identical bitstream. jpeg_cluster_bench's host_bench verifies
     * that exhaustively over the whole int16 range.
     */
    uint32_t qrecip[64];
    uint16_t qhalf[64]; /* d/2, the rounding term */
    uint8_t qshift[64];
} jpeg_gray_t;

/* Exact quantization of one coefficient via the reciprocal tables above. */
static inline int jpeg_quant_recip(const jpeg_gray_t *e, int i, int coef)
{
    int neg = coef < 0;
    uint32_t a = (uint32_t)(neg ? -coef : coef) + e->qhalf[i];
    int q = (int)((a * e->qrecip[i]) >> e->qshift[i]);
    return neg ? -q : q;
}

/*
 * quality: 1..100 (50 = the unscaled Annex-K luma table).
 * restart_interval: MCUs per independent segment. 0 disables restarts, which
 *   also makes parallel encoding impossible -- it is only useful as a
 *   single-core reference.
 */
void jpeg_gray_init(jpeg_gray_t *e, int width, int height, int stride,
                    int quality, int restart_interval);

/* SOI, DQT, SOF0, DHT x2, DRI, SOS. Returns bytes written. */
size_t jpeg_gray_header(const jpeg_gray_t *e, uint8_t *out, size_t cap);

/* EOI. Returns bytes written. */
size_t jpeg_gray_footer(uint8_t *out, size_t cap);

/*
 * Encode MCUs [mcu_start, mcu_end) into a self-contained, byte-aligned entropy
 * segment. If mcu_start > 0 the segment *opens* with its own RSTn marker, so
 * merging is a pure memcpy of the ranges in order -- the merger needs no
 * knowledge of markers or bit offsets.
 *
 * mcu_start and mcu_end must be multiples of restart_interval (mcu_end may also
 * be mcu_count). Returns bytes written, or 0 if cap was too small.
 */
size_t jpeg_gray_encode_range(const jpeg_gray_t *e, const uint8_t *img,
                              uint32_t mcu_start, uint32_t mcu_end,
                              uint8_t *out, size_t cap);

/* Whole image, single core: header + one range covering everything + EOI. */
size_t jpeg_gray_encode(const jpeg_gray_t *e, const uint8_t *img,
                        uint8_t *out, size_t cap);

/*
 * Split mcu_count into `n` contiguous chunks aligned to the restart interval,
 * as evenly as the interval allows. Fills starts[0..n] (n+1 entries, the last
 * being mcu_count). Shared by the host reference and the cluster so both agree
 * on the partition.
 */
void jpeg_gray_partition(const jpeg_gray_t *e, int n, uint32_t *starts);

/*
 * The restart interval that makes the partition exactly even: one self-contained
 * segment per core. Beats one-segment-per-MCU-row on both axes -- perfect load
 * balance (no core does 160 MCUs while another does 120) AND fewer markers, so the
 * file is smaller too. Use this unless you have a reason not to.
 */
static inline int jpeg_gray_auto_restart(int width, int height, int cores)
{
    int mcus = ((width + 7) / 8) * ((height + 7) / 8);
    return (mcus + cores - 1) / cores;
}

/* --- worst-case output size -------------------------------------------------
 *
 * Derived from the actual quantization and Huffman tables, NOT from whichever
 * adversarial image we thought to test. The DCT writes into an int16_t, so
 * |coef| <= 32767 by construction; that caps the quantized magnitude at each
 * coefficient, which caps its Huffman symbol, which caps its bit cost. The worst
 * block has all 63 AC coefficients nonzero with zero runs (a run of zeros is only
 * ever cheaper), and 0xFF stuffing can at most double the byte count.
 *
 * Against real camera frames this is enormously conservative, which is exactly the
 * point: it is a bound, not an estimate. Use it to decide whether "cannot overflow"
 * is affordable at a given resolution:
 *
 *   96x96  : ~38 kB total. Cheap. Size for it and overflow becomes impossible.
 *   QVGA   : ~315 kB total. Not payable alongside a camera buffer and DORY, so
 *            QVGA must instead size generously and handle overflow gracefully.
 */
size_t jpeg_gray_max_bytes(const jpeg_gray_t *e);
size_t jpeg_gray_max_segment_bytes(const jpeg_gray_t *e, int cores);

/*
 * The same bounds from dimensions alone, for callers sizing buffers before any
 * encoder exists.
 *
 * Use THESE rather than constructing a jpeg_gray_t to ask it. sizeof(jpeg_gray_t) is
 * 2068 bytes and a FreeRTOS task on this chip gets a 2048-byte stack -- one local
 * instance is larger than the entire stack, and the chip dies instantly and silently
 * with no output. These need 64 bytes.
 */
size_t jpeg_gray_max_bytes_for(int width, int height, int quality, int cores);
size_t jpeg_gray_max_segment_bytes_for(int width, int height, int quality, int cores);

/* --- internals, exposed so the cluster kernel can reuse them --- */

/* In-place 8x8 forward DCT on level-shifted samples. Output is the true DCT
 * scaled by 8 (libjpeg "islow" convention); the quantizers divide by qtab[i] << 3. */
void jpeg_fdct_8x8(int16_t data[64]);

#endif /* JPEG_GRAY_H */
