#include "downsample.h"

#include <string.h>

/*
 * Pick the source window. SQUASH uses the whole frame; CROP takes the largest
 * centered rectangle whose aspect matches the destination.
 */
static void ds_window(int sw, int sh, int dw, int dh, ds_mode_e mode,
                      int *ox, int *oy, int *ww, int *hh)
{
    if (mode == DS_CROP && dw > 0 && dh > 0) {
        /* Compare aspects as a cross-multiplication to stay in integers. */
        if (sw * dh > sh * dw) {
            /* source is wider than the target: trim left/right */
            *hh = sh;
            *ww = (sh * dw) / dh;
        } else {
            /* source is taller: trim top/bottom */
            *ww = sw;
            *hh = (sw * dh) / dw;
        }
        *ox = (sw - *ww) / 2;
        *oy = (sh - *hh) / 2;
    } else {
        *ox = 0;
        *oy = 0;
        *ww = sw;
        *hh = sh;
    }
}

/*
 * ceil(2^32 / n), so that (num * m) >> 32 == num / n: exact while num * n < 2^32,
 * which a band's rounded sum (under 256 n) meets for any band under 4096
 * pixels. Zero means divide instead. On the GAP8 FC the multiply is cheaper than
 * the divide.
 */
static uint32_t ds_reciprocal(uint32_t n)
{
    return n >= 2u && n < 4096u ? 0xFFFFFFFFu / n + 1u : 0u;
}

/* Round to nearest rather than truncating: truncation would darken the whole
 * image by ~0.5 LSB and bias anything the network learns. */
static uint8_t ds_mean(uint32_t sum, uint32_t n, uint32_t m)
{
    const uint32_t num = sum + n / 2u;
    return (uint8_t)(m != 0u ? (uint32_t)(((uint64_t)num * m) >> 32) : num / n);
}

/*
 * One destination row when downscaling: walk the band's columns four at a
 * time, each a single 32-bit load per source row (every L2 load is slow on the
 * GAP8 FC, so a byte at a time costs far more), and close a destination pixel
 * whenever the walk reaches its right edge. The column sums ride in 16-bit lanes,
 * which hold bands of up to 257 rows.
 */
static void ds_row_packed(const uint8_t *band, int sstride, uint32_t ny,
                          int ww, int dw, uint8_t *out)
{
    const int xstep = ww / dw, xrem = ww % dw;
    /* A band is xstep columns wide or one more: two divisors per row. */
    const uint32_t m_narrow = ds_reciprocal(ny * (uint32_t)xstep);
    const uint32_t m_wide = ds_reciprocal(ny * (uint32_t)(xstep + 1));
    int edge = xstep, rem = xrem, start = 0, x = 0, c = 0;
    uint32_t sum = 0;

#define DS_TAKE(column_sum)                                                  \
    do {                                                                     \
        sum += (column_sum);                                                 \
        if (++c == edge) {                                                   \
            const int nx = edge - start;                                     \
            out[x++] = ds_mean(sum, ny * (uint32_t)nx,                       \
                               nx == xstep ? m_narrow : m_wide);             \
            sum = 0;                                                         \
            start = edge;                                                    \
            edge += xstep;                                                   \
            rem += xrem;                                                     \
            if (rem >= dw) {                                                 \
                edge++;                                                      \
                rem -= dw;                                                   \
            }                                                                \
        }                                                                    \
    } while (0)

    for (; c + 4 <= ww;) {
        const uint8_t *p = band + c;
        uint32_t even = 0, odd = 0; /* columns 0 and 2, 1 and 3 */
        for (uint32_t r = 0; r < ny; r++, p += sstride) {
            uint32_t w;
            memcpy(&w, p, sizeof(w)); /* little-endian: column 0 is the low byte */
            even += w & 0x00FF00FFu;
            odd += (w >> 8) & 0x00FF00FFu;
        }
        DS_TAKE(even & 0xFFFFu);
        DS_TAKE(odd & 0xFFFFu);
        DS_TAKE(even >> 16);
        DS_TAKE(odd >> 16);
    }
    for (; c < ww;) {
        const uint8_t *p = band + c;
        uint32_t column = 0;
        for (uint32_t r = 0; r < ny; r++, p += sstride)
            column += *p;
        DS_TAKE(column);
    }
#undef DS_TAKE
}

void ds_resize_gray(const uint8_t *src, int sw, int sh, int sstride,
                    uint8_t *dst, int dw, int dh, int dstride,
                    ds_mode_e mode)
{
    int ox, oy, ww, hh;

    if (dw <= 0 || dh <= 0 || sw <= 0 || sh <= 0)
        return;

    ds_window(sw, sh, dw, dh, mode, &ox, &oy, &ww, &hh);

    /* Band edges are floor(i * size / count), stepped with a remainder rather
     * than divided out per pixel: exact, so rounding cannot accumulate down or
     * across the image. */
    const int xstep = ww / dw, xrem = ww % dw;
    const int ystep = hh / dh, yrem = hh % dh;
    const int packed = xstep >= 1 && ystep >= 1 && ystep < 257;
    int yq = 0, yr = 0;

    for (int y = 0; y < dh; y++) {
        const int y0 = oy + yq;
        yq += ystep;
        yr += yrem;
        if (yr >= dh) {
            yq++;
            yr -= dh;
        }
        int y1 = oy + yq;
        if (y1 <= y0)
            y1 = y0 + 1; /* upscaling: never let the band go empty */

        const uint32_t ny = (uint32_t)(y1 - y0);
        const uint8_t *band = src + (size_t)y0 * sstride;
        uint8_t *out = dst + (size_t)y * dstride;

        if (packed) {
            ds_row_packed(band + ox, sstride, ny, ww, dw, out);
            continue;
        }

        /* Upscaling, or bands too tall for the packed sums. */
        int xq = 0, xr = 0;
        for (int x = 0; x < dw; x++) {
            const int x0 = ox + xq;
            xq += xstep;
            xr += xrem;
            if (xr >= dw) {
                xq++;
                xr -= dw;
            }
            int x1 = ox + xq;
            if (x1 <= x0)
                x1 = x0 + 1;

            const int nx = x1 - x0;
            const uint8_t *row = band + x0;
            uint32_t sum = 0;
            for (uint32_t yy = 0; yy < ny; yy++, row += sstride)
                for (int xx = 0; xx < nx; xx++)
                    sum += row[xx];
            out[x] = ds_mean(sum, ny * (uint32_t)nx, 0u);
        }
    }
}
