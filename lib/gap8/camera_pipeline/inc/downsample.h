/*
 * downsample.h -- integer box-filter downsampler.
 *
 * One resized buffer can feed both the network AND the preview's JPEG encoder, so
 * the preview shows *exactly* what the network sees. It runs on the FC before the
 * encode.
 *
 * Integer-only on purpose: the GAP8 has no FPU, so anything with a float in it
 * turns into a soft-float call per pixel.
 *
 * Two ways to fit a 4:3 sensor frame into a square network input:
 *
 *   DS_SQUASH  scale the whole frame to the target, distorting the aspect ratio.
 *              Keeps the entire field of view. This is the default: for a drone,
 *              throwing away the left and right thirds of what the camera can see
 *              is a much worse trade than a stretched image.
 *
 *   DS_CROP    take the largest centered region with the target's aspect ratio,
 *              then scale. Keeps the aspect ratio, loses field of view.
 *
 * Both are kept because the two give different network behaviour and we want to
 * be able to A/B them.
 */
#ifndef CF_GAP8_DOWNSAMPLE_H
#define CF_GAP8_DOWNSAMPLE_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    DS_SQUASH = 0, /* full FOV, aspect distorted (default) */
    DS_CROP = 1    /* aspect preserved, FOV cropped */
} ds_mode_e;

/*
 * Box-filter (area-average) resize of an 8-bit grayscale image.
 *
 * Every destination pixel is the mean of the source rectangle that maps to it,
 * so this is a proper decimation filter -- not point sampling, which would alias
 * badly on a noisy sensor and hand the JPEG encoder high-frequency garbage that
 * costs bits and helps the network not at all.
 *
 * Arbitrary ratios are fine (320x240 -> 96x96 is 3.33 x 2.5). Upscaling works too
 * (the source rect degenerates to a single pixel).
 */
void ds_resize_gray(const uint8_t *src, int sw, int sh, int sstride,
                    uint8_t *dst, int dw, int dh, int dstride,
                    ds_mode_e mode);

#endif /* CF_GAP8_DOWNSAMPLE_H */
