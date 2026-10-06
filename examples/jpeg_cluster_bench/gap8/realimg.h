#ifndef JPEG_REALIMG_H
#define JPEG_REALIMG_H

#include <stdint.h>

/* A real Himax frame; see realimg.c for why the bench profiles against this. */
#define REAL_IMG_W 320
#define REAL_IMG_H 240
extern const uint8_t REAL_IMG_320x240[320 * 240];

#endif
