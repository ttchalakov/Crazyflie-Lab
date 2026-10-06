/*
 * img_protocol.h -- the wire format for streaming an image off the GAP8.
 *
 * Shared by every GAP8 app that ships frames and by the host that decodes them, so
 * that there is exactly ONE definition of the header.
 *
 * ---------------------------------------------------------------------------
 * THE WIRE
 * ---------------------------------------------------------------------------
 *
 *   [ cf_img_header_t | ext_len bytes | size bytes ]
 *     12 bytes          app-specific    the image
 *
 * The header is fixed and self-describing: any host can decode the image from any
 * cf_lab app without knowing what that app is. The extension is opaque to the
 * library -- it is where an app puts whatever *it* cares about (camera_stream puts
 * capture/encode timings and drop counters; an app with nothing to add sends
 * none). ext_len makes the payload findable regardless, so a
 * generic viewer never has to guess.
 *
 * That split is the point: apps get to differ without forking the framing.
 */
#ifndef CF_LAB_IMG_PROTOCOL_H
#define CF_LAB_IMG_PROTOCOL_H

#include <stdint.h>

#define CF_IMG_MAGIC 0xBC

/* Payload encoding. */
#define CF_IMG_TYPE_RAW_GRAY 0
#define CF_IMG_TYPE_JPEG 1

/*
 * Field order groups the 8-bit fields, then 16, then 32. The struct is packed, so
 * this is not about alignment -- it is so that the layout reads the same in C and in
 * the host's struct format string ("<BBBBHHI") without anyone having to count.
 */
typedef struct {
    uint8_t magic;   /* CF_IMG_MAGIC; resynchronisation point for the host */
    uint8_t type;    /* CF_IMG_TYPE_* */
    uint8_t depth;   /* bytes per pixel; 1 = grayscale (the Himax is monochrome) */
    uint8_t ext_len; /* app-specific bytes between this header and the payload */
    uint16_t width;
    uint16_t height;
    uint32_t size; /* payload bytes following the extension */
} __attribute__((packed)) cf_img_header_t;

#define CF_IMG_HEADER_SIZE 12 /* keep the host's struct in step */
_Static_assert(sizeof(cf_img_header_t) == CF_IMG_HEADER_SIZE,
               "cf_img_header_t must be CF_IMG_HEADER_SIZE bytes");

#endif /* CF_LAB_IMG_PROTOCOL_H */
