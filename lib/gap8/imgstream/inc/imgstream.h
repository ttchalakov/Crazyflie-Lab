/*
 * imgstream.h -- ship an image off the GAP8 over CPX.
 *
 * The other half of lib/shared/include/cf_lab/img_protocol.h.
 *
 * Usage:
 *
 *     imgstream_t s;
 *     imgstream_init(&s, CPX_T_WIFI_HOST);      // or CPX_T_STM32, ...
 *
 *     imgstream_frame_t f = {
 *         .payload = jpegBuf, .size = n,
 *         .width = 320, .height = 240,
 *         .type = CF_IMG_TYPE_JPEG,
 *         .ext = &myTelemetry, .ext_len = sizeof(myTelemetry),   // optional
 *     };
 *     imgstream_send(&s, &f);
 *
 * `ext` is whatever *this* app wants to attach -- capture timings, classifier logits,
 * nothing at all. The library neither knows nor cares; it just length-delimits it so
 * the host can always find the payload. That is what lets apps differ without forking
 * the framing.
 *
 * Blocking: uses cpxSendPacketBlocking, so call it from a task that is allowed to
 * wait on the CPX tx queue. To send a frame without stopping half-way, compare
 * imgstream_packets_for() with com_tx_free_slots() first (examples/camera_stream's exact
 * policy waits for room for the whole frame).
 */
#ifndef CF_LAB_IMGSTREAM_H
#define CF_LAB_IMGSTREAM_H

#include <stdint.h>

#include "cf_lab/img_protocol.h"
#include "cpx.h"

typedef struct {
    CPXPacket_t packet; /* reused across sends; ~1 kB, so not a stack local */
    CPXTarget_t dst;
} imgstream_t;

typedef struct {
    const uint8_t *payload;
    uint32_t size;
    uint16_t width;
    uint16_t height;
    uint8_t type;  /* CF_IMG_TYPE_* */
    uint8_t depth; /* 0 is taken as 1 (grayscale) */

    const void *ext; /* optional app-specific trailer; NULL for none */
    uint8_t ext_len; /* at most 255, so it always fits the header packet */
} imgstream_frame_t;

void imgstream_init(imgstream_t *s, CPXTarget_t dst);

/*
 * Header (+ extension) in one packet, then the payload chunked to the CPX MTU.
 * Returns the number of CPX packets sent, or 0 for a frame with no payload.
 */
uint32_t imgstream_send(imgstream_t *s, const imgstream_frame_t *f);

/*
 * How many CPX packets a frame of `size` payload bytes will cost, header included.
 * Use it to check com_tx_free_slots() before committing to a send -- the alternative
 * is blocking half-way through a frame with the queue full.
 */
uint32_t imgstream_packets_for(uint32_t size);

#endif /* CF_LAB_IMGSTREAM_H */
