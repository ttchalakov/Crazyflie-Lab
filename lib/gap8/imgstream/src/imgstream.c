#include <string.h>

#include "imgstream.h"

void imgstream_init(imgstream_t *s, CPXTarget_t dst)
{
    memset(s, 0, sizeof(*s));
    s->dst = dst;
}

uint32_t imgstream_packets_for(uint32_t size)
{
    const uint32_t mtu = (uint32_t)sizeof(((CPXPacket_t *)0)->data);

    return 1 + (size + mtu - 1) / mtu; /* 1 header packet + the chunked payload */
}

uint32_t imgstream_send(imgstream_t *s, const imgstream_frame_t *f)
{
    const uint32_t mtu = (uint32_t)sizeof(s->packet.data);

    if (f->payload == NULL || f->size == 0)
        return 0;

    cpxInitRoute(CPX_T_GAP8, s->dst, CPX_F_APP, &s->packet.route);

    /* Header, plus the app's opaque extension, in one packet. */
    cf_img_header_t *h = (cf_img_header_t *)s->packet.data;

    h->magic = CF_IMG_MAGIC;
    h->type = f->type;
    h->depth = f->depth ? f->depth : 1;
    h->ext_len = f->ext_len;
    h->width = f->width;
    h->height = f->height;
    h->size = f->size;

    if (f->ext_len && f->ext)
        memcpy(s->packet.data + CF_IMG_HEADER_SIZE, f->ext, f->ext_len);

    s->packet.dataLength = (uint16_t)(CF_IMG_HEADER_SIZE + f->ext_len);
    cpxSendPacketBlocking(&s->packet);

    /* Payload, chunked to the CPX MTU. */
    uint32_t sent = 1;

    for (uint32_t off = 0; off < f->size; ) {
        uint32_t n = f->size - off;

        if (n > mtu)
            n = mtu;
        memcpy(s->packet.data, f->payload + off, n);
        s->packet.dataLength = (uint16_t)n;
        cpxSendPacketBlocking(&s->packet);
        off += n;
        sent++;
    }
    return sent;
}
