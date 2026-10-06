/*
 * jpeg_cl.c -- optimized GAP8 cluster encoder. See jpeg_cl.h for the idea, and for
 * why the L1 working set is borrowed rather than owned.
 *
 * Must emit the exact same bytes as jpeg_gray_encode_range(). Every optimization
 * here is a data-movement or strength-reduction change, never an arithmetic one, so
 * the bitstream hash is unchanged and the test harness can prove it.
 */

#include "pmsis.h"
#include "jpeg_cl.h"

#define L1_OUT_BYTES JPEG_CL_STAGING_BYTES
#define L1_OUT_HEADROOM 32 /* worst a single block can append before we recheck */

/*
 * Tables staged in L1, shared read-only by all cores.
 *
 * The quantization tables are stored in ZIG-ZAG order, not natural order. The block
 * kernel walks coefficients in zig-zag order, so permuting once here makes its inner
 * loop read them sequentially. Leaving them natural costs an indirect load per
 * coefficient. Only px[] stays indirect, which is unavoidable: the DCT emits its
 * output in natural order.
 */
typedef struct {
    uint16_t dc_code[256];
    uint8_t dc_size[256];
    uint16_t ac_code[256];
    uint8_t ac_size[256];
    uint32_t qrecip_z[64];
    uint16_t qhalf_z[64];
    uint8_t qshift_z[64];
    uint8_t zigzag[64];
} l1_tables_t;

/* Mirrors jpeg_gray.c's ZIGZAG. */
static const uint8_t ZIGZAG[64] = {
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63};

#define ZRL 0xF0
#define EOB 0x00

/*
 * Ablation switches. Every optimization here has to earn its place by showing up in
 * a cycle count on a REAL frame (a synthetic checkerboard is dense in high-frequency
 * coefficients, which hides early-EOB's gain), so each can be turned off from the
 * build (-DJPEG_CL_EARLY_EOB=0; examples/jpeg_cluster_bench's exhaustive set runs
 * each off).
 */
#ifndef JPEG_CL_FAST_GATHER
#define JPEG_CL_FAST_GATHER 1 /* skip per-pixel edge clamping on interior blocks */
#endif
#ifndef JPEG_CL_EARLY_EOB
#define JPEG_CL_EARLY_EOB 1 /* stop the AC scan at the last nonzero coefficient */
#endif

/*
 * Where the ~2 kB table block gets staged into L1.
 *
 *   1 (default) core 0 stages it from inside the cluster fork, then a barrier.
 *   0           the FC stages it during jpeg_cl_l1_acquire().
 *
 * The FC version looks better in GVSOC and is much worse on silicon: an FC write
 * into cluster L1 crosses the cluster interconnect, a latency GVSOC does not model.
 * Staging from core 0 keeps the writes L1-local and costs only a team barrier.
 *
 * Kept switchable because GVSOC mispredicts it, and that is worth being able to
 * re-demonstrate.
 */
#ifndef JPEG_CL_STAGE_ON_CLUSTER
#define JPEG_CL_STAGE_ON_CLUSTER 1
#endif

/* ------------------------------------------------------- L1 working set */

uint32_t jpeg_cl_l1_footprint(const jpeg_gray_t *e, int cores)
{
    uint32_t tile = (uint32_t)(8 * e->width + 3) & ~3u; /* 4-byte aligned per core */

    return (uint32_t)sizeof(l1_tables_t) +
           tile * (uint32_t)cores +
           (uint32_t)L1_OUT_BYTES * (uint32_t)cores;
}

uint32_t jpeg_cl_l1_held(const jpeg_cl_l1_t *m)
{
    return (m->tables ? m->tables_bytes : 0) +
           (m->tiles ? m->tiles_bytes : 0) +
           (m->staging ? m->staging_bytes : 0);
}

/* Fill the L1 tables from the encoder's L2 tables. Run by cluster core 0 inside the
 * fork, or by the FC in the acquire under JPEG_CL_STAGE_ON_CLUSTER=0. */
static void stage_tables(l1_tables_t *t, const jpeg_gray_t *e)
{
    memcpy(t->dc_code, e->dc.code, sizeof(t->dc_code));
    memcpy(t->dc_size, e->dc.size, sizeof(t->dc_size));
    memcpy(t->ac_code, e->ac.code, sizeof(t->ac_code));
    memcpy(t->ac_size, e->ac.size, sizeof(t->ac_size));
    memcpy(t->zigzag, ZIGZAG, sizeof(t->zigzag));

    for (int k = 0; k < 64; k++) {
        int i = ZIGZAG[k];

        t->qrecip_z[k] = e->qrecip[i];
        t->qhalf_z[k] = e->qhalf[i];
        t->qshift_z[k] = e->qshift[i];
    }
}

int jpeg_cl_l1_acquire(struct pi_device *cluster, jpeg_cl_l1_t *m,
                       const jpeg_gray_t *e, int cores)
{
    uint32_t tile = (uint32_t)(8 * e->width + 3) & ~3u;

    m->tile_bytes = tile;

    /* A cached allocation is only reusable if it is still the right size -- the
     * frame dimensions can change between calls (QVGA preview vs 96x96 NN input). */
    if (m->tiles && m->tiles_bytes != tile * (uint32_t)cores) {
        pi_cl_l1_free(cluster, m->tiles, (int)m->tiles_bytes);
        m->tiles = NULL;
    }

    if (!m->tables) {
        m->tables_bytes = (uint32_t)sizeof(l1_tables_t);
        m->tables = pi_cl_l1_malloc(cluster, m->tables_bytes);
        m->tables_staged = 0;
        if (!m->tables)
            return -1;
    }
    if (!m->tiles) {
        m->tiles_bytes = tile * (uint32_t)cores;
        m->tiles = pi_cl_l1_malloc(cluster, m->tiles_bytes);
        if (!m->tiles)
            return -1;
    }
    if (!m->staging) {
        m->staging_bytes = (uint32_t)L1_OUT_BYTES * (uint32_t)cores;
        m->staging = pi_cl_l1_malloc(cluster, m->staging_bytes);
        if (!m->staging)
            return -1;
    }

    /* Re-staging is the only real cost of having released. Under
     * JPEG_CL_STAGE_ON_CLUSTER it is deferred into the fork, where the writes are
     * L1-local instead of crossing the cluster interconnect from the FC. */
    if (!m->tables_staged && !JPEG_CL_STAGE_ON_CLUSTER) {
        stage_tables((l1_tables_t *)m->tables, e);
        m->tables_staged = 1;
    }
    return 0;
}

void jpeg_cl_l1_release(struct pi_device *cluster, jpeg_cl_l1_t *m,
                        jpeg_cl_l1_policy_e policy)
{
    if (policy == JPEG_CL_L1_KEEP_ALL)
        return;

    /* The bulk of the footprint: the pixel tiles and the bitstream staging. Both are
     * pure scratch -- nothing in them needs to survive a frame. */
    if (m->tiles) {
        pi_cl_l1_free(cluster, m->tiles, (int)m->tiles_bytes);
        m->tiles = NULL;
    }
    if (m->staging) {
        pi_cl_l1_free(cluster, m->staging, (int)m->staging_bytes);
        m->staging = NULL;
    }

    if (policy == JPEG_CL_L1_RELEASE_ALL && m->tables) {
        pi_cl_l1_free(cluster, m->tables, (int)m->tables_bytes);
        m->tables = NULL;
        m->tables_staged = 0;
    }
}

/* ------------------------------------------------------- L1 bit writer */

typedef struct {
    uint8_t *l1;     /* this core's staging buffer */
    uint32_t l1_len;
    uint8_t *l2;     /* this core's L2 segment */
    uint32_t l2_len; /* bytes already committed to L2 */
    uint32_t l2_cap;
    uint32_t acc;
    int nbits;
    int overflow;
} clbw_t;

static inline void clbw_init(clbw_t *b, uint8_t *l1, uint8_t *l2, uint32_t cap)
{
    b->l1 = l1;
    b->l1_len = 0;
    b->l2 = l2;
    b->l2_len = 0;
    b->l2_cap = cap;
    b->acc = 0;
    b->nbits = 0;
    b->overflow = 0;
}

/*
 * Push the staged bytes out to L2.
 *
 * This is a memcpy, NOT an MCHAN DMA: pi_cl_dma_cmd*() is SINGLE-CORE ONLY, whatever
 * its per-core cmd argument implies, and this runs on every core.
 *
 * GAP SDK hal/dma/dma.h enqueues a DMA command as several consecutive word-writes to
 * ONE hardware register, DMAMCHAN->CMD, which MCHAN reassembles positionally. The
 * only guard is disable_irq(), which masks interrupts on the calling core -- and
 * cluster cores run in genuine parallel, so it serializes nothing. Cores calling it
 * at once interleave their word-writes and MCHAN executes a command spliced from two
 * transfers; the completion bit never sets, the waiter spins forever, the fork never
 * returns, and the whole fabric wedges.
 *
 * The cluster can address L2 directly anyway, so a memcpy costs little and cannot
 * hang. Do not "optimize" this back into a DMA.
 */
static void clbw_flush(clbw_t *b)
{
    if (b->l1_len == 0)
        return;
    if (b->l2_len + b->l1_len > b->l2_cap) {
        b->overflow = 1;
        b->l1_len = 0;
        return;
    }

    memcpy(b->l2 + b->l2_len, b->l1, b->l1_len);
    b->l2_len += b->l1_len;
    b->l1_len = 0;
}

static inline void clbw_byte(clbw_t *b, uint8_t v)
{
    b->l1[b->l1_len++] = v;
    if (b->l1_len >= L1_OUT_BYTES - L1_OUT_HEADROOM)
        clbw_flush(b);
}

static inline void clbw_bits(clbw_t *b, uint32_t v, int n)
{
    if (n == 0)
        return;
    b->acc = (b->acc << n) | (v & ((1u << n) - 1));
    b->nbits += n;
    while (b->nbits >= 8) {
        b->nbits -= 8;
        uint8_t byte = (uint8_t)(b->acc >> b->nbits);
        clbw_byte(b, byte);
        if (byte == 0xFF)
            clbw_byte(b, 0x00);
    }
}

static void clbw_align(clbw_t *b)
{
    if (b->nbits > 0) {
        int pad = 8 - b->nbits;
        uint8_t byte = (uint8_t)(((b->acc << pad) | ((1u << pad) - 1)) & 0xFF);

        clbw_byte(b, byte);
        if (byte == 0xFF)
            clbw_byte(b, 0x00);
    }
    b->acc = 0;
    b->nbits = 0;
}

/* ---------------------------------------------------------- block kernel */

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
    return (uint32_t)(v < 0 ? (v - 1) : v) & ((1u << size) - 1);
}

/*
 * Gather one 8x8 block out of the L1 tile and level-shift it.
 *
 * `full` means the block lies entirely inside the image, which is every block when
 * the width is a multiple of 8. Then there is no edge to clamp against and the inner
 * loop loses its per-pixel compare-and-branch.
 *
 * The clamped path is kept and stays correct: the encoder is general, and the
 * Himax's raw window (324 wide) is not a multiple of 8. Multiple-of-8 is a fast
 * path, not a requirement -- though it is worth choosing multiple-of-8 dimensions
 * for the network input, since we generate that buffer ourselves.
 */
static inline void gather_l1(const uint8_t *tile, int tile_w, int bx, int width,
                             int full, int16_t px[64])
{
    if (JPEG_CL_FAST_GATHER && full) {
        for (int i = 0; i < 8; i++) {
            const uint8_t *row = tile + i * tile_w + bx;
            int16_t *o = &px[i * 8];

            o[0] = (int16_t)row[0] - 128;
            o[1] = (int16_t)row[1] - 128;
            o[2] = (int16_t)row[2] - 128;
            o[3] = (int16_t)row[3] - 128;
            o[4] = (int16_t)row[4] - 128;
            o[5] = (int16_t)row[5] - 128;
            o[6] = (int16_t)row[6] - 128;
            o[7] = (int16_t)row[7] - 128;
        }
        return;
    }

    for (int i = 0; i < 8; i++) {
        const uint8_t *row = tile + i * tile_w;

        for (int j = 0; j < 8; j++) {
            int x = bx + j;

            if (x >= width)
                x = width - 1;
            px[i * 8 + j] = (int16_t)row[x] - 128;
        }
    }
}

/* Everything this touches -- pixels, tables, output -- is in L1. */
static void encode_block_l1(clbw_t *b, const l1_tables_t *t, const uint8_t *tile,
                            int tile_w, int bx, int width, int full, int *pred_dc)
{
    int16_t px[64];
    int qz[64];
    int last = 0;

    gather_l1(tile, tile_w, bx, width, full, px);
    jpeg_fdct_8x8(px);

    /*
     * Quantize straight into zig-zag order, remembering the highest nonzero index.
     *
     * Two wins, both free. The AC loop becomes a linear scan instead of 63 indirect
     * loads through the zig-zag table; and it can STOP at the last nonzero
     * coefficient rather than walking a tail of zeros to discover nothing is there.
     * On real imagery a block's coefficients go zero early, so that tail is most of
     * the scan.
     */
    for (int k = 0; k < 64; k++) {
        int coef = px[t->zigzag[k]];
        int neg = coef < 0;
        uint32_t a = (uint32_t)(neg ? -coef : coef) + t->qhalf_z[k];
        int v = (int)((a * t->qrecip_z[k]) >> t->qshift_z[k]);

        v = neg ? -v : v;
        qz[k] = v;
        if (v)
            last = k;
    }

    int diff = qz[0] - *pred_dc;
    *pred_dc = qz[0];

    int size = coef_size(diff);
    clbw_bits(b, t->dc_code[size], t->dc_size[size]);
    clbw_bits(b, coef_bits(diff, size), size);

    int stop = JPEG_CL_EARLY_EOB ? last : 63;
    int run = 0;

    for (int k = 1; k <= stop; k++) {
        int coef = qz[k];

        if (coef == 0) {
            run++;
            continue;
        }
        while (run > 15) {
            clbw_bits(b, t->ac_code[ZRL], t->ac_size[ZRL]);
            run -= 16;
        }
        size = coef_size(coef);

        int rs = (run << 4) | size;
        clbw_bits(b, t->ac_code[rs], t->ac_size[rs]);
        clbw_bits(b, coef_bits(coef, size), size);
        run = 0;
    }
    /* Anything past `last` is zero, so a nonempty tail means EOB. Written so both
     * ablation settings emit the identical bitstream: with EARLY_EOB off the scan
     * walks the tail and `run` ends > 0; with it on the scan stops and `last < 63`
     * catches the same case. */
    if (run > 0 || last < 63)
        clbw_bits(b, t->ac_code[EOB], t->ac_size[EOB]);
}

/* ------------------------------------------------------------- the worker */

/*
 * Pull the 8 image rows of MCU row `my` into this core's L1 tile, replicating rows
 * past the bottom edge (matching the reference encoder's clamping).
 *
 * memcpy, not pi_cl_dma_cmd_2d -- see clbw_flush(). The win here is not the
 * transfer: it is that the blocks of this MCU row are then read 8 times each out of
 * L1 instead of L2.
 */
static void fetch_tile(const jpeg_gray_t *e, const uint8_t *img, uint8_t *tile, int my)
{
    int y0 = my * 8;
    int rows = e->height - y0;

    if (rows > 8)
        rows = 8;

    for (int i = 0; i < rows; i++)
        memcpy(tile + i * e->width, img + (size_t)(y0 + i) * e->stride, e->width);

    for (int i = rows; i < 8; i++)
        memcpy(tile + i * e->width, tile + (rows - 1) * e->width, e->width);
}

static void cl_worker(void *arg)
{
    jpeg_cl_job_t *job = (jpeg_cl_job_t *)arg;
    const jpeg_gray_t *e = job->enc;
    const jpeg_cl_l1_t *m = job->l1;
    int c = pi_core_id();

    /* Stage the tables from inside the cluster: core 0 writes, everyone waits. The
     * writes land in L1 at L1 speed, rather than being pushed across the cluster
     * interconnect by the FC. Every core evaluates the same condition, so they all
     * reach the barrier. */
    if (JPEG_CL_STAGE_ON_CLUSTER && !m->tables_staged) {
        if (c == 0)
            stage_tables((l1_tables_t *)m->tables, job->enc);
        pi_cl_team_barrier();
        if (c == 0)
            ((jpeg_cl_l1_t *)m)->tables_staged = 1;
    }

    uint32_t start = job->starts[c];
    uint32_t end = job->starts[c + 1];

    if (start >= end) {
        job->seg_len[c] = 0;
        return;
    }

    const l1_tables_t *t = (const l1_tables_t *)m->tables;
    uint8_t *tile = (uint8_t *)m->tiles + (size_t)c * m->tile_bytes;
    uint8_t *stage = (uint8_t *)m->staging + (size_t)c * L1_OUT_BYTES;

    clbw_t bw;
    int pred_dc = 0;
    int cur_row = -1;

    /* Width a multiple of 8 => every block is interior, so no block needs edge
     * clamping. Decided once per frame, not once per pixel. */
    int width_aligned = (e->width & 7) == 0;

    clbw_init(&bw, stage, job->seg[c], job->seg_cap);

    for (uint32_t m_i = start; m_i < end; m_i++) {
        int my = (int)(m_i / e->mcus_x);

        if (my != cur_row) {
            fetch_tile(e, job->img, tile, my);
            cur_row = my;
        }

        /* Restart boundary: byte-align, emit this segment's RSTn, reset the DC
         * predictor. At m_i == start this is the segment's leading marker, which is
         * what lets the merger just memcpy the segments together. */
        if (e->restart_interval > 0 && m_i > 0 &&
            (m_i % e->restart_interval) == 0) {
            clbw_align(&bw);
            clbw_byte(&bw, 0xFF);
            clbw_byte(&bw,
                      (uint8_t)(0xD0 + (((m_i / e->restart_interval) - 1) & 7)));
            pred_dc = 0;
        }

        int bx = (int)(m_i % e->mcus_x) * 8;

        encode_block_l1(&bw, t, tile, e->width, bx, e->width,
                        width_aligned || (bx + 8 <= e->width), &pred_dc);
    }

    clbw_align(&bw);
    clbw_flush(&bw);

    job->seg_len[c] = bw.overflow ? 0 : bw.l2_len;
}

void jpeg_cl_entry(void *arg)
{
    jpeg_cl_job_t *job = (jpeg_cl_job_t *)arg;

    pi_cl_team_fork(job->cores, cl_worker, arg);
}
