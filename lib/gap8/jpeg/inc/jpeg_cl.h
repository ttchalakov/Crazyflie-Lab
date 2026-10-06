/*
 * jpeg_cl.h -- optimized GAP8 cluster encoder.
 *
 * The same decomposition as jpeg_gray_encode_range() (restart-aligned MCU ranges,
 * one per core, merged with memcpy) and the same output -- byte for byte. That
 * function reads the pixels and tables and writes the bitstream in L2, which from
 * 8 cores at once means every access crosses the shared interconnect. This one
 * keeps the working data in L1:
 *
 *   - Huffman + quant tables staged in L1, shared read-only by all cores
 *   - the MCU row a core is working on is copied into that core's L1 tile
 *   - the bitstream is built in an L1 staging buffer and flushed to L2 in chunks
 *   - quantization multiplies by a reciprocal instead of dividing
 *   - the AC scan stops at the last nonzero coefficient
 *
 * The CUDA analogy is exact: it is the kernel that stages its data in shared
 * memory first instead of reading everything from global memory.
 *
 * NO CLUSTER DMA. See the note on clbw_flush() in jpeg_cl.c: pi_cl_dma_cmd*() is
 * not safe to call from multiple cluster cores and wedges the whole fabric.
 *
 * ---------------------------------------------------------------------------
 * L1 OWNERSHIP AND THE DORY HANDOFF
 * ---------------------------------------------------------------------------
 *
 * The encoder does not own L1. It borrows it.
 *
 * The point of making the encoder fast is to get off the cluster sooner so a DORY
 * network can have it. But DORY does not just want cluster *time* -- it wants
 * cluster *L1*, and its tiler is generally configured assuming it has nearly all of
 * it. An encoder that permanently parks tens of kB of L1 in static arrays can
 * silently change DORY's tiling, or stop a network fitting at all.
 *
 * So the L1 working set is allocated, not static, and the caller chooses what
 * happens to it between frames:
 *
 *   KEEP_ALL      hold everything. Fastest per frame; withholds the most L1.
 *   KEEP_TABLES   release the pixel tiles and staging buffers, keep the tables.
 *                 The tables are the only part that costs anything to rebuild.
 *   RELEASE_ALL   give all of it back. DORY sees a completely free L1 between
 *                 frames; we pay a malloc and a table re-stage every frame.
 *
 * Which is right depends on how tight DORY's L1 budget is, so jpeg_cluster_bench
 * MEASURES the cost of each rather than assuming. jpeg_cl_l1_held() reports how
 * many bytes are currently being withheld from DORY.
 */
#ifndef JPEG_CL_H
#define JPEG_CL_H

#include "pmsis.h"
#include "jpeg_gray.h"

#define JPEG_CL_MAX_CORES 8

/* Per-core bitstream staging in L1. Flushed to L2 whenever it fills, so a
 * pathological frame cannot overrun it -- it just flushes more often. */
#define JPEG_CL_STAGING_BYTES 512

typedef enum {
    JPEG_CL_L1_KEEP_ALL = 0,
    JPEG_CL_L1_KEEP_TABLES = 1,
    JPEG_CL_L1_RELEASE_ALL = 2
} jpeg_cl_l1_policy_e;

/* The encoder's L1 working set. */
typedef struct {
    void *tables;  /* Huffman + quant, staged once, shared read-only by all cores */
    void *tiles;   /* cores x tile_bytes: the MCU row each core is working on */
    void *staging; /* cores x JPEG_CL_STAGING_BYTES */

    uint32_t tables_bytes;
    uint32_t tiles_bytes;
    uint32_t staging_bytes;
    uint32_t tile_bytes; /* per core */

    int tables_staged; /* contents still valid? survives a KEEP_TABLES release */
} jpeg_cl_l1_t;

typedef struct {
    const jpeg_gray_t *enc;
    const uint8_t *img;                     /* L2 */
    uint8_t *seg[JPEG_CL_MAX_CORES];        /* L2, one output segment per core */
    uint32_t seg_cap;
    uint32_t starts[JPEG_CL_MAX_CORES + 1]; /* restart-aligned MCU ranges */
    uint32_t seg_len[JPEG_CL_MAX_CORES];    /* filled in by the cores */
    int cores;
    jpeg_cl_l1_t *l1; /* must be acquired before the task is sent */
} jpeg_cl_job_t;

/*
 * Take the L1 working set for this frame. Idempotent: whatever is still held from
 * the previous frame is reused, so under KEEP_ALL this is nearly free.
 *
 * Returns 0, or -1 if L1 is too full or too fragmented -- a real possibility once
 * DORY is allocating too, so do not ignore it.
 *
 * Call from the FC with the cluster open.
 */
int jpeg_cl_l1_acquire(struct pi_device *cluster, jpeg_cl_l1_t *m,
                       const jpeg_gray_t *e, int cores);

/* Give L1 back, per policy. Safe to call when nothing is held. */
void jpeg_cl_l1_release(struct pi_device *cluster, jpeg_cl_l1_t *m,
                        jpeg_cl_l1_policy_e policy);

/* Bytes of cluster L1 currently withheld from everyone else, DORY included. */
uint32_t jpeg_cl_l1_held(const jpeg_cl_l1_t *m);

/* What an acquire will ask for, without allocating. For budgeting. */
uint32_t jpeg_cl_l1_footprint(const jpeg_gray_t *e, int cores);

/* Cluster entry point: pass this to pi_cluster_task() with a jpeg_cl_job_t. */
void jpeg_cl_entry(void *arg);

#endif /* JPEG_CL_H */
