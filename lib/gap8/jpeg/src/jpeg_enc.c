/*
 * jpeg_enc.c -- runtime FC-vs-cluster path selection over one shared jpeg_gray_t.
 *
 * No encoding logic lives here. Both paths take the tables and the restart interval
 * from the same jpeg_gray_t: the FC path encodes the frame with jpeg_gray_encode(),
 * and the cluster path hands restart-aligned MCU ranges to jpeg_cl.c's kernel, which
 * must emit the same bytes as jpeg_gray_encode_range(). That is what keeps the two
 * byte-identical.
 */

#include "jpeg_enc.h"

/* Each cluster core's stack, master and slaves. */
#define JPEG_ENC_CL_STACK 2048

/*
 * Both bounds are computed from the quantization table (64 bytes) and the standard
 * Annex-K Huffman spec, so no jpeg_gray_t is constructed and nothing large lands on a
 * stack. Reentrant, and safe to call from any task.
 */
size_t jpeg_enc_seg_cap(int width, int height, int quality, int cores)
{
    return jpeg_gray_max_segment_bytes_for(width, height, quality, cores);
}

size_t jpeg_enc_out_cap(int width, int height, int quality, int cores)
{
    return jpeg_gray_max_bytes_for(width, height, quality, cores);
}

int jpeg_enc_init(jpeg_enc_t *j, const jpeg_cfg_t *cfg,
                  int width, int height, int stride,
                  struct pi_device *cluster,
                  uint8_t *segs, size_t seg_cap)
{
    int cores = cfg->cores;
    int slices = cfg->slices > 1 ? cfg->slices : 1;
    int segments = cores * slices;
    int restart;

    if (cores < 1 || cores > JPEG_CL_MAX_CORES || slices > JPEG_ENC_MAX_SLICES)
        return -1;
    if (cfg->path == JPEG_PATH_CLUSTER && (!cluster || !segs))
        return -1;

    memset(j, 0, sizeof(*j));
    j->cfg = *cfg;
    j->cluster = cluster;
    j->segs = segs;
    j->seg_cap = (uint32_t)seg_cap;

    /*
     * The restart interval is part of the bitstream, so BOTH paths must use the same
     * one or they would emit different (still valid) files and the byte-identical
     * invariant would be meaningless. Auto derives it from the segment count (cores x
     * slices) even on the FC path, precisely so that switching path mid-stream changes
     * nothing.
     */
    restart = cfg->restart_interval > 0
                  ? cfg->restart_interval
                  : jpeg_gray_auto_restart(width, height, segments);

    jpeg_gray_init(&j->enc, width, height, stride, cfg->quality, restart);

    j->slices = slices;
    j->segments = segments;
    jpeg_gray_partition(&j->enc, segments, j->starts);

    j->job.enc = &j->enc;
    j->job.cores = cores;
    j->job.seg_cap = j->seg_cap;
    j->job.l1 = &j->l1;

    pi_cluster_task(&j->task, jpeg_cl_entry, &j->job);
    j->task.stack_size = JPEG_ENC_CL_STACK;
    j->task.slave_stack_size = JPEG_ENC_CL_STACK;

    return 0;
}

void jpeg_enc_set_path(jpeg_enc_t *j, jpeg_path_e path)
{
    j->cfg.path = path;
}

void jpeg_enc_deinit(jpeg_enc_t *j)
{
    if (j->cluster)
        jpeg_cl_l1_release(j->cluster, &j->l1, JPEG_CL_L1_RELEASE_ALL);
}

/* Header + the byte-aligned segments, in order + EOI.
 *
 * Each segment already opens with its own RSTn marker, so this is a plain memcpy --
 * no bit stitching, no marker bookkeeping. That is the whole payoff of the
 * restart-interval decomposition. */
static size_t merge(jpeg_enc_t *j, uint8_t *out, size_t cap)
{
    size_t n = jpeg_gray_header(&j->enc, out, cap);

    if (n == 0)
        return 0;

    for (int s = 0; s < j->segments; s++) {
        uint32_t len = j->seg_len[s];

        if (j->starts[s + 1] > j->starts[s] && len == 0)
            return 0; /* a core overflowed its segment */
        if (len > j->peak_seg_bytes)
            j->peak_seg_bytes = len;
        if (n + len + 2 > cap)
            return 0;

        memcpy(out + n, j->segs + (size_t)s * j->seg_cap, len);
        n += len;
    }

    size_t foot = jpeg_gray_footer(out + n, cap - n);

    return foot ? n + foot : 0;
}

void jpeg_enc_begin(jpeg_enc_t *j, const uint8_t *img)
{
    j->img = img;
    j->next_slice = 0;
    j->l1_failed = 0;
}

int jpeg_enc_step(jpeg_enc_t *j)
{
    if (j->cfg.path == JPEG_PATH_FC || j->next_slice >= j->slices)
        return 0;
    /* Idempotent while held: only the first slice, or one after a yield, pays. */
    if (jpeg_cl_l1_acquire(j->cluster, &j->l1, &j->enc, j->job.cores) != 0) {
        /* L1 is full or fragmented -- entirely possible once DORY is also
         * allocating. Report it; do not scribble somewhere. */
        j->l1_failed = 1;
        return -1;
    }

    const int base = j->next_slice * j->job.cores;

    for (int c = 0; c <= j->job.cores; c++)
        j->job.starts[c] = j->starts[base + c];
    for (int c = 0; c < j->job.cores; c++)
        j->job.seg[c] = j->segs + (size_t)(base + c) * j->seg_cap;
    j->job.img = j->img;
    /* The SDK zeroes stack_size when it frees a finished task's stacks. */
    j->task.stack_size = JPEG_ENC_CL_STACK;
    j->task.slave_stack_size = JPEG_ENC_CL_STACK;
    if (pi_cluster_send_task_to_cl(j->cluster, &j->task) != 0) {
        j->l1_failed = 1; /* no L1 for the stacks */
        return -1;
    }
    for (int c = 0; c < j->job.cores; c++)
        j->seg_len[base + c] = j->job.seg_len[c];

    j->next_slice++;
    return j->slices - j->next_slice;
}

void jpeg_enc_yield(jpeg_enc_t *j)
{
    if (j->cfg.path == JPEG_PATH_CLUSTER)
        jpeg_cl_l1_release(j->cluster, &j->l1, JPEG_CL_L1_RELEASE_ALL);
}

size_t jpeg_enc_end(jpeg_enc_t *j, uint8_t *out, size_t cap)
{
    size_t n;

    if (j->cfg.path == JPEG_PATH_FC) {
        /* No cluster, no L1. Free to overlap with a DORY inference. */
        n = jpeg_gray_encode(&j->enc, j->img, out, cap);
    } else if (j->l1_failed) {
        j->overflows++;
        return 0;
    } else {
        n = j->next_slice == j->slices ? merge(j, out, cap) : 0;
        jpeg_cl_l1_release(j->cluster, &j->l1, j->cfg.l1_policy);
    }

    j->frames++;
    if (n == 0) {
        j->overflows++;
    } else if (n > j->peak_frame_bytes) {
        j->peak_frame_bytes = (uint32_t)n;
    }
    return n;
}

size_t jpeg_enc_frame(jpeg_enc_t *j, const uint8_t *img, uint8_t *out, size_t cap)
{
    jpeg_enc_begin(j, img);
    while (jpeg_enc_step(j) > 0) {
    }
    return jpeg_enc_end(j, out, cap);
}
