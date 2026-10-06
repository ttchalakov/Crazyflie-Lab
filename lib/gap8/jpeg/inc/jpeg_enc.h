/*
 * jpeg_enc.h -- the one public entry point. This is what a caller (e.g.
 * lib/gap8/jpeg_pipeline) should use; everything else in lib/gap8/jpeg/ is an
 * implementation detail.
 *
 * Encode a grayscale frame to a JPEG, on either the fabric controller or the
 * 8-core cluster, chosen at RUNTIME:
 *
 *   JPEG_PATH_FC        single core, no cluster, no L1. Slower per frame, but it
 *                       costs zero cluster cycles and zero cluster L1 -- so it can
 *                       run *concurrently with a DORY inference*.
 *
 *   JPEG_PATH_CLUSTER   8 cores. Much faster, but it takes the cluster, which means
 *                       it serializes against DORY.
 *
 * Which is right is genuinely application-dependent -- it is a question about what
 * else is competing for the cluster, not about the encoder -- so it is a runtime
 * knob, not a build-time one. Both paths are held to the same invariant:
 *
 *   THE TWO PATHS EMIT BYTE-IDENTICAL JPEGS.
 *
 * jpeg_cluster_bench asserts this on every frame. Switching path can change your
 * timing and your power; it can never change your output.
 */
#ifndef JPEG_ENC_H
#define JPEG_ENC_H

#include "pmsis.h"
#include "jpeg_gray.h"
#include "jpeg_cl.h"

typedef enum {
    JPEG_PATH_FC = 0,     /* single core; leaves the cluster free for DORY */
    JPEG_PATH_CLUSTER = 1 /* 8 cores; much faster; takes the cluster */
} jpeg_path_e;

typedef struct {
    jpeg_path_e path;
    int cores;                      /* cluster path; 1..JPEG_CL_MAX_CORES */
    int quality;                    /* 1..100; 50 = the unscaled Annex-K table */
    int restart_interval;           /* 0 = auto: one segment per core (recommended:
                                     * perfectly balanced AND fewer markers, so it is
                                     * both faster and smaller) */
    jpeg_cl_l1_policy_e l1_policy;  /* cluster path. RELEASE_ALL unless you have a
                                     * measured reason: it hands DORY all 64 kB of L1
                                     * back between frames at a small cost. */
    int slices;                     /* cluster path: 0 or 1 encodes a frame in one
                                     * cluster job; n encodes it in n jobs, each a
                                     * strip of the image, so something else can use
                                     * the cluster between them (jpeg_enc_step). The
                                     * frame then has cores x n restart segments,
                                     * on both paths, so they stay byte-identical. */
} jpeg_cfg_t;

#define JPEG_ENC_MAX_SLICES 16
#define JPEG_ENC_MAX_SEGMENTS (JPEG_CL_MAX_CORES * JPEG_ENC_MAX_SLICES)

typedef struct {
    jpeg_gray_t enc;
    jpeg_cfg_t cfg;

    struct pi_device *cluster; /* may be NULL if only the FC path is ever used */
    jpeg_cl_l1_t l1;
    jpeg_cl_job_t job;
    struct pi_cluster_task task;

    uint8_t *segs;    /* L2, cores * slices * seg_cap, caller-owned */
    uint32_t seg_cap;

    /* Every restart segment of the frame, slice after slice. */
    int slices;
    int segments;
    uint32_t starts[JPEG_ENC_MAX_SEGMENTS + 1];
    uint32_t seg_len[JPEG_ENC_MAX_SEGMENTS];
    /* The frame being encoded by begin/step/end. */
    const uint8_t *img;
    int next_slice;
    int l1_failed;

    /* Running evidence that the buffer sizing is right. Overflow means a frame was
     * dropped -- never silently. */
    uint32_t frames;
    uint32_t overflows;
    uint32_t peak_frame_bytes;
    uint32_t peak_seg_bytes;
} jpeg_enc_t;

/* The restart segments a frame is cut into: cores x slices on either path. */
static inline int jpeg_enc_segments(const jpeg_cfg_t *cfg)
{
    return cfg->cores * (cfg->slices > 1 ? cfg->slices : 1);
}

/*
 * Worst-case buffer sizes, derived from the quantization and Huffman tables -- not
 * from a sample of images. Size your L2 with these and overflow becomes structurally
 * impossible.
 *
 * Affordable at 96x96 (38 kB total) and QQVGA (80 kB). NOT affordable at QVGA
 * (315 kB): there, size to taste and watch jpeg_enc_t::overflows, which is then
 * load-bearing. See the jpeg_cluster_bench README.
 */
size_t jpeg_enc_out_cap(int width, int height, int quality, int cores);
size_t jpeg_enc_seg_cap(int width, int height, int quality, int cores);
/* Pass jpeg_enc_segments(cfg) as `cores` to size a sliced encoder: seg_cap is then
 * per segment, and segs must hold segments of them. */

/*
 * `cluster` may be NULL if cfg->path is JPEG_PATH_FC and you never switch. It must
 * already be opened (pi_cluster_open) -- and note that its FLL cannot be programmed
 * before the open, or the chip hangs at boot with no output.
 *
 * `segs` is caller-owned L2 of at least cores * seg_cap bytes. It is the per-core
 * entropy-segment scratch; the FC path does not use it.
 *
 * Returns 0, or -1 on a bad config.
 */
int jpeg_enc_init(jpeg_enc_t *j, const jpeg_cfg_t *cfg,
                  int width, int height, int stride,
                  struct pi_device *cluster,
                  uint8_t *segs, size_t seg_cap);

/* Switch path between frames. Cheap: no reallocation, no re-init. */
void jpeg_enc_set_path(jpeg_enc_t *j, jpeg_path_e path);

/*
 * Encode one frame. Returns bytes written, or 0 if the whole-frame output or a
 * per-core segment buffer was too small, or if cluster L1 could not be acquired.
 * In each failure case j->overflows is incremented -- a dropped frame, not a
 * corrupt one. The counter is therefore a conservative aggregate despite its
 * name; it does not distinguish capacity failures from L1 contention.
 *
 * On the cluster path this acquires L1, runs the 8 cores, merges, and releases L1
 * per cfg.l1_policy. Call from the FC.
 */
size_t jpeg_enc_frame(jpeg_enc_t *j, const uint8_t *img, uint8_t *out, size_t cap);

/*
 * The same encode, one slice at a time (cfg.slices), so the cluster can run
 * something else between slices:
 *
 *     jpeg_enc_begin(j, img);
 *     while ((left = jpeg_enc_step(j)) > 0) {
 *         jpeg_enc_yield(j);    // only if what runs next needs the encoder's L1
 *         ...                   // e.g. a DORY inference
 *     }
 *     n = jpeg_enc_end(j, out, cap);
 *
 * step() returns the slices still to encode, or -1 if L1 could not be acquired
 * (end() then returns 0). `img` must stay unchanged until end(). On the FC path
 * step() does nothing and end() encodes the whole frame. jpeg_enc_frame() is
 * exactly this loop, without yields.
 */
void jpeg_enc_begin(jpeg_enc_t *j, const uint8_t *img);
int jpeg_enc_step(jpeg_enc_t *j);
void jpeg_enc_yield(jpeg_enc_t *j);
size_t jpeg_enc_end(jpeg_enc_t *j, uint8_t *out, size_t cap);

/* Hand back any L1 still held (only matters under KEEP_ALL / KEEP_TABLES). */
void jpeg_enc_deinit(jpeg_enc_t *j);

#endif /* JPEG_ENC_H */
