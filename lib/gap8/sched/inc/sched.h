/*
 * sched.h -- run a DORY policy on the AI-deck beside the camera stream.
 *
 * One application has one camera, one cluster and one WiFi link, so this is a
 * single process-wide scheduler, used at one of three levels:
 *
 *   1. sched_run(network, on_output, context): the camera, the video stream and
 *      the scheduler come from build settings (sched.mk); give it the network
 *      and where its outputs go. Never returns unless setup fails.
 *   2. sched_config_t: the same, with the settings worth tuning at run time.
 *   3. Hooks in sched_config_t and the accessors at the end, for full control.
 *
 * A policy takes its input from the camera (SCHED_INPUT_IMAGE: the freshest
 * frame, whose size must equal the network's input) or from the application
 * (SCHED_INPUT_STATE: the newest vector passed to sched_post_state()).
 *
 * Two schedulers, chosen at build time (SCHED=PIPELINED or SEQUENTIAL; a bench
 * may build both):
 *   SEQUENTIAL  capture, infer, encode, send and wait for the send to drain, in
 *               turn. The baseline.
 *   PIPELINED   the camera, the cluster, the FC and the WiFi link work at once.
 *               One task owns the cluster and runs, by priority: a posted
 *               state's inference, a new frame's inference, then the frame's
 *               JPEG encode in slices, serving a state that arrives between
 *               them. A background task copies the frame into the CPX queue
 *               while the cluster works. A frame is sent only when the previous
 *               one has gone (the send policy).
 *
 * Instrumentation is compiled in only when asked for, so a demo pays nothing:
 *   SCHED_METRICS  per-job timing and per-resource busy time (sched_get_stats)
 *   SCHED_VERIFY   the network checked against DORY's golden output at start
 *                  and every SCHED_VERIFY_PERIOD_MS
 *   SCHED_LOG      a report line every report_ms
 */
#ifndef CF_GAP8_SCHED_H
#define CF_GAP8_SCHED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "camera_stream.h"
#include "jpeg_pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SCHED_METRICS
#define SCHED_METRICS 0
#endif
#ifndef SCHED_VERIFY
#define SCHED_VERIFY 0
#endif
#ifndef SCHED_VERIFY_PERIOD_MS
#define SCHED_VERIFY_PERIOD_MS 5000u
#endif
#ifndef SCHED_LOG
#define SCHED_LOG 0
#endif

typedef enum {
  SCHED_OK = 0,
  SCHED_ERR_ARGUMENT = -301,
  SCHED_ERR_STATE = -302,
  SCHED_ERR_MEMORY = -303,
  SCHED_ERR_CAMERA = -304,
  SCHED_ERR_JPEG = -305,
  SCHED_ERR_WIFI = -306,
  SCHED_ERR_CLUSTER = -307,
  SCHED_ERR_NETWORK = -308,
  SCHED_ERR_RTOS = -309,
  SCHED_ERR_SIZE = -310,   /* the camera frame does not match the network input */
  SCHED_ERR_VERIFY = -311, /* the network did not reproduce DORY's golden output */
} sched_status_t;

/* --------------------------------------------------------------------------
 * The network: a DORY graph, described by a sched_network_t.
 */
typedef enum {
  SCHED_INPUT_STATE = 0, /* a vector from sched_post_state() */
  SCHED_INPUT_IMAGE = 1, /* the camera frame, grayscale, width x height */
} sched_input_t;

typedef struct {
  const char *name;
  sched_input_t input;
  uint16_t image_width; /* SCHED_INPUT_IMAGE: must equal the camera's frame */
  uint16_t image_height;
  uint32_t input_bytes;
  uint32_t output_bytes;
  uint32_t arena_bytes; /* DORY's L2 buffer */
  /* 3-level target: weights in HyperRAM, the input at the base of the arena.
   * 2-level target: weights linked into L2, the input passed by pointer. */
  uint8_t three_level;
  int initial_dir;
  /* Mounts the flash and loads the weights into HyperRAM (3-level); NULL on
   * the 2-level target. Runs once, from a task. */
  void (*initialize)(void);
  void (*cluster_entry)(void *args); /* DORY's network_run_cluster */
  uint16_t cluster_stack;
  uint16_t cluster_slave_stack;
  /* Golden check (SCHED_VERIFY): writes the input DORY's golden output was
   * computed from, input_bytes long, to dst. NULL: no check. */
  void (*stage_golden)(void *dst);
  uint32_t golden_output_sum; /* byte sum of the final output */
} sched_network_t;

/* One inference's result, valid only during the callback. */
typedef struct {
  const void *data; /* DORY's final output, output_bytes long */
  uint32_t bytes;
  uint32_t sequence;  /* since sched_start() */
  uint32_t input_us;  /* when the input appeared: frame captured, state posted */
  uint32_t start_us;  /* inference start */
  uint32_t done_us;   /* inference end */
  uint32_t frame_id;  /* SCHED_INPUT_IMAGE: the camera frame */
  uint32_t tag;       /* SCHED_INPUT_STATE: the tag posted with the state */
} sched_output_t;

typedef void (*sched_output_fn)(const sched_output_t *output, void *context);

/* --------------------------------------------------------------------------
 * Level 1.
 */
int sched_run(const sched_network_t *network, sched_output_fn on_output,
              void *context);

/* State policies: the next inference uses the newest state posted. A state
 * posted while an older one waits replaces it. `tag` comes back with its
 * output (sched_output_t.tag), to tell which request an output answers. Any
 * task may call this. */
int sched_post_state(const void *state, uint32_t bytes, uint32_t tag);

/* Notes that a message stamped `remote_ms` on another processor's ms clock
 * (the STM32's tick) has just arrived. Every frame sent carries the newest
 * such pair, which lets the host put the frames on that clock
 * (cf_lab.telemetry.sync). Call it as soon as the message is received. Any
 * task may call this. */
void sched_clock_anchor(uint32_t remote_ms);

/* --------------------------------------------------------------------------
 * Level 2.
 */
typedef enum {
  SCHED_SEQUENTIAL = 0,
  SCHED_PIPELINED = 1,
} sched_kind_t;

/* What the send policy sees when a frame could be sent. */
typedef struct {
  uint32_t frame_id;
  uint32_t payload_bytes;   /* the frame as it would be sent */
  uint32_t packets;         /* CPX packets it would take */
  uint32_t free_packets;    /* CPX packets free in the send queue */
  uint32_t queue_packets;   /* the send queue's size */
  bool previous_drained;    /* the previous frame has left the queue */
} sched_send_view_t;

/* Return true to send this frame. The default sends only when the previous
 * frame has drained (skip while sending). */
typedef bool (*sched_send_policy_fn)(const sched_send_view_t *view, void *context);

typedef struct {
  sched_kind_t kind;
  const sched_network_t *network;
  sched_output_fn on_output;
  void *output_context;

  bool stream;           /* send frames to the WiFi host */
  uint8_t jpeg_quality;  /* JPEG builds */
  /* JPEG builds: cluster jobs per encode, so a posted state waits for one
   * slice, not a whole encode. 0 chooses: one per 4800 pixels for a state
   * network (4 at QQVGA, 16 at QVGA), else 1. */
  uint8_t jpeg_slices;
  uint32_t cluster_hz;   /* 0 leaves the SDK default */
  uint32_t report_ms;    /* SCHED_LOG period */

  /* Level 3. NULL keeps the default. */
  sched_send_policy_fn send_policy;
  void *send_policy_context;
} sched_config_t;

void sched_config_init(sched_config_t *config);

/* Opens the camera, CPX (it calls cpxInit(), after the camera, which must come
 * first; do not call it yourself), the network, cluster, JPEG encoder and WiFi:
 * everything that allocates. Call once, from a task. */
int sched_open(const sched_config_t *config);
/* Starts scheduling with `kind` (must be built in). */
int sched_start(sched_kind_t kind);
/* Stops scheduling; the camera, the cluster and WiFi stay open, so a bench can
 * start the other scheduler on the same boot. */
int sched_stop(void);

/* --------------------------------------------------------------------------
 * Measurements (SCHED_METRICS). Times are kept as histograms: 100 us bins up
 * to 12.8 ms, then 1 ms bins up to 140.8 ms (the last bin holds the rest).
 */
#define SCHED_HIST_BINS 256u
#define SCHED_HIST_FINE_BINS 128u
#define SCHED_HIST_FINE_US 100u
#define SCHED_HIST_COARSE_US 1000u

typedef struct {
  uint32_t count;
  uint32_t max_us;
  uint64_t sum_us;
  uint16_t bins[SCHED_HIST_BINS];
} sched_hist_t;

typedef struct {
  uint32_t elapsed_us;
  uint32_t frames;            /* frames taken from the camera */
  uint32_t frames_sent;
  uint32_t frames_not_sent;   /* skipped by the send policy */
  uint32_t encode_failures;
  uint32_t encode_yields;     /* states served between two slices of an encode */
  uint32_t cluster_errors;    /* inferences the cluster could not start (no L1) */
  uint32_t states_posted;
  uint32_t states_replaced;   /* posted over one still waiting */
  uint32_t inferences;
  uint32_t verify_runs;
  uint32_t verify_failures;
  uint32_t cluster_busy_us;
  uint32_t camera_drops;      /* frames the camera completed and nobody took */
  sched_hist_t latency;       /* input appeared -> inference done */
  sched_hist_t wait;          /* input appeared -> inference start */
  sched_hist_t inference;     /* inference start -> done */
  sched_hist_t encode;        /* JPEG encode */
  sched_hist_t send;          /* queueing a frame for the link, on the FC */
  sched_hist_t period;        /* between consecutive inferences */
} sched_stats_t;

/* Copies the counters; `reset` starts a new interval. */
int sched_get_stats(sched_stats_t *stats, bool reset);
/* The time below which `percent` of a histogram's samples fall: the top of
 * that sample's bin, or the maximum if lower. */
uint32_t sched_hist_percentile(const sched_hist_t *hist, uint32_t percent);

/* --------------------------------------------------------------------------
 * Level 3: the objects underneath, open after sched_open().
 */
camera_stream_t *sched_camera(void);
jpeg_pipeline_t *sched_jpeg(void);
struct pi_device *sched_cluster(void);
/* After a failed sched_open(): the status of the call underneath that failed
 * (camera_stream, jpeg_pipeline, cpx_wifi_session or the cluster), and which. */
int sched_error_detail(const char **what);
/* Microseconds since boot: the clock of sched_output_t. Unlike
 * pi_time_get_us(), never steps back. */
uint32_t sched_time_us(void);
/* Whether a WiFi host is connected: where a report line can go. */
bool sched_client_ready(void);
/* Free L2 in total and in the largest block: what a network arena could get. */
void sched_l2_free(uint32_t *total, uint32_t *largest);

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_SCHED_H */
