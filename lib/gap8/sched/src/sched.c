/*
 * sched.c -- see sched.h.
 *
 * Who owns what (pipelined scheduler):
 *   the camera        camera_stream, exact buffers; one frame is leased at a time
 *                     (the frame task leases the newest whenever it is free)
 *   the cluster       the worker task, which runs every network inference and
 *                     every JPEG slice on it (the encoder borrows the device)
 *   the FC copies     the link task, below every other task: it copies a
 *                     prepared frame into the CPX send queue while the worker
 *                     waits for the cluster; the CPX task then clocks it out
 *                     over SPI. Nothing holds the SPI quiet around cluster jobs.
 *   the state         a one-deep mailbox; a newer state replaces a waiting one
 */
#include "sched.h"

#include <string.h>

#include "FreeRTOS.h"
#include "task.h"

#include "pmsis.h"
#include "pmsis/rtos/malloc/l2_malloc.h"
#include "pmsis/rtos/malloc/malloc_internal.h"

extern malloc_t __l2_malloc; /* the SDK's L2 heap, for sched_l2_free() */

#include "com.h"
#include "cpx.h"
#include "cpx_config.h"
#include "cpx_wifi_session.h"
#include "imgstream.h"
#include "sched_build.h"

/* Frame metadata carried in each image's extension, for the host
 * (cf_lab.camera.ext.SchedFrameExt). */
typedef struct __attribute__((packed)) {
  uint8_t version;
  uint8_t kind; /* sched_kind_t */
  uint16_t encode_us_div8;
  uint32_t frame_id;
  uint32_t captured_ms; /* frame complete in its buffer, GAP8 ms tick */
  uint32_t inferences;
  /* The newest sched_clock_anchor(): another clock's stamp, and when it
   * arrived on the GAP8 ms tick. Both 0 before the first. */
  uint32_t anchor_remote_ms;
  uint32_t anchor_local_ms;
} sched_frame_ext_t;
#define SCHED_FRAME_EXT_VERSION 2

/* A frame ready for the link: raw pixels or the encoded JPEG, and its metadata. */
typedef struct {
  const uint8_t *payload;
  uint32_t bytes;
  sched_frame_ext_t ext;
} sched_outgoing_t;

/* A frame has drained when the send queue is back to (almost) empty: a couple
 * of console lines may share it. */
#define SCHED_TXQ_SLACK 2u

#if !SCHED_BUILD_PIPELINED && !SCHED_BUILD_SEQUENTIAL
#error "build at least one scheduler: SCHED=PIPELINED, SEQUENTIAL or BOTH"
#endif

static struct {
  sched_config_t config;
  const sched_network_t *net;
  bool is_open;
  volatile bool running;
  volatile bool stop;
  sched_kind_t kind;

  camera_stream_t camera;
  uint16_t frame_width;
  uint16_t frame_height;
  uint16_t frame_stride;
#if SCHED_STREAM_FORMAT == SCHED_FORMAT_JPEG
  jpeg_pipeline_t jpeg;
  uint32_t jpeg_budget;
#endif
  cpx_wifi_session_t wifi;
  imgstream_t images;
  uint32_t announced_sequence; /* the WiFi event sequence last told the camera config */

  struct pi_device cluster;
  struct pi_cluster_task cluster_task;
  unsigned int cluster_args[6];
  uint8_t *arena;
  uint8_t *output;
  uint8_t *packed;    /* a frame without row padding, when the camera pads rows */
  uint8_t *state_in;  /* the state being inferred on */

  /* The state mailbox. */
  uint8_t *mailbox;
  uint32_t mailbox_bytes;
  volatile bool state_waiting;
  uint32_t state_posted_us;
  uint32_t state_tag;

  /* The newest clock anchor (sched_clock_anchor). */
  uint32_t anchor_remote_ms;
  uint32_t anchor_local_ms;

  int detail;          /* the failing call's own status, after an open error */
  const char *detail_what;

  TaskHandle_t worker;
  TaskHandle_t frame_task;
  TaskHandle_t log_task;
  volatile uint32_t sequence;
  uint32_t last_done_us;
  uint32_t last_verify_ms;

  /* The pipelined scheduler's handoffs: frame task -> worker -> link task. */
  camera_stream_frame_t frame;
  volatile bool frame_ready;
  volatile bool lease_held;
  sched_outgoing_t outgoing;     /* the frame being sent */
  volatile bool link_busy;       /* the link task is queueing `outgoing` */
  camera_stream_frame_t link_frame;
  volatile bool link_holds_frame;
  TaskHandle_t link_task;
  volatile int alive_tasks;

#if SCHED_METRICS
  sched_stats_t stats;
  uint32_t stats_started_us;
  uint32_t camera_drops_base;
#endif
} S;

/* -------------------------------------------------------------------------- */
/* Measurements                                                               */
/* -------------------------------------------------------------------------- */

/* pi_time_get_us() adds the SysTick timer to the tick count, with interrupts
 * off; when the timer wraps between the two reads, its tick interrupt is still
 * pending and the time steps back a whole tick. Here a pending tick is
 * counted. */
uint32_t sched_time_us(void) {
  const uint32_t irq = pi_irq_disable();
  uint32_t ticks = xTaskGetTickCount();
  uint32_t count = pi_timer_value_read(SYS_TIMER);
  if (NVIC->BUFFER & (1u << SYSTICK_IRQN)) {
    ticks++;
    count = pi_timer_value_read(SYS_TIMER); /* after the wrap */
  }
  pi_irq_restore(irq);
  return ticks * (1000000u / configTICK_RATE_HZ) + count * 1000u / (system_core_clock_get() / 1000u);
}

#define HIST_FINE_TOP_US (SCHED_HIST_FINE_BINS * SCHED_HIST_FINE_US)

#if SCHED_METRICS
static uint32_t hist_bin(uint32_t us) {
  if (us < HIST_FINE_TOP_US) return us / SCHED_HIST_FINE_US;
  const uint32_t bin = SCHED_HIST_FINE_BINS + (us - HIST_FINE_TOP_US) / SCHED_HIST_COARSE_US;
  return bin < SCHED_HIST_BINS ? bin : SCHED_HIST_BINS - 1u;
}

static void hist_add(sched_hist_t *h, uint32_t us) {
  const uint32_t bin = hist_bin(us);
  taskENTER_CRITICAL();
  if (h->bins[bin] != UINT16_MAX) h->bins[bin]++;
  h->count++;
  h->sum_us += us;
  if (us > h->max_us) h->max_us = us;
  taskEXIT_CRITICAL();
}
#define COUNT(field) \
  do {               \
    taskENTER_CRITICAL(); \
    S.stats.field++; \
    taskEXIT_CRITICAL(); \
  } while (0)
#define HIST(field, us) hist_add(&S.stats.field, (us))
#define BUSY(us) \
  do {           \
    taskENTER_CRITICAL(); \
    S.stats.cluster_busy_us += (us); \
    taskEXIT_CRITICAL(); \
  } while (0)
#else
#define COUNT(field) ((void)0)
#define HIST(field, us) ((void)(us))
#define BUSY(us) ((void)(us))
#endif

uint32_t sched_hist_percentile(const sched_hist_t *hist, uint32_t percent) {
  if (hist == NULL || hist->count == 0) return 0;
  const uint64_t want = ((uint64_t)hist->count * percent + 99u) / 100u;
  uint64_t seen = 0;
  for (uint32_t i = 0; i < SCHED_HIST_BINS; i++) {
    seen += hist->bins[i];
    if (seen >= want) {
      const uint32_t top = i < SCHED_HIST_FINE_BINS
                               ? (i + 1u) * SCHED_HIST_FINE_US
                               : HIST_FINE_TOP_US + (i + 1u - SCHED_HIST_FINE_BINS) * SCHED_HIST_COARSE_US;
      return top < hist->max_us ? top : hist->max_us;
    }
  }
  return hist->max_us;
}

static void reset_stats(void) {
#if SCHED_METRICS
  camera_stream_stats_t camera;
  memset(&camera, 0, sizeof(camera));
  (void)camera_stream_get_stats(&S.camera, &camera);
  taskENTER_CRITICAL();
  memset(&S.stats, 0, sizeof(S.stats));
  S.stats_started_us = sched_time_us();
  S.camera_drops_base = camera.dropped_ready_frames + camera.dropped_capture_completions;
  taskEXIT_CRITICAL();
#endif
}

int sched_get_stats(sched_stats_t *stats, bool reset) {
#if SCHED_METRICS
  if (stats == NULL) return SCHED_ERR_ARGUMENT;
  camera_stream_stats_t camera;
  memset(&camera, 0, sizeof(camera));
  (void)camera_stream_get_stats(&S.camera, &camera);
  const uint32_t drops = camera.dropped_ready_frames + camera.dropped_capture_completions;
  const uint32_t now = sched_time_us();
  taskENTER_CRITICAL();
  *stats = S.stats;
  stats->elapsed_us = now - S.stats_started_us;
  stats->camera_drops = drops - S.camera_drops_base;
  if (reset) {
    memset(&S.stats, 0, sizeof(S.stats));
    S.stats_started_us = now;
    S.camera_drops_base = drops;
  }
  taskEXIT_CRITICAL();
  return SCHED_OK;
#else
  (void)stats;
  (void)reset;
  return SCHED_ERR_STATE;
#endif
}

/* -------------------------------------------------------------------------- */
/* The network                                                                */
/* -------------------------------------------------------------------------- */

static uint32_t byte_sum(const uint8_t *data, uint32_t bytes) {
  uint32_t sum = 0;
  for (uint32_t i = 0; i < bytes; i++) sum += data[i];
  return sum;
}

/* One inference on the cluster, returning when it is done. The 3-level target
 * reads its input from the base of the arena (DORY's allocator puts layer 0's
 * input there because initial_dir is 1); the 2-level target takes a pointer. */
static void infer(const uint8_t *input, uint32_t input_us, uint32_t frame_id,
                  uint32_t tag, bool report) {
  const sched_network_t *net = S.net;
  if (net->three_level && input != S.arena) {
    memcpy(S.arena, input, net->input_bytes);
  }
  S.cluster_args[0] = (unsigned int)S.arena;
  S.cluster_args[1] = (unsigned int)net->arena_bytes;
  S.cluster_args[2] = (unsigned int)S.output;
  S.cluster_args[3] = 0u;
  S.cluster_args[4] = (unsigned int)net->initial_dir;
  S.cluster_args[5] = net->three_level ? 0u : (unsigned int)input;

  /* The SDK frees a task's L1 stacks when it ends and zeroes stack_size (not
   * slave_stack_size), so without this every inference after the first runs
   * DORY's master core on the SDK's 2 KB default. */
  S.cluster_task.stack_size = net->cluster_stack;
  S.cluster_task.slave_stack_size = net->cluster_slave_stack;
  const uint32_t start = sched_time_us();
  if (pi_cluster_send_task_to_cl(&S.cluster, &S.cluster_task) != 0) {
    COUNT(cluster_errors); /* no L1 for the stacks: nothing ran */
    return;
  }
  const uint32_t done = sched_time_us();
  BUSY(done - start);
  if (!report) return;

  const uint32_t sequence = S.sequence++;
  COUNT(inferences);
  HIST(wait, start - input_us);
  HIST(inference, done - start);
  HIST(latency, done - input_us);
  if (sequence > 0) HIST(period, done - S.last_done_us);
  S.last_done_us = done;
  if (S.config.on_output != NULL) {
    const sched_output_t output = {
        .data = S.output,
        .bytes = net->output_bytes,
        .sequence = sequence,
        .input_us = input_us,
        .start_us = start,
        .done_us = done,
        .frame_id = frame_id,
        .tag = tag,
    };
    S.config.on_output(&output, S.config.output_context);
  }
}

/* Runs DORY's golden input and compares the output's byte sum with DORY's.
 * The 3-level input goes to the base of the arena; the 2-level one to
 * `scratch`, input_bytes of L2 that are free for the moment: the state buffer,
 * or a camera frame about to be released. */
static bool verify_network(uint8_t *scratch) {
  const sched_network_t *net = S.net;
  if (net->stage_golden == NULL) return true;
  uint8_t *input = net->three_level ? S.arena : scratch;
  net->stage_golden(input);
  infer(input, sched_time_us(), 0u, 0u, false);
  const bool ok = byte_sum(S.output, net->output_bytes) == net->golden_output_sum;
  COUNT(verify_runs);
  if (!ok) COUNT(verify_failures);
  return ok;
}

/* Every SCHED_VERIFY_PERIOD_MS, with the frame just used (before its release). */
static void maybe_verify(const camera_stream_frame_t *frame) {
#if SCHED_VERIFY
  const uint32_t now = sched_time_us() / 1000u;
  if (now - S.last_verify_ms < SCHED_VERIFY_PERIOD_MS) return;
  S.last_verify_ms = now;
  if (S.net->input == SCHED_INPUT_STATE) {
    (void)verify_network(S.state_in);
  } else {
    (void)verify_network(S.packed != NULL ? S.packed : frame->data);
  }
#else
  (void)frame;
#endif
}

/* The state waiting in the mailbox, if any, moved to state_in. */
static bool take_state(uint32_t *posted_us, uint32_t *tag) {
  bool taken = false;
  taskENTER_CRITICAL();
  if (S.state_waiting) {
    memcpy(S.state_in, S.mailbox, S.mailbox_bytes);
    *posted_us = S.state_posted_us;
    *tag = S.state_tag;
    S.state_waiting = false;
    taken = true;
  }
  taskEXIT_CRITICAL();
  return taken;
}

static bool serve_state(void) {
  uint32_t posted_us, tag;
  if (S.net->input != SCHED_INPUT_STATE || !take_state(&posted_us, &tag)) return false;
  infer(S.state_in, posted_us, 0u, tag, true);
  return true;
}

int sched_post_state(const void *state, uint32_t bytes, uint32_t tag) {
  if (!S.is_open || S.net->input != SCHED_INPUT_STATE) return SCHED_ERR_STATE;
  if (state == NULL || bytes != S.mailbox_bytes) return SCHED_ERR_ARGUMENT;
  const uint32_t now = sched_time_us();
  bool replaced;
  taskENTER_CRITICAL();
  replaced = S.state_waiting;
  memcpy(S.mailbox, state, bytes);
  S.state_posted_us = now;
  S.state_tag = tag;
  S.state_waiting = true;
  taskEXIT_CRITICAL();
  COUNT(states_posted);
  if (replaced) COUNT(states_replaced);
  if (S.worker != NULL) xTaskNotifyGive(S.worker);
  return SCHED_OK;
}

/* The camera's ms tick: the clock of camera_stream's frame times. */
void sched_clock_anchor(uint32_t remote_ms) {
  const uint32_t local_ms = (uint32_t)xTaskGetTickCount() * (uint32_t)portTICK_PERIOD_MS;
  taskENTER_CRITICAL();
  S.anchor_remote_ms = remote_ms;
  S.anchor_local_ms = local_ms;
  taskEXIT_CRITICAL();
}

/* The frame as the network wants it: contiguous rows. */
static const uint8_t *frame_pixels(const camera_stream_frame_t *frame) {
  if (S.packed == NULL) return frame->data;
  for (uint16_t row = 0; row < S.frame_height; row++) {
    memcpy(S.packed + (uint32_t)row * S.frame_width,
           frame->data + (uint32_t)row * frame->stride, S.frame_width);
  }
  return S.packed;
}

static void infer_frame(const camera_stream_frame_t *frame) {
  if (S.net->input != SCHED_INPUT_IMAGE) return;
  const uint32_t captured_us = frame->capture_completed_ms * 1000u;
  infer(frame_pixels(frame), captured_us, frame->camera_frame_id, 0u, true);
}

/* -------------------------------------------------------------------------- */
/* The video stream                                                           */
/* -------------------------------------------------------------------------- */

static uint32_t queue_free(void) { return com_tx_free_slots(); }
static bool queue_drained(void) { return queue_free() + SCHED_TXQ_SLACK >= TXQ_SIZE; }

static bool skip_while_sending(const sched_send_view_t *view, void *context) {
  (void)context;
  return view->previous_drained;
}

/* Whether to send this frame: a client listens, the previous frame has left
 * this library (a frame's bytes are not ours to touch until it has), and the
 * send policy agrees (`always` skips the policy). */
static bool want_send(const camera_stream_frame_t *frame, bool always) {
  if (!S.config.stream || !cpx_wifi_session_client_ready(&S.wifi)) return false;
  if (S.link_busy) {
    COUNT(frames_not_sent);
    return false;
  }
  if (always) return true;
#if SCHED_STREAM_FORMAT == SCHED_FORMAT_JPEG
  /* JPEG size is unknown before encoding: ask the policy with the largest
   * frame, so a skipped frame costs no encode. */
  const uint32_t expect = S.jpeg_budget;
#else
  const uint32_t expect = (uint32_t)S.frame_width * S.frame_height;
#endif
  const sched_send_view_t view = {
      .frame_id = frame->camera_frame_id,
      .payload_bytes = expect,
      .packets = imgstream_packets_for(expect),
      .free_packets = queue_free(),
      .queue_packets = TXQ_SIZE,
      .previous_drained = queue_drained(),
  };
  const sched_send_policy_fn policy =
      S.config.send_policy != NULL ? S.config.send_policy : skip_while_sending;
  if (!policy(&view, S.config.send_policy_context)) {
    COUNT(frames_not_sent);
    return false;
  }
  return true;
}

/* Prepares a frame for the link: JPEG builds encode it, in slices; with
 * `serve_between`, a state that arrives meanwhile is served between two
 * slices. Returns false when the frame cannot be sent. */
static bool prepare_frame(const camera_stream_frame_t *frame, sched_outgoing_t *out,
                          bool serve_between) {
  out->payload = frame_pixels(frame);
  out->bytes = (uint32_t)S.frame_width * S.frame_height;
  uint32_t encode_us = 0;
#if SCHED_STREAM_FORMAT == SCHED_FORMAT_JPEG
  jpeg_pipeline_output_t jpeg = {0};
  int status = jpeg_pipeline_encode_begin(&S.jpeg, out->payload);
  if (status == JPEG_PIPELINE_OK) {
    while (1) {
      const uint32_t start = sched_time_us();
      const int left = jpeg_pipeline_encode_step(&S.jpeg);
      encode_us += sched_time_us() - start;
      if (left <= 0) break;
      if (serve_between && S.state_waiting) {
        jpeg_pipeline_encode_yield(&S.jpeg); /* DORY needs the L1 */
        if (serve_state()) COUNT(encode_yields);
      }
    }
    const uint32_t start = sched_time_us();
    status = jpeg_pipeline_encode_end(&S.jpeg, &jpeg);
    encode_us += sched_time_us() - start;
  }
  BUSY(encode_us);
  HIST(encode, encode_us);
  if (status != JPEG_PIPELINE_OK) {
    COUNT(encode_failures);
    return false;
  }
  out->payload = jpeg.data;
  out->bytes = (uint32_t)jpeg.size_bytes;
#else
  (void)serve_between;
#endif
  if (imgstream_packets_for(out->bytes) > queue_free()) {
    COUNT(frames_not_sent);
    return false;
  }
  out->ext = (sched_frame_ext_t){
      .version = SCHED_FRAME_EXT_VERSION,
      .kind = (uint8_t)S.kind,
      .encode_us_div8 = (uint16_t)(encode_us / 8u > UINT16_MAX ? UINT16_MAX : encode_us / 8u),
      .frame_id = frame->camera_frame_id,
      .captured_ms = frame->capture_completed_ms,
      .inferences = S.sequence,
  };
  taskENTER_CRITICAL();
  out->ext.anchor_remote_ms = S.anchor_remote_ms;
  out->ext.anchor_local_ms = S.anchor_local_ms;
  taskEXIT_CRITICAL();
  return true;
}

/* Tells a new client, once and before its first frame, every camera and stream
 * setting of this build; the host records the line with a run's telemetry. A
 * new connection bumps the WiFi session's event sequence. */
static void announce_camera(void) {
  cpx_wifi_session_status_t wifi;
  if (cpx_wifi_session_get_status(&S.wifi, &wifi) != CPX_WIFI_SESSION_OK ||
      wifi.event_sequence == S.announced_sequence) {
    return;
  }
  S.announced_sequence = wifi.event_sequence;
  static char camera[384];
  (void)camera_stream_describe(&S.camera, camera, sizeof(camera));
  camera_stream_info_t info;
  const unsigned fps = camera_stream_get_info(&S.camera, &info) == CAMERA_STREAM_OK
                           ? info.named_fps
                           : 0u;
  cpxPrintToConsole(LOG_TO_WIFI,
                    "camera config: %s out=%ux%u fps=%u format=%s jpeg=cluster quality=%u "
                    "transport=%s channel=%d\n",
                    camera, S.frame_width, S.frame_height, fps,
                    SCHED_STREAM_FORMAT == SCHED_FORMAT_JPEG ? "jpeg" : "raw",
                    (unsigned)S.config.jpeg_quality,
                    WIFI_TRANSPORT == CPX_WIFI_TRANSPORT_UDP ? "udp" : "tcp", WIFI_CHANNEL);
}

/* Copies a prepared frame into the CPX send queue, packet by packet, on the
 * FC; the CPX task then clocks it out over SPI. */
static void queue_frame(const sched_outgoing_t *out) {
  announce_camera();
  const imgstream_frame_t wire = {
      .payload = out->payload,
      .size = out->bytes,
      .width = S.frame_width,
      .height = S.frame_height,
      .type = SCHED_STREAM_FORMAT == SCHED_FORMAT_JPEG ? CF_IMG_TYPE_JPEG : CF_IMG_TYPE_RAW_GRAY,
      .ext = &out->ext,
      .ext_len = sizeof(out->ext),
  };
  const uint32_t start = sched_time_us();
  imgstream_send(&S.images, &wire);
  HIST(send, sched_time_us() - start);
  COUNT(frames_sent);
}

/* -------------------------------------------------------------------------- */
/* The schedulers                                                             */
/* -------------------------------------------------------------------------- */

#define COMPILER_BARRIER() __asm__ volatile("" ::: "memory")

static void task_exit(void) {
  taskENTER_CRITICAL();
  S.alive_tasks--;
  taskEXIT_CRITICAL();
  vTaskDelete(NULL);
}

#if SCHED_BUILD_SEQUENTIAL
/* Capture, infer, encode, send, wait for the send to drain; repeat. */
static void sequential_task(void *argument) {
  (void)argument;
  while (!S.stop) {
    camera_stream_frame_t frame;
    if (camera_stream_acquire(&S.camera, &frame, 100u) != CAMERA_STREAM_OK) continue;
    COUNT(frames);
    infer_frame(&frame);
    (void)serve_state();
    if (want_send(&frame, true) && prepare_frame(&frame, &S.outgoing, false)) {
      queue_frame(&S.outgoing);
      while (!queue_drained() && !S.stop) vTaskDelay(1);
    }
    maybe_verify(&frame);
    (void)camera_stream_release(&S.camera, &frame);
  }
  task_exit();
}
#endif

#if SCHED_BUILD_PIPELINED
/* The camera lease: one frame at a time, held by the worker, or by the link
 * task while a raw frame is copied out of it. Whoever holds it releases it
 * here, and the frame task then leases the newest frame. */
static void lease_release(camera_stream_frame_t *frame) {
  (void)camera_stream_release(&S.camera, frame);
  S.lease_held = false;
  if (S.frame_task != NULL) xTaskNotifyGive(S.frame_task);
}

/* Leases the newest frame whenever the lease is free. Blocking rather than
 * polling keeps the FC idle between frames, which the HyperBus needs: every
 * 512 B burst of an L3 transfer is restarted by an FC interrupt. */
static void frame_task(void *argument) {
  (void)argument;
  while (!S.stop) {
    if (S.lease_held) {
      (void)ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));
      continue;
    }
    camera_stream_frame_t frame;
    if (camera_stream_acquire(&S.camera, &frame, 100u) != CAMERA_STREAM_OK) continue;
    S.frame = frame;
    S.lease_held = true;
    COMPILER_BARRIER();
    S.frame_ready = true;
    if (S.worker != NULL) xTaskNotifyGive(S.worker);
  }
  task_exit();
}

/* Queues prepared frames into CPX in the background, below every other task:
 * it runs while the worker waits for the cluster, so the FC copies overlap
 * cluster work, and the worker never waits on the link. */
static void link_task(void *argument) {
  (void)argument;
  while (1) {
    if (!S.link_busy) {
      if (S.stop) break;
      (void)ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));
      continue;
    }
    queue_frame(&S.outgoing);
    if (S.link_holds_frame) {
      S.link_holds_frame = false;
      lease_release(&S.link_frame);
    }
    COMPILER_BARRIER();
    S.link_busy = false;
    if (S.worker != NULL) xTaskNotifyGive(S.worker);
  }
  task_exit();
}

/* The cluster's single owner, by priority: a waiting state; then a new frame's
 * inference; then the frame's encode for the video, in slices, serving any
 * state that arrives between them. The link task queues the encoded frame. */
static void pipelined_task(void *argument) {
  (void)argument;
  while (!S.stop) {
    if (serve_state()) continue;
    if (!S.frame_ready) {
      (void)ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(100));
      continue;
    }
    camera_stream_frame_t frame = S.frame;
    S.frame_ready = false;
    COUNT(frames);
    infer_frame(&frame);
    bool handed_over = false;
    if (want_send(&frame, false) && prepare_frame(&frame, &S.outgoing, true)) {
#if SCHED_STREAM_FORMAT == SCHED_FORMAT_RAW
      /* The raw frame is sent from the camera buffer: the link releases it. */
      S.link_frame = frame;
      S.link_holds_frame = true;
      handed_over = true;
#endif
      COMPILER_BARRIER();
      S.link_busy = true;
      if (S.link_task != NULL) xTaskNotifyGive(S.link_task);
    }
    if (!handed_over) {
      maybe_verify(&frame);
      lease_release(&frame);
    }
  }
  if (S.frame_ready) {
    S.frame_ready = false;
    lease_release(&S.frame);
  }
  task_exit();
}
#endif

#if SCHED_LOG
static void log_task(void *argument) {
  (void)argument;
  while (!S.stop) {
    vTaskDelay(pdMS_TO_TICKS(S.config.report_ms));
    static sched_stats_t st; /* 3 KB: not on this task's stack */
    if (sched_get_stats(&st, true) != SCHED_OK) continue;
    const CPXConsoleTarget_t to =
        cpx_wifi_session_client_ready(&S.wifi) ? LOG_TO_WIFI : LOG_TO_CRTP;
    const uint32_t ms = st.elapsed_us / 1000u ? st.elapsed_us / 1000u : 1u;
    cpxPrintToConsole(to,
                      "SB|k=%s ms=%lu fr=%lu sent=%lu skip=%lu inf=%lu st=%lu rep=%lu clpm=%lu\n",
                      S.kind == SCHED_PIPELINED ? "pipe" : "seq", (unsigned long)ms,
                      (unsigned long)st.frames, (unsigned long)st.frames_sent,
                      (unsigned long)st.frames_not_sent, (unsigned long)st.inferences,
                      (unsigned long)st.states_posted, (unsigned long)st.states_replaced,
                      (unsigned long)(st.cluster_busy_us / ms));
    cpxPrintToConsole(to, "SB|lat50=%lu lat99=%lu latmax=%lu inf50=%lu enc50=%lu per99=%lu\n",
                      (unsigned long)sched_hist_percentile(&st.latency, 50),
                      (unsigned long)sched_hist_percentile(&st.latency, 99),
                      (unsigned long)st.latency.max_us,
                      (unsigned long)sched_hist_percentile(&st.inference, 50),
                      (unsigned long)sched_hist_percentile(&st.encode, 50),
                      (unsigned long)sched_hist_percentile(&st.period, 99));
    cpxPrintToConsole(to, "SB|drop=%lu encfail=%lu vf=%lu/%lu\n",
                      (unsigned long)st.camera_drops, (unsigned long)st.encode_failures,
                      (unsigned long)st.verify_failures, (unsigned long)st.verify_runs);
  }
  task_exit();
}
#endif

/* -------------------------------------------------------------------------- */
/* Opening                                                                    */
/* -------------------------------------------------------------------------- */

/* Records which call failed and how, then returns `status`. */
static int open_failed(int status, const char *what, int detail) {
  S.detail = detail;
  S.detail_what = what;
  return status;
}

void sched_config_init(sched_config_t *config) {
  if (config == NULL) return;
  memset(config, 0, sizeof(*config));
  config->kind = SCHED_DEFAULT_KIND;
  config->stream = SCHED_STREAM != 0;
  config->jpeg_quality = SCHED_JPEG_QUALITY;
  config->jpeg_slices = SCHED_JPEG_SLICES;
  config->cluster_hz = SCHED_CLUSTER_HZ;
  config->report_ms = SCHED_REPORT_MS;
}

static int open_cluster(void) {
  /* static: the driver keeps a pointer to the conf after the open. */
  static struct pi_cluster_conf conf;
  pi_cluster_conf_init(&conf);
  conf.id = 0;
  pi_open_from_conf(&S.cluster, &conf);
  const int status = pi_cluster_open(&S.cluster);
  if (status != 0) return open_failed(SCHED_ERR_CLUSTER, "cluster_open", status);
  if (S.config.cluster_hz != 0u) {
    (void)pi_freq_set(PI_FREQ_DOMAIN_CL, S.config.cluster_hz);
  }
  pi_cluster_task(&S.cluster_task, S.net->cluster_entry, S.cluster_args);
  S.cluster_task.stack_size = S.net->cluster_stack;
  S.cluster_task.slave_stack_size = S.net->cluster_slave_stack;
  return SCHED_OK;
}

static int open_camera(void) {
  camera_stream_config_t config;
  int status = camera_stream_config_init(&config);
  if (status != CAMERA_STREAM_OK) return open_failed(SCHED_ERR_CAMERA, "camera_config", status);
  config.sensor_mode = (camera_stream_sensor_mode_t)SCHED_SENSOR_MODE;
  config.policy = CAMERA_STREAM_POLICY_EXACT_FRAME;
  const bool qvga = config.sensor_mode == CAMERA_STREAM_SENSOR_QVGA_30 ||
                    config.sensor_mode == CAMERA_STREAM_SENSOR_QVGA_60;
  config.capture_buffer_count = SCHED_CAMERA_BUFFERS ? SCHED_CAMERA_BUFFERS : (qvga ? 2u : 3u);
  config.orientation = SCHED_CAMERA_ORIENTATION;
  config.watchdog_timeout_ms = SCHED_CAMERA_WATCHDOG_MS;
  status = camera_stream_init(&S.camera, &config);
  if (status != CAMERA_STREAM_OK) return open_failed(SCHED_ERR_CAMERA, "camera_init", status);
  status = camera_stream_open(&S.camera);
  if (status != CAMERA_STREAM_OK) return open_failed(SCHED_ERR_CAMERA, "camera_open", status);
  camera_stream_info_t info;
  status = camera_stream_get_info(&S.camera, &info);
  if (status != CAMERA_STREAM_OK) return open_failed(SCHED_ERR_CAMERA, "camera_info", status);
  S.frame_width = info.width;
  S.frame_height = info.height;
  S.frame_stride = info.stride;
  return SCHED_OK;
}

#if SCHED_STREAM_FORMAT == SCHED_FORMAT_JPEG
static int open_jpeg(void) {
  jpeg_pipeline_config_t config;
  if (jpeg_pipeline_config_init(&config) != JPEG_PIPELINE_OK) return SCHED_ERR_JPEG;
  config.width = S.frame_width;
  config.height = S.frame_height;
  config.stride = S.frame_width; /* encoded from the packed frame */
  config.encoder.path = JPEG_PATH_CLUSTER;
  config.encoder.cores = SCHED_JPEG_CORES;
  config.encoder.quality = S.config.jpeg_quality;
  config.encoder.restart_interval = 0;
  config.encoder.l1_policy = JPEG_CL_L1_RELEASE_ALL; /* L1 back to DORY each frame */
  /* A state served between slices waits for at most one slice. */
  const uint32_t auto_slices = S.net->input == SCHED_INPUT_STATE
                                   ? ((uint32_t)S.frame_width * S.frame_height + 4799u) / 4800u
                                   : 1u;
  const uint32_t slices = S.config.jpeg_slices ? S.config.jpeg_slices : auto_slices;
  config.encoder.slices = (int)(slices > JPEG_ENC_MAX_SLICES ? JPEG_ENC_MAX_SLICES : slices);
  S.jpeg_budget = SCHED_JPEG_BUDGET_BYTES
                      ? SCHED_JPEG_BUDGET_BYTES
                      : (((uint32_t)S.frame_width * S.frame_height / 3u) + 1023u) & ~1023u;
  config.output_budget_bytes = S.jpeg_budget;
  config.cluster_mode = JPEG_PIPELINE_CLUSTER_BORROWED;
  config.borrowed_cluster = &S.cluster;
  int status = jpeg_pipeline_init(&S.jpeg, &config);
  if (status != JPEG_PIPELINE_OK) return open_failed(SCHED_ERR_JPEG, "jpeg_init", status);
  status = jpeg_pipeline_open(&S.jpeg);
  if (status != JPEG_PIPELINE_OK) return open_failed(SCHED_ERR_JPEG, "jpeg_open", status);
  return SCHED_OK;
}
#endif

static int open_wifi(void) {
  cpx_wifi_session_config_t config;
  if (cpx_wifi_session_config_init(&config) != CPX_WIFI_SESSION_OK) return SCHED_ERR_WIFI;
  config.ssid = WIFI_SSID;
  config.key = WIFI_KEY;
  config.mode = (cpx_wifi_mode_t)WIFI_MODE;
  config.transport = (cpx_wifi_transport_t)WIFI_TRANSPORT;
  config.channel = WIFI_CHANNEL;
  config.hostname = WIFI_HOSTNAME;
  config.name = WIFI_NAME;
  int status = cpx_wifi_session_init(&S.wifi, &config);
  if (status != CPX_WIFI_SESSION_OK) return open_failed(SCHED_ERR_WIFI, "wifi_init", status);
  status = cpx_wifi_session_open(&S.wifi);
  if (status != CPX_WIFI_SESSION_OK) return open_failed(SCHED_ERR_WIFI, "wifi_open", status);
  status = cpx_wifi_session_connect(&S.wifi);
  if (status != CPX_WIFI_SESSION_OK) return open_failed(SCHED_ERR_WIFI, "wifi_connect", status);
  imgstream_init(&S.images, CPX_T_WIFI_HOST);
  return SCHED_OK;
}

static void *alloc_l2(uint32_t bytes) {
  return bytes ? pi_l2_malloc((bytes + 3u) & ~3u) : NULL;
}

int sched_open(const sched_config_t *config) {
  if (config == NULL || config->network == NULL || config->network->cluster_entry == NULL) {
    return SCHED_ERR_ARGUMENT;
  }
  if (S.is_open) return SCHED_ERR_STATE;
  memset(&S, 0, sizeof(S));
  S.announced_sequence = UINT32_MAX;
  S.config = *config;
  if (S.config.report_ms == 0u) S.config.report_ms = SCHED_REPORT_MS;
  S.net = config->network;
  const sched_network_t *net = S.net;

  /* The camera before CPX: the HM01B0 fails to configure once cpxInit() has
   * run. CPX comes up even when the camera fails, so the application can
   * report it. */
  int status = open_camera();
  cpxInit();
  if (status != SCHED_OK) return status;
  /* DORY's order: the flash and HyperRAM before the arena (they allocate L2
   * themselves). */
  if (net->initialize != NULL) net->initialize();
  if (net->input == SCHED_INPUT_IMAGE &&
      (net->image_width != S.frame_width || net->image_height != S.frame_height ||
       net->input_bytes != (uint32_t)S.frame_width * S.frame_height)) {
    return SCHED_ERR_SIZE;
  }
  status = open_cluster();
  if (status != SCHED_OK) return status;
#if SCHED_STREAM_FORMAT == SCHED_FORMAT_JPEG
  status = open_jpeg();
  if (status != SCHED_OK) return status;
#endif
  status = open_wifi();
  if (status != SCHED_OK) return status;

  S.arena = alloc_l2(net->arena_bytes);
  S.output = alloc_l2(net->output_bytes);
  if (S.frame_stride != S.frame_width) {
    S.packed = alloc_l2((uint32_t)S.frame_width * S.frame_height);
    if (S.packed == NULL) return SCHED_ERR_MEMORY;
  }
  if (net->input == SCHED_INPUT_STATE) {
    S.mailbox_bytes = net->input_bytes;
    S.mailbox = alloc_l2(net->input_bytes);
    S.state_in = alloc_l2(net->input_bytes);
    if (S.mailbox == NULL || S.state_in == NULL) return SCHED_ERR_MEMORY;
  }
  if (S.arena == NULL || S.output == NULL) return SCHED_ERR_MEMORY;

  S.is_open = true;
#if SCHED_METRICS
  S.stats_started_us = sched_time_us();
#endif
  status = camera_stream_start(&S.camera);
  if (status != CAMERA_STREAM_OK) return open_failed(SCHED_ERR_CAMERA, "camera_start", status);
#if SCHED_VERIFY
  /* An image network's check runs in a camera frame's buffer. */
  camera_stream_frame_t frame = {0};
  if (net->input == SCHED_INPUT_IMAGE) {
    status = camera_stream_acquire(&S.camera, &frame, 1000u);
    if (status != CAMERA_STREAM_OK) return open_failed(SCHED_ERR_CAMERA, "camera_acquire", status);
  }
  const bool ok = verify_network(net->input == SCHED_INPUT_STATE ? S.state_in : frame.data);
  if (net->input == SCHED_INPUT_IMAGE) (void)camera_stream_release(&S.camera, &frame);
  if (!ok) return SCHED_ERR_VERIFY;
  S.last_verify_ms = sched_time_us() / 1000u;
#endif
  return SCHED_OK;
}

int sched_start(sched_kind_t kind) {
  if (!S.is_open || S.running) return SCHED_ERR_STATE;
  if ((kind == SCHED_PIPELINED && !SCHED_BUILD_PIPELINED) ||
      (kind == SCHED_SEQUENTIAL && !SCHED_BUILD_SEQUENTIAL)) {
    return SCHED_ERR_ARGUMENT;
  }
  S.kind = kind;
  S.stop = false;
  S.sequence = 0; /* also restarts the period measurement */
  S.state_waiting = false; /* posted before this start: stale */
  S.frame_ready = false;
  S.lease_held = false;
  S.link_busy = false;
  S.link_holds_frame = false;
  S.alive_tasks = 0;
  reset_stats();

  BaseType_t ok = pdPASS;
  const UBaseType_t priority = tskIDLE_PRIORITY + 1;
#if SCHED_BUILD_PIPELINED
  if (kind == SCHED_PIPELINED) {
    /* In the order they notify each other: the frame task leases at once. */
    S.alive_tasks = 3;
    ok = xTaskCreate(pipelined_task, "sched", configMINIMAL_STACK_SIZE * 4, NULL, priority,
                     &S.worker);
    if (ok == pdPASS) {
      ok = xTaskCreate(link_task, "sched_link", configMINIMAL_STACK_SIZE * 3, NULL,
                       tskIDLE_PRIORITY, &S.link_task);
    }
    if (ok == pdPASS) {
      ok = xTaskCreate(frame_task, "sched_frame", configMINIMAL_STACK_SIZE * 2, NULL, priority,
                       &S.frame_task);
    }
  }
#endif
#if SCHED_BUILD_SEQUENTIAL
  if (kind == SCHED_SEQUENTIAL) {
    S.alive_tasks = 1;
    ok = xTaskCreate(sequential_task, "sched", configMINIMAL_STACK_SIZE * 4, NULL, priority,
                     &S.worker);
  }
#endif
#if SCHED_LOG
  if (ok == pdPASS) {
    S.alive_tasks++;
    ok = xTaskCreate(log_task, "sched_log", configMINIMAL_STACK_SIZE * 3, NULL, priority,
                     &S.log_task);
  }
#endif
  if (ok != pdPASS) return SCHED_ERR_RTOS;
  S.running = true;
  return SCHED_OK;
}

int sched_stop(void) {
  if (!S.running) return SCHED_ERR_STATE;
  S.stop = true;
  if (S.worker != NULL) xTaskNotifyGive(S.worker);
  if (S.frame_task != NULL) xTaskNotifyGive(S.frame_task);
  if (S.link_task != NULL) xTaskNotifyGive(S.link_task);
  while (S.alive_tasks > 0) vTaskDelay(pdMS_TO_TICKS(10));
  S.worker = NULL;
  S.frame_task = NULL;
  S.link_task = NULL;
  S.log_task = NULL;
  S.running = false;
  return SCHED_OK;
}

int sched_run(const sched_network_t *network, sched_output_fn on_output, void *context) {
  sched_config_t config;
  sched_config_init(&config);
  config.network = network;
  config.on_output = on_output;
  config.output_context = context;
  int status = sched_open(&config);
  if (status == SCHED_OK) status = sched_start(config.kind);
  if (status != SCHED_OK) {
    cpxPrintToConsole(LOG_TO_CRTP, "sched: setup failed (%d)\n", status);
    return status;
  }
  vTaskSuspend(NULL);
  return SCHED_OK;
}

camera_stream_t *sched_camera(void) { return S.is_open ? &S.camera : NULL; }

jpeg_pipeline_t *sched_jpeg(void) {
#if SCHED_STREAM_FORMAT == SCHED_FORMAT_JPEG
  return S.is_open ? &S.jpeg : NULL;
#else
  return NULL;
#endif
}

struct pi_device *sched_cluster(void) { return S.is_open ? &S.cluster : NULL; }

int sched_error_detail(const char **what) {
  if (what != NULL) *what = S.detail_what != NULL ? S.detail_what : "";
  return S.detail;
}

bool sched_client_ready(void) { return S.is_open && cpx_wifi_session_client_ready(&S.wifi); }

void sched_l2_free(uint32_t *total, uint32_t *largest) {
  int32_t size = 0, chunks = 0;
  void *first = NULL;
  __malloc_info(&__l2_malloc, &size, &first, &chunks);
  uint32_t big = 0;
  for (const malloc_chunk_t *chunk = __l2_malloc.first_free; chunk != NULL; chunk = chunk->next) {
    if ((uint32_t)chunk->size > big) big = (uint32_t)chunk->size;
  }
  if (total != NULL) *total = (uint32_t)size;
  if (largest != NULL) *largest = big;
}
