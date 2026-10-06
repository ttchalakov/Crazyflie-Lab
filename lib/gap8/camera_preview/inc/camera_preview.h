#ifndef CF_GAP8_CAMERA_PREVIEW_H
#define CF_GAP8_CAMERA_PREVIEW_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "camera_stream.h"
#include "cpx_wifi_session.h"
#include "jpeg_pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CAMERA_PREVIEW_DEFAULT_WIDTH 320u
#define CAMERA_PREVIEW_DEFAULT_HEIGHT 240u
#define CAMERA_PREVIEW_DEFAULT_TARGET_FPS 60u
#define CAMERA_PREVIEW_DEFAULT_JPEG_BUDGET (64u * 1024u)

/* The camera settings (orientation, exposure, gains, flicker, settle time) come
 * from camera_stream's defaults, so the -DCAMERA_STREAM_DEFAULT_* hooks set them
 * here too; the initialized config stays editable before init(). */

typedef enum {
  CAMERA_PREVIEW_OK = 0,
  CAMERA_PREVIEW_ERR_INVALID_ARGUMENT = -201,
  CAMERA_PREVIEW_ERR_INVALID_STATE = -202,
  CAMERA_PREVIEW_ERR_BUSY = -203,
  CAMERA_PREVIEW_ERR_CAMERA = -204,
  CAMERA_PREVIEW_ERR_JPEG = -205,
  CAMERA_PREVIEW_ERR_WIFI = -206,
  CAMERA_PREVIEW_ERR_RTOS = -207,
  CAMERA_PREVIEW_ERR_TRANSPORT_CONFLICT = -208,
} camera_preview_status_t;

typedef enum {
  CAMERA_PREVIEW_TRANSPORT_TCP = 0,
  CAMERA_PREVIEW_TRANSPORT_UDP = 1,
} camera_preview_transport_t;

typedef struct {
  /*
   * The facade is deliberately a QVGA60 native live-preview preset. These
   * nested configurations expose orientation/exposure/JPEG ownership knobs
   * without duplicating the lower-level APIs. Changing the sensor geometry or
   * capture policy is rejected; use camera_stream directly for those cases.
   */
  camera_stream_config_t camera;
  jpeg_pipeline_config_t jpeg;

  const char *ssid;
  const char *key;
  camera_preview_transport_t transport;
  cpx_wifi_mode_t wifi_mode;
  uint32_t client_settle_ms;

  /*
   * true lets the facade initialize process-wide CPX after Himax opens, which
   * keeps the camera-before-transport allocation order. Set false only when the
   * application has already called cpxInit() exactly once.
   */
  bool initialize_cpx;

  /*
   * NULL selects the facade's process-lifetime Wi-Fi control session. A
   * non-NULL session is borrowed and must already be opened/connected by the
   * application; the facade never closes or reconnects it. Select existing
   * CPX with camera_preview_config_use_existing_cpx() when borrowing.
   */
  cpx_wifi_session_t *borrowed_wifi_session;

  uint32_t target_fps;
  /* Zero selects 4 * configMINIMAL_STACK_SIZE and tskIDLE_PRIORITY + 1. */
  uint32_t worker_task_stack_words;
  uint32_t worker_task_priority;

  /* Attach camera_stream_frame_ext_t (timings and drop counters). */
  bool include_frame_metadata;
} camera_preview_config_t;

typedef struct {
  camera_stream_memory_plan_t camera;
  jpeg_pipeline_memory_plan_t jpeg;
  /* Raw capture + JPEG output/segment allocations. */
  size_t bulk_l2_bytes;
  /* Caller-owned facade storage and process-lifetime internal Wi-Fi BSS. */
  size_t object_bytes;
  size_t internal_wifi_object_bytes;
  /* Queue payload storage allocated by CPX/Wi-Fi on the first owned start. */
  size_t cpx_queue_payload_bytes;
  size_t wifi_queue_payload_bytes;
  /*
   * Known L2 footprint including the fields above and task stack storage.
   * RTOS control blocks/queue metadata, lower-layer static BSS, driver state,
   * and cluster L1 are platform-dependent and deliberately excluded.
   */
  size_t known_l2_bytes;
  uint32_t camera_task_stack_words;
  uint32_t cpx_task_stack_words;
  uint32_t wifi_task_stack_words;
  uint32_t worker_task_stack_words;
  uint32_t total_task_stack_words;
} camera_preview_memory_plan_t;

typedef struct {
  uint32_t acquired_frames;
  uint32_t no_client_drops;
  uint32_t jpeg_drops;
  uint32_t send_failures;
  uint32_t sent_frames;
  uint32_t sent_packets;
  uint32_t last_jpeg_us;
  uint32_t max_jpeg_us;
} camera_preview_stats_t;

/*
 * Caller-owned storage. The internal Wi-Fi session is process-global because
 * CPX_F_WIFI_CTRL has one receiver and CPX has no queue teardown operation.
 */
#define CAMERA_PREVIEW_OBJECT_BYTES                                      \
  (CAMERA_STREAM_OBJECT_BYTES + JPEG_PIPELINE_OBJECT_BYTES +             \
   sizeof(CPXPacket_t) + 1280u)

typedef union {
  uint64_t _alignment;
  uint8_t _storage[CAMERA_PREVIEW_OBJECT_BYTES];
} camera_preview_t;

/*
 * QVGA60/native/JPEG-cluster/live-preview defaults. Application-wide PMU and
 * FC clock policy is intentionally left untouched.
 */
int camera_preview_config_init(camera_preview_config_t *config);

/* Convert an initialized config to FC JPEG so the cluster remains untouched. */
int camera_preview_config_use_fc_jpeg(camera_preview_config_t *config);

/* Select application-owned CPX; the caller must call cpxInit() exactly once. */
int camera_preview_config_use_existing_cpx(camera_preview_config_t *config);

/* Pure validation/planning; no allocation, task creation, CPX, or camera I/O. */
int camera_preview_memory_plan(const camera_preview_config_t *config,
                               camera_preview_memory_plan_t *plan);

/* Validate/copy only. The object must not be copied after this call. */
int camera_preview_init(camera_preview_t *preview,
                        const camera_preview_config_t *config);

/*
 * start() opens Himax before the large transport/JPEG consumers, optionally
 * initializes CPX, prepares Wi-Fi, creates one sender task, and starts capture.
 * Only one facade may run at a time.
 */
int camera_preview_start(camera_preview_t *preview);

/* One-call configuration + init + start using the exact default preset. */
int camera_preview_start_default(camera_preview_t *preview,
                                 camera_preview_transport_t transport);

/*
 * Stop the sender, close JPEG/owned cluster, and physically close the camera.
 * The call waits for an in-flight blocking imgstream send. CPX and the internal
 * Wi-Fi control receiver intentionally remain alive for the application.
 * Lifecycle calls must be serialized and close() must not run from the facade's
 * worker or a Wi-Fi event callback. A lower-layer BUSY/error is retryable.
 */
int camera_preview_close(camera_preview_t *preview);

int camera_preview_get_stats(const camera_preview_t *preview,
                             camera_preview_stats_t *stats);
bool camera_preview_is_running(const camera_preview_t *preview);

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_CAMERA_PREVIEW_H */
