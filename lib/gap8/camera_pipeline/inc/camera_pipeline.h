#ifndef CF_GAP8_CAMERA_PIPELINE_H
#define CF_GAP8_CAMERA_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#include "camera_stream.h"
#include "downsample.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * camera_pipeline gives a caller frames at the size it asked for, above
 * camera_stream. It deliberately does not own a task, JPEG encoder, neural
 * network, transport, or logger: the caller's task (or scheduler) calls
 * acquire(), and a resized frame is resized right there, from the newest
 * capture the camera's policy allows. Nothing is resized that is not taken.
 */
#define CAMERA_PIPELINE_OBJECT_BYTES 320u

typedef enum {
  CAMERA_PIPELINE_OK = 0,
  CAMERA_PIPELINE_ERR_INVALID_ARGUMENT = -101,
  CAMERA_PIPELINE_ERR_INVALID_STATE = -102,
  CAMERA_PIPELINE_ERR_NO_MEMORY = -103,
  CAMERA_PIPELINE_ERR_RTOS = -104,
  CAMERA_PIPELINE_ERR_TIMEOUT = -105,
  CAMERA_PIPELINE_ERR_STOPPED = -106,
  CAMERA_PIPELINE_ERR_BUSY = -107,
  CAMERA_PIPELINE_ERR_STALE_FRAME = -108,
  CAMERA_PIPELINE_ERR_CAMERA = -110,
} camera_pipeline_status_t;

typedef struct {
  /* A camera_stream object that has already been initialized by the caller. */
  camera_stream_t *camera;

  /*
   * The output size. Zero, or the camera's own size, hands out the camera's
   * buffers without a copy. Any other size is resized with ds_resize_gray()
   * into one output buffer: fit DS_SQUASH keeps the whole view, DS_CROP keeps
   * the aspect ratio.
   */
  uint16_t output_width;
  uint16_t output_height;
  ds_mode_e fit;
} camera_pipeline_config_t;

typedef struct {
  uint8_t is_native;
  uint16_t output_width;
  uint16_t output_height;
  uint16_t output_stride;
  size_t output_frame_bytes;
  size_t output_l2_bytes; /* allocated by open(): zero for native frames */
} camera_pipeline_memory_plan_t;

typedef struct {
  uint8_t *data;
  size_t size_bytes;
  uint16_t width;
  uint16_t height;
  uint16_t stride;
  uint32_t capture_started_ms;
  uint32_t capture_completed_ms;
  uint32_t capture_duration_ms;
  uint32_t resize_duration_ms; /* zero for native frames */
  uint32_t camera_frame_id;

  /* Opaque lease fields. Preserve them unchanged until release(). */
  uintptr_t token;
  camera_stream_frame_t _source_frame;
  uint8_t _source_lease;
} camera_pipeline_frame_t;

typedef union {
  uint64_t _alignment;
  uint8_t _storage[CAMERA_PIPELINE_OBJECT_BYTES];
} camera_pipeline_t;

/*
 * Do not copy an object after init(). Serialize lifecycle calls. One task at a
 * time may acquire, and each acquired frame must be released before the next.
 */

void camera_pipeline_config_init(camera_pipeline_config_t *config);

/* init() validates and copies configuration; it allocates nothing. */
int camera_pipeline_init(camera_pipeline_t *pipeline,
                         const camera_pipeline_config_t *config);
int camera_pipeline_memory_plan(const camera_pipeline_config_t *config,
                                camera_pipeline_memory_plan_t *plan);

/*
 * The camera must be open before this allocates the output buffer of a resized
 * pipeline. Reconfiguring/reinitializing the referenced camera after pipeline
 * init is rejected; initialize the pipeline again from the new camera contract.
 */
int camera_pipeline_open(camera_pipeline_t *pipeline);
int camera_pipeline_start(camera_pipeline_t *pipeline);

/*
 * Waits up to timeout_ms for the camera's newest frame, then either
 * leases the camera buffer (native) or resizes it into the output buffer and
 * gives the camera buffer straight back. A resized frame is unchanged until
 * release(); a preview capture can tear (its one buffer keeps being written).
 */
int camera_pipeline_acquire(camera_pipeline_t *pipeline,
                            camera_pipeline_frame_t *frame,
                            uint32_t timeout_ms);
int camera_pipeline_release(camera_pipeline_t *pipeline,
                            camera_pipeline_frame_t *frame);

/*
 * These control handing out frames only; camera_stream start/stop remains
 * explicit. stop() and close() wake a blocked acquire() within 50 ms. close()
 * is idempotent after init and returns BUSY while a frame is still leased.
 */
int camera_pipeline_stop(camera_pipeline_t *pipeline);
int camera_pipeline_close(camera_pipeline_t *pipeline);

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_CAMERA_PIPELINE_H */
