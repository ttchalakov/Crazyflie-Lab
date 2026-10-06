#ifndef CF_GAP8_CAMERA_STREAM_LOG_H
#define CF_GAP8_CAMERA_STREAM_LOG_H

#include <stdint.h>

#include "camera_stream.h"
#include "cpx.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Optional status line for an app built on camera_stream: camera counters,
 * CPX/SPI transport counters, and two app counters, as one "status k=v ..."
 * console line. Send it to CPX_T_WIFI_HOST to put it on the image stream's
 * socket, where cf_lab records it beside the frames, or to CPX_T_STM32 for the
 * radio console.
 *
 * The logger allocates nothing and creates no task or timer: the app calls
 * camera_stream_log_emit() at its own cadence, typically from a small task of
 * its own. Formatting the line takes about 650 bytes of stack, too much for a
 * FreeRTOS timer callback: the timer task has 1 KB and needs some of it itself.
 */

/* App counters reported beside the camera's own. */
typedef struct {
  uint32_t frames_sent;
  /* Frames the app dropped because encoding produced no output. */
  uint32_t encode_failures;
} camera_stream_log_progress_t;

/* Caller-owned state. The persistent packet keeps an MTU-sized object off the
 * caller's stack. */
typedef struct {
  const camera_stream_t *camera_stream;
  uint32_t send_timeout_ms;
  CPXPacket_t packet;
} camera_stream_log_t;

typedef enum {
  CAMERA_STREAM_LOG_EMIT_QUEUED = 1,
  CAMERA_STREAM_LOG_EMIT_DROPPED = 2,
} camera_stream_log_emit_result_t;

/*
 * CPX itself must already be initialized by the app. send_timeout_ms is how long
 * emit waits for CPX queue space: 0 never waits, which suits a caller that must
 * not stall; a status task of its own can wait (100 ms is typical), which keeps
 * the line from being dropped while a stream fills the queue.
 */
int camera_stream_log_init(camera_stream_log_t *logger,
                           const camera_stream_t *camera_stream,
                           CPXTarget_t destination,
                           uint32_t send_timeout_ms);

/*
 * Send one status line; the caller sets the cadence.
 *
 * Waits at most send_timeout_ms for CPX queue space; a queue still full then
 * drops the line and returns CAMERA_STREAM_LOG_EMIT_DROPPED. Negative returns
 * are camera_stream status codes.
 */
int camera_stream_log_emit(camera_stream_log_t *logger,
                           const camera_stream_log_progress_t *progress);

#ifdef __cplusplus
}
#endif

#endif /* CF_GAP8_CAMERA_STREAM_LOG_H */
