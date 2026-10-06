// camera_bench -- camera_stream's pipeline, instrumented for the bench.
//
// Streams exactly as camera_stream does, from the same camera settings and the
// same wire format, so the host measures a camera_stream build. On top it:
//
//   - adds app=camera_bench (and its own settings) to the config line, which the
//     runner checks against the case it built before measuring anything;
//   - sends two status lines every second: the camera/transport counters from
//     lib/gap8/camera_stream_log, then "bench stage=... l2_free=..." with the
//     stream task's current stage, free L2 and each task's stack headroom (in
//     words, the least it has had);
//   - TRACE=1: a "trace" line for each of the first frames;
//   - LIFECYCLE=1: instead of streaming, runs the camera start/stop/resume/close
//     self-test at boot and reports the result to each client that connects.

#include <stdint.h>
#include <string.h>

#include "pmsis.h"
#include "bsp/bsp.h"
#include "pmsis/rtos/malloc/l2_malloc.h"

#include "camera_pipeline.h"
#include "camera_stream.h"
#include "camera_stream_log.h"
#include "cpx.h"
#include "cpx_wifi_log.h"
#include "cpx_wifi_session.h"
#include "imgstream.h"
#include "jpeg_pipeline.h"

#include "camera_bench_config.h"

#if OUT_WIDTH < 1 || OUT_HEIGHT < 1 || OUT_WIDTH > CAM_WIDTH || OUT_HEIGHT > CAM_HEIGHT
#error "OUT_WIDTH x OUT_HEIGHT must be at least 1x1 and at most the capture size"
#endif
#if TARGET_FPS < 1
#error "TARGET_FPS must be >= 1"
#endif

#define OUT_PIXELS (OUT_WIDTH * OUT_HEIGHT)


// Where the stream task is, for the status line: a stage that stops changing
// names the call that hung.
enum {
  STAGE_IDLE = 0,
  STAGE_ACQUIRE = 1,
  STAGE_WAIT_CLIENT = 2,
  STAGE_ENCODE = 3,
  STAGE_ADMIT = 4,
  STAGE_SEND = 5,
  STAGE_PACE = 6,
};

extern malloc_t __l2_malloc;

static camera_stream_t cameraStream;
static camera_pipeline_t framePipeline;
static cpx_wifi_session_t wifiSession;
static imgstream_t imgStream;
#if STREAM_FORMAT == FORMAT_JPEG
static jpeg_pipeline_t jpegPipeline;
#endif

static int cameraOpenStatus = CAMERA_STREAM_ERR_INVALID_STATE;
static uint32_t framesSent;
static uint32_t encodeFailures;
static volatile uint8_t stage;
static volatile uint32_t stageEnteredMs;
static uint32_t l2FreeMin = UINT32_MAX;
static TaskHandle_t streamTaskHandle;
static TaskHandle_t cameraTaskHandle;
static TaskHandle_t statusTaskHandle;

static camera_stream_log_t statusLog;
static CPXPacket_t benchPacket;

#if BENCH_LIFECYCLE
static char lifecycleResult[200];
#endif

static void setStage(uint8_t next) {
  stage = next;
  stageEnteredMs = xTaskGetTickCount();
}

static void fail(const char *what, int status) {
  while (1) {
    cpxPrintToConsole(LOG_TO_CRTP, "camera_bench: %s (status %d)\n", what, status);
    vTaskDelay(pdMS_TO_TICKS(10000));
  }
}

// An open that ran out of L2: say so, since a status number does not.
static void failOpen(const char *what, int status, int noMemory) {
  if (status == noMemory) {
    fail("out of L2 memory; this configuration needs too much (use fewer buffers, "
         "FC JPEG or a smaller SIZE)", status);
  }
  fail(what, status);
}

// camera_stream's config line with the bench's fields appended.
static void printConfig(CPXConsoleTarget_t target) {
  static char camera[384];
  (void)camera_stream_describe(&cameraStream, camera, sizeof(camera));
  cpxPrintToConsole(
      target,
      "camera config: %s out=%dx%d fit=%s fps=%d format=%s jpeg=%s quality=%d "
      "transport=%s channel=%d log=1 app=camera_bench lifecycle=%d trace=%d txq=%d\n",
      camera, OUT_WIDTH, OUT_HEIGHT,
      OUT_IS_NATIVE ? "native" : (OUT_FIT == FIT_CROP ? "crop" : "squash"), TARGET_FPS,
      STREAM_FORMAT == FORMAT_JPEG ? "jpeg" : "raw",
      JPEG_ENCODER == JPEG_ENCODER_CLUSTER ? "cluster" : "fc", JPEG_QUALITY,
      WIFI_TRANSPORT == CPX_WIFI_TRANSPORT_UDP ? "udp" : "tcp", WIFI_CHANNEL, BENCH_LIFECYCLE,
      BENCH_TRACE, TXQ_SIZE);
}

// Free L2 in total and in its largest block (what the next allocation can get).
static void l2Free(uint32_t *total, uint32_t *largest) {
  int32_t size = 0;
  int32_t chunks = 0;
  void *first = NULL;
  __malloc_info(&__l2_malloc, &size, &first, &chunks);
  *total = (uint32_t)size;
  *largest = 0;
  for (const malloc_chunk_t *chunk = __l2_malloc.first_free; chunk != NULL;
       chunk = chunk->next) {
    if ((uint32_t)chunk->size > *largest) {
      *largest = (uint32_t)chunk->size;
    }
  }
  if (*total < l2FreeMin) {
    l2FreeMin = *total;
  }
}

static uint32_t stackFreeWords(TaskHandle_t task) {
  return task == NULL ? 0 : (uint32_t)uxTaskGetStackHighWaterMark(task);
}

static void sendStatus(void) {
  const camera_stream_log_progress_t progress = {
      .frames_sent = framesSent,
      .encode_failures = encodeFailures,
  };
  (void)camera_stream_log_emit(&statusLog, &progress);

  const uint32_t now = xTaskGetTickCount();
  uint32_t free = 0;
  uint32_t largest = 0;
  l2Free(&free, &largest);
  const int length = snprintf(
      (char *)benchPacket.data, sizeof(benchPacket.data),
      "bench stage=%u stage_ms=%lu l2_free=%lu l2_free_min=%lu l2_largest=%lu "
      "stack_stream=%lu stack_camera=%lu stack_status=%lu\n",
      (unsigned int)stage, (unsigned long)(now - stageEnteredMs),
      (unsigned long)free, (unsigned long)l2FreeMin, (unsigned long)largest,
      (unsigned long)stackFreeWords(streamTaskHandle),
      (unsigned long)stackFreeWords(cameraTaskHandle),
      (unsigned long)stackFreeWords(statusTaskHandle));
  if (length > 0) {
    benchPacket.dataLength = (uint16_t)(length + 1);
    (void)cpxSendPacket(&benchPacket, STATUS_SEND_TIMEOUT_MS);
  }
}

// Its own task, above the stream's priority, so a stalled stream still reports.
static void statusTask(void *parameters) {
  (void)parameters;
  while (1) {
    vTaskDelay(pdMS_TO_TICKS(BENCH_STATUS_INTERVAL_MS));
    sendStatus();
  }
}

static int openCamera(void) {
  camera_stream_config_t config;
  int status = camera_stream_config_init(&config);
  if (status != CAMERA_STREAM_OK) {
    return status;
  }
  config.sensor_mode = (camera_stream_sensor_mode_t)SENSOR_MODE;
  config.policy = (camera_stream_policy_t)CAPTURE_POLICY;
  config.capture_buffer_count = RAW_CAPTURE_BUFFER_COUNT;

  status = camera_stream_init(&cameraStream, &config);
  if (status == CAMERA_STREAM_OK) {
    status = camera_stream_open(&cameraStream);
  }
  if (status != CAMERA_STREAM_OK) {
    return status;
  }

  camera_pipeline_config_t pipeline;
  camera_pipeline_config_init(&pipeline);
  pipeline.camera = &cameraStream;
  pipeline.output_width = OUT_WIDTH;
  pipeline.output_height = OUT_HEIGHT;
  pipeline.fit = OUT_FIT == FIT_CROP ? DS_CROP : DS_SQUASH;
  status = camera_pipeline_init(&framePipeline, &pipeline);
  if (status != CAMERA_PIPELINE_OK) {
    (void)camera_stream_close(&cameraStream);
  }
  return status;
}

static int openWiFi(void) {
  cpx_wifi_session_config_t config;
  int status = cpx_wifi_session_config_init(&config);
  if (status != CPX_WIFI_SESSION_OK) {
    return status;
  }
  config.ssid = WIFI_SSID;
  config.key = WIFI_KEY;
  config.mode = (cpx_wifi_mode_t)WIFI_MODE;
  config.transport = (cpx_wifi_transport_t)WIFI_TRANSPORT;
  config.channel = WIFI_CHANNEL;
  config.client_settle_ms = STREAM_CLIENT_SETTLE_MS;
  config.event_callback = cpx_wifi_log_event_callback;
  return cpx_wifi_session_init(&wifiSession, &config);
}

#if STREAM_FORMAT == FORMAT_JPEG
static int openJpeg(void) {
  jpeg_pipeline_config_t config;
  int status = jpeg_pipeline_config_init(&config);
  if (status != JPEG_PIPELINE_OK) {
    return status;
  }
  config.width = OUT_WIDTH;
  config.height = OUT_HEIGHT;
  config.stride = OUT_WIDTH;
  config.encoder.path = JPEG_ENCODER == JPEG_ENCODER_CLUSTER ? JPEG_PATH_CLUSTER : JPEG_PATH_FC;
  config.encoder.cores = JPEG_CLUSTER_CORES;
  config.encoder.quality = JPEG_QUALITY;
  config.encoder.restart_interval = 0;
  config.encoder.l1_policy = JPEG_CL_L1_RELEASE_ALL;
  config.output_budget_bytes = JPEG_OUT_BUDGET;
#if JPEG_ENCODER == JPEG_ENCODER_CLUSTER
  config.cluster_mode = JPEG_PIPELINE_CLUSTER_OWNED;
  config.owned_cluster_frequency_hz = 150000000;
#else
  config.cluster_mode = JPEG_PIPELINE_CLUSTER_NONE;
#endif
  status = jpeg_pipeline_init(&jpegPipeline, &config);
  if (status == JPEG_PIPELINE_OK) {
    status = jpeg_pipeline_open(&jpegPipeline);
  }
  return status;
}
#endif

static void sendFrame(const camera_pipeline_frame_t *frame, const uint8_t *data,
                      uint32_t size, uint16_t jpegMs) {
  camera_stream_stats_t stats;
  memset(&stats, 0, sizeof(stats));
  (void)camera_stream_get_stats(&cameraStream, &stats);
  const camera_stream_frame_ext_t ext = {
      .frame_id = framesSent,
      .timestamp_ms = frame->capture_started_ms,
      .encode_ms = (uint16_t)(frame->capture_duration_ms +
                              frame->resize_duration_ms + jpegMs),
      .capture_ms = (uint16_t)frame->capture_duration_ms,
      .preprocess_ms = (uint16_t)frame->resize_duration_ms,
      .jpeg_ms = jpegMs,
      .camera_frame_id = frame->camera_frame_id,
      .camera_done_ms = frame->capture_completed_ms,
      .dropped_capture_completions = stats.dropped_capture_completions,
      .dropped_ready_frames = stats.dropped_ready_frames,
      .dropped_while_streaming = stats.dropped_while_leased,
  };
  const imgstream_frame_t wire = {
      .payload = data,
      .size = size,
      .width = OUT_WIDTH,
      .height = OUT_HEIGHT,
      .type = STREAM_FORMAT == FORMAT_JPEG ? CF_IMG_TYPE_JPEG : CF_IMG_TYPE_RAW_GRAY,
      .ext = &ext,
      .ext_len = sizeof(ext),
  };
  imgstream_send(&imgStream, &wire);
  framesSent++;
}

// Describe the build to each new client. True while a client is ready.
static bool announceToClient(uint32_t *announcedSequence) {
  cpx_wifi_session_status_t wifi;
  if (!cpx_wifi_session_client_ready(&wifiSession) ||
      cpx_wifi_session_get_status(&wifiSession, &wifi) != CPX_WIFI_SESSION_OK) {
    return false;
  }
  if (wifi.event_sequence != *announcedSequence) {
    printConfig(LOG_TO_WIFI);
#if BENCH_LIFECYCLE
    cpxPrintToConsole(LOG_TO_WIFI, "%s", lifecycleResult);
#endif
    *announcedSequence = wifi.event_sequence;
  }
  return true;
}

static void streamTask(void *parameters) {
  (void)parameters;
  const uint32_t periodBase = 1000 / TARGET_FPS;
  const uint32_t periodRemainder = 1000 % TARGET_FPS;
  uint32_t remainder = 0;
  uint32_t announcedSequence = UINT32_MAX;
  TickType_t nextFrame = xTaskGetTickCount();

  while (1) {
    camera_pipeline_frame_t frame;
    setStage(STAGE_ACQUIRE);
    if (camera_pipeline_acquire(&framePipeline, &frame, CAMERA_STREAM_WAIT_FOREVER) != 0) {
      vTaskDelay(1);
      continue;
    }
    setStage(STAGE_WAIT_CLIENT);
    if (!announceToClient(&announcedSequence)) {
      (void)camera_pipeline_release(&framePipeline, &frame);
      vTaskDelay(20);
      continue;
    }

    const uint8_t *data = frame.data;
    uint32_t size = OUT_PIXELS;
    uint16_t jpegMs = 0;
    int ok = 1;
#if STREAM_FORMAT == FORMAT_JPEG
    setStage(STAGE_ENCODE);
    const uint32_t start = xTaskGetTickCount();
    jpeg_pipeline_output_t jpeg;
    const int status = jpeg_pipeline_encode(&jpegPipeline, frame.data, &jpeg);
    jpegMs = (uint16_t)(xTaskGetTickCount() - start);
    if (status == JPEG_PIPELINE_OK) {
      data = jpeg.data;
      size = (uint32_t)jpeg.size_bytes;
    } else {
      ok = 0;
      encodeFailures++;
    }
#endif

    setStage(STAGE_ADMIT);
    const uint32_t packets = imgstream_packets_for(size);
    uint32_t freeSlots = com_tx_free_slots();
#if CAPTURE_POLICY == CAPTURE_POLICY_EXACT
    while (ok && freeSlots < packets) {
      vTaskDelay(1);
      freeSlots = com_tx_free_slots();
    }
#endif
#if BENCH_TRACE
    if (framesSent < BENCH_TRACE_FRAMES) {
      cpxPrintToConsole(LOG_TO_WIFI,
                        "trace frame=%lu camera_frame=%lu ok=%d size=%lu packets=%lu "
                        "tx_free=%lu\n",
                        (unsigned long)framesSent, (unsigned long)frame.camera_frame_id,
                        ok, (unsigned long)size, (unsigned long)packets,
                        (unsigned long)freeSlots);
    }
#else
    (void)packets;
    (void)freeSlots;
#endif
    if (ok) {
      setStage(STAGE_SEND);
      sendFrame(&frame, data, size, jpegMs);
    }
    (void)camera_pipeline_release(&framePipeline, &frame);

    setStage(STAGE_PACE);
    uint32_t period = periodBase;
    remainder += periodRemainder;
    if (remainder >= TARGET_FPS) {
      period++;
      remainder -= TARGET_FPS;
    }
    // The schedule covers the whole loop, acquire (and so the resize)
    // included. A stream a period behind restarts it rather than burst.
    nextFrame += period;
    const TickType_t now = xTaskGetTickCount();
    if ((int32_t)(nextFrame - now) > 0) {
      vTaskDelay(nextFrame - now);
    } else if (now - nextFrame > period) {
      nextFrame = now;
    }
  }
}

#if BENCH_LIFECYCLE
static void lifecycleWait(uint32_t ms) { vTaskDelay(pdMS_TO_TICKS(ms)); }

// Start, take a frame, soft-stop, check captures pause, resume, take a frame,
// then hard-close. Uses only the public camera_stream API.
static void lifecycleTest(void) {
  camera_stream_frame_t frame;
  camera_stream_stats_t beforeStop;
  camera_stream_stats_t whileStopped;
  uint32_t firstId = 0;
  uint32_t resumedId = 0;
  memset(&beforeStop, 0, sizeof(beforeStop));
  memset(&whileStopped, 0, sizeof(whileStopped));

  int status = camera_stream_start(&cameraStream);
  if (status == CAMERA_STREAM_OK) {
    status = camera_stream_acquire(&cameraStream, &frame, 5000);
  }
  if (status == CAMERA_STREAM_OK) {
    firstId = frame.camera_frame_id;
    status = camera_stream_release(&cameraStream, &frame);
  }
  if (status == CAMERA_STREAM_OK) {
    status = camera_stream_get_stats(&cameraStream, &beforeStop);
  }
  if (status == CAMERA_STREAM_OK) {
    status = camera_stream_stop(&cameraStream);
  }
  if (status == CAMERA_STREAM_OK) {
    lifecycleWait(250);
    status = camera_stream_get_stats(&cameraStream, &whileStopped);
  }
  if (status == CAMERA_STREAM_OK) {
    status = camera_stream_start(&cameraStream);
  }
  if (status == CAMERA_STREAM_OK) {
    status = camera_stream_acquire(&cameraStream, &frame, 5000);
  }
  if (status == CAMERA_STREAM_OK) {
    resumedId = frame.camera_frame_id;
    status = camera_stream_release(&cameraStream, &frame);
  }
  // A soft stop keeps the sensor running but discards its frames, so captures
  // continue while stopped, all of them discarded, and the frame id moves on.
  const uint32_t stoppedCaptures =
      whileStopped.capture_completions - beforeStop.capture_completions;
  const uint32_t stoppedDiscards = whileStopped.paused_discards - beforeStop.paused_discards;
  const int softStopOk = status == CAMERA_STREAM_OK && stoppedCaptures != 0 &&
                         stoppedDiscards != 0 && resumedId != firstId;
  const int closeStatus = camera_stream_close(&cameraStream);
  const int passed = softStopOk && closeStatus == CAMERA_STREAM_OK;
  snprintf(lifecycleResult, sizeof(lifecycleResult),
           "lifecycle result=%s status=%d close=%d first_frame=%lu resumed_frame=%lu "
           "stopped_captures=%lu stopped_discards=%lu\n",
           passed ? "pass" : "fail", status, closeStatus, (unsigned long)firstId,
           (unsigned long)resumedId, (unsigned long)stoppedCaptures,
           (unsigned long)stoppedDiscards);
  cpxPrintToConsole(LOG_TO_CRTP, "camera_bench: %s", lifecycleResult);
}
#endif

static void cameraTask(void *parameters) {
  (void)parameters;
  if (cameraOpenStatus != CAMERA_STREAM_OK) {
    failOpen("camera open failed", cameraOpenStatus, CAMERA_STREAM_ERR_NO_MEMORY);
  }
  printConfig(LOG_TO_CRTP);

  int status = cpx_wifi_session_connect(&wifiSession);
  if (status != CPX_WIFI_SESSION_OK) {
    fail("WiFi setup failed", status);
  }
  cpxInitRoute(CPX_T_GAP8, CPX_T_WIFI_HOST, CPX_F_CONSOLE, &benchPacket.route);
  if (camera_stream_log_init(&statusLog, &cameraStream, CPX_T_WIFI_HOST,
                             STATUS_SEND_TIMEOUT_MS) != CAMERA_STREAM_OK ||
      xTaskCreate(statusTask, "status", configMINIMAL_STACK_SIZE * 4, NULL,
                  tskIDLE_PRIORITY + 3, &statusTaskHandle) != pdPASS) {
    fail("status task start failed", 0);
  }

#if BENCH_LIFECYCLE
  lifecycleTest();
  uint32_t announcedSequence = UINT32_MAX;
  while (1) {
    (void)announceToClient(&announcedSequence);
    vTaskDelay(pdMS_TO_TICKS(100));
  }
#else
  status = camera_pipeline_open(&framePipeline);
  if (status != CAMERA_PIPELINE_OK) {
    failOpen("frame pipeline open failed", status, CAMERA_PIPELINE_ERR_NO_MEMORY);
  }
#if STREAM_FORMAT == FORMAT_JPEG
  status = openJpeg();
  if (status != JPEG_PIPELINE_OK) {
    failOpen("JPEG encoder open failed", status, JPEG_PIPELINE_ERR_NO_MEMORY);
  }
#endif
  if (xTaskCreate(streamTask, "stream", configMINIMAL_STACK_SIZE * 4, NULL,
                  tskIDLE_PRIORITY + 1, &streamTaskHandle) != pdPASS) {
    fail("stream task creation failed", 0);
  }
  status = camera_pipeline_start(&framePipeline);
  if (status == CAMERA_PIPELINE_OK) {
    status = camera_stream_start(&cameraStream);
  }
  if (status != CAMERA_STREAM_OK) {
    fail("camera start failed", status);
  }
  cpxPrintToConsole(LOG_TO_CRTP, "camera_bench: streaming\n");

  // The stream task takes each frame from the pipeline, which resizes it there.
  while (1) {
    vTaskDelay(1000);
  }
#endif
}

static void startTasks(void) {
  cameraOpenStatus = openCamera();
  int status = openWiFi();
  cpxInit();
  imgstream_init(&imgStream, CPX_T_WIFI_HOST);
  if (status == CPX_WIFI_SESSION_OK) {
    status = cpx_wifi_session_open(&wifiSession);
  }
  if (status != CPX_WIFI_SESSION_OK) {
    cpxPrintToConsole(LOG_TO_CRTP, "camera_bench: WiFi session open failed (status %d)\n",
                      status);
    pmsis_exit(-1);
  }
  if (xTaskCreate(cameraTask, "camera", configMINIMAL_STACK_SIZE * 2, NULL,
                  tskIDLE_PRIORITY + 2, &cameraTaskHandle) != pdPASS) {
    cpxPrintToConsole(LOG_TO_CRTP, "camera_bench: task creation failed\n");
    pmsis_exit(-1);
  }
  while (1) {
    pi_yield();
  }
}

int main(void) {
  pi_bsp_init();
  __pi_pmu_voltage_set(PI_PMU_DOMAIN_FC, 1200);
  pi_freq_set(PI_FREQ_DOMAIN_FC,
              JPEG_ENCODER == JPEG_ENCODER_CLUSTER ? 150000000 : 200000000);
  return pmsis_kickoff((void *)startTasks);
}
