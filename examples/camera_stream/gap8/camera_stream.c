// camera_stream -- the AI-deck camera as a WiFi video server.
//
// Captures the Himax camera, optionally resizes and JPEG-encodes each frame, and
// sends it to the connected host over CPX and WiFi in the cf_lab image-stream
// format (lib/shared/include/cf_lab/img_protocol.h). Settings and their defaults
// are in camera_stream_config.h; the build sets them with -D flags.
//
// Before the first frame of every connection the deck sends one console line on
// the same socket, "camera config: k=v ...", so the host knows every camera and
// stream setting of this build. With LOG=1 it also sends a "status k=v
// ..." line every second (lib/gap8/camera_stream_log). Boot progress and errors
// go to the radio console.

#include <stdint.h>
#include <string.h>

#include "pmsis.h"
#include "bsp/bsp.h"

#include "camera_pipeline.h"
#include "camera_stream.h"
#include "cpx.h"
#include "cpx_wifi_log.h"
#include "cpx_wifi_session.h"
#include "imgstream.h"
#include "jpeg_pipeline.h"
#if CAMERA_STREAM_LOG
#include "camera_stream_log.h"
#endif

#include "camera_stream_config.h"

// The camera settings are checked by lib/gap8/camera_stream itself.
#if OUT_WIDTH < 1 || OUT_HEIGHT < 1 || OUT_WIDTH > CAM_WIDTH || OUT_HEIGHT > CAM_HEIGHT
#error "OUT_WIDTH x OUT_HEIGHT must be at least 1x1 and at most the capture size"
#endif
#if TARGET_FPS < 1
#error "TARGET_FPS must be >= 1"
#endif
#if JPEG_QUALITY < 1 || JPEG_QUALITY > 100
#error "JPEG_QUALITY must be in [1, 100]"
#endif
#if defined(WINDOW_TOP) && (WINDOW_TOP < 0 || WINDOW_TOP > 84)
#error "WINDOW must be in [0, 84]"
#endif

#define OUT_PIXELS (OUT_WIDTH * OUT_HEIGHT)


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

#if CAMERA_STREAM_LOG
static camera_stream_log_t statusLog;
#endif

// Report a failure on the radio console and stop. It repeats, so a console that
// attaches later still learns why nothing is streaming.
static void fail(const char *what, int status) {
  while (1) {
    cpxPrintToConsole(LOG_TO_CRTP, "camera_stream: %s (status %d)\n", what, status);
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

// One line describing this build: the camera's settings as camera_stream runs
// them, then this app's. The host parses it (cf_lab.camera.ext) and records it
// with a run's telemetry.
static void printConfig(CPXConsoleTarget_t target) {
  static char camera[384];
  (void)camera_stream_describe(&cameraStream, camera, sizeof(camera));
  cpxPrintToConsole(
      target,
      "camera config: %s out=%dx%d fit=%s fps=%d format=%s jpeg=%s quality=%d "
      "transport=%s channel=%d log=%d\n",
      camera, OUT_WIDTH, OUT_HEIGHT,
      OUT_IS_NATIVE ? "native" : (OUT_FIT == FIT_CROP ? "crop" : "squash"), TARGET_FPS,
      STREAM_FORMAT == FORMAT_JPEG ? "jpeg" : "raw",
      JPEG_ENCODER == JPEG_ENCODER_CLUSTER ? "cluster" : "fc", JPEG_QUALITY,
      WIFI_TRANSPORT == CPX_WIFI_TRANSPORT_UDP ? "udp" : "tcp", WIFI_CHANNEL, CAMERA_STREAM_LOG);
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
#ifdef WINDOW_TOP
  config.window_top = WINDOW_TOP;
#endif

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
  config.hostname = WIFI_HOSTNAME;
  config.name = WIFI_NAME;
  config.client_settle_ms = STREAM_CLIENT_SETTLE_MS;
  // "WiFi up (a.b.c.d)" on the radio console: the deck's address, from DHCP in
  // station mode.
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

static void streamTask(void *parameters) {
  (void)parameters;
  const uint32_t periodBase = 1000 / TARGET_FPS;
  const uint32_t periodRemainder = 1000 % TARGET_FPS;
  uint32_t remainder = 0;
  uint32_t announcedSequence = UINT32_MAX;
  TickType_t nextFrame = xTaskGetTickCount();

  while (1) {
    camera_pipeline_frame_t frame;
    if (camera_pipeline_acquire(&framePipeline, &frame, CAMERA_STREAM_WAIT_FOREVER) != 0) {
      vTaskDelay(1);
      continue;
    }

    cpx_wifi_session_status_t wifi;
    if (!cpx_wifi_session_client_ready(&wifiSession) ||
        cpx_wifi_session_get_status(&wifiSession, &wifi) != CPX_WIFI_SESSION_OK) {
      (void)camera_pipeline_release(&framePipeline, &frame);
      vTaskDelay(20);
      continue;
    }
    // A new connection bumps the event sequence: describe the build to it first.
    if (wifi.event_sequence != announcedSequence) {
      printConfig(LOG_TO_WIFI);
      announcedSequence = wifi.event_sequence;
    }

    const uint8_t *data = frame.data;
    uint32_t size = OUT_PIXELS;
    uint16_t jpegMs = 0;
    int ok = 1;
#if STREAM_FORMAT == FORMAT_JPEG
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

#if CAPTURE_POLICY == CAPTURE_POLICY_EXACT
    // Exact frames go out whole: wait until the queue can take every packet.
    const uint32_t packets = imgstream_packets_for(size);
    while (ok && com_tx_free_slots() < packets) {
      vTaskDelay(1);
    }
#endif
    if (ok) {
      sendFrame(&frame, data, size, jpegMs);
    }
    (void)camera_pipeline_release(&framePipeline, &frame);

    // Pace to TARGET_FPS, carrying the sub-millisecond remainder.
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

#if CAMERA_STREAM_LOG
// Its own task, above the stream's priority, so a stalled stream still reports.
// Not a FreeRTOS timer: formatting the line takes about 650 bytes of stack,
// more than the timer task's 1 KB has to spare.
static void statusTask(void *parameters) {
  (void)parameters;
  while (1) {
    vTaskDelay(pdMS_TO_TICKS(CAMERA_STREAM_LOG_INTERVAL_MS));
    const camera_stream_log_progress_t progress = {
        .frames_sent = framesSent,
        .encode_failures = encodeFailures,
    };
    (void)camera_stream_log_emit(&statusLog, &progress);
  }
}
#endif

static void cameraTask(void *parameters) {
  (void)parameters;
  if (cameraOpenStatus != CAMERA_STREAM_OK) {
    if (cameraOpenStatus == CAMERA_STREAM_ERR_INVALID_ARGUMENT) {
      fail("camera settings rejected: WINDOW, an exposure time or a gain is out "
           "of range for this SENSOR", cameraOpenStatus);
    }
    failOpen("camera open failed", cameraOpenStatus, CAMERA_STREAM_ERR_NO_MEMORY);
  }
  printConfig(LOG_TO_CRTP);

  int status = cpx_wifi_session_connect(&wifiSession);
  if (status != CPX_WIFI_SESSION_OK) {
    fail("WiFi setup failed", status);
  }
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
                  tskIDLE_PRIORITY + 1, NULL) != pdPASS) {
    fail("stream task creation failed", 0);
  }
  status = camera_pipeline_start(&framePipeline);
  if (status == CAMERA_PIPELINE_OK) {
    status = camera_stream_start(&cameraStream);
  }
  if (status != CAMERA_STREAM_OK) {
    fail("camera start failed", status);
  }
#if CAMERA_STREAM_LOG
  // The task can wait for queue space, which a stream keeps nearly full.
  if (camera_stream_log_init(&statusLog, &cameraStream, CPX_T_WIFI_HOST,
                             STATUS_SEND_TIMEOUT_MS) != CAMERA_STREAM_OK ||
      xTaskCreate(statusTask, "status", configMINIMAL_STACK_SIZE * 4, NULL,
                  tskIDLE_PRIORITY + 3, NULL) != pdPASS) {
    fail("status log start failed", 0);
  }
#endif
  cpxPrintToConsole(LOG_TO_CRTP, "camera_stream: streaming\n");

  // The stream task takes each frame from the pipeline, which resizes it there.
  while (1) {
    vTaskDelay(1000);
  }
}

static void startTasks(void) {
  // Open the camera before CPX, so its driver and buffers get L2 first.
  cameraOpenStatus = openCamera();
  int status = openWiFi();
  cpxInit();
  imgstream_init(&imgStream, CPX_T_WIFI_HOST);
  if (status == CPX_WIFI_SESSION_OK) {
    status = cpx_wifi_session_open(&wifiSession);
  }
  if (status != CPX_WIFI_SESSION_OK) {
    cpxPrintToConsole(LOG_TO_CRTP, "camera_stream: WiFi session open failed (status %d)\n",
                      status);
    pmsis_exit(-1);
  }
  if (xTaskCreate(cameraTask, "camera", configMINIMAL_STACK_SIZE * 2, NULL,
                  tskIDLE_PRIORITY + 2, NULL) != pdPASS) {
    cpxPrintToConsole(LOG_TO_CRTP, "camera_stream: task creation failed\n");
    pmsis_exit(-1);
  }
  while (1) {
    pi_yield();
  }
}

int main(void) {
  pi_bsp_init();
  __pi_pmu_voltage_set(PI_PMU_DOMAIN_FC, 1200);
  // The cluster JPEG encoder runs the fabric controller at 150 MHz beside the
  // cluster; everything else runs it at 200 MHz.
  pi_freq_set(PI_FREQ_DOMAIN_FC,
              JPEG_ENCODER == JPEG_ENCODER_CLUSTER ? 150000000 : 200000000);
  return pmsis_kickoff((void *)startTasks);
}
