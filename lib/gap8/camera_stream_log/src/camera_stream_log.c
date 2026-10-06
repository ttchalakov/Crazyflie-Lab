/* pmsis.h remaps snprintf onto the SDK's freestanding printf; newlib's stdio is
 * not linked into the image. Including it first is what resolves snprintf. */
#include "pmsis.h"

#include "camera_stream_log.h"

#include <stdio.h>
#include <string.h>

int camera_stream_log_init(camera_stream_log_t *logger,
                           const camera_stream_t *camera_stream,
                           CPXTarget_t destination,
                           uint32_t send_timeout_ms) {
  if (logger == NULL || camera_stream == NULL) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }

  memset(logger, 0, sizeof(*logger));
  logger->camera_stream = camera_stream;
  logger->send_timeout_ms = send_timeout_ms;
  cpxInitRoute(CPX_T_GAP8, destination, CPX_F_CONSOLE, &logger->packet.route);
  return CAMERA_STREAM_OK;
}

int camera_stream_log_emit(camera_stream_log_t *logger,
                           const camera_stream_log_progress_t *progress) {
  camera_stream_info_t info;
  camera_stream_stats_t stats;
  camera_stream_exposure_t exposure;
  uint32_t tx_queued = 0;
  uint32_t tx_sent = 0;
  uint32_t rx_received = 0;
  uint32_t spi_soft = 0;
  uint32_t spi_hard = 0;
  uint32_t spi_dropped = 0;
  int length;
  int status;

  if (logger == NULL || logger->camera_stream == NULL || progress == NULL) {
    return CAMERA_STREAM_ERR_INVALID_ARGUMENT;
  }
  status = camera_stream_get_info(logger->camera_stream, &info);
  if (status == CAMERA_STREAM_OK) {
    status = camera_stream_get_stats(logger->camera_stream, &stats);
  }
  if (status != CAMERA_STREAM_OK) {
    return status;
  }
  if (camera_stream_get_exposure(logger->camera_stream, &exposure) !=
      CAMERA_STREAM_OK) {
    memset(&exposure, 0, sizeof(exposure));
  }
  com_debug_counters(&tx_queued, &tx_sent, &rx_received);
  com_spi_recovery_counters(&spi_soft, &spi_hard, &spi_dropped);

  length = snprintf(
      (char *)logger->packet.data, sizeof(logger->packet.data),
      "status frames=%lu encode_failures=%lu captures=%lu "
      "dropped_capture=%lu dropped_ready=%lu dropped_leased=%lu "
      "watchdog_restarts=%lu faulted=%u tx_free=%lu tx_queued=%lu tx_sent=%lu "
      "spi_soft=%lu spi_hard=%lu spi_dropped=%lu "
      "exposure_us=%lu analog_gain=%u digital_gain=%u\n",
      (unsigned long)progress->frames_sent,
      (unsigned long)progress->encode_failures,
      (unsigned long)stats.capture_completions,
      (unsigned long)stats.dropped_capture_completions,
      (unsigned long)stats.dropped_ready_frames,
      (unsigned long)stats.dropped_while_leased,
      (unsigned long)stats.watchdog_restarts,
      (unsigned int)info.is_faulted,
      (unsigned long)com_tx_free_slots(),
      (unsigned long)tx_queued,
      (unsigned long)tx_sent,
      (unsigned long)spi_soft,
      (unsigned long)spi_hard,
      (unsigned long)spi_dropped,
      (unsigned long)exposure.exposure_us, 1u << exposure.analog_gain,
      (unsigned int)exposure.digital_gain);
  if (length <= 0) {
    return CAMERA_STREAM_ERR_INVALID_STATE;
  }
  if (length >= (int)sizeof(logger->packet.data)) {
    length = (int)sizeof(logger->packet.data) - 1;
  }
  logger->packet.dataLength = (uint16_t)(length + 1);

  return cpxSendPacket(&logger->packet, logger->send_timeout_ms)
             ? CAMERA_STREAM_LOG_EMIT_QUEUED
             : CAMERA_STREAM_LOG_EMIT_DROPPED;
}
