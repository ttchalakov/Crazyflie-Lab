/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * AI-deck GAP8
 *
 * Copyright (C) 2022 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#include "pmsis.h"
#include "cpx.h"
#include "com.h"
#include "cpx_config.h"

typedef struct {
  CPXTarget_t destination : 3;
  CPXTarget_t source : 3;
  bool lastPacket : 1;
  bool reserved : 1;
  CPXFunction_t function : 6;
  uint8_t version : 2;
} __attribute__((packed)) CPXRoutingPacked_t;

typedef struct {
    uint16_t wireLength;
    CPXRoutingPacked_t route;
    uint8_t data[MTU - CPX_HEADER_SIZE];
} __attribute__((packed)) CPXPacketPacked_t;

static xQueueHandle queues[CPX_F_LAST];

static CPXPacket_t rxp;
static CPXPacket_t txp;

static CPXPacketPacked_t rxpPacked;
static CPXPacketPacked_t txpPacked;

static SemaphoreHandle_t consoleSemaphore = NULL;
static SemaphoreHandle_t txSemaphore = NULL;

static bool packetToPacked(const CPXPacket_t *packet, CPXPacketPacked_t *packed) {
  if (packet == NULL || packed == NULL ||
      packet->dataLength > sizeof(packed->data) ||
      packet->route.function >= CPX_F_LAST ||
      packet->route.version != CPX_VERSION) {
    return false;
  }

  packed->wireLength = packet->dataLength + CPX_HEADER_SIZE;
  packed->route.destination = packet->route.destination;
  packed->route.source = packet->route.source;
  packed->route.function = packet->route.function;
  packed->route.version = packet->route.version;
  packed->route.lastPacket = packet->route.lastPacket;
  packed->route.reserved = 0;
  memcpy(packed->data, packet->data, packet->dataLength);
  return true;
}

static void cpx_rx_task(void *parameters) {
  while (1) {
    com_read((packet_t*)&rxpPacked);

    if (rxpPacked.wireLength < CPX_HEADER_SIZE || rxpPacked.wireLength > MTU ||
        rxpPacked.route.version != CPX_VERSION ||
        rxpPacked.route.function >= CPX_F_LAST) {
      continue;
    }

    rxp.route.version = rxpPacked.route.version;
    rxp.dataLength = rxpPacked.wireLength - CPX_HEADER_SIZE;

    rxp.route.destination = rxpPacked.route.destination;
    rxp.route.source = rxpPacked.route.source;
    rxp.route.function = rxpPacked.route.function;
    rxp.route.lastPacket = rxpPacked.route.lastPacket;
    memcpy(rxp.data, rxpPacked.data, rxp.dataLength);

    if (queues[rxp.route.function] != 0) {
      xQueueSend(queues[rxp.route.function], &rxp, 0);
    }
  }
}

void cpxEnableFunction(CPXFunction_t function) {
  if (function >= CPX_F_LAST) {
    return;
  }
  if (queues[function] != 0) {
    return;
  }
  queues[function] = xQueueCreate(CPX_FUNCTION_QUEUE_LENGTH, sizeof(CPXPacket_t));
  configASSERT(queues[function] != 0);
}

void cpxReceivePacketBlocking(CPXFunction_t function, CPXPacket_t * packet) {
  if (function >= CPX_F_LAST || queues[function] == 0) {
    memset(packet, 0, sizeof(*packet));
    return;
  }
  xQueueReceive(queues[function], packet, (TickType_t)portMAX_DELAY);
}

void cpxSendPacketBlocking(const CPXPacket_t * packet) {
  (void)cpxSendPacket(packet, (uint32_t)portMAX_DELAY);
}

// Pack into the shared txpPacked under txSemaphore, then hand it to one of the
// com layer's two tx queues.
static bool sendPacket(const CPXPacket_t * packet, uint32_t timeout,
                       bool (*write)(packet_t *, uint32_t)) {
  if (txSemaphore == NULL) {
    return false;
  }

  if (xSemaphoreTake(txSemaphore, (TickType_t)timeout) != pdTRUE) {
    return false;
  }

  const bool packed = packetToPacked(packet, &txpPacked);
  bool sent = false;
  if (packed) {
    sent = write((packet_t*)&txpPacked, timeout);
  }
  xSemaphoreGive(txSemaphore);
  return sent;
}

bool cpxSendPacket(const CPXPacket_t * packet, uint32_t timeout) {
  return sendPacket(packet, timeout, com_write_timeout);
}

bool cpxSendPacketUrgent(const CPXPacket_t * packet, uint32_t timeout) {
  return sendPacket(packet, timeout, com_write_urgent);
}

static CPXPacket_t consoleTx;
void cpxPrintToConsole(CPXConsoleTarget_t target, const char * fmt, ...) {
  if( xSemaphoreTake( consoleSemaphore, ( TickType_t )portMAX_DELAY) == pdTRUE )
  {
    va_list ap;
    int len;

    va_start(ap, fmt);
    len = vsnprintf((char*)consoleTx.data, sizeof(consoleTx.data), fmt, ap);
    va_end(ap);

    if (len < 0) {
      xSemaphoreGive( consoleSemaphore );
      return;
    }
    if (len >= (int)sizeof(consoleTx.data)) {
      len = sizeof(consoleTx.data) - 1;
      consoleTx.data[len] = '\0';
    }

    cpxInitRoute(CPX_T_GAP8, target, CPX_F_CONSOLE, &consoleTx.route);
    consoleTx.dataLength = len + 1;

    cpxSendPacketBlocking(&consoleTx);
    xSemaphoreGive( consoleSemaphore );
  }
}

void cpxInitRoute(const CPXTarget_t source, const CPXTarget_t destination, const CPXFunction_t function, CPXRouting_t* route) {
    route->source = source;
    route->destination = destination;
    route->function = function;
    route->version = CPX_VERSION;
    route->lastPacket = true;
}

void cpxInit(void) {

  com_init();

  memset(queues, 0, sizeof(queues));
  BaseType_t rxTask = xTaskCreate(cpx_rx_task, "rx_task", CPX_RX_TASK_STACK_WORDS,
                      NULL, CPX_TASK_PRIORITY, NULL);

  if (rxTask != pdPASS)
  {
    printf("Could not start router rx tasks!\n");
    pmsis_exit(-1);
  }
  consoleSemaphore = xSemaphoreCreateBinary();
  xSemaphoreGive( consoleSemaphore );
  txSemaphore = xSemaphoreCreateBinary();
  xSemaphoreGive( txSemaphore );
}
