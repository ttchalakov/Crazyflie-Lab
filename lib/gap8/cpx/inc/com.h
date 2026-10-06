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
#pragma once
#include <stdint.h>
#include <stdbool.h>

#define MTU (1022)

typedef struct
{
  uint16_t len; // Of data (max 1022)
  uint8_t data[MTU];
} __attribute__((packed)) packet_t;

/* Initialize the communication */
void com_init();

void com_read(packet_t * p);

bool com_write_timeout(packet_t * p, uint32_t timeout);

// Enqueue on the strict-priority tx class, drained ahead of com_write_timeout()
// bulk traffic. For latency-critical packets (e.g. a 100 Hz
// control reply) that must not sit behind a multi-chunk image already in the
// bulk queue. Worst-case wait is one already-committed bulk SPI transaction.
bool com_write_urgent(packet_t * p, uint32_t timeout);

// Free slots in the outgoing (tx) queue. Lets a producer check there is room for
// a whole message before sending, instead of blocking forever when it is full.
uint32_t com_tx_free_slots(void);

// Transport debug counters: packets accepted into the tx queue, packets clocked
// out over SPI, and packets received over SPI. Diverging queued/sent values
// mean the SPI link itself stalled rather than the producer.
void com_debug_counters(uint32_t *txQueued, uint32_t *txSent, uint32_t *rxReceived);

// SPI watchdog outcomes: transfers whose completion event was lost but had
// finished on the wire (soft), stuck transfers that forced an SPI
// close/reopen (hard), and transactions abandoned after all retries.
void com_spi_recovery_counters(uint32_t *soft, uint32_t *hard, uint32_t *dropped);

// While held (nonzero), com_task defers starting SPI transactions; packets
// queue up and the ESP stays armed, so the SPI uDMA stays quiet. The holder
// must release it: the GAP8's console and telemetry wait until it does.
void com_spi_hold(int hold);

// Take the hold AND wait out any SPI transaction already in flight (com_task
// dequeues a packet before clocking it, so an empty tx queue does not mean
// the bus is idle). Returns 0 once the bus is provably quiet; -1 if it stayed
// busy past the timeout — the hold is left set either way and the caller must
// release it with com_spi_hold(0).
int com_spi_hold_acquire(uint32_t timeoutMs);

