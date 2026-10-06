
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
 *
 *
 * com.c - SPI interface for ESP32 communication
 */

#include "pmsis.h"
#include "com.h"
#include "cpx_config.h"

#define max(a, b)               \
  (                             \
      {                         \
        __typeof__(a) _a = (a); \
        __typeof__(b) _b = (b); \
        _a > _b ? _a : _b;      \
      })

// Nina RTT (goes high when Nina is ready to talk)
#define CONFIG_NINA_GPIO_NINA_ACK 18
#define CONFIG_NINA_GPIO_NINA_ACK_PAD PI_PAD_32_A13_TIMER0_CH1
#define CONFIG_NINA_GPIO_NINA_ACK_PAD_FUNC PI_PAD_32_A13_GPIO_A18_FUNC1

// Structs for setting up interrut for the Nina RTT pin
static struct pi_gpio_conf cts_gpio_conf;
static pi_gpio_callback_t cb_gpio;

// GAP8 RTT (pull high when we want to talk)
#define CONFIG_NINA_GPIO_NINA_NOTIF 3
#define CONFIG_NINA_GPIO_NINA_NOTIF_PAD PI_PAD_15_B1_RF_PACTRL3
#define CONFIG_NINA_GPIO_NINA_NOTIF_PAD_FUNC PI_PAD_15_B1_GPIO_A3_FUNC1

#define GPIO_HIGH ((uint32_t)1)
#define GPIO_LOW ((uint32_t)0)

static pi_device_t spi_dev, nina_rtt_dev, gap8_rtt_dev;

// Queues for interacting with COM layer
static QueueHandle_t txq = NULL;
// Strict-priority tx queue drained ahead of txq. Latency-critical packets
// (e.g. a 100 Hz control reply) go here so they cannot sit behind an image
// that is queued as many MTU-sized bulk chunks in txq. com_task empties
// txqUrgent before taking any bulk packet and re-checks it between every
// packet, so the worst-case wait for an urgent packet is one already-committed
// bulk SPI transaction (an in-flight transfer cannot be preempted).
static QueueHandle_t txqUrgent = NULL;
static QueueHandle_t rxq = NULL;

static EventGroupHandle_t evGroup;
#define NINA_RTT_BIT (1 << 0)
#define TX_QUEUE_BIT (1 << 1)
#define SPI_XFER_DONE_BIT (1 << 2)

#define INITIAL_TRANSFER_SIZE (4)

// A 1 KB transfer at 10 MHz takes ~1 ms; this bounds scheduling jitter, not
// the wire. Hitting it means the transfer-complete event was lost, not that
// the transfer is slow.
#define SPI_XFER_TIMEOUT_MS (50)
#define SPI_XFER_MAX_ATTEMPTS (3)

// The ESP32 raises Nina RTT each time it arms one SPI slave transaction and
// drops it when that transaction ends. Bytes clocked while it is not armed are
// discarded, and neither side can tell. So the rising edges are counted here, in
// the interrupt itself, and com_task clocks only against an arming it has not
// used yet. NINA_RTT_BIT only wakes com_task. It reaches the event group through
// the FreeRTOS timer daemon, which shares the top priority with com_task and the
// application, so it can arrive after the arming it announces has been used.
// Neither that bit nor the line's level shows whether the ESP32 is armed now:
// the line stays high for a moment after each transaction, so com_task trusts
// the level only where none has just ended (at startup, after a recovery).
static volatile uint32_t ninaArmings;
static uint32_t ninaArmingsUsed;

static bool nina_armed(void)
{
  return ninaArmings != ninaArmingsUsed;
}

static void vDataReadyISR(void *args)
{
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  ninaArmings++;
  xEventGroupSetBitsFromISR(evGroup, NINA_RTT_BIT, &xHigherPriorityTaskWoken);
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

static void setup_nina_rtt_pin(pi_device_t *device)
{
  // Configure Nina RTT
  pi_gpio_conf_init(&cts_gpio_conf);
  pi_open_from_conf(device, &cts_gpio_conf);
  pi_gpio_open(device);
  pi_gpio_pin_configure(device, CONFIG_NINA_GPIO_NINA_ACK, PI_GPIO_INPUT);
  pi_gpio_pin_notif_configure(device, CONFIG_NINA_GPIO_NINA_ACK, PI_GPIO_NOTIF_RISE);
  pi_pad_set_function(CONFIG_NINA_GPIO_NINA_ACK_PAD, CONFIG_NINA_GPIO_NINA_ACK_PAD_FUNC);

  // Set up interrupt
  uint32_t gpio_mask = (1 << (CONFIG_NINA_GPIO_NINA_ACK & PI_GPIO_NUM_MASK));                   // create pin mask
  pi_gpio_callback_init(&cb_gpio, gpio_mask, vDataReadyISR, (void *)CONFIG_NINA_GPIO_NINA_ACK); // setup callback and handler
  pi_gpio_callback_add(device, &cb_gpio);                                                   // Attach callback to gpio pin
}

static void setup_gap8_rtt_pin(pi_device_t *device)
{
  // Set up GAP8 RTT
  struct pi_gpio_conf gpio_conf;
  pi_gpio_conf_init(&gpio_conf);
  pi_open_from_conf(device, &gpio_conf);
  pi_gpio_open(device);
  pi_gpio_pin_configure(device, CONFIG_NINA_GPIO_NINA_NOTIF, PI_GPIO_OUTPUT);
  pi_pad_set_function(CONFIG_NINA_GPIO_NINA_NOTIF_PAD, CONFIG_NINA_GPIO_NINA_NOTIF_PAD_FUNC);
}

static void set_gap8_rtt_pin(pi_device_t *device, uint32_t val)
{
  // Set the GAP8 RTT pin
  if (pi_gpio_pin_write(device, CONFIG_NINA_GPIO_NINA_NOTIF, val))
  {
    pmsis_exit(-1);
  }
}

static void init_spi(pi_device_t *device)
{
  struct pi_spi_conf spi_conf = {0};

  pi_spi_conf_init(&spi_conf);
  spi_conf.wordsize = PI_SPI_WORDSIZE_8;
  spi_conf.big_endian = 1;
  spi_conf.max_baudrate = 10000000;
  spi_conf.polarity = 0;
  spi_conf.phase = 0;
  spi_conf.itf = 1; // SPI1
  spi_conf.cs = 0;  // CS0

  pi_open_from_conf(device, &spi_conf);

  if (pi_spi_open(device))
  {
    pmsis_exit(-1);
  }
}

static packet_t rx_buff;
static packet_t tx_buff;

// Debug counters for localizing transport wedges: packets accepted into txq,
// packets actually clocked out over SPI, and packets received over SPI.
static volatile uint32_t comTxQueuedCount;
static volatile uint32_t comTxSentCount;
static volatile uint32_t comRxCount;

void com_debug_counters(uint32_t *txQueued, uint32_t *txSent, uint32_t *rxReceived)
{
  if (txQueued) *txQueued = comTxQueuedCount;
  if (txSent) *txSent = comTxSentCount;
  if (rxReceived) *rxReceived = comRxCount;
}

// SPI watchdog outcomes. "Soft" = the transfer finished on the wire but its
// completion event was lost (ESP RTT already low) and we carried on. "Hard" =
// the transfer was genuinely stuck and the SPI device was closed/reopened.
// "Dropped" = a transaction still failed after SPI_XFER_MAX_ATTEMPTS.
static volatile uint32_t comSpiSoftRecoveries;
static volatile uint32_t comSpiHardRecoveries;
static volatile uint32_t comSpiDroppedTransactions;

void com_spi_recovery_counters(uint32_t *soft, uint32_t *hard, uint32_t *dropped)
{
  if (soft) *soft = comSpiSoftRecoveries;
  if (hard) *hard = comSpiHardRecoveries;
  if (dropped) *dropped = comSpiDroppedTransactions;
}

// While held, com_task will not start a new SPI transaction (an in-flight one
// finishes normally), so the SPI uDMA stays silent. The ESP side simply stays
// armed until the hold is released, so nothing is lost - transfers are just
// deferred.
static volatile int comSpiHold;

// Set by com_task for the duration of one SPI transaction (claimed *before*
// re-checking comSpiHold, cleared after the last transfer). Together with the
// acquire loop below this forms a Dekker-style handshake: on the single-core
// FC, whichever side wins, a holder never observes "idle" while a transfer is
// (or is about to be) on the wire.
static volatile int comSpiTxnActive;

void com_spi_hold(int hold)
{
  comSpiHold = hold;
}

int com_spi_hold_acquire(uint32_t timeoutMs)
{
  comSpiHold = 1;
  const uint32_t deadline = xTaskGetTickCount() + pdMS_TO_TICKS(timeoutMs);
  while (comSpiTxnActive) {
    if ((int32_t)(deadline - xTaskGetTickCount()) <= 0) {
      // Bus would not go quiet; hold stays set, caller decides whether to
      // release or proceed regardless.
      return -1;
    }
    vTaskDelay(1);
  }
  return 0;
}

// Generation tag lets a late completion from a recovered transfer be told
// apart from the current one: a stale callback publishes a stale generation
// and the waiter keeps waiting instead of treating the new transfer as done.
static volatile uint32_t spiXferDoneGen;
static uint32_t spiXferGen;

static void spiXferDoneCb(void *arg)
{
  spiXferDoneGen = (uint32_t)(uintptr_t)arg;
  xEventGroupSetBits(evGroup, SPI_XFER_DONE_BIT);
}

// Bounded replacement for the SDK's blocking pi_spi_transfer, whose internal
// completion wait has no timeout: one lost completion event would park
// com_task forever mid-transaction and silently cut every GAP8<->ESP route
// (images, console, WiFi control). Returns 0 on completion, -1 on timeout.
static int spi_transfer_with_timeout(void *tx, void *rx, size_t lenBits, pi_spi_flags_e flags)
{
  // Rotate task structs so a timed-out transfer's late completion cannot
  // collide with the struct already re-enqueued for the next transfer.
  static pi_task_t xferTasks[4];

  spiXferGen++;
  const uint32_t gen = spiXferGen;
  pi_task_t *xferTask = &xferTasks[gen & 3];
  xEventGroupClearBits(evGroup, SPI_XFER_DONE_BIT);
  pi_spi_transfer_async(&spi_dev, tx, rx, lenBits, flags,
                        pi_task_callback(xferTask, spiXferDoneCb,
                                         (void *)(uintptr_t)gen));

  const uint32_t deadline = xTaskGetTickCount() + SPI_XFER_TIMEOUT_MS;
  while (spiXferDoneGen != gen) {
    const uint32_t now = xTaskGetTickCount();
    if ((int32_t)(deadline - now) <= 0) {
      return -1;
    }
    xEventGroupWaitBits(evGroup, SPI_XFER_DONE_BIT, pdTRUE, pdFALSE,
                        (TickType_t)(deadline - now));
  }
  return 0;
}

// Force the link out of a stuck transfer: drop our request line, close the
// SPI master (aborting the wedged uDMA transaction and releasing CS so the
// ESP slave side also completes), and reopen it fresh.
static void spi_link_recover(void)
{
  comSpiHardRecoveries++;
  set_gap8_rtt_pin(&gap8_rtt_dev, GPIO_LOW);
  pi_spi_close(&spi_dev);
  pi_time_wait_us(2000);
  init_spi(&spi_dev);
}

// Total packets waiting across both tx priority classes.
static uint32_t com_tx_pending(void)
{
  return uxQueueMessagesWaiting(txqUrgent) + uxQueueMessagesWaiting(txq);
}

// Dequeue the next tx packet, urgent class first. Returns false if both empty.
static bool com_tx_take(packet_t *out)
{
  if (xQueueReceive(txqUrgent, out, 0) == pdTRUE) {
    return true;
  }
  return xQueueReceive(txq, out, 0) == pdTRUE;
}

static void com_task(void *parameters)
{
  uint32_t startupESPRTTValue;

  pi_gpio_pin_read(&nina_rtt_dev, CONFIG_NINA_GPIO_NINA_ACK, &startupESPRTTValue);

  // An ESP32 that armed before com_init attached the edge interrupt shows only
  // as a high line; one that armed since is counted as well (the first
  // transaction uses up both). Either way it reads as an arming not used yet.
  // No transaction has run, so a high line is a live arming.
  if (startupESPRTTValue > 0) {
    ninaArmingsUsed = ninaArmings - 1;
  }

  while (1)
  {

    // Check if we have more to send, if not then wait until we have or Nina wants to send
    if (com_tx_pending() == 0) {
      // Wait for a packet to send, or for the ESP32 to arm a transaction (it
      // has a packet for us). A wake-up bit can be late or lost (the timer
      // daemon's queue can be full), so re-check every 10 ms.
      while (com_tx_pending() == 0 && !nina_armed()) {
        xEventGroupWaitBits(evGroup, NINA_RTT_BIT | TX_QUEUE_BIT, pdTRUE, pdFALSE,
                            pdMS_TO_TICKS(10));
      }
    }

    if (!com_tx_take(&tx_buff))
    {
      memset(&tx_buff, 0, sizeof(packet_t));
    }

    // Check if we have a package to send (all 0 otherwise)
    if (tx_buff.len > 0)
    {
      set_gap8_rtt_pin(&gap8_rtt_dev, GPIO_HIGH);
      // Wait for an arming not used yet (see vDataReadyISR), re-checking
      // every 10 ms in case the wake-up bit is late or lost.
      while (!nina_armed()) {
        xEventGroupWaitBits(evGroup, NINA_RTT_BIT, pdTRUE, pdFALSE, pdMS_TO_TICKS(10));
      }
    }
    if (nina_armed()) {
      // This transaction uses the ESP32's arming. It cannot arm again until
      // the transaction ends, so every edge counted so far is used up here.
      ninaArmingsUsed = ninaArmings;

      // Defer the whole transaction while the link is held (see
      // com_spi_hold). The claim is raised before each re-check of the hold
      // so com_spi_hold_acquire() can wait out a transaction we have already
      // committed to — checking the hold alone would race: the packet is
      // dequeued above, so the tx queue reads empty while the transfer is
      // still on the wire.
      comSpiTxnActive = 1;
      while (comSpiHold) {
        comSpiTxnActive = 0;
        vTaskDelay(1);
        comSpiTxnActive = 1;
      }

      int tx_len = 0;
      int xferOk = 0;
      for (int attempt = 0; attempt < SPI_XFER_MAX_ATTEMPTS && !xferOk; attempt++) {
        if (attempt > 0) {
          // Recovered mid-transaction: redo the RTT handshake. Raise our
          // request again when we have a packet, then wait for the ESP slave
          // to re-arm (its RTT high) before re-clocking.
          if (tx_buff.len > 0) {
            set_gap8_rtt_pin(&gap8_rtt_dev, GPIO_HIGH);
          }
          uint32_t ninaRtt = 0;
          const uint32_t armDeadline = xTaskGetTickCount() + 100;
          while ((int32_t)(armDeadline - xTaskGetTickCount()) > 0) {
            pi_gpio_pin_read(&nina_rtt_dev, CONFIG_NINA_GPIO_NINA_ACK, &ninaRtt);
            if (ninaRtt > 0) {
              break;
            }
            vTaskDelay(1);
          }
          if (ninaRtt == 0) {
            continue;
          }
          // The ESP32 is armed again, or still holds the arming the failed
          // transfer never reached: this attempt uses it.
          ninaArmingsUsed = ninaArmings;
        }

        if (spi_transfer_with_timeout(&tx_buff, &rx_buff,
                                      INITIAL_TRANSFER_SIZE * 8,
                                      PI_SPI_LINES_SINGLE | PI_SPI_CS_KEEP) != 0) {
          spi_link_recover();
          continue;
        }

        tx_len = tx_buff.len;
        // The ESP32 never sends more than MTU and the link has no CRC, so a
        // longer length is a corrupt header: clocking it would write past
        // rx_buff. Size the transfer by our own packet and drop theirs.
        int rx_len = rx_buff.len;
        if (rx_len > MTU) {
          rx_len = 0;
          rx_buff.len = 0;
        }

        int sizeLeft = max(tx_len - INITIAL_TRANSFER_SIZE + 2, rx_len - INITIAL_TRANSFER_SIZE + 2);

        // Set minumum size left, this works with 0 bytes as well
        sizeLeft = max(0, sizeLeft);

        // We only support transfers which are multiples of 4
        if ((sizeLeft % 4) > 0) {
          sizeLeft += (4-sizeLeft%4); // Pad upwards
        }

        // Protect against the case where the ESP might signal
        // on the RTT line that it wants to send, but actually has
        // no length. Calling the SPI transfer function with size = 0
        // will corrupt the following transaction. Sending random data
        // is ok, since the length is set to 0 and the ESP will ignore it.
        if (sizeLeft == 0) {
          sizeLeft = 4;
        }

        // Set GAP8 RTT low before we end the transfer
        set_gap8_rtt_pin(&gap8_rtt_dev, GPIO_LOW);

        // Transfer the remaining bytes
        if (spi_transfer_with_timeout(((uint8_t*)&tx_buff) + INITIAL_TRANSFER_SIZE,
                                      ((uint8_t*)&rx_buff) + INITIAL_TRANSFER_SIZE,
                                      sizeLeft * 8,
                                      PI_SPI_LINES_SINGLE | PI_SPI_CS_AUTO) != 0) {
          // The ESP slave holds its RTT high for the whole transaction and
          // drops it in its end-of-transaction hook, so a LOW level here
          // proves the payload finished on the wire and only our completion
          // event was lost: keep the data. HIGH is ambiguous (stuck, or
          // completed and already re-armed), so recover and retransmit; the
          // worst case is one duplicate packet, which beats a dead link.
          uint32_t ninaRtt = 0;
          pi_gpio_pin_read(&nina_rtt_dev, CONFIG_NINA_GPIO_NINA_ACK, &ninaRtt);
          if (ninaRtt == 0) {
            comSpiSoftRecoveries++;
          } else {
            spi_link_recover();
            continue;
          }
        }
        xferOk = 1;
      }
      comSpiTxnActive = 0;

      if (!xferOk)
      {
        // Transaction abandoned; do not forward a torn rx buffer.
        comSpiDroppedTransactions++;
        memset(&rx_buff, 0, sizeof(rx_buff));
      }
      if (xferOk && tx_len > 0)
      {
        comTxSentCount++;
      }
      if (xferOk && rx_buff.len > 0)
      {
        comRxCount++;
        xQueueSend(rxq, &rx_buff, (TickType_t)portMAX_DELAY);
      }

      // Do not wait for Nina RTT to go low, we trigger on rising edge anyway
    }
  }
}

void com_init()
{
  setup_gap8_rtt_pin(&gap8_rtt_dev);
  init_spi(&spi_dev);

  txq = xQueueCreate(TXQ_SIZE, sizeof(packet_t));
  txqUrgent = xQueueCreate(TXQ_URGENT_SIZE, sizeof(packet_t));
  rxq = xQueueCreate(RXQ_SIZE, sizeof(packet_t));

  if (txq == NULL || txqUrgent == NULL || rxq == NULL)
  {
    printf("Could not allocate txq and/or rxq in com\n");
    pmsis_exit(1);
  }

  evGroup = xEventGroupCreate();

  // Before com_task, which preempts us as soon as it is created: its startup
  // read needs the pin opened, and the edge interrupt already attached so
  // that no arming can fall between the two (see com_task).
  setup_nina_rtt_pin(&nina_rtt_dev);

  BaseType_t xTask;
  xTask = xTaskCreate(com_task, "com_task", CPX_COM_TASK_STACK_WORDS,
                      NULL, CPX_TASK_PRIORITY, NULL);
  if (xTask != pdPASS)
  {
    pmsis_exit(-1);
  }
}

void com_read(packet_t *p)
{
  xQueueReceive(rxq, p, (TickType_t)portMAX_DELAY);
}

bool com_write_timeout(packet_t *p, uint32_t timeout)
{
  if (xQueueSend(txq, p, (TickType_t)timeout) != pdPASS) {
    return false;
  }
  comTxQueuedCount++;
  xEventGroupSetBits(evGroup, TX_QUEUE_BIT);
  return true;
}

bool com_write_urgent(packet_t *p, uint32_t timeout)
{
  if (txqUrgent == NULL) {
    return false;
  }
  if (xQueueSend(txqUrgent, p, (TickType_t)timeout) != pdPASS) {
    return false;
  }
  comTxQueuedCount++;
  xEventGroupSetBits(evGroup, TX_QUEUE_BIT);
  return true;
}

uint32_t com_tx_free_slots(void)
{
  return (txq != NULL) ? (uint32_t)uxQueueSpacesAvailable(txq) : 0;
}
