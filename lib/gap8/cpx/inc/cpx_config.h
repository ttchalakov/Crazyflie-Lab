/*
 * cpx_config.h -- the CPX transport's sizes and knobs.
 *
 * TXQ_SIZE and CPX_TASK_PRIORITY are #ifndef-guarded, so an application sets
 * them from its own build, never by editing this file:
 *
 *     APP_CFLAGS += -DTXQ_SIZE=160               # one knob
 *     APP_CFLAGS += -include my_app_config.h     # a whole app config header
 *
 * A library never includes an app-named header: the app pushes its config in.
 */
#pragma once

/*
 * Outgoing packet queue depth, in packets (sizeof(packet_t), ~1 kB of L2 each).
 *
 * 80 is the stock value and suits a console/telemetry link. Anything shipping images
 * wants this sized against a whole frame: 160 lets a raw QVGA frame (77 packets)
 * be admitted while the previous one is still draining, rather than being dropped.
 * examples/camera_stream derives it from its capture policy and injects the result with
 * -include.
 */
#ifndef TXQ_SIZE
#define TXQ_SIZE (80)
#endif

#if TXQ_SIZE < 1
#error "TXQ_SIZE must be positive"
#endif

/*
 * The other two queues, also in packets. The urgent queue is drained ahead of
 * the outgoing one (cpxSendPacketUrgent) and only ever holds a handful of
 * one-at-a-time control replies; the receive queue holds what arrived from the
 * ESP32 until rx_task routes it. All three come out of L2 at cpxInit().
 */
#define TXQ_URGENT_SIZE (8)
#define RXQ_SIZE (5)

/* Each enabled function's receive queue (cpxEnableFunction), in packets. */
#define CPX_FUNCTION_QUEUE_LENGTH (2)

/* The transport's two tasks: com_task drives the SPI link, rx_task routes. */
#define CPX_COM_TASK_STACK_WORDS (configMINIMAL_STACK_SIZE * 6)
#define CPX_RX_TASK_STACK_WORDS (configMINIMAL_STACK_SIZE * 2)

/*
 * Priority of the transport's two tasks: com_task, which drives the SPI link to
 * the ESP32, and rx_task, which routes what arrives.
 *
 * The highest priority by default. Every wait in both tasks blocks (event groups,
 * queues, vTaskDelay), so they take the CPU only to move a packet. At the same
 * priority as the application's own tasks they could not preempt one: each wake
 * (the ESP32's ready line, a finished SPI transfer) would wait for the next 1 ms
 * tick whenever the application is busy, for example encoding JPEG on the FC,
 * capping the link at about one SPI step per millisecond. Link throughput is what
 * a raw camera stream or a log of every frame an onboard network sees is limited
 * by; an application has no need to set this.
 */
#ifndef CPX_TASK_PRIORITY
#define CPX_TASK_PRIORITY (configMAX_PRIORITIES - 1)
#endif
