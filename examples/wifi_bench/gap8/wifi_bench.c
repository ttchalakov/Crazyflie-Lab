/* Synthetic CPX traffic only. WIFI_* settings come from cf-lab --config. */
#include "pmsis.h"
#include "cpx.h"
#include "cpx_wifi_session.h"
#include <string.h>

#ifndef WIFI_TRANSPORT
#define WIFI_TRANSPORT 0
#endif
#ifndef WIFI_MODE
#define WIFI_MODE CPX_WIFI_MODE_ACCESS_POINT
#endif
#ifndef WIFI_SSID
#define WIFI_SSID "wifi-bench"
#endif
#ifndef WIFI_KEY
#define WIFI_KEY ""
#endif
#ifndef WIFI_CHANNEL
#define WIFI_CHANNEL 1
#endif
#define MAGIC 0x31424657u /* WFB1, little endian */
enum { PING=1, PONG, START, STOP, DATA, STATS, SPI_BAD };
typedef struct __attribute__((packed)) {
  uint32_t magic;
  uint8_t kind, version;
  uint16_t size;
  uint32_t session, sequence, tick_us, crc;
} Header;
typedef struct {
  uint32_t session, rate, size, until_ms, sequence, accepted, dropped;
  bool running;
} Stream;
static Stream stream;
static cpx_wifi_session_t wifi;
static CPXPacket_t incoming, reply, outgoing;
static uint32_t crc_table[256], received, invalid;

static uint32_t now_ms(void) { return xTaskGetTickCount() * portTICK_PERIOD_MS; }
static uint32_t checksum(const uint8_t *p, unsigned size) {
  uint32_t crc = ~0u;
  for (unsigned i=0; i<size; ++i)
    if (i<20 || i>=24) crc = crc_table[(crc ^ p[i]) & 255] ^ (crc >> 8);
  return ~crc;
}
static void prepare(CPXPacket_t *p, unsigned kind, uint32_t session,
                    uint32_t sequence, unsigned size) {
  cpxInitRoute(CPX_T_GAP8, CPX_T_WIFI_HOST, CPX_F_APP, &p->route);
  p->dataLength = size;
  Header h = {MAGIC, kind, 1, size, session, sequence, pi_time_get_us(), 0};
  memcpy(p->data, &h, sizeof(h));
}
static void seal(CPXPacket_t *p) {
  uint32_t crc=checksum(p->data, p->dataLength);
  memcpy(p->data+20, &crc, sizeof(crc));
}
static void stats(uint32_t session, uint32_t sequence) {
  uint32_t values[10];
  taskENTER_CRITICAL();
  values[0]=stream.sequence; values[1]=stream.accepted; values[2]=stream.dropped;
  values[3]=received; values[4]=invalid; values[5]=stream.session;
  values[6]=stream.running;
  taskEXIT_CRITICAL();
  values[7]=cpx_wifi_session_client_connected(&wifi);
  values[8]=WIFI_TRANSPORT; values[9]=WIFI_CHANNEL;
  prepare(&reply, STATS, session, sequence, sizeof(Header)+sizeof(values));
  memcpy(reply.data+sizeof(Header), values, sizeof(values));
  seal(&reply);
  cpxSendPacketUrgent(&reply, pdMS_TO_TICKS(100));
}
static void receive_task(void *unused) {
  (void)unused;
  while (1) {
    cpxReceivePacketBlocking(CPX_F_APP, &incoming);
    Header h;
    if (incoming.dataLength<sizeof(h)) { invalid++; continue; }
    memcpy(&h, incoming.data, sizeof(h));
    if (h.magic!=MAGIC || h.version!=1 || h.size!=incoming.dataLength ||
        h.crc!=checksum(incoming.data, incoming.dataLength)) { invalid++; continue; }
    received++;
    if (h.kind==PING) {
      memcpy(reply.data, incoming.data, incoming.dataLength);
      prepare(&reply, PONG, h.session, h.sequence, h.size);
      seal(&reply);
      cpxSendPacketUrgent(&reply, pdMS_TO_TICKS(100));
    } else if (h.kind==START && h.size==sizeof(h)+12) {
      uint32_t args[3];
      memcpy(args, incoming.data+sizeof(h), sizeof(args));
      if (args[0]<1 || args[0]>5000 || args[1]<sizeof(h) || args[1]>1020 ||
          args[2]<1 || args[2]>86400000) { invalid++; continue; }
      taskENTER_CRITICAL();
      stream=(Stream){.session=h.session, .rate=args[0], .size=args[1],
                      .until_ms=now_ms()+args[2], .running=true};
      taskEXIT_CRITICAL();
      stats(h.session, h.sequence);
    } else if (h.kind==SPI_BAD) {
      /* A one-byte route is invalid, but the transfer stays inside our buffer.
       * Deliberately use the public raw transport for this single fault case. */
      static packet_t bad;
      memset(&bad, 0, sizeof(bad));
      bad.len=1;
      com_write_timeout(&bad, pdMS_TO_TICKS(100));
      stats(h.session, h.sequence);
    } else if (h.kind==STOP || h.kind==STATS) {
      if (h.kind==STOP) {
        taskENTER_CRITICAL();
        stream.running=false;
        taskEXIT_CRITICAL();
      }
      stats(h.session, h.sequence);
    } else { invalid++; }
  }
}
static void bench(void) {
  pi_freq_set(PI_FREQ_DOMAIN_FC, 150000000);
  for (unsigned i=0; i<256; ++i) {
    uint32_t c=i;
    for (unsigned j=0; j<8; ++j) c=(c>>1)^((c&1)?0xedb88320u:0);
    crc_table[i]=c;
  }
  cpxInit();
  cpxEnableFunction(CPX_F_APP);
  cpx_wifi_session_config_t config;
  cpx_wifi_session_config_init(&config);
  config.ssid=WIFI_SSID;
  config.key=WIFI_KEY;
  config.mode=WIFI_MODE;
  config.transport=WIFI_TRANSPORT;
  config.channel=WIFI_CHANNEL;
  if (cpx_wifi_session_init(&wifi, &config)!=0 || cpx_wifi_session_open(&wifi)!=0)
    pmsis_exit(-1);
  if (xTaskCreate(receive_task, "bench rx", 1024, NULL, 2, NULL)!=pdPASS)
    pmsis_exit(-2);
  if (cpx_wifi_session_connect(&wifi)!=0) pmsis_exit(-3);
  uint32_t next=pi_time_get_us(), last_session=0;
  while (1) {
    Stream s;
    taskENTER_CRITICAL();
    if (stream.running && (int32_t)(now_ms()-stream.until_ms)>=0) stream.running=false;
    s=stream;
    taskEXIT_CRITICAL();
    if (!s.running || !cpx_wifi_session_client_ready(&wifi)) {
      vTaskDelay(pdMS_TO_TICKS(5)); next=pi_time_get_us(); continue;
    }
    if (s.session!=last_session) { next=pi_time_get_us(); last_session=s.session; }
    int32_t wait=(int32_t)(next-pi_time_get_us());
    if (wait>0) { pi_time_wait_us((uint32_t)wait); continue; }
    prepare(&outgoing, DATA, s.session, s.sequence, s.size);
    for (unsigned i=sizeof(Header); i<s.size; ++i) outgoing.data[i]=(uint8_t)(s.sequence+i);
    seal(&outgoing);
    bool ok=cpxSendPacket(&outgoing, 0);
    taskENTER_CRITICAL();
    if (stream.session==s.session) {
      stream.sequence++;
      if (ok) stream.accepted++; else stream.dropped++;
    }
    taskEXIT_CRITICAL();
    /* Pace start-to-start; skip missed deadlines instead of accumulating bursts. */
    next+=1000000u/s.rate;
    uint32_t now=pi_time_get_us();
    if ((int32_t)(now-next)>0) next=now;
  }
}
int main(void) { return pmsis_kickoff((void *)bench); }
