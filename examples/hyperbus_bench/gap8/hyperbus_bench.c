// hyperbus_bench: the AI-deck's HyperMem (Infineon S71KS512SC0B: a 512 Mbit
// HyperFlash and a 64 Mbit HyperRAM in one package, two chip selects on one
// HyperBus), checked from the GAP8. See README.md.
//
//   1. HyperFlash raw reads. The radio flasher writes the image at 0x40000, so
//      the reads at 0x40000 and 0x48000 must return the image's own bytes; the
//      host compares them. 0x0 holds the bootloader.
//   2. The flash filesystem (readfs): pi_fs_mount(), as DORY's mem_init() calls
//      it, and a read-back of hb_probe.txt.
//   3. HyperRAM: fixed and walking patterns written and read back.
//   4. The L3 weight load (HB_LOAD_KB > 0): readfs/hb_load.bin copied into HyperRAM
//      exactly as our DORY fork's load_file_to_ram() does it (pi_fs_direct_read into
//      an L2 heap buffer, then pi_ram_write, chunk by chunk), then read back and
//      checked. The file is a xorshift32 stream from HB_LOAD_SEED, so the GAP8
//      regenerates every expected byte itself. HB_PLAN lists the steps,
//      "chunk:spi:fc_mhz,...": spi is quiet (CPX SPI held during the load), burst
//      (~25 console lines queued just before it) or chatter (a task printing
//      throughout). Each step opens the flash, readfs and HyperRAM as mem_init()
//      does, and closes them after. A load that stops making progress is reported
//      as STUCK by a watchdog task.
//   5. The tiled layer (HB_TILE_PLAN, with the load): DORY's L3 weight tiling
//      over the loaded file; see run_tile_plan().
//
// gapy can end the image with a zero-length area. A zero-length HyperBus transfer
// leaves the read address unprogrammed, so every later pi_flash_read() ignores its
// address and the mount fails. The flasher drops such areas (repair_gap8_image()
// in src/cf_lab/aideck/flash.py).
//
// The report is printed once ("HB: ") and then again forever ("HB| "), so a
// console that attaches late still gets all of it. Load lines are printed once.

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "pmsis.h"
#include "bsp/bsp.h"
#include "bsp/flash.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/ram.h"
#include "bsp/ram/hyperram.h"
#include "cpx.h"

#include "com.h"

#ifndef HB_FC_MHZ
#define HB_FC_MHZ 150  // nn_bench's
#endif
#ifndef HB_LOAD_KB
#define HB_LOAD_KB 0
#endif
#ifndef HB_LOAD_SEED
#define HB_LOAD_SEED 0x12345678u
#endif
#ifndef HB_LOAD_ITERS
#define HB_LOAD_ITERS 5
#endif
#ifndef HB_PLAN
#define HB_PLAN "128:burst:150"
#endif
#define HB_MAX_CHUNK 4096

static struct pi_device flash;
static struct pi_hyperflash_conf flash_conf;
static struct pi_device fs;
static struct pi_readfs_conf fs_conf;
static struct pi_device ram;
static struct pi_hyperram_conf ram_conf;

// The report: filled once, re-emitted forever.
#define MAX_LINES 40
#define LINE_LEN 96
static char g_lines[MAX_LINES][LINE_LEN];
static int g_count = 0;

#define BUF 2048
static uint8_t wbuf[BUF];
static uint8_t rbuf[BUF];

static void emit(const char *fmt, ...) {
  int slot = (g_count < MAX_LINES) ? g_count : (MAX_LINES - 1);
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_lines[slot], LINE_LEN, fmt, ap);
  va_end(ap);
  if (g_count < MAX_LINES) g_count++;
  cpxPrintToConsole(LOG_TO_CRTP, "HB: %s\n", g_lines[slot]);
  pi_time_wait_us(70000);  // let the console drain
}

static void hex16(const uint8_t *b, char *out, int outlen) {
  int n = 0;
  for (int i = 0; i < 16 && n < outlen - 3; i++) {
    n += snprintf(out + n, outlen - n, "%02x ", b[i]);
  }
}

static void test_hyperram(void) {
  emit("== TEST HyperRAM (S27KS0641, CS=MBR0) ==");
  pi_hyperram_conf_init(&ram_conf);
  pi_open_from_conf(&ram, &ram_conf);
  if (pi_ram_open(&ram)) {
    emit("RAM open: FAIL");
    return;
  }
  emit("RAM open: OK");

  uint32_t base = 0;
  if (pi_ram_alloc(&ram, &base, (uint32_t)(1 << 20))) {
    emit("RAM alloc 1MB: FAIL");
    pi_ram_close(&ram);
    return;
  }
  emit("RAM alloc 1MB @0x%06lx: OK", (unsigned long)base);

  const uint8_t pats[4] = {0x00u, 0xFFu, 0x55u, 0xAAu};
  for (int p = 0; p < 4; p++) {
    memset(wbuf, pats[p], BUF);
    memset(rbuf, (uint8_t)~pats[p], BUF);
    pi_ram_write(&ram, base, wbuf, BUF);
    pi_ram_read(&ram, base, rbuf, BUF);
    int bad = 0, first = -1;
    for (int i = 0; i < BUF; i++) {
      if (rbuf[i] != pats[p]) { bad++; if (first < 0) first = i; }
    }
    emit("RAM pat=0x%02x: %s (%d/%d bad, first@%d rd=0x%02x)", pats[p],
         bad ? "FAIL" : "OK", bad, BUF, first,
         first >= 0 ? rbuf[first] : 0);
  }

  // An address-dependent pattern catches stuck or shifted bytes.
  for (int i = 0; i < BUF; i++) wbuf[i] = (uint8_t)(i * 7 + 3);
  memset(rbuf, 0, BUF);
  pi_ram_write(&ram, base, wbuf, BUF);
  pi_ram_read(&ram, base, rbuf, BUF);
  int bad = 0, first = -1;
  for (int i = 0; i < BUF; i++) {
    if (rbuf[i] != wbuf[i]) { bad++; if (first < 0) first = i; }
  }
  emit("RAM walking: %s (%d/%d bad, first@%d exp=0x%02x rd=0x%02x)",
       bad ? "FAIL" : "OK", bad, BUF, first, first >= 0 ? wbuf[first] : 0,
       first >= 0 ? rbuf[first] : 0);

  pi_ram_free(&ram, base, (uint32_t)(1 << 20));
  pi_ram_close(&ram);
}

static void test_hyperflash_read(void) {
  emit("== TEST HyperFlash (S26KS512S, CS=MBR1) ==");
  pi_hyperflash_conf_init(&flash_conf);
  pi_open_from_conf(&flash, &flash_conf);
  if (pi_flash_open(&flash)) {
    emit("FLASH open: FAIL");
    return;
  }
  emit("FLASH open: OK");

  const uint32_t offs[3] = {0x0u, 0x40000u, 0x48000u};
  for (int k = 0; k < 3; k++) {
    memset(rbuf, 0xDE, 16);
    pi_flash_read(&flash, offs[k], rbuf, 16);
    char hx[56];
    hex16(rbuf, hx, sizeof(hx));
    emit("FLASH @0x%06lx: %s", (unsigned long)offs[k], hx);
  }
}

// Uses the flash device test_hyperflash_read() opened.
static void test_readfs_mount(void) {
  emit("== TEST readfs mount (FS_PARTITIONTABLE_OFFSET=0x%x) ==",
       (unsigned)FS_PARTITIONTABLE_OFFSET);
  pi_readfs_conf_init(&fs_conf);
  fs_conf.fs.flash = &flash;
  pi_open_from_conf(&fs, &fs_conf);
  if (pi_fs_mount(&fs)) {
    emit("MOUNT: FAIL (pi_fs_mount != 0)");
    return;
  }
  emit("MOUNT: OK");

  pi_fs_file_t *f = pi_fs_open(&fs, "hb_probe.txt", 0);
  if (f == NULL) {
    emit("OPEN hb_probe.txt: FAIL");
    return;
  }
  emit("OPEN hb_probe.txt: OK (size=%u)", (unsigned)f->size);
  int n = pi_fs_direct_read(f, rbuf, 32);
  rbuf[n > 0 ? (n < BUF ? n : BUF - 1) : 0] = '\0';
  emit("READ 32B: n=%d '%.*s'", n, n > 0 ? (n < 40 ? n : 40) : 0, (char *)rbuf);
  pi_fs_close(f);
}

// ---------------------------------------------------------------------------
// The L3 weight load.
// ---------------------------------------------------------------------------
static inline uint32_t xorshift32(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

static uint8_t load_buf[HB_MAX_CHUNK];  // verify()'s
static uint32_t expect_buf[HB_MAX_CHUNK / 4];
static volatile int chatter_on;

static void chatter_task(void *arg) {
  (void)arg;
  int i = 0;
  while (chatter_on) {
    cpxPrintToConsole(LOG_TO_CRTP, "HB~ %d\n", i++);
    pi_time_wait_us(2000);
  }
  vTaskDelete(NULL);
}

// The load's progress, for the watchdog: a stuck load cannot report itself.
static volatile uint32_t g_load_offset;
static volatile int g_load_step = -1, g_load_it;

// Our DORY fork's load_file_to_ram(), with a guard against a zero-length read.
// pi_fs_direct_read(), never pi_fs_read(): readfs's pi_fs_read() races the PMSIS
// event kernel and hangs or lands a stale chunk.
static int load_file(uint32_t base, uint32_t chunk) {
  pi_fs_file_t *f = pi_fs_open(&fs, "hb_load.bin", 0);
  if (f == NULL) return -1;
  uint8_t *buf = pi_l2_malloc(chunk);
  if (buf == NULL) {
    pi_fs_close(f);
    return -3;
  }
  const uint32_t size = f->size;
  uint32_t offset = 0;
  do {
    g_load_offset = offset;
    const int n = pi_fs_direct_read(f, buf, chunk);
    if (n <= 0) break;
    pi_ram_write(&ram, base + offset, buf, n);
    offset += n;
  } while (offset < size);
  pi_l2_free(buf, chunk);
  pi_fs_close(f);
  return offset == size ? 0 : -2;
}

typedef struct {
  int bad_chunks;   // chunks whose HyperRAM copy is wrong
  int flash_bad;    // of those, chunks whose flash copy is also wrong
  int first_bad;    // byte offset of the first wrong chunk, or -1
} verify_t;

// Read the HyperRAM copy back against the xorshift stream; recheck the flash copy
// of any wrong chunk, which tells a bad flash read from a bad HyperRAM write/read.
static verify_t verify(uint32_t base, uint32_t size, uint32_t chunk) {
  verify_t v = {0, 0, -1};
  uint32_t state = HB_LOAD_SEED;
  for (uint32_t offset = 0; offset < size; offset += chunk) {
    const uint32_t n = (size - offset < chunk) ? size - offset : chunk;
    for (uint32_t w = 0; w < n / 4; w++) expect_buf[w] = xorshift32(&state);
    pi_ram_read(&ram, base + offset, load_buf, n);
    if (memcmp(load_buf, expect_buf, n) == 0) continue;
    v.bad_chunks++;
    if (v.first_bad < 0) v.first_bad = (int)offset;
    pi_fs_file_t *f = pi_fs_open(&fs, "hb_load.bin", 0);
    if (f != NULL) {
      pi_fs_seek(f, offset);
      if (pi_fs_direct_read(f, load_buf, n) != (int)n || memcmp(load_buf, expect_buf, n) != 0) v.flash_bad++;
      pi_fs_close(f);
    }
  }
  return v;
}

typedef struct {
  uint32_t chunk;
  char spi[8];
  uint32_t fc_mhz;
} step_t;

// The SDK's libc has no strtoul.
static uint32_t parse_uint(const char **p) {
  uint32_t v = 0;
  while (**p >= '0' && **p <= '9') v = v * 10 + (uint32_t)(*(*p)++ - '0');
  return v;
}

static void parse_word(const char **p, char *out, int size) {
  int k = 0;
  while (**p && **p != ':' && **p != ',' && k < size - 1) out[k++] = *(*p)++;
  out[k] = '\0';
}

// "128:burst:150,4096:quiet:50" -> steps; returns the count.
static int parse_plan(const char *plan, step_t *steps, int max) {
  int n = 0;
  const char *p = plan;
  while (*p && n < max) {
    step_t *s = &steps[n];
    s->chunk = parse_uint(&p);
    if (*p != ':') break;
    p++;
    parse_word(&p, s->spi, sizeof s->spi);
    if (*p != ':') break;
    p++;
    s->fc_mhz = parse_uint(&p);
    if (s->chunk == 0 || s->chunk > HB_MAX_CHUNK || s->chunk % 4) break;
    n++;
    if (*p == ',') p++;
  }
  return n;
}

static int open_devices(void) {
  pi_hyperflash_conf_init(&flash_conf);
  pi_open_from_conf(&flash, &flash_conf);
  if (pi_flash_open(&flash)) return -1;
  pi_readfs_conf_init(&fs_conf);
  fs_conf.fs.flash = &flash;
  pi_open_from_conf(&fs, &fs_conf);
  if (pi_fs_mount(&fs)) return -2;
  pi_hyperram_conf_init(&ram_conf);
  pi_open_from_conf(&ram, &ram_conf);
  if (pi_ram_open(&ram)) return -3;
  return 0;
}

static void close_devices(void) {
  pi_ram_close(&ram);
  pi_fs_unmount(&fs);
  pi_flash_close(&flash);
}

static void run_load_plan(void) {
  static step_t steps[16];
  const int count = parse_plan(HB_PLAN, steps, 16);
  emit("== TEST L3 load: %d KB, %d steps x %d, plan %.40s ==", HB_LOAD_KB, count, HB_LOAD_ITERS, HB_PLAN);
  for (int s = 0; s < count; s++) {
    const step_t *st = &steps[s];
    pi_freq_set(PI_FREQ_DOMAIN_FC, st->fc_mhz * 1000000u);
    const int err = open_devices();
    uint32_t base = 0;
    if (err || pi_ram_alloc(&ram, &base, HB_LOAD_KB * 1024u)) {
      emit("LOAD step=%d open: FAIL (%d)", s, err);
      if (!err) close_devices();
      continue;
    }
    int total_bad = 0, done = 0;
    for (int it = 0; it < HB_LOAD_ITERS; it++) {
      if (!strcmp(st->spi, "burst")) {
        for (int i = 0; i < 25; i++) {
          cpxPrintToConsole(LOG_TO_CRTP, "HB~ burst %d of 25 before the load\n", i);
          pi_time_wait_us(3000);
        }
      } else if (!strcmp(st->spi, "chatter")) {
        chatter_on = 1;
        xTaskCreate(chatter_task, "hb_chatter", configMINIMAL_STACK_SIZE * 2, NULL, tskIDLE_PRIORITY + 1, NULL);
      } else if (!strcmp(st->spi, "quiet")) {
        (void)com_spi_hold_acquire(100);
      }
      g_load_step = s;
      g_load_it = it;
      const uint32_t t0 = pi_time_get_us();
      const int load = load_file(base, st->chunk);
      const uint32_t t1 = pi_time_get_us();
      g_load_step = -1;
      const verify_t v = verify(base, HB_LOAD_KB * 1024u, st->chunk);
      const uint32_t t2 = pi_time_get_us();
      com_spi_hold(0);
      chatter_on = 0;
      cpxPrintToConsole(LOG_TO_CRTP,
                        "HB: LOAD step=%d chunk=%lu spi=%s fc=%lu it=%d load=%d load_us=%lu verify_us=%lu "
                        "bad=%d flash_bad=%d first=%d\n",
                        s, (unsigned long)st->chunk, st->spi, (unsigned long)st->fc_mhz, it, load,
                        (unsigned long)(t1 - t0), (unsigned long)(t2 - t1), v.bad_chunks, v.flash_bad, v.first_bad);
      pi_time_wait_us(100000);
      total_bad += v.bad_chunks + (load != 0);
      done++;
    }
    pi_ram_free(&ram, base, HB_LOAD_KB * 1024u);
    close_devices();
    emit("LOAD step=%d chunk=%lu spi=%s fc=%lu: %d/%d loads, %d bad", s, (unsigned long)st->chunk, st->spi,
         (unsigned long)st->fc_mhz, done, HB_LOAD_ITERS, total_bad);
  }
  pi_freq_set(PI_FREQ_DOMAIN_FC, HB_FC_MHZ * 1000000u);
}

// ---------------------------------------------------------------------------
// The tiled layer: DORY's L3 weight tiling. Core 0 asks the FC to read the next
// 16 KB tile from HyperRAM (pi_cl_ram_read: a remote call the FC turns into a
// HyperBus transfer into L2) while all 8 cores compute on the current one, which
// core 0 has moved L2 -> L1 with MCHAN. Every tile's sum is checked.
// ---------------------------------------------------------------------------
#ifndef HB_TILE_PLAN
#define HB_TILE_PLAN ""  // "cl_mhz:spi,..."; empty leaves the test out
#endif
#ifndef HB_TILE_ITERS
#define HB_TILE_ITERS 20
#endif
#define HB_TILE 16384
#define HB_TILES_MAX 64
#define HB_TILE_WORK 24  // compute repeats per tile: about DORY's time per tile

static PI_L2 uint8_t tile_l2[2][HB_TILE];
static uint32_t tile_sums[HB_TILES_MAX];
static PI_L1 uint32_t core_sums[8];
static PI_L1 volatile uint32_t mac_sink[8];  // keeps the busy loop from being optimised away

typedef struct {
  uint32_t base, tiles;
  uint8_t *l1;
  volatile int bad, first_bad;
} tile_args_t;

static tile_args_t tile_args;

static void tile_worker(void *arg) {
  tile_args_t *a = (tile_args_t *)arg;
  const int core = pi_core_id();
  pi_cl_ram_req_t req;
  int cur = 0;
  if (core == 0) {
    pi_cl_ram_read(&ram, a->base, tile_l2[0], HB_TILE, &req);
    pi_cl_ram_read_wait(&req);
  }
  pi_cl_team_barrier(0);
  for (uint32_t k = 0; k < a->tiles; k++) {
    if (core == 0) {
      if (k + 1 < a->tiles) pi_cl_ram_read(&ram, a->base + (k + 1) * HB_TILE, tile_l2[!cur], HB_TILE, &req);
      pi_cl_dma_cmd_t cmd;  // one core issues MCHAN transfers (the multi-core push race)
      pi_cl_dma_cmd((uint32_t)tile_l2[cur], (uint32_t)a->l1, HB_TILE, PI_CL_DMA_DIR_EXT2LOC, &cmd);
      pi_cl_dma_cmd_wait(&cmd);
    }
    pi_cl_team_barrier(0);
    const uint32_t *words = (const uint32_t *)a->l1 + core * (HB_TILE / 32);
    uint32_t sum = 0, mac = 0;
    for (int r = 0; r < HB_TILE_WORK; r++) {
      for (int w = 0; w < HB_TILE / 32; w++) {
        if (r == 0) sum += words[w];
        mac += (words[w] & 0xff) * (uint32_t)(w + r);
      }
    }
    core_sums[core] = sum;
    mac_sink[core] = mac;
    pi_cl_team_barrier(0);
    if (core == 0) {
      uint32_t total = 0;
      for (int c = 0; c < 8; c++) total += core_sums[c];
      if (total != tile_sums[k]) {
        if (a->first_bad < 0) a->first_bad = (int)k;
        a->bad++;
      }
      if (k + 1 < a->tiles) pi_cl_ram_read_wait(&req);
    }
    cur = !cur;
    pi_cl_team_barrier(0);
  }
}

// Runs on cluster core 0, like DORY's execute_layer_fork(): L1 is allocated per task.
static void tile_entry(void *arg) {
  tile_args_t *a = (tile_args_t *)arg;
  a->l1 = pmsis_l1_malloc(HB_TILE);
  if (a->l1 == NULL) {
    a->bad = -1;
    return;
  }
  pi_cl_team_fork(8, tile_worker, arg);
  pmsis_l1_malloc_free(a->l1, HB_TILE);
}

static void run_tile_plan(void) {
  static struct { uint32_t cl_mhz; char spi[8]; } steps[16];
  int count = 0;
  for (const char *p = HB_TILE_PLAN; *p && count < 16;) {
    steps[count].cl_mhz = parse_uint(&p);
    if (*p != ':') break;
    p++;
    int k = 0;
    while (*p && *p != ',' && k < 7) steps[count].spi[k++] = *p++;
    steps[count].spi[k] = '\0';
    count++;
    if (*p == ',') p++;
  }
  const uint32_t size = HB_LOAD_KB * 1024u, tiles = size / HB_TILE;
  emit("== TEST tiled layer: %lu tiles of %d B, %d steps x %d ==", (unsigned long)tiles, HB_TILE, count,
       HB_TILE_ITERS);
  // Expected per-tile sums from the xorshift stream.
  uint32_t state = HB_LOAD_SEED;
  for (uint32_t k = 0; k < tiles && k < HB_TILES_MAX; k++) {
    uint32_t sum = 0;
    for (int w = 0; w < HB_TILE / 4; w++) sum += xorshift32(&state);
    tile_sums[k] = sum;
  }
  pi_freq_set(PI_FREQ_DOMAIN_FC, HB_FC_MHZ * 1000000u);
  uint32_t base = 0;
  if (open_devices() || pi_ram_alloc(&ram, &base, size) || load_file(base, 4096) ||
      verify(base, size, 4096).bad_chunks) {
    emit("TILE setup: FAIL");
    return;
  }
  struct pi_device cluster;
  struct pi_cluster_conf conf;
  pi_cluster_conf_init(&conf);
  pi_open_from_conf(&cluster, &conf);
  if (pi_cluster_open(&cluster)) {
    emit("TILE cluster open: FAIL");
    return;
  }
  tile_args.base = base;
  tile_args.tiles = tiles;
  for (int s = 0; s < count; s++) {
    pi_freq_set(PI_FREQ_DOMAIN_CL, steps[s].cl_mhz * 1000000u);
    int total_bad = 0;
    for (int it = 0; it < HB_TILE_ITERS; it++) {
      if (!strcmp(steps[s].spi, "burst")) {
        for (int i = 0; i < 25; i++) {
          cpxPrintToConsole(LOG_TO_CRTP, "HB~ burst %d of 25 before the pass\n", i);
          pi_time_wait_us(3000);
        }
      } else if (!strcmp(steps[s].spi, "chatter")) {
        chatter_on = 1;
        xTaskCreate(chatter_task, "hb_chatter", configMINIMAL_STACK_SIZE * 2, NULL, tskIDLE_PRIORITY + 1, NULL);
      } else if (!strcmp(steps[s].spi, "quiet")) {
        (void)com_spi_hold_acquire(100);
      }
      tile_args.bad = 0;
      tile_args.first_bad = -1;
      struct pi_cluster_task task;
      pi_cluster_task(&task, tile_entry, &tile_args);
      const uint32_t t0 = pi_time_get_us();
      pi_cluster_send_task_to_cl(&cluster, &task);
      const uint32_t t1 = pi_time_get_us();
      com_spi_hold(0);
      chatter_on = 0;
      cpxPrintToConsole(LOG_TO_CRTP, "HB: TILE step=%d cl=%lu spi=%s it=%d us=%lu bad=%d first=%d\n", s,
                        (unsigned long)steps[s].cl_mhz, steps[s].spi, it, (unsigned long)(t1 - t0), tile_args.bad,
                        tile_args.first_bad);
      pi_time_wait_us(50000);
      total_bad += tile_args.bad;
    }
    emit("TILE step=%d cl=%lu spi=%s: %d passes, %d bad tiles", s, (unsigned long)steps[s].cl_mhz,
         steps[s].spi, HB_TILE_ITERS, total_bad);
  }
  pi_cluster_close(&cluster);
  pi_ram_free(&ram, base, size);
  close_devices();
}

static void report_forever(void) {
  emit("==== HYPERBUS BENCH done: %d lines ====", g_count);
  while (1) {
    for (int i = 0; i < g_count; i++) {
      cpxPrintToConsole(LOG_TO_CRTP, "HB| %s\n", g_lines[i]);
      pi_time_wait_us(60000);
    }
    pi_time_wait_us(700000);
  }
}

// A load whose offset has not moved for 2 s is stuck (a healthy 128 B chunk takes
// about 70 us). Report where, then finish the report from here: the loading task
// never returns.
static void watchdog_task(void *arg) {
  (void)arg;
  uint32_t last = 0;
  int still = 0;
  while (1) {
    vTaskDelay(pdMS_TO_TICKS(250));
    const int step = g_load_step;
    const uint32_t offset = g_load_offset;
    still = (step >= 0 && offset == last) ? still + 1 : 0;
    last = offset;
    if (still == 8) {
      emit("LOAD step=%d STUCK it=%d at=%lu", step, g_load_it, (unsigned long)offset);
      report_forever();
    }
  }
}

static void startTests(void) {
  cpxInit();
  cpxEnableFunction(CPX_F_APP);

  emit("==== HYPERBUS BENCH start (FC=%dMHz) ====", HB_FC_MHZ);
  test_hyperflash_read();
  test_readfs_mount();
  test_hyperram();
  if (HB_LOAD_KB > 0) {
    // Close what the tests left open: each load step opens the devices itself,
    // as mem_init() does at boot.
    pi_fs_unmount(&fs);
    pi_flash_close(&flash);
    xTaskCreate(watchdog_task, "hb_watchdog", configMINIMAL_STACK_SIZE * 2, NULL, tskIDLE_PRIORITY + 1, NULL);
    run_load_plan();
    if (HB_TILE_PLAN[0]) run_tile_plan();
  }
  report_forever();
}

int main(void) {
  pi_bsp_init();
  __pi_pmu_voltage_set(PI_PMU_DOMAIN_FC, 1200);
  (void)pi_freq_set(PI_FREQ_DOMAIN_FC, HB_FC_MHZ * 1000000u);
  return pmsis_kickoff((void *)startTests);
}
