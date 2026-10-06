/*
 * nn_bench -- runs one generated network on the GAP8 and checks it.
 *
 * The network is DORY's generated app (runners pulp_nn and lib_kernels) or a
 * fused executor (fused/, runner fused). It runs NNBENCH_RUNS times on one
 * cluster kept open (the pattern every application uses), each time on the
 * golden input: xorshift bytes, regenerated here exactly as host/graph.py
 * generates them. After each layer the network calls back (dory_layer_done(),
 * or nnbench_layer() from an executor); the callback takes the CRC-32 of the
 * layer's output for comparison with the host's (nn_bench_golden.h) and the
 * layer's cluster cycles, and the CRC is taken out of the timing. A layer the
 * network never stores (an executor fuses it away) is reported skipped. The
 * results are latched and printed after the last run as "NN|" lines: to stdout
 * in GVSOC, twice, ending in #DONE; over CPX to the radio console on the deck,
 * replayed every few seconds because the console attaches late. The host keeps
 * the first copy of each line.
 */
#include "pmsis.h"
#include "bsp/bsp.h"

#include <stdio.h>
#include <string.h>

#include "cf_dory_profile.h"
#include "nn_bench_golden.h" /* generated: each layer's output bytes and CRC-32 */
#if NNBENCH_FUSED
#include "fused.h"
#include "nn_bench_fused.h"  /* generated: the executor's tables and tensors   */
#else
#include "nn_bench_net.h"    /* generated: the memory model, arena and stacks    */
#endif
#if NNBENCH_HAS_L3
#include "bsp/ram.h"
#include "mem.h"
#endif
#if NNBENCH_CPX
#include "cpx.h"
#endif

#define CLUSTER_HZ 150000000u /* as lib/gap8/sched runs the cluster */

#if NNBENCH_CPX
#define BOOT_DELAY_MS 12000 /* the host opens the radio link after the deck boots */
/* Paced: a report is hundreds of lines, more than the CPX queue holds. */
#define PRINT(...)                                                             \
  do {                                                                         \
    char line_[96];                                                            \
    snprintf(line_, sizeof line_, __VA_ARGS__);                                \
    cpxPrintToConsole(LOG_TO_CRTP, "%s", line_);                               \
    pi_time_wait_us(3000);                                                     \
  } while (0)
#else
#define PRINT(...) printf(__VA_ARGS__)
#endif

static uint32_t gCrcTable[256];
static uint32_t gLayerCrc[NNBENCH_RUNS][NNBENCH_NUM_LAYERS];
static uint32_t gLayerCycles[NNBENCH_RUNS][NNBENCH_NUM_LAYERS];
static uint8_t gLayerSeen[NNBENCH_RUNS][NNBENCH_NUM_LAYERS];
static uint32_t gRunCycles[NNBENCH_RUNS];
static int gRun, gRunsDone;
static uint32_t gStart, gCheck; /* cycle stamps within the current run */
static uint32_t gHz = CLUSTER_HZ;

static uint32_t crc_update(uint32_t c, const uint8_t *p, size_t n) {
  while (n--) c = gCrcTable[(c ^ *p++) & 0xFFu] ^ (c >> 8);
  return c;
}

/* A layer ended at cycle `now` with output CRC `crc`; the check is not timed. */
static void layer_end(int id, uint32_t now, uint32_t crc) {
  if (id >= 0 && id < NNBENCH_NUM_LAYERS) {
    gLayerCycles[gRun][id] = now - gStart;
    gLayerCrc[gRun][id] = crc;
    gLayerSeen[gRun][id] = 1;
  }
  gStart = pi_perf_read(PI_PERF_CYCLES);
  gCheck += gStart - now;
}

/* A layer's output is complete, in L1 or L2; on cluster core 0. */
void nnbench_layer(int id, const void *out, size_t bytes) {
  const uint32_t now = pi_perf_read(PI_PERF_CYCLES);
  layer_end(id, now, ~crc_update(~0u, (const uint8_t *)out, bytes));
}

/* Overrides DORY's weak hook, called after every layer while its output is in L2. */
void dory_layer_done(int id, const char *name, void *out, size_t size) {
  (void)name;
  nnbench_layer(id, out, size);
}

#if NNBENCH_HAS_L3
/* The same for a layer whose output DORY keeps in HyperRAM: read it back. */
void dory_layer_done_l3(int id, const char *name, unsigned int l3_output) {
  (void)name;
  static uint8_t chunk[2048] __attribute__((aligned(4)));
  const uint32_t now = pi_perf_read(PI_PERF_CYCLES);
  const uint32_t bytes = id >= 0 && id < NNBENCH_NUM_LAYERS ? nnbenchLayerBytes[id] : 0;
  uint32_t c = ~0u;
  for (uint32_t done = 0, n; done < bytes; done += n) {
    pi_cl_ram_req_t req;
    n = bytes - done < sizeof chunk ? bytes - done : sizeof chunk;
    pi_cl_ram_read(get_ram_ptr(), l3_output + done, chunk, n, &req);
    pi_cl_ram_read_wait(&req);
    c = crc_update(c, chunk, n);
  }
  layer_end(id, now, ~c);
}

/* mem_init() reports a failed mount on a UART nothing reads, then exits. */
void dory_mem_init_step(const char *step, int err) {
  if (err) PRINT("NN| FATAL mem_init %s err=%d\n", step, err);
}
#endif

/* --------------------------------------------------------------------------
 * The network: DORY's app, or a fused executor.
 */
#if NNBENCH_FUSED
static int net_init(void) { return fused_init(); }
static uint8_t *net_input(void) { return fused_input(); }
static void *net_args(void) { return NULL; }
static void net_run(void *args) { (void)args; fused_run(); }
#else
static void *gArena;

static int net_init(void) {
#if NNBENCH_HAS_L3
  mem_init();
  network_initialize();
#endif
  gArena = pi_l2_malloc(NNBENCH_L2_ARENA); /* after mem_init: the mount allocates L2 too */
  return gArena ? 0 : -1;
}

/* DORY's 3-level network reads its input from the base of the arena (where its
 * own main.c stages it, and where initial_dir = 1 points layer 0); the 2-level
 * one from its L2_input_h array. */
static uint8_t *net_input(void) {
#if NNBENCH_HAS_L3
  return (uint8_t *)gArena;
#else
  return (uint8_t *)L2_input_h;
#endif
}

static void *net_args(void) {
  static uint8_t output[NNBENCH_OUTPUT_BYTES] __attribute__((aligned(4)));
  static unsigned int args[6];
  args[0] = (unsigned int)gArena;
  args[1] = NNBENCH_L2_ARENA;
  args[2] = (unsigned int)output;
  args[3] = 0; /* exec */
  args[4] = 1; /* initial_dir */
  args[5] = (unsigned int)net_input(); /* the 2-level network's input argument */
  return args;
}

static void net_run(void *args) { network_run_cluster(args); }
#endif

static void run_cluster(void *args) {
  pi_perf_conf(1u << PI_PERF_CYCLES);
  pi_perf_reset();
  pi_perf_start();
  gStart = gCheck = 0u;
  cf_dory_prof_reset();
  net_run(args);
  gRunCycles[gRun] = pi_perf_read(PI_PERF_CYCLES) - gCheck;
  pi_perf_stop();
}

/* --------------------------------------------------------------------------
 * The runs and the report.
 */
static void stage_input(uint8_t *dst) {
  uint32_t x = 2463534242u; /* graph.py XORSHIFT_SEED */
  for (uint32_t i = 0; i < NNBENCH_INPUT_LEN; i++) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    dst[i] = (uint8_t)x;
  }
}

static unsigned long us(uint32_t cycles) {
  return (unsigned long)(((uint64_t)cycles * 1000000u + gHz / 2u) / gHz);
}

static void report(void) {
  for (int r = 0; r < gRunsDone; r++) {
    for (int i = 0; i < NNBENCH_NUM_LAYERS; i++) {
      if (gLayerSeen[r][i])
        PRINT("NN| r=%d L%d ok=%d us=%lu\n", r, i, gLayerCrc[r][i] == nnbenchLayerHash[i], us(gLayerCycles[r][i]));
      else
        PRINT("NN| r=%d L%d skip\n", r, i);
    }
    PRINT("NN| r=%d us=%lu\n", r, us(gRunCycles[r]));
  }
}

static void bench_task(void *arg) {
  (void)arg;
  static struct pi_device cluster;
  static struct pi_cluster_task task;
  struct pi_cluster_conf conf;
  int err;
#if NNBENCH_CPX
  PRINT("NN| boot_delay_ms=%d\n", BOOT_DELAY_MS);
  pi_time_wait_us(BOOT_DELAY_MS * 1000);
#endif
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & -(c & 1u));
    gCrcTable[i] = c;
  }
  PRINT("NN| runner=%s runs=%d layers=%d\n", NNBENCH_RUNNER, NNBENCH_RUNS, NNBENCH_NUM_LAYERS);

  pi_cluster_conf_init(&conf);
  pi_open_from_conf(&cluster, &conf);
  if (pi_cluster_open(&cluster)) {
    PRINT("NN| FATAL cluster_open\n");
    goto done;
  }
  pi_freq_set(PI_FREQ_DOMAIN_CL, CLUSTER_HZ);
  gHz = pi_freq_get(PI_FREQ_DOMAIN_CL);
  if ((err = net_init()) != 0) { /* after the cluster opens: an executor allocates L1 */
    PRINT("NN| FATAL init err=%d\n", err);
    goto done;
  }
  PRINT("NN| cluster_hz=%lu\n", (unsigned long)gHz);

  for (gRun = 0; gRun < NNBENCH_RUNS; gRun++) {
    stage_input(net_input()); /* every run: the network reuses the buffer */
    pi_cluster_task(&task, run_cluster, net_args());
    task.stack_size = NNBENCH_CL_MASTER_STACK; /* PMSIS forgets it after each run */
    task.slave_stack_size = NNBENCH_CL_SLAVE_STACK;
    pi_cluster_send_task_to_cl(&cluster, &task);
    cf_dory_prof_report(gRunCycles[gRun]); /* a no-op unless CF_DORY_PROFILE=1 */
    gRunsDone = gRun + 1;
  }
  pi_cluster_close(&cluster);

done:
  report();
  PRINT("NN| DONE\n");
#if NNBENCH_CPX
  for (;;) {
    pi_time_wait_us(3000000);
    report();
  }
#else
  report(); /* again: GVSOC's semihosting now and then drops a line */
  printf("#DONE\n");
  pmsis_exit(0);
#endif
}

static void start(void) {
#if NNBENCH_CPX
  cpxInit();
#endif
  if (xTaskCreate(bench_task, "nnbench", 8192 / sizeof(StackType_t), NULL, tskIDLE_PRIORITY + 1, NULL) != pdPASS)
    pmsis_exit(-1);
  vTaskDelete(NULL); /* not a spinning kickoff task: the HyperBus needs FC interrupts */
}

int main(void) {
  pi_bsp_init();
  __pi_pmu_voltage_set(PI_PMU_DOMAIN_FC, 1200);
  pi_freq_set(PI_FREQ_DOMAIN_FC, 150000000);
  return pmsis_kickoff((void *)start);
}
