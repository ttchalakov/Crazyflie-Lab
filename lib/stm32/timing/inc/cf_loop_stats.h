#ifndef CF_LAB_LOOP_STATS_H
#define CF_LAB_LOOP_STATS_H

// Timing statistics of a periodic loop (the stabilizer, a control task) from cf_cycles() stamps: the period between
// ticks against a nominal period, and the duration of whatever each tick measures. Header only; one writer at a
// time (a reader copies the struct inside a critical section).

#include <stdint.h>

#define CF_LOOP_HIST_BINS 32

typedef struct {
  uint32_t nominal;  // nominal period, cycles
  uint32_t last;     // stamp of the previous tick, 0 before the first
  uint32_t ticks;
  uint32_t periodMin, periodMax;
  uint32_t late10, late50, late100;  // periods more than 10 %, 50 %, 100 % above nominal
  uint32_t spans;
  uint32_t spanMin, spanMax;
  uint64_t spanSum;
  uint32_t binWidth;                       // histogram of spans, cycles per bin
  uint16_t hist[CF_LOOP_HIST_BINS + 1];  // the last bin counts everything above
} cf_loop_stats_t;

static inline void cf_loop_stats_reset(cf_loop_stats_t *s, uint32_t nominal, uint32_t binWidth) {
  *s = (cf_loop_stats_t){.nominal = nominal, .periodMin = UINT32_MAX, .spanMin = UINT32_MAX, .binWidth = binWidth};
}

static inline void cf_loop_tick(cf_loop_stats_t *s, uint32_t now) {
  if (s->last != 0u) {
    const uint32_t p = now - s->last;
    s->periodMin = p < s->periodMin ? p : s->periodMin;
    s->periodMax = p > s->periodMax ? p : s->periodMax;
    s->late10 += p > s->nominal + s->nominal / 10u;
    s->late50 += p > s->nominal + s->nominal / 2u;
    s->late100 += p > 2u * s->nominal;
    s->ticks++;
  }
  s->last = now == 0u ? 1u : now;
}

static inline void cf_loop_span(cf_loop_stats_t *s, uint32_t cycles) {
  s->spans++;
  s->spanSum += cycles;
  s->spanMin = cycles < s->spanMin ? cycles : s->spanMin;
  s->spanMax = cycles > s->spanMax ? cycles : s->spanMax;
  if (s->binWidth != 0u) {
    const uint32_t bin = cycles / s->binWidth;
    s->hist[bin < CF_LOOP_HIST_BINS ? bin : CF_LOOP_HIST_BINS]++;
  }
}

// Upper edge (cycles) of the histogram bin holding the q-quantile of the spans, q in [0, 1].
static inline uint32_t cf_loop_span_quantile(const cf_loop_stats_t *s, float q) {
  const uint32_t target = (uint32_t)(q * (float)s->spans);
  uint32_t seen = 0;
  for (uint32_t b = 0; b <= CF_LOOP_HIST_BINS; b++) {
    seen += s->hist[b];
    if (seen > target) {
      return b < CF_LOOP_HIST_BINS ? (b + 1u) * s->binWidth : s->spanMax;
    }
  }
  return s->spanMax;
}

#endif
