#ifndef TELEMETRY_BENCH_GAP8_SAMPLE_H
#define TELEMETRY_BENCH_GAP8_SAMPLE_H
#include <stdint.h>

/* Local wire contract: little-endian uint32 + IEEE-754 float32 fields. */
typedef struct {
    uint32_t tick;
    float sin1;
    float cos1;
    float saw;
    float ramp;
    float square;
    float tri;
} gap8_sample_t;

#endif
