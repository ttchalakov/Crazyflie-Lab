#ifndef NNB_CONTROLLER_H
#define NNB_CONTROLLER_H

#include "cf_loop_stats.h"

typedef struct {
  cf_loop_stats_t tick;     // period between stabilizer ticks; spans: the controller's execution time
  cf_loop_stats_t latency;  // spans: IMU data-ready interrupt to controller output
} nnb_inner_stats_t;

extern uint8_t nnbInnerIndi;

void nnbInnerReset(void);
void nnbInnerSnapshot(nnb_inner_stats_t *out);

#endif
