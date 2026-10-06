#ifndef CF_LAB_HOVER_NN_PROTOCOL_H
#define CF_LAB_HOVER_NN_PROTOCOL_H

#include <stdint.h>

// CPX_F_APP contract shared by the Crazyflie STM32 and AI-deck GAP8: the state
// is the 11-float yaw-invariant observation, the reply a collective-thrust /
// body-rate setpoint. The version byte is only there so flashing one processor
// and not the other fails loudly instead of decoding garbage.
#define CF_LAB_HNN_PROTOCOL_VERSION 3u
#define CF_LAB_HNN_STATE_FLOATS 11u
#define CF_LAB_HNN_CONTROL_FLOATS 4u

#define CF_LAB_HNN_CONTROL_THRUST 0u
#define CF_LAB_HNN_CONTROL_RATE_X 1u
#define CF_LAB_HNN_CONTROL_RATE_Y 2u
#define CF_LAB_HNN_CONTROL_RATE_Z 3u

typedef enum {
  CF_LAB_HNN_MSG_STATE = 1,
  CF_LAB_HNN_MSG_CONTROL = 2,
} cfLabHoverNnMessageType_t;

// STM32 -> GAP8 at 100 Hz. The state is ctbr.observe(): body-frame position
// error, body-frame velocity, world-up in the body frame, and the heading error
// as (sin, cos). SI units, FLU body frame.
typedef struct __attribute__((packed)) {
  uint8_t version;
  uint8_t msgType;
  uint32_t stateTimestampMs;
  float state[CF_LAB_HNN_STATE_FLOATS];
} cfLabHoverNnState_t;

// GAP8 -> STM32, one per state the deck infers on. Control order is total
// thrust [N], then body rates x/y/z [rad/s]. stateTimestampMs echoes the state
// it answers, which is what the STM32 ages to decide whether the command is
// still fresh enough to fly. A state the deck skips gets no reply.
typedef struct __attribute__((packed)) {
  uint8_t version;
  uint8_t msgType;
  uint32_t stateTimestampMs;
  float control[CF_LAB_HNN_CONTROL_FLOATS];
} cfLabHoverNnControl_t;

#endif
