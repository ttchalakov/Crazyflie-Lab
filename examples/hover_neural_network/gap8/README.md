# Neural Hover: GAP8

The deck answers the STM32's policy requests and streams video, and both share
one cluster. [lib/gap8/sched](../../../lib/gap8/sched/README.md) schedules them;
this app supplies the network, the CPX plumbing and the codec.

## Requests and replies

`stateTask` receives each request on CPX, checks its version and that every
value is finite, encodes it, and posts it with `sched_post_state()`, tagged with
the request's timestamp. A request that arrives while an older one still waits
replaces it: a stale state is not worth flying. Each request's timestamp is
also passed to `sched_clock_anchor()`, so recorded video lines up with the
STM32's logs.

The scheduler's worker runs the inference and calls `onOutput()`, which decodes
the four outputs and replies on the urgent CPX class, ahead of any queued video
packets. The reply echoes the timestamp, which the STM32 ages to decide whether
the command is fresh enough to fly. A request the deck drops gets no reply, and
the STM32 keeps the PID. There is nothing else on the link: no heartbeat, no
handshake.

The worker serves a waiting request first. Then it encodes the newest camera
frame, if a video client is connected and the previous frame has gone out. The
encode runs in 16 slices, and a request that arrives meanwhile is served at the
next slice boundary, so it waits for at most one slice.

## Camera and video

Set in the Makefile: QVGA at 60 fps, JPEG at quality 50, auto exposure with the
60 Hz anti-flicker step for the mocap cameras' IR strobes. WiFi comes from the
`camera:` section of the drone config, which `cf-lab build gap8` and `cf-lab
flash gap8` pass in.

## Memory

The policy is built for DORY's `PULP.GAP8_L2` target only: its weights are
resident L2 arrays, so inference never touches HyperRAM and there is no flash
filesystem to mount. The Makefile refuses a 3-level network.

| | bytes |
|---|---|
| camera, two 320×240 buffers | 153,600 |
| JPEG output (a third of a frame) | 25,600 |
| CPX send queue, 30 packets | ~30,000 |
| DORY arena (activations only) | 1,024 |

## Codec

The top of `hover_neural_network.c` holds the whole boundary between the
integer graph and the vehicle: DORY eats a `uint8[11]` and returns an
`int32[4]`. `encodeState` is the same mapping as `_encode_state` in
`hover_policy.py`. If they disagree, the network sees a differently-scaled
world than it trained in, and nothing reports it.

Every constant there mirrors the top of `hover_policy.py` and stays the same on
every retrain. The one weight-dependent value, `HNN_POLICY_OUTPUT_EPSILON`,
arrives as a `-D` from the Makefile. The build fails with an explicit `#error`
if you build without a deploy.

Generate the policy first: see [policy/README.md](policy/README.md).
