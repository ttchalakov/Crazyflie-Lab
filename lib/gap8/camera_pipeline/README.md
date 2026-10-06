# GAP8 camera frame pipeline

`camera_pipeline` hands out camera frames at the size the caller configured,
above `camera_stream`. It deliberately does not own a task, JPEG encoder, neural
network, transport, or logger: whichever task or scheduler calls
`camera_pipeline_acquire()` gets the next frame there, resized there if a size
was set. Nothing is resized that is not taken, however fast the sensor runs.

```c
camera_stream_open(&camera);            /* Himax and raw L2 buffers first */

camera_pipeline_config_init(&config);
config.camera = &camera;
config.output_width = 96;               /* zero or the camera's size: no copy */
config.output_height = 96;
config.fit = DS_SQUASH;                 /* or DS_CROP to keep the aspect */

camera_pipeline_init(&pipeline, &config); /* validate/copy only */
camera_pipeline_memory_plan(&config, &plan);
camera_pipeline_open(&pipeline);          /* the one output buffer */
camera_pipeline_start(&pipeline);
camera_stream_start(&camera);

camera_pipeline_frame_t frame;
if (camera_pipeline_acquire(&pipeline, &frame, CAMERA_STREAM_WAIT_FOREVER) == 0) {
  use_pixels(frame.data, frame.width, frame.height); /* resize_duration_ms too */
  camera_pipeline_release(&pipeline, &frame);
}
```

- `acquire()` takes the camera's newest frame. A native pipeline leases that
  camera buffer until `release()`. A resized one resizes it into its output
  buffer and gives the camera buffer straight back, so the camera keeps
  capturing at the sensor rate while the caller encodes, infers or sends, and
  two exact camera buffers are always enough.
- Frames are timed per stage: `capture_duration_ms` from the camera,
  `resize_duration_ms` for the resize, and the caller times whatever comes
  next (JPEG, a network, the send).
- A resized frame is unchanged until `release()`. A preview capture can tear:
  preview's single camera buffer keeps being written while it is read.
- A finite `acquire()` timeout bounds the wait for a camera frame, which suits a
  scheduler polling with a timeout of zero.
- The output buffer is allocated in `open()`. Acquiring allocates nothing.
- `open()` verifies that the referenced camera still has the sensor/ownership
  contract observed by `init()`. Reinitialize the pipeline if the camera object
  has been reconfigured in between.
- `start()`/`stop()` control handing out frames only. Camera lifecycle remains
  explicit, so a user can pause without hiding a Himax transition.
- `stop()` and `close()` wake a blocked `acquire()` within 50 ms; `close()`
  returns busy while a frame is still leased. Keep the referenced camera open
  for the pipeline lifetime and close it only after closing the pipeline.

Lease tokens remain unique across `close()`/`init()` reuse of the same storage,
so a frame retained from an earlier pipeline lifetime cannot release a newer
lease. As with raw camera leases, preserve the returned token unchanged and pair
each successful acquire with exactly one release.

## The resize

`ds_resize_gray()` (`inc/downsample.h`) is the one resize: a box filter, where
each output pixel is the rounded mean of the source rectangle that maps to it.
It is integer-only and also compiles on the host, so a host tool sees exactly
the pixels the deck sends. The GAP8 fabric controller pays about 10 cycles for
every L2 load, so it reads four source columns per 32-bit load, steps band
edges with a remainder and multiplies by a reciprocal instead of dividing, with
output byte-identical to the plain box filter. Call it directly for a resize
outside the pipeline.

Include the module after the camera core:

```make
include $(CF_LAB)/lib/gap8/camera_stream/camera_stream.mk
include $(CF_LAB)/lib/gap8/camera_pipeline/camera_pipeline.mk
```
