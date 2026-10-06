# GAP8 camera stream status line

An optional status line for an app built on `camera_stream`: the camera's
counters (captures, drops, watchdog restarts, fault), the CPX/SPI transport's
counters, and two app counters, as one `status k=v ...` console line. The camera
driver itself stays free of CPX and formatted strings.

```c
camera_stream_log_t logger;
// Wait up to 100 ms for CPX queue space.
camera_stream_log_init(&logger, &camera, CPX_T_WIFI_HOST, 100);

// From the app's own status task, once a second:
const camera_stream_log_progress_t progress = {
    .frames_sent = framesSent,
    .encode_failures = encodeFailures,
};
camera_stream_log_emit(&logger, &progress);
```

Send it to `CPX_T_WIFI_HOST` to put it on the image stream's socket, where
`cf_lab.camera` parses it and the telemetry recorder stores it with the frames;
`CPX_T_STM32` sends it to the radio console instead. The send waits at most the
timeout given to init for CPX queue space, then drops the line: pass 0 when the
caller must never stall, and a short wait from a status task of your own, since
a stream keeps the queue nearly full and would otherwise crowd most lines out.

The logger allocates nothing and creates no task or timer. Formatting the line
takes about 650 bytes of stack, too much for a FreeRTOS timer callback (the timer
task has 1 KB and needs some of it itself), so call it from a task of your own.

```make
include $(CF_LAB)/lib/gap8/cpx/cpx.mk
include $(CF_LAB)/lib/gap8/camera_stream/camera_stream.mk
include $(CF_LAB)/lib/gap8/camera_stream_log/camera_stream_log.mk
```

`examples/camera_stream` enables it with `LOG=1`; `examples/camera_bench` always
does.
