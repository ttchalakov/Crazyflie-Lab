# GAP8 JPEG Pipeline

`jpeg_pipeline` is a caller-owned lifecycle wrapper around `jpeg_enc`. It plans
and allocates the encoder's L2 buffers once, optionally owns the GAP8 cluster,
and performs no per-frame L2 allocation.

```c
jpeg_pipeline_config_t config;
jpeg_pipeline_memory_plan_t memory;
jpeg_pipeline_t pipeline;
jpeg_pipeline_output_t output;

jpeg_pipeline_config_init(&config);
config.width = 96;
config.height = 96;
config.stride = 96;
config.output_budget_bytes = 0; /* rigorous ceiling */

jpeg_pipeline_memory_plan(&config, &memory);
jpeg_pipeline_init(&pipeline, &config); /* validate/copy only */
jpeg_pipeline_open(&pipeline);          /* all allocation happens here */

if (jpeg_pipeline_encode(&pipeline, gray, &output) == JPEG_PIPELINE_OK) {
  send(output.data, output.size_bytes);
}
jpeg_pipeline_close(&pipeline);
```

The output pointer is owned by the pipeline and remains valid until the next
`encode()` or `close()`.

## Cluster ownership

- `JPEG_PIPELINE_CLUSTER_OWNED` opens the cluster during `open()`, optionally
  sets its clock, and closes it after releasing encoder L1 during `close()`.
- `JPEG_PIPELINE_CLUSTER_BORROWED` uses an already-open caller-owned device. It
  never opens, clocks, or closes that device, which permits an application to
  coordinate the encoder with a persistent DORY cluster.
- FC encoding requires `JPEG_PIPELINE_CLUSTER_NONE` and consumes no cluster L1.

Cluster JPEG and DORY must still be serialized by the application. Use
`JPEG_CL_L1_RELEASE_ALL` unless retained encoder L1 has been explicitly included
in the network's memory budget.

The JPEG implementation is linked into the application; it is not dynamically
loaded. `jpeg_enc_frame()` uses the cluster L1 allocator to acquire its working
set during an encode and releases it according to `l1_policy`; with
`JPEG_CL_L1_RELEASE_ALL`, those L1 allocations occur again on the next frame.
The cluster device itself stays open for the pipeline lifetime. Per-frame
cluster open/close is intentionally unsupported because its latency and
power-domain/FLL transitions are incompatible with a stable streaming hot path.

## Slices

`config.encoder.slices = n` encodes a frame in `n` cluster jobs instead of one,
each a horizontal strip, so the caller can use the cluster between them:

```c
jpeg_pipeline_encode_begin(&pipeline, gray);
while (jpeg_pipeline_encode_step(&pipeline) > 0) {
  if (urgent()) {
    jpeg_pipeline_encode_yield(&pipeline); /* gives the encoder's L1 back */
    run_urgent_job();                      /* e.g. a DORY inference */
  }
}
jpeg_pipeline_encode_end(&pipeline, &output);
```

The frame then has `cores x n` restart segments rather than `cores`, on the FC
path too, so the two paths stay byte-identical; segment budgets are per segment.
`jpeg_pipeline_encode()` runs the slices back to back. lib/gap8/sched serves
posted states between slices: more slices add a little encode time and cut the
longest wait for the cluster from a whole encode to one slice.

## Buffer budgets

An output budget of zero allocates the rigorous whole-frame ceiling. A nonzero
budget is capped at that ceiling and may drop an unusually large frame. On the
cluster path, `segment_budget_bytes` is per core. Zero divides the resolved
output budget evenly across the cores, matching the camera demo's bounded-memory
policy. `jpeg_pipeline_memory_plan()` reports both rigorous ceilings, resolved
allocations, and the two overflow-risk flags before hardware is touched.

The rigorous ceilings come from the encoder's own quantization and Huffman tables:

| Encode size | Frame ceiling | Per-core segment ceiling (8 cores) |
| --- | ---: | ---: |
| 96x80 | 32 kB | 3.9 kB |
| 160x120 | 79 kB | 10 kB |
| 320x240 | 315 kB | 39 kB |

Real frames are far smaller, and the worst frame in jpeg_cluster_bench's corpus
(maximum-amplitude salt and pepper) fits with room to spare, which the bench
checks on every run. So camera_stream caps the output at 64 kB and each of the 8
segments at 8 kB. Where
a ceiling is below its budget, overflow cannot happen; where it is above (QVGA), a
frame that does not fit produces no output and is counted as an encode failure,
never truncated. The segments must stay within the output budget in total: a
merged frame is the segments plus header, so segment memory beyond the output
budget can never be used, and sizing segments from their raw ceiling (8 x 39 kB)
exhausts L2 at QVGA. An equal per-core share bounds imbalance too: a frame whose
detail piles into one core's strip can overflow that segment although the whole
frame would fit.

An L2 allocation failure at open shows up on the host as a deck that accepts the
connection and then sends nothing, which looks like a hang; the GAP8's radio
console says which allocation failed.
