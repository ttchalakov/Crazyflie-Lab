# jpeg_cluster_bench

The verification bench for the GAP8's grayscale JPEG encoder,
[`lib/gap8/jpeg`](../../lib/gap8/jpeg), which camera_stream, camera_bench,
hover_neural_network and scheduler_bench use. The encoder runs on the fabric controller (FC) or on the 8-core cluster, and
the two must emit the same bytes: the bench holds both to a host build of the same
sources, in GVSOC and on the AI-deck, and measures what each costs.

```bash
python examples/jpeg_cluster_bench/host/bench.py                    # the common set (~3 min)
python examples/jpeg_cluster_bench/host/bench.py --set exhaustive   # every case, a 15 min soak among them (~25 min)
python examples/jpeg_cluster_bench/host/bench.py --target gvsoc     # the host and the simulator only; no drone
python examples/jpeg_cluster_bench/host/bench.py --list
```

It needs gcc, Docker (the `cf-lab/aideck-gap8` image), and for the deck the drone
on and a Crazyradio; the radio address comes from `configs/drone.yaml`.

## What it checks

A **case** (`host/bench.yaml`) is a window of a real camera frame
(`gap8/realimg.c`, 320×240) and optional `-D` switches for the encoder, named
`<W>x<H>` plus `-no_<switch>` for each switch turned off: `320x240` (the full
frame), `316x236` (neither side a multiple of 8: partial edge blocks), `96x96` (a
network's input), and `320x240-no_fast_gather`, `320x240-no_early_eob` and
`320x240-no_stage_on_cluster` (each of the encoder's optimisations off,
`-DJPEG_CL_<SWITCH>=0`). `soak` is `320x240` for 15 minutes on the deck.

| where | checks |
| --- | --- |
| host | `host_bench.c`: the reciprocal quantizer equals division for every int16 coefficient; encoding in 8 ranges and concatenating them gives the serial encode's bytes; the JPEG decodes (PIL) above 28 dB PSNR. Its bytes and hash are the reference. |
| host, `frames/` | `profile_frames.c` on 12 captured frames and 3 adversarial patterns: the 8-range merge holds on every frame, and the worst frame and worst segment fit the buffers in `gap8/main.c` |
| GVSOC | one frame on each path; the FC and cluster JPEGs are identical and equal the host's bytes |
| AI-deck | rounds of 100 frames for 60 s (the soak: 900 s): every round's FC and cluster JPEGs identical, with the host's hash; no overflow; rounds still arriving at the end (a wedged cluster stops them) |

The deck runs one case at a time over the radio, after the idle controller is
put on the STM32 (`--keep-controller` skips that); GVSOC runs several containers
at once. One case by hand: `host/run.py --out <dir> --window 316x236`.

Every run writes `artifacts/bench_<stamp>/`: `summary.md`, `results.jsonl`,
`run.json`, `logs/<job>.log` (jobs `gvsoc-<case>` and `aideck-<case>`), `host/`
(the reference JPEGs), `gvsoc/<case>/` and `aideck/<case>/` (`build.log`,
`jpeg_cluster_bench.img`, `console.log`, `report.json`).

## How the encoder works

JPEG has two scan-order dependencies, the differential DC coding and the
bit-packed Huffman stream. A **restart marker** resets both, so a run of blocks
that starts and ends on a restart boundary encodes on its own. The encoder splits
the frame into one such run per core; each core encodes its run into its own
buffer, with no locks and no shared state, and the runs are concatenated with
`memcpy`. The markers cost a little size.

The cluster path adds only speed, never different arithmetic: quantization by an
exact reciprocal multiply instead of 64 divides a block, and the Huffman and
quantization tables, the pixel rows and the bitstream staged in L1. The FC path
calls the same range encoder, so the two cannot diverge; the choice is a runtime
argument, because it depends on whether something else (a DORY network) wants the
cluster:

```c
jpeg_cfg_t cfg = {.path = JPEG_PATH_CLUSTER, .cores = 8, .quality = 50,
                  .restart_interval = 0 /* one run per core */,
                  .l1_policy = JPEG_CL_L1_RELEASE_ALL};
jpeg_enc_init(&enc, &cfg, w, h, stride, &cluster, segs, seg_cap);
size_t n = jpeg_enc_frame(&enc, img, out, cap);
jpeg_enc_set_path(&enc, JPEG_PATH_FC);   /* between frames, no re-init */
```

`jpeg_enc_out_cap()` and `jpeg_enc_seg_cap()` give the provable buffer bounds.

## What the encoder is built around

- **Release all of L1 between frames** (`JPEG_CL_L1_RELEASE_ALL`), so a DORY
  network gets the whole 64 kB; restaging the encoder's working set each frame
  costs little.
- **Buffers come from the corpus at QVGA.** There the provable ceiling (315 kB)
  is unaffordable, so the 64 kB output and 8 kB per-core buffers are sized from
  the frame corpus, which every run checks, and an overflow is counted, never
  truncated. At 96×96 the buffers clear the provable ceiling.
- **Cluster DMA from several cores wedges the whole chip.** `pi_cl_dma_cmd()` writes
  a command as several words to one MCHAN register without a lock, so two cores
  issuing at once splice their commands, a completion never arrives, and the
  cluster never returns. The encoder copies with `memcpy`; issue cluster DMA from
  one core only.
- **GVSOC does not model the cost of FC writes into cluster L1.** Restaging the
  tables from the FC every frame is free in GVSOC but not on the deck, so core 0
  does it (`JPEG_CL_STAGE_ON_CLUSTER`, on by default;
  `320x240-no_stage_on_cluster` turns it off).
- **A `jpeg_gray_t` (2,068 B) is larger than a default FreeRTOS task stack
  (2,048 B)**: one on the stack kills the chip silently. The app reports its stack
  high-water mark each round.
- **The benchmark cannot run on the main task**: CPX sends console lines from its
  own task, and a busy main task starves it (the ESP32 then reports `rxp 0`).
- **Program the cluster clock only after `pi_cluster_open()`**; before, the chip
  hangs at boot without output. The FC runs at 150 MHz, not 200.

## Files

```text
host/bench.py, bench.yaml   the bench and its sets
host/run.py                 one case on the deck: build, flash, record, judge
host/host_bench.c           the host reference
host/profile_frames.c       the frame corpus check
gap8/main.c, Makefile       the app: both paths every round, compared on the device
gap8/realimg.c              the frame it encodes (frames/qvga_00.pgm)
frames/                     12 frames from the deck's camera and 3 adversarial patterns
```
