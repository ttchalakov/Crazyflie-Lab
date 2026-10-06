"""Render a recorded camera stream to constant-rate H.264 or lossless video, or to files.

The feed is bursty: frames arrive when WiFi lets them, not on a grid. Playing
them back at a fixed rate as if evenly spaced would show a lie -- motion would
speed up and slow down with the network. So each output frame at time
``t = t0 + i/fps`` shows the frame that was *current* at t: the last one whose
timestamp is at or before it. Bursts land in their true slots and gaps hold the
previous image, so wall-clock timing is preserved.

Frames are written at the resolution they were captured at. That resolution is a
GAP8 compile-time decision this side never second-guesses.

H.264 (the default) is small and plays everywhere, but its compression smears
the fine detail of a low-resolution frame. ``lossless`` writes VP9 in lossless
mode instead: every decoded pixel equals the recorded gray value. It still plays
in a browser, VSCode and Slack.

Timestamps come from ``t_s`` (``cf_lab.telemetry.sync.load_run``), which is the
deck's own capture clock fitted onto the host timeline when the app supplies an
extension. That matters here: the deck's clock has the true capture cadence,
while host arrival times carry the network's jitter on top. Runs without an
extension fall back to arrival time automatically.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cf_lab.camera.feed import decode_frame
from cf_lab.telemetry.schema import CAMERA_BLOB_FILE

FILL_HOLD = "hold"
FILL_BLEND = "blend"

# H.264 in yuv420p: what a browser-class player (VSCode's preview, anything
# Electron/Chromium, QuickTime) will actually decode. Both halves matter -- an
# H.264 stream in some other pixel format still will not play in a browser.
#
# OpenCV's VideoWriter cannot produce this. The FFmpeg inside the opencv-python
# wheel ships no libx264 and no openh264, only h264_v4l2m2m, which wants
# hardware that is not there, so every H.264 fourcc fails to even open the
# writer and the only thing it will write is mp4v (MPEG-4 Part 2) -- which is
# exactly what VSCode refuses to play. Frames therefore go to a real ffmpeg,
# which imageio-ffmpeg provides as an ordinary pip dependency (a static build
# with libx264), so this needs nothing installed on the system.
#
# Piping is also less work than VideoWriter: ffmpeg accepts the grayscale bytes
# directly, so there is no per-frame BGR conversion.
H264_CRF = "23"
H264_PRESET = "veryfast"


def ffmpeg_executable() -> str:
    """The static ffmpeg imageio-ffmpeg brings, which encodes H.264 and VP9."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass(frozen=True)
class VideoStats:
    """What the render actually did.

    ``held`` and ``skipped`` are the two ways output stops being a 1:1 view of
    the capture, and both are silent unless reported. ``skipped`` counts source
    frames that never appeared: two landed in one output slot -- raise ``fps``
    if that number is not near zero and you wanted every frame -- or the frame
    did not decode, and its slots held the previous image.
    """

    output_frames: int
    source_frames: int
    held: int
    skipped: int
    width: int
    height: int
    fps: float
    duration_s: float


@dataclass(frozen=True)
class RecordedFrame:
    t_s: float
    payload: bytes
    width: int
    height: int
    is_jpeg: bool
    frame_id: int

    def decode(self):
        """Decode to an ``(h, w)`` uint8 ndarray, or None; see ``decode_frame``."""
        return decode_frame(self.payload, self.width, self.height, self.is_jpeg)


def load_frames(run_dir: str | Path) -> list[RecordedFrame]:
    """Read camera.parquet + camera_frames.bin into timestamped frames.

    Payloads are sliced out of the blob by offset, so nothing decodes here and
    memory stays proportional to the run's bytes, not to any decoded video.
    """
    from cf_lab.telemetry.sync import load_run

    run_dir = Path(run_dir)
    run = load_run(run_dir)
    if "camera" not in run.streams:
        raise ValueError(f"{run_dir} has no camera stream.")
    table = run.streams["camera"]
    blob = (run_dir / CAMERA_BLOB_FILE).read_bytes()

    columns = {name: table.column(name).to_pylist() for name in
               ("t_s", "size", "blob_offset", "width", "height", "pixel_format", "frame_id")}
    frames = []
    for i in range(table.num_rows):
        offset, size = columns["blob_offset"][i], columns["size"][i]
        frames.append(
            RecordedFrame(
                t_s=columns["t_s"][i],
                payload=blob[offset : offset + size],
                width=columns["width"][i],
                height=columns["height"][i],
                is_jpeg=columns["pixel_format"][i] == 1,
                frame_id=columns["frame_id"][i],
            )
        )
    frames.sort(key=lambda f: f.t_s)
    return frames


# A frame captured exactly on the output grid lands within an ULP of its slot
# (3*0.1 != 3/10.0 in binary), and a strict compare would show the *previous*
# frame instead -- silently dropping one, in precisely the well-behaved case
# where the deck's rate divides the target rate. A nanosecond is finer than any
# clock here, so it can only ever fix that tie.
_TIE_EPS_S = 1e-9


def hold_schedule(timestamps: list[float], fps: float) -> list[int]:
    """Index of the frame to show in each output slot (zero-order hold).

    Pure and list-only so the timing rule can be tested without a codec: for
    each output instant, the last frame at or before it.
    """
    if not timestamps:
        return []
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    start, end = timestamps[0], timestamps[-1]
    # Ceil, not round: the last slot must reach the last frame, or a run ending
    # just past a slot boundary would never render its final frames at all.
    count = max(1, math.ceil((end - start) * fps - _TIE_EPS_S) + 1)
    schedule = []
    source = 0
    for i in range(count):
        t = start + i / fps
        # Advance while the *next* frame has already happened by t.
        while (
            source + 1 < len(timestamps)
            and timestamps[source + 1] <= t + _TIE_EPS_S
        ):
            source += 1
        schedule.append(source)
    return schedule


# Playback rate for a run whose deck never reported its frame rate.
DEFAULT_FPS = 30.0


def deck_fps(run_dir: str | Path) -> float | None:
    """The frame rate the deck's build reported in its config line, if recorded."""
    import json

    from cf_lab.telemetry.sync import read_stream

    events = read_stream(Path(run_dir), "events")
    if events is None:
        return None
    fps = None
    for kind, data in zip(events.column("kind").to_pylist(),
                          events.column("data_json").to_pylist()):
        if kind == "deck_config":
            fps = json.loads(data).get("fps", fps)
    return float(fps) if fps else None


def render_video(
    run_dir: str | Path,
    out: str | Path,
    *,
    fps: float | None = None,
    fill: str = FILL_HOLD,
    lossless: bool = False,
) -> VideoStats:
    """Render a run's camera stream to a constant-rate H.264, or lossless VP9, mp4.

    ``fps`` defaults to the rate the deck's build reported, else ``DEFAULT_FPS``.
    ``fill=hold`` repeats the current frame across gaps -- every pixel written
    was really captured. ``fill=blend`` cross-fades between neighbours, which
    looks smoother and *fabricates images that never existed*: fine for a demo,
    never for perception or analysis input. ``lossless`` writes lossless VP9
    instead of H.264 (see the module docstring).
    """
    import numpy as np

    if fps is None:
        fps = deck_fps(run_dir) or DEFAULT_FPS
    if fill not in (FILL_HOLD, FILL_BLEND):
        raise ValueError(f"fill must be {FILL_HOLD!r} or {FILL_BLEND!r}, got {fill!r}.")
    frames = load_frames(run_dir)
    if not frames:
        raise ValueError(f"{run_dir} recorded no camera frames.")

    width, height = frames[0].width, frames[0].height
    mismatched = [f for f in frames if (f.width, f.height) != (width, height)]
    if mismatched:
        raise ValueError(
            f"{run_dir} mixes frame sizes ({width}x{height} and "
            f"{mismatched[0].width}x{mismatched[0].height}); a run should be one "
            f"GAP8 build."
        )

    timestamps = [f.t_s for f in frames]
    schedule = hold_schedule(timestamps, fps)
    encoder = _open_encoder(out, width=width, height=height, fps=fps, lossless=lossless)

    held = 0
    written = 0
    shown: set[int] = set()
    cache: dict[int, object] = {}
    previous = None
    try:
        for i, source in enumerate(schedule):
            if source not in cache:
                cache = {source: frames[source].decode()}  # only ever need the current
            image = cache[source]
            if image is None:
                # A frame that does not decode holds the previous image, so every
                # later frame keeps its time; before the first one that decodes
                # there is nothing to hold, and the video starts there instead.
                if previous is None:
                    continue
                image = previous
                held += 1
            else:
                if source in shown:
                    held += 1
                shown.add(source)
                if fill == FILL_BLEND and source + 1 < len(frames):
                    image = _blend(frames, source, timestamps[0] + i / fps, cache)
                previous = image
            encoder.stdin.write(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
            written += 1
    except BrokenPipeError:
        pass  # ffmpeg died; _close_encoder reports why
    _close_encoder(encoder, out)

    return VideoStats(
        output_frames=written,
        source_frames=len(frames),
        held=held,
        skipped=len(frames) - len(shown),
        width=width,
        height=height,
        fps=fps,
        duration_s=written / fps,
    )


def _open_encoder(out, *, width: int, height: int, fps: float, lossless: bool):
    """Start ffmpeg reading raw gray frames on stdin, writing H.264 or lossless VP9 mp4."""
    if lossless:
        # VP9 in lossless mode, in mp4: plays in Chromium (VSCode, Slack
        # desktop, Chrome, Edge) and Firefox.
        #
        # Full range keeps Y equal to the gray value and tags the stream so
        # players show it as such; the default limited range would squeeze
        # 0-255 into 16-235 and merge neighbouring levels.
        full_range = "scale=out_range=full,"
        # The matrix tag is what makes Chromium honour the range flag: with the
        # matrix unspecified it stretches 16-235 regardless, up to 20 levels
        # off. BT.601 is the matrix swscale converts with and the one JPEG
        # uses; chroma is flat, so any matrix gives R=G=B=Y. row-mt threads
        # within a frame: same pixels, faster on wide frames (little at QVGA).
        codec = ["-c:v", "libvpx-vp9", "-lossless", "1", "-row-mt", "1",
                 "-colorspace", "bt470bg"]
    else:
        full_range = ""
        codec = ["-c:v", "libx264", "-preset", H264_PRESET, "-crf", H264_CRF]
    command = [
        ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y",
        # Input: exactly the bytes we hand over, no container, no probing.
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
        "-an",  # the deck sends no audio; do not let ffmpeg look for any
        *codec,
        # yuv420p is the half people forget: H.264 in another pixel format
        # still will not play in a browser. It needs even dimensions, and the
        # camera can be built for odd ones (96x96 is even, 99x99 would not be),
        # so pad rather than fail. A no-op for every current mode.
        "-vf", f"pad=ceil(iw/2)*2:ceil(ih/2)*2,{full_range}format=yuv420p",
        # Put the index at the front so the file plays before it is fully read.
        "-movflags", "+faststart",
        str(out),
    ]
    return subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _close_encoder(encoder, out) -> None:
    try:
        encoder.stdin.close()
    except BrokenPipeError:
        pass
    stderr = encoder.stderr.read().decode("utf-8", "replace").strip()
    encoder.stderr.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {out}:\n{stderr}")


def _blend(frames, source, t, cache):
    import cv2

    a, b = frames[source], frames[source + 1]
    span = b.t_s - a.t_s
    if span <= 0:
        return cache[source]
    alpha = min(1.0, max(0.0, (t - a.t_s) / span))
    nxt = b.decode()
    if nxt is None:
        return cache[source]
    return cv2.addWeighted(cache[source], 1.0 - alpha, nxt, alpha, 0.0)


def extract_frames(run_dir: str | Path, out_dir: str | Path, *, limit: int = 0) -> int:
    """Write each recorded frame as its own file, as received.

    The blob is the right storage for a 120 FPS run; this is the escape hatch
    for when you want to look at, or feed out, individual images. JPEG frames
    are written verbatim (no re-encode); raw frames become PNG.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_frames(run_dir)
    if limit:
        frames = frames[:limit]
    for i, frame in enumerate(frames):
        stem = f"{i:06d}_id{frame.frame_id}"
        if frame.is_jpeg:
            (out_dir / f"{stem}.jpg").write_bytes(frame.payload)
        else:
            import cv2

            image = frame.decode()
            if image is not None:
                cv2.imwrite(str(out_dir / f"{stem}.png"), image)
    return len(frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="A telemetry run directory")
    parser.add_argument("-o", "--out", default="",
                        help="Output video (default <run>/camera.mp4, "
                             "or <run>/camera_lossless.mp4 with --lossless)")
    parser.add_argument("--fps", type=float,
                        help=f"Playback rate (default: the rate the deck reported, else {DEFAULT_FPS:g})")
    parser.add_argument("--fill", choices=(FILL_HOLD, FILL_BLEND), default=FILL_HOLD,
                        help="hold repeats real frames; blend fabricates in-between ones")
    parser.add_argument("--lossless", action="store_true",
                        help="Lossless VP9 instead of H.264")
    parser.add_argument("--extract", default="", help="Instead write frames to this directory")
    args = parser.parse_args(argv)

    if args.extract:
        count = extract_frames(args.run_dir, args.extract)
        print(f"Wrote {count} frame(s) to {args.extract}")
        return 0

    name = "camera_lossless.mp4" if args.lossless else "camera.mp4"
    out = args.out or str(Path(args.run_dir) / name)
    stats = render_video(args.run_dir, out, fps=args.fps, fill=args.fill, lossless=args.lossless)
    print(
        f"{out}: {stats.output_frames} frames @ {stats.fps:g} fps "
        f"({stats.duration_s:.1f}s, {stats.width}x{stats.height})\n"
        f"  {stats.source_frames} recorded, {stats.held} held, {stats.skipped} skipped"
    )
    if stats.skipped:
        print(f"  {stats.skipped} frame(s) never shown; raise --fps to see them all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
