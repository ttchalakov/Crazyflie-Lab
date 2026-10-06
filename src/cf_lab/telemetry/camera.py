"""Record the AI-deck image stream into a run directory.

Frames land in two files. Payloads are appended verbatim to
``camera_frames.bin``; ``camera.parquet`` gets one index row per frame pointing
into it. Pixels stay out of Parquet on purpose -- ``RunWriter`` finalizes by
reading a whole table into memory, and raw QVGA at 60 FPS is ~276 MB/minute.
The blob is written *before* its index row, so a row always points at bytes that
exist and a crash costs at most the trailing frame.

Nothing here decodes or re-encodes: whatever resolution and format the GAP8 was
compiled for is what reaches disk.

Backpressure follows the rule the rest of telemetry follows: the feed's receive
thread only ever does a ``put_nowait`` here. A full queue drops the frame and
counts it -- disk I/O must never stall the socket and start dropping datagrams
inside the kernel instead, where nothing can count them.
"""

from __future__ import annotations

import queue
import threading

from cf_lab.camera.ext import DeckLine
from cf_lab.camera.feed import CapturedFrame
from cf_lab.telemetry.schema import CAMERA_BLOB_FILE
from cf_lab.telemetry.writer import RunWriter

# ~2 s of QVGA at 30 FPS. Deep enough to ride out a disk hiccup, shallow enough
# that a wedged disk is visible as drops rather than unbounded memory growth.
DEFAULT_QUEUE_SIZE = 64


class CameraRecorder:
    """A ``FrameSink`` that writes frames to the run directory.

    Owns its own queue and writer thread rather than borrowing ``RunWriter``'s:
    the blob append is this stream's alone, and ``RunWriter.append`` is already
    non-blocking and thread-safe, so the index row can simply be handed over
    from here with no change to the writer at all.
    """

    def __init__(self, writer: RunWriter, *, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self._writer = writer
        self._queue: queue.Queue[CapturedFrame | None] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._blob = None
        self._offset = 0
        self.frames_written = 0
        self.dropped = 0
        self.error: Exception | None = None

    def start(self) -> None:
        self._blob = (self._writer.run_dir / CAMERA_BLOB_FILE).open("wb")
        self._thread = threading.Thread(
            target=self._run, name="telemetry-camera", daemon=True
        )
        self._thread.start()

    def offer(self, frame: CapturedFrame) -> None:
        """Called on the feed's receive thread. Never blocks, never raises."""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def offer_console(self, line: DeckLine) -> None:
        """Record a deck console line as a ``deck_<kind>`` event; never blocks."""
        self._writer.event(f"deck_{line.kind}", line.text, line.fields)

    def stop(self) -> None:
        """Drain, flush, and record the drop count. Idempotent."""
        if self._thread is None:
            return
        self._queue.put(None)  # sentinel: drain what is queued, then exit
        self._thread.join(timeout=5.0)
        self._thread = None
        if self._blob is not None:
            self._blob.close()
            self._blob = None
        self._writer.event(
            "camera_recorded",
            f"{self.frames_written} frame(s), {self._offset} byte(s)",
            {"frames": self.frames_written, "dropped": self.dropped,
             "bytes": self._offset, "error": str(self.error) if self.error else None},
        )

    def _run(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            if self.error is not None:
                # Recording has stopped; emptying the queue still lets stop()
                # hand over its sentinel.
                self.dropped += 1
                continue
            try:
                self._write(frame)
            except Exception as exc:  # recording stops; the flight does not
                self.error = exc

    def _write(self, frame: CapturedFrame) -> None:
        offset = self._offset
        self._blob.write(frame.payload)
        self._offset += len(frame.payload)
        self.frames_written += 1
        ext = frame.ext
        self._writer.append_at(
            "camera",
            (
                ext.frame_id if ext is not None else -1,
                ext.done_ms if ext is not None else -1,
                frame.header.width,
                frame.header.height,
                frame.header.type,
                frame.header.depth,
                len(frame.payload),
                offset,
                frame.raw_ext,
                getattr(ext, "stm32_anchor_ms", -1),
                getattr(ext, "deck_anchor_ms", -1),
            ),
            host_ns=frame.host_ns,
        )
