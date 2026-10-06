"""Decoders for the app-specific extension on a cf_lab image frame.

``cf_lab.imgstream`` deliberately hands the extension back as opaque bytes: the
12-byte header is shared by every app, and what each app bolts on top is its own
business. That separation lets an app change its extension without touching the
header, and nothing here is allowed to erode it -- ``imgstream`` must never import
this module.

What lives here is the other half: given that you *know* which app you are
talking to, how to read its extension. Frames whose extension is a different
size (another app, or a build without one) decode to ``None`` and still carry a
perfectly good image.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable, Protocol


class FrameExt(Protocol):
    """What the recorder needs from any app's extension.

    An app whose extension cannot supply these can decode to ``None``; the
    frame is still recorded, with the columns left at -1.
    """

    frame_id: int
    deck_ms: int

    @property
    def done_ms(self) -> int:
        """When the frame was complete in its buffer, on the deck's clock: the
        frame's timestamp for telemetry."""


ExtDecoder = Callable[[bytes], "FrameExt | None"]

# camera_stream_frame_ext_t in lib/gap8/camera_stream. Capture/encode timings plus
# the deck's cumulative drop counters.
CAMERA_STREAM_EXT = struct.Struct("<IIHHHHIIIII")


@dataclass(frozen=True)
class CameraStreamExt:
    """camera_stream's extension.

    ``frame_id``/``deck_ms`` identify and timestamp the frame *on the deck's own
    clock*, which shares no origin with the host's. Their spacing is the deck's
    true capture cadence, free of network jitter; their absolute value means
    nothing here. The ``dropped_*`` counters are cumulative and are the only
    account of frames the deck decided not to send at all -- no host-side
    measurement can see those.
    """

    frame_id: int
    deck_ms: int
    encode_ms: int
    capture_ms: int
    preprocess_ms: int
    jpeg_ms: int
    camera_frame_id: int
    camera_done_ms: int
    dropped_capture: int
    dropped_ready: int
    dropped_streaming: int

    @property
    def done_ms(self) -> int:
        # deck_ms is when the buffer was queued, about a frame earlier.
        return self.camera_done_ms


def decode_camera_stream(ext: bytes) -> CameraStreamExt | None:
    if len(ext) != CAMERA_STREAM_EXT.size:
        return None
    return CameraStreamExt(*CAMERA_STREAM_EXT.unpack(ext))


# sched_frame_ext_t in lib/gap8/sched (hover_neural_network, scheduler_bench).
SCHED_EXT = struct.Struct("<BBHIIIII")
SCHED_EXT_VERSION = 2  # SCHED_FRAME_EXT_VERSION in sched.c


@dataclass(frozen=True)
class SchedFrameExt:
    """lib/gap8/sched's extension.

    ``deck_ms`` is when the frame was complete in its buffer. ``stm32_anchor_ms``
    and ``deck_anchor_ms`` are the newest STM32-stamped message the deck had
    received and when it arrived on the deck's clock (-1 when none): pairs that
    let telemetry.sync map the deck's clock onto the STM32's.
    """

    version: int
    kind: int
    encode_us: int
    frame_id: int
    deck_ms: int
    inferences: int
    stm32_anchor_ms: int
    deck_anchor_ms: int

    @property
    def done_ms(self) -> int:
        return self.deck_ms


def decode_sched(ext: bytes) -> SchedFrameExt | None:
    if len(ext) != SCHED_EXT.size or ext[0] != SCHED_EXT_VERSION:
        return None
    v = SCHED_EXT.unpack(ext)
    anchored = v[6] != 0 or v[7] != 0
    return SchedFrameExt(v[0], v[1], v[2] * 8, v[3], v[4], v[5],
                         v[6] if anchored else -1, v[7] if anchored else -1)


def decode_deck_ext(ext: bytes) -> CameraStreamExt | SchedFrameExt | None:
    """Any extension this repo's deck apps send, told apart by size."""
    return decode_camera_stream(ext) or decode_sched(ext)


# The deck's console lines on the image socket. "config" lists every camera and
# stream setting of the build (camera_stream_describe plus the app's own; sent
# before the first frame of each connection); "status" is the LOG=1 counter line
# from lib/gap8/camera_stream_log. Both are "k=v" words after a prefix.
DECK_LINE_PREFIXES = (("camera config:", "config"), ("status ", "status"))


@dataclass(frozen=True)
class DeckLine:
    """One console line from the deck; ``fields`` holds its ``k=v`` words."""

    kind: str  # "config", "status" or "console"
    text: str
    fields: dict[str, int | str]


def parse_deck_line(text: str) -> DeckLine:
    for prefix, kind in DECK_LINE_PREFIXES:
        if text.startswith(prefix):
            fields: dict[str, int | str] = {}
            for word in text[len(prefix):].split():
                key, sep, value = word.partition("=")
                if sep:
                    fields[key] = int(value) if value.removeprefix("-").isdigit() else value
            return DeckLine(kind, text, fields)
    return DeckLine("console", text, {})
