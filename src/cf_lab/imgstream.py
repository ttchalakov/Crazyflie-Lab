"""Decode a cf_lab image stream off a CPX link.

The host half of ``lib/shared/include/cf_lab/img_protocol.h``. One decoder, shared by
every example that streams frames.

The wire, at CPX packet-payload boundaries::

    [ one CPX payload: 12-byte header | ext_len bytes ]
                                         app-specific
    [ one or more CPX payloads: size image bytes total ]

The header is fixed and self-describing, so this decoder handles frames from *any*
cf_lab app without knowing which one. The extension is opaque: it is handed back as
raw bytes for the app to interpret (camera_stream puts capture timings there; an app
with nothing to add sends none). ``ext_len`` is what
lets a generic viewer find the payload regardless. Keeping the header and extension
in their own CPX packet also provides an in-band recovery boundary: after UDP loss,
the decoder can abandon an incomplete frame when the next structurally valid header
packet arrives. This is best-effort recovery, not an integrity layer; the wire has no
per-chunk sequence number or checksum, so compound payload/header loss or datagram
reordering cannot always be detected. Use a reliable transport when payload integrity
is required.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable, Iterator

CF_IMG_MAGIC = 0xBC
# No frame is this large (GAP8 L2 is 512 KB): a header claiming more is corrupt.
MAX_IMAGE_SIZE = 4 << 20

CF_IMG_TYPE_RAW_GRAY = 0
CF_IMG_TYPE_JPEG = 1

# Must match cf_img_header_t. Field order in the C struct groups 8-bit, then 16, then
# 32, precisely so this format string reads the same as the struct.
IMG_HEADER = struct.Struct("<BBBBHHI")
assert IMG_HEADER.size == 12, "cf_img_header_t is 12 bytes (CF_IMG_HEADER_SIZE)"


class StreamDesync(Exception):
    """The packet decoder cannot recover safely, for example from an implausible
    declared size or a transport closure in the middle of a frame."""


@dataclass(frozen=True)
class ImgHeader:
    magic: int
    type: int
    depth: int
    ext_len: int
    width: int
    height: int
    size: int

    @property
    def is_jpeg(self) -> bool:
        return self.type == CF_IMG_TYPE_JPEG


@dataclass(frozen=True)
class ImgFrame:
    header: ImgHeader
    payload: bytes
    ext: bytes  # app-specific; empty unless the app attached one

    @property
    def width(self) -> int:
        return self.header.width

    @property
    def height(self) -> int:
        return self.header.height


def parse_header(data: bytes) -> ImgHeader | None:
    """Parse a header packet, or return None if it is not one.

    Returning None rather than raising is deliberate: a reader that starts mid-stream
    sees payload chunks before it sees a header, and it should skip them and
    resynchronise on the next structurally valid header packet rather than die. Magic
    byte 0xBC by itself is not enough; the packet length must also match ``ext_len``.
    """
    if len(data) < IMG_HEADER.size:
        return None
    magic, typ, depth, ext_len, width, height, size = IMG_HEADER.unpack_from(data)
    if magic != CF_IMG_MAGIC:
        return None
    if len(data) != IMG_HEADER.size + ext_len:
        return None
    return ImgHeader(magic, typ, depth, ext_len, width, height, size)


def frames(
    recv_packet: Callable[[], bytes],
    *,
    recover_incomplete: bool = False,
) -> Iterator[ImgFrame]:
    """Yield ``ImgFrame`` from a stream of CPX packet payloads.

    ``recv_packet()`` must return one whole CPX packet's data each call. The sender
    puts the header and extension in one dedicated packet, followed by image packets.
    With ``recover_incomplete=True``, a structurally valid header is an in-band
    boundary even if the previous body is incomplete, and an overshot body is
    abandoned while decoding waits for the next boundary. Camera UDP enables this so
    isolated packet loss discards a frame without terminating the session. The
    default is strict: a header-shaped packet inside a body remains payload and an
    overshoot raises ``StreamDesync``, preserving reliable/TCP behavior and avoiding
    any header-versus-payload ambiguity on a lossless link.

    Packet alignment is not end-to-end integrity: without per-chunk sequence numbers
    or checksums, compound payload/header loss or UDP reordering can be undetectable.
    Factoring packet receipt out also lets tests drive the parser with canned bytes
    and lets the decoder work over TCP, UDP, or another packet-preserving transport.
    """
    header: ImgHeader | None = None
    ext = b""
    body = bytearray()

    while True:
        packet = recv_packet()
        next_header = parse_header(packet)
        if recover_incomplete and next_header is not None and next_header.size > MAX_IMAGE_SIZE:
            # Image data that only looks like a header: no frame is that large.
            next_header = None

        if next_header is not None and (header is None or recover_incomplete):
            if next_header.size > MAX_IMAGE_SIZE:
                raise StreamDesync(f"implausible image size {next_header.size}")

            # In recovery mode a header is also an in-band recovery point. UDP can
            # lose one whole CPX payload packet while retaining packet alignment; if
            # the following frame's header arrives before the old body reaches its
            # advertised size, abandon the incomplete frame and start cleanly.
            header = next_header
            ext = bytes(packet[IMG_HEADER.size : IMG_HEADER.size + header.ext_len])
            body.clear()

            if header.size == 0:
                yield ImgFrame(header=header, payload=b"", ext=ext)
                header = None
            continue

        if header is None:
            continue  # stray payload while attaching mid-stream; wait for a header.
        if not packet:
            raise StreamDesync("stream closed mid-frame")

        body.extend(packet)
        if len(body) == header.size:
            yield ImgFrame(header=header, payload=bytes(body), ext=ext)
            header = None
            ext = b""
            body.clear()
        elif len(body) > header.size:
            if not recover_incomplete:
                raise StreamDesync(
                    f"got {len(body)} bytes, header said {header.size}"
                )
            # The advertised body cannot be recovered by splitting this CPX payload:
            # image headers are their own packet.  Drop the corrupt frame and wait
            # for the next structurally valid header boundary instead of terminating
            # a UDP run.
            header = None
            ext = b""
            body.clear()
