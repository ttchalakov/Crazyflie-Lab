"""The shared image wire protocol: header parsing and frame assembly (strict TCP and lossy UDP)."""

import pytest

from cf_lab.imgstream import IMG_HEADER, ImgHeader, StreamDesync, frames, parse_header


def _take(source, count, **kwargs):
    gen = frames(source, **kwargs)
    return [next(gen) for _ in range(count)]


@pytest.mark.parametrize("type_, depth, ext, width, height, size", [
    (0, 1, b"", 320, 240, 76800),
    (1, 1, bytes(range(36)), 324, 244, 3400),
    (1, 1, bytes(255), 65535, 65535, 0xFFFFFFFF),
])
def test_parse_header_round_trips_every_field(type_, depth, ext, width, height, size):
    packet = IMG_HEADER.pack(0xBC, type_, depth, len(ext), width, height, size) + ext

    assert parse_header(packet) == ImgHeader(0xBC, type_, depth, len(ext), width, height, size)


def test_header_layout_matches_the_c_struct():
    # cf_img_header_t: magic, type, depth, ext_len (u8), width, height (u16 LE), size (u32 LE).
    packet = bytes([0xBC, 1, 1, 0, 0x40, 0x01, 0xF0, 0x00, 0x78, 0x56, 0x34, 0x12])

    assert parse_header(packet) == ImgHeader(0xBC, 1, 1, 0, 320, 240, 0x12345678)


@pytest.mark.parametrize("packet", [
    b"",
    IMG_HEADER.pack(0xBC, 1, 1, 0, 320, 240, 100)[:11],                  # truncated
    IMG_HEADER.pack(0xBD, 1, 1, 0, 320, 240, 100),                       # wrong magic
    IMG_HEADER.pack(0xBC, 1, 1, 4, 320, 240, 100) + b"abc",              # ext short
    IMG_HEADER.pack(0xBC, 1, 1, 4, 320, 240, 100) + b"abcde",            # ext long
    IMG_HEADER.pack(0xBC, 1, 1, 0, 320, 240, 100) + bytes(1008),         # MTU payload chunk
], ids=["empty", "truncated", "magic", "ext-short", "ext-long", "payload-chunk"])
def test_parse_header_rejects_non_header_packets(packet):
    assert parse_header(packet) is None


def test_real_jpeg_stream_decodes_byte_exact(wire, deck_frames):
    packets = []
    for f in deck_frames:
        packets += wire.frame_payloads(f["payload"], width=320, height=240, ext=f["ext"])

    got = _take(wire.packet_source(packets), len(deck_frames))

    assert [(g.payload, g.ext) for g in got] == [(f["payload"], f["ext"]) for f in deck_frames]
    assert all(g.header.is_jpeg and (g.width, g.height) == (320, 240) for g in got)


def test_attaching_mid_frame_skips_stray_chunks_until_a_header(wire, deck_frames):
    first, second = deck_frames[:2]
    tail_of_first = wire.frame_payloads(first["payload"], width=320, height=240)[2:]
    packets = tail_of_first + wire.frame_payloads(second["payload"], width=320, height=240)

    (got,) = _take(wire.packet_source(packets), 1)

    assert got.payload == second["payload"]


def test_zero_size_frame_is_yielded_and_the_next_frame_still_decodes(wire):
    packets = [IMG_HEADER.pack(0xBC, 0, 1, 2, 0, 0, 0) + b"xy",
               *wire.frame_payloads(b"\x07" * 2500, width=50, height=50, type_=0)]

    empty, full = _take(wire.packet_source(packets), 2)

    assert (empty.payload, empty.ext) == (b"", b"xy")
    assert full.payload == b"\x07" * 2500


def test_strict_mode_keeps_a_header_shaped_chunk_as_payload(wire):
    # A raw frame whose 12-byte tail chunk happens to look exactly like a header.
    lookalike = IMG_HEADER.pack(0xBC, 0, 1, 0, 8, 8, 64)
    payload = bytes(1020) + lookalike
    packets = wire.frame_payloads(payload, width=29, height=36, type_=0)

    (got,) = _take(wire.packet_source(packets), 1)

    assert got.payload == payload


def test_strict_overshoot_raises_instead_of_yielding_a_corrupt_frame(wire):
    packets = wire.frame_payloads(bytes(1500), width=10, height=10, size=1400)

    with pytest.raises(StreamDesync):
        _take(wire.packet_source(packets), 1)


def test_strict_short_frame_raises_instead_of_gluing_the_next_frame_on(wire, deck_frames):
    a, b = deck_frames[:2]
    short_a = wire.frame_payloads(a["payload"], width=320, height=240, ext=a["ext"])[:-1]
    packets = short_a + wire.frame_payloads(b["payload"], width=320, height=240, ext=b["ext"])

    got = []
    with pytest.raises(StreamDesync):
        for frame in frames(wire.packet_source(packets)):
            got.append(frame)
    assert got == []


def test_empty_packet_mid_frame_reports_closed_stream(wire):
    packets = wire.frame_payloads(bytes(3000), width=10, height=10)[:2] + [b""]

    with pytest.raises(StreamDesync, match="closed"):
        _take(wire.packet_source(packets), 1)


def test_implausible_declared_size_raises_before_buffering(wire):
    packets = [IMG_HEADER.pack(0xBC, 1, 1, 0, 320, 240, (4 << 20) + 1)]

    with pytest.raises(StreamDesync, match="implausible"):
        _take(wire.packet_source(packets), 1)


def _udp_frames(wire, deck_frames, n):
    return [wire.frame_payloads(f["payload"], width=320, height=240, ext=f["ext"])
            for f in deck_frames[:n]]


@pytest.mark.parametrize("lost", ["middle-chunk", "last-chunk", "header"])
def test_recovery_drops_only_the_damaged_frame(wire, deck_frames, lost):
    per_frame = _udp_frames(wire, deck_frames, 3)
    damaged = per_frame[1]
    index = {"middle-chunk": 2, "last-chunk": len(damaged) - 1, "header": 0}[lost]
    del damaged[index]
    packets = [p for frame in per_frame for p in frame]

    got = _take(wire.packet_source(packets), 2, recover_incomplete=True)

    assert [g.payload for g in got] == [deck_frames[0]["payload"], deck_frames[2]["payload"]]
    assert got[1].ext == deck_frames[2]["ext"]


def test_recovery_abandons_an_overshot_frame_and_resumes_at_the_next_header(wire, deck_frames):
    # Losing frame 1's last chunk *and* frame 2's header glues frame 2's body onto frame 1.
    per_frame = _udp_frames(wire, deck_frames, 4)
    del per_frame[1][-1]
    del per_frame[2][0]
    packets = [p for frame in per_frame for p in frame]

    got = _take(wire.packet_source(packets), 2, recover_incomplete=True)

    assert [g.payload for g in got] == [deck_frames[0]["payload"], deck_frames[3]["payload"]]


def test_recovery_survives_a_header_shaped_tail_chunk(wire, deck_frames):
    # A JPEG whose final chunk starts 0xBC with byte 3 == len - 12: parse_header accepts it,
    # and its random "size" is almost always > 4 MiB.
    tail = IMG_HEADER.pack(0xBC, 0x5A, 0x11, 4, 0x1234, 0x5678, 0xDEADBEEF) + b"\xff\xd9\x00\x00"
    odd = deck_frames[0]["payload"][:1020] + tail
    packets = [*wire.frame_payloads(odd, width=320, height=240),
               *wire.frame_payloads(deck_frames[1]["payload"], width=320, height=240)]

    got = []
    with pytest.raises(wire.EndOfStream):  # not StreamDesync: the session must survive
        for frame in frames(wire.packet_source(packets), recover_incomplete=True):
            got.append(frame.payload)

    # Whether the odd frame is kept or abandoned is the decoder's choice; the next one must arrive.
    assert got[-1] == deck_frames[1]["payload"]
