"""Mocap telemetry tap: record every pose the consumer pulls from a source.

Wraps any ``MocapSampleSource`` (NatNet, VRPN/rosbridge, fakes) so it slots
into ``ExtposeStreamer`` unchanged. The tap sits *before* the streamer's
rate-cap decimation and ``OriginZeroer`` transform, so the recording is the
full raw mocap stream in lab frame — even when the radio uplink is capped.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from cf_lab.mocap.stream import iter_pose_samples
from cf_lab.mocap.types import PoseSample
from cf_lab.telemetry.writer import RunWriter


class TappedPoseSource:
    def __init__(self, source, writer: RunWriter, *, stream: str = "mocap") -> None:
        self._source = source
        self._writer = writer
        self._stream = stream

    def samples(self, *, start_s: float = 0.0, dt_s: float = 0.01) -> Iterator[PoseSample]:
        for pose in iter_pose_samples(self._source, start_s=start_s, dt_s=dt_s):
            q = pose.orientation
            # Stamped when the packet arrived, if the source knows; otherwise now.
            self._writer.append_at(
                self._stream,
                (
                    pose.timestamp_s,
                    -1 if pose.source_seq is None else pose.source_seq,
                    *pose.position_m,
                    q.qx,
                    q.qy,
                    q.qz,
                    q.qw,
                    -1.0 if pose.latency_s is None else pose.latency_s * 1e6,
                ),
                host_ns=time.monotonic_ns() if pose.host_ns is None else pose.host_ns,
            )
            yield pose

    def connected(self, within_s: float) -> bool:
        connected = getattr(self._source, "connected", None)
        return connected is not None and connected(within_s)

    def close(self) -> None:
        close = getattr(self._source, "close", None)
        if close is not None:
            close()


class TappedExtposeBackend:
    """Record every pose forwarded to a backend's ``send_extpose``.

    This is the ground truth of what went up the radio: post origin-zeroing,
    post rate-cap. ``source_seq`` cross-references rows to the raw ``mocap``
    stream. Everything except ``send_extpose`` delegates to the real backend,
    so the wrapper drops into ``ExtposeStreamer`` (and session code holding the
    backend) unchanged.
    """

    def __init__(self, backend, writer: RunWriter, *, stream: str = "extpose") -> None:
        self._backend = backend
        self._writer = writer
        self._stream = stream

    def send_extpose(self, pose: PoseSample) -> None:
        q = pose.orientation
        self._writer.append(
            self._stream,
            (
                pose.timestamp_s,
                -1 if pose.source_seq is None else pose.source_seq,
                *pose.position_m,
                q.qx,
                q.qy,
                q.qz,
                q.qw,
            ),
        )
        self._backend.send_extpose(pose)

    def __getattr__(self, name: str):
        return getattr(self._backend, name)
