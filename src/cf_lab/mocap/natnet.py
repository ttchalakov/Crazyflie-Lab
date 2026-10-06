"""Pure-Python NatNet pose source for OptiTrack/Motive — no ROS, no deps.

Motive streams rigid-body pose over its native NatNet protocol (UDP). This
module joins the NatNet stream directly with the standard library and yields
``PoseSample`` objects, so the host can read OptiTrack pose without
``vrpn_client_ros`` + ``rosbridge`` (the chain documented in
examples/mocap_positioning/README.md). It is a drop-in replacement for
:class:`RosbridgePoseSource`.

Conventions: the stream is passed through unchanged, so set Motive's streaming
**Up Axis to Z** (Data Streaming pane) — exactly the gotcha the VRPN path also
relies on — and the frame is already +z up in metres to match the estimator.

Both modes query Motive's command port to resolve the asset name before reading
poses. Multicast data needs no registration; unicast registers for streaming.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time
from collections.abc import Iterator
from queue import Queue

from cf_lab.config import NatNetConfig
from cf_lab.mocap.types import PoseSample, Quaternion

# NatNet message ids (header is "<HH" = message_id, payload_byte_count).
NAT_CONNECT = 0
NAT_SERVERINFO = 1
NAT_REQUEST_MODELDEF = 4
NAT_MODELDEF = 5
NAT_FRAMEOFDATA = 7
NAT_KEEPALIVE = 10


def _rigid_body_descriptions(data: bytes, major: int, minor: int) -> list[tuple[str, int]]:
    """Read standalone asset names/IDs from NatNet 2.x–4.x model descriptions."""
    offset = 0

    def skip(size: int) -> None:
        nonlocal offset
        if size < 0 or offset + size > len(data):
            raise ValueError("Truncated NatNet model description")
        offset += size

    def integer() -> int:
        start = offset
        skip(4)
        return struct.unpack_from("<i", data, start)[0]

    def count() -> int:
        value = integer()
        if value < 0 or value > len(data):
            raise ValueError("Invalid NatNet description count")
        return value

    def string() -> str:
        start = offset
        end = data.find(b"\0", start)
        if end < 0:
            raise ValueError("Unterminated NatNet description name")
        skip(end - start + 1)
        return data[start:end].decode("utf-8")

    def rigid_body() -> tuple[str, int]:
        name = string()
        ident = integer()
        skip(16)  # parent ID + position offset
        if major >= 3:
            markers = count()
            skip(markers * 16)  # XYZ offsets + active labels
            if major >= 4:
                for _ in range(markers):
                    string()
        return name, ident

    if major < 2:
        raise ValueError("Rigid-body name lookup requires NatNet 2.0 or newer")
    bodies = []
    for _ in range(count()):
        kind = integer()
        if (major, minor) >= (4, 1):
            size = count()
            end = offset + size
            if end > len(data):
                raise ValueError("Truncated NatNet description record")
            if kind == 1:
                name, ident = string(), integer()
                if offset + 16 > end:
                    raise ValueError("Truncated rigid-body description")
                bodies.append((name, ident))
            offset = end  # Skip other versioned asset types by their record size.
        elif kind == 0:  # marker set
            string()
            for _ in range(count()):
                string()
        elif kind == 1:
            bodies.append(rigid_body())
        elif kind == 2:  # skeleton: bones are not standalone assets
            string()
            integer()
            for _ in range(count()):
                rigid_body()
        elif kind == 3 and major >= 3:  # force plate
            integer()
            string()
            skip(8 + 12 + 12 * 12 * 4 + 4 * 3 * 4 + 8)
            for _ in range(count()):
                string()
        elif kind == 4 and major >= 3:  # device
            integer()
            string()
            string()
            skip(8)
            for _ in range(count()):
                string()
        elif kind == 5:  # camera
            string()
            skip(28)
        else:
            raise ValueError(f"Unsupported NatNet description type {kind}")
    return bodies


class NatNetPoseSource:
    """Read rigid-body pose straight from a Motive NatNet stream."""

    def __init__(
        self,
        *,
        server_ip: str,
        rigid_body: str = "cf21bl_01",
        multicast_ip: str = NatNetConfig.multicast_ip,
        command_port: int = NatNetConfig.command_port,
        data_port: int = NatNetConfig.data_port,
        local_ip: str = "",
        use_multicast: bool = True,
    ) -> None:
        self._server_ip = server_ip
        self._rigid_body = rigid_body
        if not rigid_body.strip():
            raise ValueError("mocap.rigid_body must be a Motive asset name")
        self._rigid_id: int | None = None
        self._multicast_ip = multicast_ip
        self._command_port = command_port
        self._data_port = data_port
        self._local_ip = local_ip or "0.0.0.0"
        self._use_multicast = use_multicast
        self._queue: Queue[PoseSample] = Queue(maxsize=1)  # only the newest pose
        self._data_sock: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Motive's NatNet protocol version; >= 3.0 dropped inline marker data
        # from the rigid-body block. Default to 3 if the probe finds nothing.
        self._nn_major = 3
        self._nn_minor = 0
        self.server_name = ""
        # Motive's high-resolution clock (NatNet 3.0+), for frame latency.
        self._clock_hz = 0
        self._arrival_ns: int | None = None

    # -- lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        self._rigid_id = None
        self._probe_server_info()
        if not self.server_name:
            raise RuntimeError("Cannot query Motive: check the NatNet server IP and command port")
        self._rigid_id = self._resolve_rigid_body()
        self._data_sock = self._open_data_socket()
        if not self._use_multicast:
            # Unicast: register with Motive (NAT_CONNECT) so it starts streaming
            # frames to us; the recv loop then keeps the registration alive with
            # periodic keepalives. (Multicast needs no handshake at all.)
            self._send_command(NAT_CONNECT)
            self._send_command(NAT_KEEPALIVE)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._recv_loop, name="natnet-recv", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._data_sock is not None:
            try:
                self._data_sock.close()
            finally:
                self._data_sock = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def __enter__(self) -> "NatNetPoseSource":
        self.connect()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def samples(self, *, start_s: float = 0.0, dt_s: float = 0.01) -> Iterator[PoseSample]:
        del start_s, dt_s
        if self._thread is None:
            self.connect()
        while not self._stop.is_set():
            yield self._queue.get()

    def connected(self, within_s: float) -> bool:
        """Whether Motive sent anything in the last ``within_s``.

        Motive streams a frame per camera frame even while the rigid body is
        untracked, so this tells an untracked body from a lost connection.
        """
        thread = self._thread
        arrival_ns = self._arrival_ns
        return (
            thread is not None
            and thread.is_alive()
            and arrival_ns is not None
            and time.monotonic_ns() - arrival_ns <= within_s * 1e9
        )

    # -- sockets -------------------------------------------------------------

    def _open_data_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        if self._use_multicast:
            sock.bind(("", self._data_port))
            mreq = socket.inet_aton(self._multicast_ip) + socket.inet_aton(self._local_ip)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        else:
            sock.bind((self._local_ip, self._data_port))
        sock.settimeout(1.0)
        return sock

    def _probe_server_info(self) -> None:
        """Ping the command port to confirm reachability and learn the version.

        connect() requires a successful reply before decoding model descriptions.
        """
        self.server_name = ""
        try:
            cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            cmd.settimeout(0.75)
            with cmd:
                cmd.bind((self._local_ip, 0))
                cmd.sendto(struct.pack("<HH", NAT_CONNECT, 0), (self._server_ip, self._command_port))
                data, _ = cmd.recvfrom(2048)
            message_id = struct.unpack("<H", data[0:2])[0]
            if message_id == NAT_SERVERINFO and len(data) >= 4 + 264:
                payload = data[4:]
                self.server_name = payload[0:256].split(b"\0", 1)[0].decode("utf-8", "replace")
                self._nn_major = payload[260]
                self._nn_minor = payload[261]
                if self._nn_major >= 3 and len(payload) >= 272:
                    (self._clock_hz,) = struct.unpack_from("<Q", payload, 264)
        except OSError:
            pass  # connect() reports failure before starting the receive task.

    def _resolve_rigid_body(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as cmd:
            cmd.bind((self._local_ip, 0))
            cmd.connect((self._server_ip, self._command_port))
            for _ in range(3):
                cmd.send(struct.pack("<HH", NAT_REQUEST_MODELDEF, 0))
                deadline = time.monotonic() + 0.75
                while (remaining := deadline - time.monotonic()) > 0:
                    cmd.settimeout(remaining)
                    try:
                        packet = cmd.recv(65535)
                    except socket.timeout:
                        break
                    if len(packet) < 4:
                        continue
                    kind, size = struct.unpack_from("<HH", packet)
                    if kind != NAT_MODELDEF:
                        continue
                    if len(packet) != size + 4:
                        raise ValueError("Truncated NatNet model descriptions")
                    bodies = _rigid_body_descriptions(packet[4:], self._nn_major, self._nn_minor)
                    matches = [ident for name, ident in bodies if name == self._rigid_body]
                    if len(matches) == 1:
                        return matches[0]
                    names = ", ".join(name for name, _ in bodies) or "(none)"
                    raise ValueError(
                        f"Motive rigid body {self._rigid_body!r} must match exactly one asset; "
                        f"found {len(matches)}. Available rigid bodies: {names}"
                    )
        raise RuntimeError("Motive did not return model descriptions for rigid-body name lookup")

    def _send_command(self, message_id: int) -> None:
        """Send a header-only NatNet command (e.g. NAT_CONNECT/NAT_KEEPALIVE)."""
        sock = self._data_sock
        if sock is None:
            return
        try:
            sock.sendto(
                struct.pack("<HH", message_id, 0), (self._server_ip, self._command_port)
            )
        except OSError:
            pass

    # -- receive + parse -----------------------------------------------------

    def _recv_loop(self) -> None:
        last_keepalive = 0.0
        while not self._stop.is_set():
            if not self._use_multicast and time.monotonic() - last_keepalive > 1.0:
                self._send_command(NAT_KEEPALIVE)
                last_keepalive = time.monotonic()
            sock = self._data_sock
            if sock is None:
                return
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            self._arrival_ns = time.monotonic_ns()
            pose = self._pose_from_packet(data)
            if pose is not None:
                self._publish(pose)

    def _publish(self, pose: PoseSample) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Exception:
                pass
        self._queue.put_nowait(pose)

    def _pose_from_packet(self, data: bytes) -> PoseSample | None:
        if len(data) < 4:
            return None
        message_id, _size = struct.unpack("<HH", data[0:4])
        if message_id != NAT_FRAMEOFDATA:
            return None
        try:
            (frame_number,) = struct.unpack_from("<i", data, 4)
            rb = self._first_matching_rigid_body(data, 4)
        except (struct.error, IndexError, ValueError):
            return None
        if rb is None:
            return None
        _rb_id, position, quat = rb
        # A corrupt frame, or an untracked body before NatNet 2.6 (no tracking
        # flag: Motive sends it as a zero pose), must neither reach the drone nor
        # end this thread, which a zero quaternion's normalization would.
        if not all(map(math.isfinite, (*position, *quat))) or not any(quat):
            return None
        return PoseSample(
            timestamp_s=time.time(),
            position_m=position,
            orientation=Quaternion(qx=quat[0], qy=quat[1], qz=quat[2], qw=quat[3]).normalized(),
            rigid_body=self._rigid_body,
            source_seq=frame_number,
            host_ns=self._arrival_ns,
            latency_s=self._latency_s(data),
        )

    def _latency_s(self, data: bytes) -> float | None:
        """Motive's mid-exposure-to-transmit time for this frame, or None.

        The frame suffix holds both stamps on Motive's high-resolution clock.
        Reaching it means skipping every section, which only NatNet 4.1+ makes
        safe (each section is prefixed with its byte size): marker sets, other
        markers, rigid bodies, skeletons, assets, labeled markers, force plates
        and devices; then timecode, timecode sub-frame and the timestamp.
        """
        if not self._size_prefixed or self._clock_hz <= 0:
            return None
        try:
            offset = 8  # message header, frame number
            for _ in range(8):
                offset = self._skip_sized_section(data, offset)
            offset += 4 + 4 + 8
            mid_exposure, _received, transmit = struct.unpack_from("<QQQ", data, offset)
        except struct.error:
            return None
        latency = (transmit - mid_exposure) / self._clock_hz
        return latency if 0.0 <= latency < 1.0 else None

    def _first_matching_rigid_body(self, data: bytes, offset: int):
        """Parse a NatNet FrameOfData far enough to return one rigid body.

        Returns ``(id, (x,y,z), (qx,qy,qz,qw))`` for the configured streaming id
        resolved from the configured asset name, else ``None``.
        Untracked bodies (Motive streams them as a zero pose) are skipped so we
        never forward a stale origin pose.
        """
        offset += 4  # frame number (int32)

        # Skip the marker-set and legacy "other" marker sections. NatNet 4.1+
        # prefixes each section with (count, byte_size), so we can jump the
        # whole block; older streams pack items inline and we walk them.
        if self._size_prefixed:
            offset = self._skip_sized_section(data, offset)  # marker sets
            offset = self._skip_sized_section(data, offset)  # other markers
            (n_rigid,) = struct.unpack_from("<i", data, offset)
            offset += 8  # rigid-body count + block byte_size (we parse items)
        else:
            (n_sets,) = struct.unpack_from("<i", data, offset)
            offset += 4
            for _ in range(n_sets):
                offset = data.index(b"\0", offset) + 1  # null-terminated name
                (n_markers,) = struct.unpack_from("<i", data, offset)
                offset += 4 + n_markers * 12
            (n_other,) = struct.unpack_from("<i", data, offset)
            offset += 4 + n_other * 12
            (n_rigid,) = struct.unpack_from("<i", data, offset)
            offset += 4

        for _ in range(n_rigid):
            (rb_id,) = struct.unpack_from("<i", data, offset)
            offset += 4
            x, y, z, qx, qy, qz, qw = struct.unpack_from("<7f", data, offset)
            offset += 28
            if 0 < self._nn_major < 3:
                # Pre-3.0 streams carried marker data inside each rigid body.
                (n_rb_markers,) = struct.unpack_from("<i", data, offset)
                offset += 4 + n_rb_markers * 12
                if self._nn_major >= 2:
                    offset += n_rb_markers * 8  # marker ids + sizes
            offset += 4  # mean marker error (float)
            tracked = True
            if self._nn_major >= 3 or (self._nn_major == 2 and self._nn_minor >= 6):
                (params,) = struct.unpack_from("<h", data, offset)
                offset += 2
                tracked = bool(params & 0x01)
            if not tracked:
                continue
            rb = (rb_id, (x, y, z), (qx, qy, qz, qw))
            if rb_id == self._rigid_id:
                return rb
        return None

    @property
    def _size_prefixed(self) -> bool:
        """NatNet 4.1 added a (count, byte_size) prefix to frame sections."""
        return (self._nn_major, self._nn_minor) >= (4, 1)

    @staticmethod
    def _skip_sized_section(data: bytes, offset: int) -> int:
        """Skip a NatNet 4.1+ section: int32 count, int32 byte_size, then data."""
        (byte_size,) = struct.unpack_from("<i", data, offset + 4)
        return offset + 8 + byte_size
