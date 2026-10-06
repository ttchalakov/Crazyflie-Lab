"""The route check for running the camera feed alongside mocap.

The hazard this exists for: the AI-deck in ``access_point`` mode *is* a WiFi
network, and the host has to join it. On a machine with one usable interface
that takes the host off the network carrying mocap -- and the failure is silent
and expensive, because the radio is USB and survives. You get a flight that
streams images beautifully while poses quietly stop arriving.

The tempting check does not work. ``ip route get 192.168.4.1`` succeeds on a
host that has never seen the deck, because the default route answers for every
destination; so "is there a route" is a false pass. The check that does work is
comparing *source* addresses: the kernel picks the source address for a
destination, so two destinations reached through different interfaces report
different sources. A dual-NIC host (wired mocap + WiFi to the deck) passes, a
single-NIC host does not, and neither answer is guessed from config.

``station`` mode sidesteps all of it: the deck joins an existing network, the
host stays put, and both share one interface legitimately.

One blind spot: VRPN through a rosbridge on this host is reached over loopback,
so the check cannot see which interface the bridge itself uses to reach the
mocap PC, and passes.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

from cf_lab.camera.find_deck import resolve_deck
from cf_lab.config import LabConfig


@dataclass(frozen=True)
class RouteCheck:
    """Which local address the kernel would use to reach each service."""

    camera_ip: str
    camera_source: str | None
    mocap_ip: str
    mocap_source: str | None
    shared_interface: bool


def source_address_for(dest_ip: str, port: int = 9) -> str | None:
    """The local address the kernel would send from, without sending anything.

    A UDP ``connect`` only fixes the destination in the socket and consults the
    routing table; no packet leaves. Returns None when nothing routes there.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((dest_ip, port))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()


def _mocap_server_ip(cfg: LabConfig) -> str | None:
    if cfg.mocap.transport == "natnet":
        return cfg.mocap.natnet.server_ip or None
    return cfg.mocap.vrpn.ros_host or None


def check_routes(cfg: LabConfig) -> RouteCheck | None:
    """Compare the paths to the deck and to the mocap server.

    Returns None when the question does not arise (no camera, or no mocap
    server address configured). Raises OSError when the deck cannot be looked up.
    """
    if not cfg.camera.enabled:
        return None
    mocap_ip = _mocap_server_ip(cfg)
    if mocap_ip is None:
        return None
    camera_ip = resolve_deck(cfg.radio.uri)
    camera_source = source_address_for(camera_ip, cfg.camera.deck_port)
    mocap_source = source_address_for(mocap_ip)
    return RouteCheck(
        camera_ip=camera_ip,
        camera_source=camera_source,
        mocap_ip=mocap_ip,
        mocap_source=mocap_source,
        shared_interface=(
            camera_source is not None and camera_source == mocap_source
        ),
    )
