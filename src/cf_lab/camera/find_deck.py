"""Find the AI-deck by the name its GAP8 app gives it on the network.

The deck's ESP32 answers mDNS for its hostname and advertises ``_cpx._tcp``. An
app built by ``cf-lab build gap8`` names the deck after the drone's radio
address, so the host finds the deck of the drone ``radio.uri`` names on
whatever network it is on, in station or access-point mode:

  radio://0/80/2M/E7E7E7E7E7  ->  cf-80-2m-e7e7e7e7e7.local, TXT name 80/2M/E7E7E7E7E7

The dongle index is left out: it belongs to this host, not the drone.

  cf-lab find-deck                                   # radio.uri of --config names the drone
  cf-lab find-deck --uri radio://0/80/2M/E7E7E7E7E7  # --uri names it, over --config

Either prints the deck's name and address.
"""

from __future__ import annotations

import argparse
import re
import socket
import sys
import time
from pathlib import Path

from cf_lab.config import radio_uri

RADIO_URI = re.compile(r"radio://[^/]+/(\d{1,3})/(250K|1M|2M)/([0-9A-Fa-f]{1,10})")


def deck_name(uri: str) -> str:
    """The drone's radio address, ``80/2M/E7E7E7E7E7``: the deck's TXT name."""
    match = RADIO_URI.fullmatch(uri)
    if match is None:
        raise ValueError(f"radio.uri {uri!r} is not radio://<dongle>/<channel>/<rate>/<address>")
    return "/".join(match.groups())


def deck_hostname(uri: str) -> str:
    """The deck's mDNS hostname, ``cf-80-2m-e7e7e7e7e7``, without ``.local``."""
    return "cf-" + deck_name(uri).lower().replace("/", "-")


def resolve_deck(uri: str, timeout_s: float = 0.0) -> str:
    """The IPv4 address of the deck on the drone ``uri`` names.

    Asks the system resolver (nss-mdns, through avahi) for ``<hostname>.local``,
    again until ``timeout_s`` has passed, for a deck that is still booting. One
    failed lookup takes about 5 s.
    """
    host = f"{deck_hostname(uri)}.local"
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return socket.getaddrinfo(host, None, socket.AF_INET)[0][4][0]
        except socket.gaierror as exc:
            if time.monotonic() >= deadline:
                raise OSError(
                    f"no deck answers to {host}, the name for {uri}. Is the deck on this "
                    f"network (booted, and this computer on its access point or its "
                    f"station network)? Was its GAP8 app built since radio.uri last "
                    f"changed (cf-lab flash gap8)? Does its ESP32 firmware take the name "
                    f"(cf-lab flash esp)?"
                ) from exc
            time.sleep(0.5)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/drone.yaml"),
                        help="its radio.uri names the drone")
    parser.add_argument("--uri", help="overrides radio.uri from --config")


def run(args: argparse.Namespace) -> int:
    """``cf-lab find-deck``: look up the deck of the drone --uri or --config names."""
    uri = radio_uri(args.config, args.uri)
    started = time.monotonic()
    try:
        ip = resolve_deck(uri)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{deck_hostname(uri)}.local ({deck_name(uri)}) is {ip}, "
          f"found in {(time.monotonic() - started) * 1e3:.0f} ms")
    return 0
