"""Deck naming from radio.uri, mDNS lookup with retries, and the camera/mocap route check."""

import socket

import pytest

from cf_lab.camera import find_deck, preflight
from cf_lab.camera.find_deck import deck_hostname, deck_name, resolve_deck
from cf_lab.config import CameraConfig, LabConfig, MocapConfig, NatNetConfig, RadioConfig, VrpnConfig

URI = "radio://0/80/2M/E7E7E7E7E7"


@pytest.mark.parametrize("uri, name, hostname", [
    (URI, "80/2M/E7E7E7E7E7", "cf-80-2m-e7e7e7e7e7"),
    ("radio://1/80/2M/E7E7E7E7E7", "80/2M/E7E7E7E7E7", "cf-80-2m-e7e7e7e7e7"),  # other dongle
    ("radio://0/125/250K/a1", "125/250K/a1", "cf-125-250k-a1"),
    ("radio://*/7/1M/0123456789", "7/1M/0123456789", "cf-7-1m-0123456789"),
])
def test_deck_names_identify_the_drone_not_the_dongle(uri, name, hostname):
    assert deck_name(uri) == name
    assert deck_hostname(uri) == hostname


@pytest.mark.parametrize("uri", [
    "radio://0/80/2M",                        # no address
    "radio://0/80/3M/E7E7E7E7E7",             # no such rate
    "radio://0/80/2M/E7E7E7E7E7E7",           # 12 hex digits
    "radio://0/80/2M/0xE7E7E7E7E7",
    "usb://0",
    "radio://0/80/2M/E7E7E7E7E7/extra",
])
def test_deck_name_rejects_uris_that_name_no_drone(uri):
    with pytest.raises(ValueError, match="radio.uri"):
        deck_hostname(uri)


def _getaddrinfo_script(monkeypatch, results):
    """Patch getaddrinfo: each call pops an address, or raises gaierror for None."""
    calls = []

    def getaddrinfo(host, port, family=0, *args):
        calls.append((host, family))
        result = results.pop(0)
        if result is None:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (result, 0))]

    monkeypatch.setattr(find_deck.socket, "getaddrinfo", getaddrinfo)
    return calls


def test_resolve_deck_asks_for_the_ipv4_address_of_the_local_name(monkeypatch):
    calls = _getaddrinfo_script(monkeypatch, ["192.0.2.10"])

    assert resolve_deck(URI) == "192.0.2.10"
    assert calls == [("cf-80-2m-e7e7e7e7e7.local", socket.AF_INET)]


def test_resolve_deck_keeps_asking_a_booting_deck_until_it_answers(monkeypatch, fake_clock):
    monkeypatch.setattr(find_deck, "time", fake_clock)
    calls = _getaddrinfo_script(monkeypatch, [None, None, None, "192.168.4.1"])

    assert resolve_deck(URI, timeout_s=10.0) == "192.168.4.1"
    assert len(calls) == 4


def test_resolve_deck_gives_up_at_the_timeout_with_an_explanation(monkeypatch, fake_clock):
    monkeypatch.setattr(find_deck, "time", fake_clock)
    _getaddrinfo_script(monkeypatch, [None] * 100)

    with pytest.raises(OSError, match="cf-80-2m-e7e7e7e7e7.local") as raised:
        resolve_deck(URI, timeout_s=2.0)
    assert isinstance(raised.value.__cause__, socket.gaierror)
    assert sum(fake_clock.sleeps) == pytest.approx(2.0)


def test_resolve_deck_without_timeout_tries_once(monkeypatch, fake_clock):
    monkeypatch.setattr(find_deck, "time", fake_clock)
    calls = _getaddrinfo_script(monkeypatch, [None, "10.0.0.1"])

    with pytest.raises(OSError):
        resolve_deck(URI)
    assert len(calls) == 1 and fake_clock.sleeps == []


def _cfg(*, ssid="dummy-ssid", transport="natnet", server_ip="10.1.1.1", ros_host="10.2.2.2"):
    return LabConfig(
        radio=RadioConfig(uri=URI),
        camera=CameraConfig(ssid=ssid),
        mocap=MocapConfig(transport=transport, natnet=NatNetConfig(server_ip=server_ip),
                          vrpn=VrpnConfig(ros_host=ros_host)),
    )


@pytest.fixture
def routes(monkeypatch):
    """Fake routing table: destination -> the local source address the kernel would pick."""
    table = {}
    monkeypatch.setattr(preflight, "resolve_deck", lambda uri: "192.168.4.1")
    monkeypatch.setattr(preflight, "source_address_for", lambda ip, port=9: table.get(ip))
    return table


@pytest.mark.parametrize("camera_src, mocap_src, shared", [
    ("192.168.4.2", "10.1.1.50", False),   # dual NIC: wired mocap + WiFi to the deck
    ("10.1.1.50", "10.1.1.50", True),      # single NIC: joining the deck's AP drops mocap
    (None, None, False),                   # nothing routes to the deck
])
def test_check_routes_flags_a_shared_interface(routes, camera_src, mocap_src, shared):
    routes.update({"192.168.4.1": camera_src, "10.1.1.1": mocap_src})

    check = preflight.check_routes(_cfg())

    assert (check.camera_ip, check.mocap_ip) == ("192.168.4.1", "10.1.1.1")
    assert check.shared_interface is shared


def test_check_routes_uses_the_vrpn_bridge_host_for_vrpn(routes):
    routes.update({"192.168.4.1": "10.2.2.50", "10.2.2.2": "10.2.2.50"})

    check = preflight.check_routes(_cfg(transport="vrpn"))

    assert check.mocap_ip == "10.2.2.2" and check.shared_interface


def test_check_routes_does_not_apply_with_the_camera_off(routes):
    assert preflight.check_routes(_cfg(ssid="")) is None


def test_source_address_for_returns_the_local_end_not_the_destination():
    # 127.0.0.2 is routed over lo from 127.0.0.1: the source differs from the destination.
    assert preflight.source_address_for("127.0.0.2") == "127.0.0.1"
