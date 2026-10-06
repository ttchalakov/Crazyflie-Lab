"""The cflib radio backend: TOC cache, connect deadlines and retries, console capture, extpose."""
import threading

import cflib.crazyflie
import cflib.crtp
import pytest

import cf_lab.backends.crazyflie as backend_mod
from cf_lab.backends.crazyflie import CrazyflieRadioBackend, connect, toc_cache
from cf_lab.mocap.types import PoseSample, Quaternion


def _checkout(root):
    (root / "lib").mkdir(parents=True)
    (root / "external").mkdir()
    return root


def test_toc_cache_lives_in_the_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("CF_LAB", str(_checkout(tmp_path / "cf-lab")))

    assert toc_cache() == str((tmp_path / "cf-lab" / "artifacts" / "cache").resolve())


def test_toc_cache_is_off_without_a_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("CF_LAB", str(tmp_path / "pip-only"))

    assert toc_cache() is None


def _no_callbacks_left(cf):
    return not cf.connected.callbacks and not cf.connection_failed.callbacks


def test_connect_returns_when_the_link_reports_connected_from_its_thread(fake_cf_class):
    def on_open(cf, uri):
        threading.Thread(target=cf.connected.call, args=(uri,)).start()

    cf = fake_cf_class(on_open=on_open)

    connect(cf, "radio://0/80/2M/E7", timeout_s=2.0)

    assert cf.opened == ["radio://0/80/2M/E7"]
    assert cf.closed == 0
    assert _no_callbacks_left(cf)


def test_connect_failure_without_retry_raises_with_the_link_message(fake_cf_class):
    cf = fake_cf_class(on_open=lambda cf, uri: cf.connection_failed.call(uri, "Too many packets lost"))

    with pytest.raises(ConnectionError, match="Too many packets lost"):
        connect(cf, "radio://0/80/2M/E7")

    assert len(cf.opened) == 1
    assert cf.closed == 1
    assert _no_callbacks_left(cf)


def test_connect_times_out_and_closes_a_hung_toc_download(fake_cf_class):
    cf = fake_cf_class(on_open=lambda cf, uri: None)  # one lost TOC packet: no callback ever

    with pytest.raises(TimeoutError):
        connect(cf, "radio://0/80/2M/E7", timeout_s=0.05)

    assert cf.closed == 1
    assert _no_callbacks_left(cf)


def test_connect_retries_a_booting_drone_until_it_answers(fake_cf_class, clock, monkeypatch):
    monkeypatch.setattr(backend_mod, "time", clock.module())
    attempts = []

    def on_open(cf, uri):
        attempts.append(uri)
        if len(attempts) < 3:
            cf.connection_failed.call(uri, "no ack")
        else:
            cf.connected.call(uri)

    cf = fake_cf_class(on_open=on_open)

    connect(cf, "radio://0/80/2M/E7", timeout_s=10.0, retry=True)

    assert len(cf.opened) == 3
    assert cf.closed == 2  # each failed attempt is closed before the next
    assert _no_callbacks_left(cf)


def test_connect_retry_gives_up_at_the_deadline(fake_cf_class, clock, monkeypatch):
    monkeypatch.setattr(backend_mod, "time", clock.module())
    cf = fake_cf_class(on_open=lambda cf, uri: cf.connection_failed.call(uri, "no ack"))

    with pytest.raises(ConnectionError, match="no ack"):
        connect(cf, "radio://0/80/2M/E7", timeout_s=2.0, retry=True)

    assert 3 <= len(cf.opened) <= 6
    assert cf.closed == len(cf.opened)
    assert _no_callbacks_left(cf)


@pytest.fixture
def radio(fake_cf_class, connected_on_open, monkeypatch, tmp_path):
    """CrazyflieRadioBackend.open() against a fake cflib Crazyflie."""
    monkeypatch.setenv("CF_LAB", str(_checkout(tmp_path / "cf-lab")))
    monkeypatch.setattr(cflib.crtp, "init_drivers", lambda *a, **k: None)
    created = []

    def make(rw_cache=None):
        cf = fake_cf_class(rw_cache, on_open=connected_on_open)
        created.append(cf)
        return cf

    monkeypatch.setattr(cflib.crazyflie, "Crazyflie", make)
    return created


def test_radio_backend_opens_with_the_toc_cache_and_closes_the_link(radio):
    with CrazyflieRadioBackend("radio://0/80/2M/E7") as backend:
        cf = backend.crazyflie
        assert cf.rw_cache == toc_cache()
        assert cf.opened == ["radio://0/80/2M/E7"]

    assert cf.closed == 1
    with pytest.raises(RuntimeError):
        backend.crazyflie


def test_failed_open_leaves_the_backend_closed_and_the_link_released(radio, fake_cf_class, monkeypatch):
    cf = fake_cf_class(on_open=lambda cf, uri: cf.connection_failed.call(uri, "no ack"))
    monkeypatch.setattr(cflib.crazyflie, "Crazyflie", lambda rw_cache=None: cf)
    backend = CrazyflieRadioBackend("radio://0/80/2M/E7", console_sink=print)

    with pytest.raises(ConnectionError):
        backend.open()

    assert cf.closed == 1
    assert not cf.console.receivedChar.callbacks
    with pytest.raises(RuntimeError):
        backend.crazyflie


@pytest.mark.parametrize(
    "chunks",
    [
        ["boot ok\nGAP8 up\npart", "ial"],
        ["bo", "ot ok\n", "GAP8", " up\n", "p", "artial"],
        ["boot ok\nGAP8 up\npartial"],
    ],
)
def test_console_chunks_are_reassembled_into_lines(radio, chunks):
    lines = []
    with CrazyflieRadioBackend("radio://0/80/2M/E7", console_sink=lines.append) as backend:
        for chunk in chunks:
            backend.crazyflie.console.receivedChar.call(chunk)
        assert lines == ["boot ok", "GAP8 up"]

    assert lines == ["boot ok", "GAP8 up", "partial"]  # the trailing partial line is flushed on close


def test_blank_console_lines_are_kept(radio):
    lines = []
    with CrazyflieRadioBackend("radio://0/80/2M/E7", console_sink=lines.append) as backend:
        backend.crazyflie.console.receivedChar.call("a\n\nb\n")

    assert lines == ["a", "", "b"]


def test_close_releases_the_link_even_when_the_console_sink_fails(radio):
    def sink(line):
        raise OSError("debug log closed")

    backend = CrazyflieRadioBackend("radio://0/80/2M/E7", console_sink=sink)
    backend.open()
    cf = backend.crazyflie
    cf.console.receivedChar.call("partial line without newline")

    with pytest.raises(OSError):
        backend.close()

    assert cf.closed == 1


def test_extpose_sends_position_and_unit_quaternion(radio):
    pose = PoseSample(
        timestamp_s=0.0,
        position_m=(0.1, -0.2, 0.3),
        orientation=Quaternion(qx=1.0, qy=2.0, qz=3.0, qw=4.0),  # distinct so a swapped axis shows
        rigid_body="cf",
    )
    with CrazyflieRadioBackend("radio://0/80/2M/E7") as backend:
        backend.send_extpose(pose)
        (name, args), = backend.crazyflie.calls

    assert name == "extpos.send_extpose"
    assert args[:3] == (0.1, -0.2, 0.3)
    assert args[3:] == pytest.approx(tuple(c / 30**0.5 for c in (1.0, 2.0, 3.0, 4.0)))
