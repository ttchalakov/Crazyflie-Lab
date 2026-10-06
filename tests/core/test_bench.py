"""cf_lab.bench: run_jobs lanes, the NetworkManager access point, flash_and_boot and the radio console."""
import contextlib
import io
import subprocess
import sys
import threading
import time
import types

import pytest

from cf_lab import bench
from cf_lab.bench import AccessPoint, BenchError, Job, RadioConsole, flash_and_boot, run_jobs


def fake_subprocess(run):
    """A stand-in for the subprocess module inside cf_lab.bench, with ``run`` replaced."""
    return types.SimpleNamespace(run=run, STDOUT=subprocess.STDOUT, DEVNULL=subprocess.DEVNULL)


# --- Job / run_jobs -----------------------------------------------------------------------------

@pytest.mark.parametrize(("codes", "ok"), [([], False), ([0], True), ([0, 0], True), ([0, 2], False),
                                           ([127], False)])
def test_job_is_ok_only_when_it_ran_and_every_command_succeeded(codes, ok):
    assert Job("j", [], codes=codes).ok is ok


def test_run_jobs_runs_every_command_and_logs_it(tmp_path, monkeypatch):
    def run(command, cwd, stdout, stderr, stdin):
        stdout.write(f"out of {command[0]} in {cwd.name}\n")
        return subprocess.CompletedProcess(command, {"sweep": 3, "report": 0}[command[0]])

    monkeypatch.setattr(bench, "subprocess", fake_subprocess(run))
    job = Job("gvsoc-a", [["sweep", "--suite", "a b"], ["report", "x.log"]])
    done = []

    run_jobs([job], tmp_path / "logs", 2, tmp_path, done.append)

    # A failing command does not skip the next one (the report reads a partial log).
    assert job.codes == [3, 0] and not job.ok
    assert done == [job] and job.minutes >= 0
    assert (tmp_path / "logs" / "gvsoc-a.log").read_text() == (
        f"$ sweep --suite 'a b'\nout of sweep in {tmp_path.name}\n"
        f"$ report x.log\nout of report in {tmp_path.name}\n"
    )


def test_run_jobs_missing_program_fails_the_job_not_the_lane(tmp_path, monkeypatch):
    def run(command, **_):
        if command[0] == "missing":
            raise FileNotFoundError(2, "No such file or directory", "missing")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bench, "subprocess", fake_subprocess(run))
    jobs = [Job("a", [["missing"]], radio=True), Job("b", [["ok"]], radio=True)]

    run_jobs(jobs, tmp_path / "logs", 1, tmp_path, lambda job: None)

    assert jobs[0].codes == [127] and jobs[1].codes == [0]
    assert "No such file or directory" in (tmp_path / "logs" / "a.log").read_text()


def test_run_jobs_radio_jobs_run_alone_in_order_beside_the_parallel_lane(tmp_path, monkeypatch):
    lock = threading.Lock()
    active = {"radio": 0, "local": 0}
    peak = {"radio": 0, "local": 0}
    started = []
    local_running = threading.Event()
    slots_full = threading.Event()

    def run(command, **_):
        lane = command[0]
        with lock:
            started.append(command[1])
            active[lane] += 1
            peak[lane] = max(peak[lane], active[lane])
            if active["local"] == 2:
                slots_full.set()
        if lane == "local":
            local_running.set()
            # Hold the first jobs until both slots are taken, so the peak never depends on timing.
            slots_full.wait(1.0)
            time.sleep(0.02)
        else:
            # The radio lane does not wait for the parallel lane to finish.
            assert local_running.wait(1.0)
            time.sleep(0.01)
        with lock:
            active[lane] -= 1
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bench, "subprocess", fake_subprocess(run))
    radio = [Job(f"r{i}", [["radio", f"r{i}"]], radio=True) for i in range(3)]
    local = [Job(f"l{i}", [["local", f"l{i}"]]) for i in range(5)]
    jobs = [local[0], radio[0], local[1], radio[1], local[2], local[3], radio[2], local[4]]
    done = []

    run_jobs(jobs, tmp_path / "logs", 2, tmp_path, lambda job: done.append(job.name))

    assert all(job.ok for job in jobs)
    assert peak == {"radio": 1, "local": 2}
    assert [name for name in started if name.startswith("r")] == ["r0", "r1", "r2"]
    assert sorted(done) == sorted(job.name for job in jobs)


def test_run_jobs_reports_one_job_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "subprocess",
                        fake_subprocess(lambda command, **_: subprocess.CompletedProcess(command, 0)))
    inside = threading.Lock()
    overlaps = []

    def on_done(job):
        if not inside.acquire(blocking=False):
            overlaps.append(job.name)
            return
        time.sleep(0.01)
        inside.release()

    jobs = [Job(f"l{i}", [["x"]]) for i in range(4)] + [Job("r", [["x"]], radio=True)]

    run_jobs(jobs, tmp_path / "logs", 4, tmp_path, on_done)

    assert overlaps == []


@pytest.mark.parametrize("radio", [False, True])
def test_run_jobs_error_reporting_one_job_still_runs_the_rest(tmp_path, monkeypatch, radio):
    monkeypatch.setattr(bench, "subprocess",
                        fake_subprocess(lambda command, **_: subprocess.CompletedProcess(command, 0)))
    monkeypatch.setattr(threading, "excepthook", lambda args: None)

    def on_done(job):
        if job.name == "first":
            raise KeyError("summary row")

    jobs = [Job("first", [["x"]], radio=radio), Job("second", [["x"]], radio=radio)]

    # Raising the error once both lanes are done is fine; skipping the next job is not.
    with contextlib.suppress(KeyError):
        run_jobs(jobs, tmp_path / "logs", 1, tmp_path, on_done)

    assert jobs[1].codes == [0]


def test_run_jobs_zero_slots_never_runs_everything_at_once(tmp_path, monkeypatch):
    lock = threading.Lock()
    active, peak = [0], [0]

    def run(command, **_):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.02)
        with lock:
            active[0] -= 1
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(bench, "subprocess", fake_subprocess(run))
    jobs = [Job(f"l{i}", [["x"]]) for i in range(4)]

    # Rejecting the value is as good as running one job at a time.
    with contextlib.suppress(ValueError):
        run_jobs(jobs, tmp_path / "logs", 0, tmp_path, lambda job: None)

    assert peak[0] <= 1


# --- AccessPoint ---------------------------------------------------------------------------------

class FakeNmcli:
    """nmcli as NetworkManager would answer it: profiles, devices and active connections."""

    def __init__(self, profiles=None, devices=("lo:loopback", "enp3s0:ethernet"), add_code=0):
        self.profiles = dict(profiles or {})
        self.devices = list(devices)
        self.active = []
        self.add_code = add_code
        self.calls = []

    def __call__(self, command, capture_output, text, timeout):
        assert command[0] == "nmcli" and capture_output and text
        args = command[1:]
        self.calls.append(args)
        out = ""
        if args == ["-t", "-f", "NAME", "connection", "show"]:
            out = "".join(f"{name}\n" for name in ["Wired connection 1", *self.profiles])
        elif args[:2] == ["-g", "802-11-wireless.ssid"]:
            out = self.profiles[args[-1]] + "\n"
        elif args == ["-t", "-f", "DEVICE,TYPE", "device", "status"]:
            out = "".join(f"{line}\n" for line in self.devices)
        elif args[:2] == ["connection", "add"]:
            return subprocess.CompletedProcess(command, self.add_code, "",
                                               "Error: failed" if self.add_code else "")
        elif args[:3] == ["-t", "-f", "NAME,STATE"]:
            out = "".join(f"{line}\n" for line in ["Wired connection 1:activated", *self.active])
        return subprocess.CompletedProcess(command, 0, out, "")

    def modifies(self):
        return [args for args in self.calls if args[:2] == ["connection", "modify"]]


@pytest.fixture
def nmcli(monkeypatch):
    def install(**kwargs):
        fake = FakeNmcli(**kwargs)
        monkeypatch.setattr(bench, "subprocess", fake_subprocess(fake))
        return fake
    return install


def option(args, name):
    return args[args.index(name) + 1]


@pytest.mark.parametrize("key", ["password1", ""])
def test_ensure_creates_a_profile_that_never_takes_the_default_route(nmcli, key):
    fake = nmcli(devices=["lo:loopback", "p2p-dev-wlo1:wifi-p2p", "enp3s0:ethernet", "wlo1:wifi",
                          "wlx00c0:wifi"])
    log = []

    AccessPoint("cf-bench", "deck-ap", key).ensure(log.append)

    (add,) = [args for args in fake.calls if args[:2] == ["connection", "add"]]
    assert option(add, "ifname") == "wlo1"
    assert option(add, "con-name") == "cf-bench" and option(add, "ssid") == "deck-ap"
    assert option(add, "ipv4.never-default") == "yes"
    assert option(add, "connection.autoconnect") == "no"
    assert option(add, "802-11-wireless.powersave") == "2"
    if key:
        assert option(add, "wifi-sec.key-mgmt") == "wpa-psk" and option(add, "wifi-sec.psk") == key
    else:
        assert not any(arg.startswith("wifi-sec") for arg in add)
    assert log == ["Created NetworkManager profile 'cf-bench' on wlo1"]


def test_ensure_without_a_wifi_device_raises(nmcli):
    nmcli(devices=["lo:loopback", "enp3s0:ethernet", "p2p-dev-wlo1:wifi-p2p"])

    with pytest.raises(BenchError, match="no WiFi device"):
        AccessPoint("cf-bench", "deck-ap").ensure(print)


def test_ensure_reports_a_failed_profile_creation(nmcli):
    nmcli(devices=["wlo1:wifi"], add_code=4)

    with pytest.raises(BenchError, match="Error: failed"):
        AccessPoint("cf-bench", "deck-ap").ensure(print)


def test_ensure_never_adds_a_second_profile_of_the_same_name(nmcli):
    # Two profiles named cf-bench make `nmcli connection up cf-bench` pick either one.
    fake = nmcli(profiles={"cf-bench": "deck-ap"})

    AccessPoint("cf-bench", "deck-ap", "password1").ensure(print)

    assert not any(args[:2] == ["connection", "add"] for args in fake.calls)


def test_ensure_points_a_stale_profile_at_the_new_secured_ssid(nmcli):
    fake = nmcli(profiles={"cf-bench": "cf-wifi-bench"})
    log = []

    AccessPoint("cf-bench", "deck-ap", "password1").ensure(log.append)

    (modify,) = fake.modifies()
    assert option(modify, "802-11-wireless.ssid") == "deck-ap"
    assert option(modify, "wifi-sec.psk") == "password1"
    assert log == ["NetworkManager profile 'cf-bench': SSID 'cf-wifi-bench' -> 'deck-ap'"]


def test_ensure_points_a_stale_profile_at_the_new_open_ssid(nmcli):
    fake = nmcli(profiles={"cf-bench": "cf-wifi-bench"})

    AccessPoint("cf-bench", "deck-ap").ensure(print)

    (modify,) = fake.modifies()
    assert option(modify, "802-11-wireless.ssid") == "deck-ap"
    assert modify[-2:] == ["remove", "802-11-wireless-security"]


def test_ensure_applies_a_changed_key_to_an_existing_profile(nmcli):
    fake = nmcli(profiles={"cf-bench": "deck-ap"})

    AccessPoint("cf-bench", "deck-ap", "new-password").ensure(print)

    assert any("new-password" in args for args in fake.modifies())


def test_is_up_needs_the_profile_activated_not_activating(nmcli):
    fake = nmcli()
    ap = AccessPoint("cf-bench", "deck-ap")
    fake.active = ["cf-bench:activating"]
    assert not ap.is_up()
    fake.active = ["cf-bench:activated"]
    assert ap.is_up()


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


def test_up_retries_until_the_profile_is_activated(nmcli, monkeypatch):
    fake = nmcli()
    clock = Clock()
    monkeypatch.setattr(bench, "time", clock)
    attempts = []
    original = fake.__call__

    def joining(command, **kwargs):
        if "up" in command:
            attempts.append(command[command.index("--wait") + 1])
            clock.now += 20
            if len(attempts) == 2:
                fake.active = ["cf-bench:activated"]
        return original(command, **kwargs)

    monkeypatch.setattr(bench, "subprocess", fake_subprocess(joining))

    AccessPoint("cf-bench", "deck-ap").up(deadline=clock.now + 45)

    # Each join waits at most 20 s.
    assert attempts == ["20", "20"]


def test_up_gives_up_at_the_deadline(nmcli, monkeypatch):
    fake = nmcli()
    clock = Clock()
    monkeypatch.setattr(bench, "time", clock)
    original = fake.__call__
    waits = []

    def never_joins(command, **kwargs):
        if "up" in command:
            waits.append(command[command.index("--wait") + 1])
            # A join that waits 0 s would never reach the deadline; fail instead of hanging.
            assert len(waits) < 10, waits
            clock.now += float(waits[-1])
        return original(command, **kwargs)

    monkeypatch.setattr(bench, "subprocess", fake_subprocess(never_joins))

    with pytest.raises(BenchError, match="never appeared"):
        AccessPoint("cf-bench", "deck-ap").up(deadline=clock.now + 25.5)

    # The joins shrink to the time left, at least 1 s, and stop once it is gone.
    assert waits == ["20", "5", "1"]


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


# --- flash_and_boot ------------------------------------------------------------------------------

@pytest.fixture
def flash_parts(monkeypatch):
    from cf_lab.aideck import flash
    from cf_lab.backends import crazyflie

    calls = []
    state = {"flash_error": None, "power_cycle": True}

    def flash_deck(artifact, uri, processor, progress_cb):
        calls.append(("flash", artifact, uri, processor))
        progress_cb("Writing to bcAI:gap8 deck memory", 50)
        if state["flash_error"]:
            raise state["flash_error"]

    def power_cycle(uri, log):
        calls.append(("power_cycle", uri))
        return state["power_cycle"]

    monkeypatch.setattr(flash, "flash_deck", flash_deck)
    monkeypatch.setattr(crazyflie, "power_cycle", power_cycle)
    return calls, state


URI = "radio://0/80/2M/E7E7E7E7E7"


def test_flash_and_boot_flashes_the_gap8_then_cold_boots(flash_parts, tmp_path):
    calls, _ = flash_parts

    flash_and_boot(tmp_path / "image.img", URI, print)

    assert calls == [("flash", tmp_path / "image.img", URI, "gap8"), ("power_cycle", URI)]


def test_flash_and_boot_failed_flash_never_boots_the_old_image(flash_parts, tmp_path, monkeypatch):
    calls, state = flash_parts
    state["flash_error"] = RuntimeError("deck flash corrupt")
    terminal = Terminal()
    monkeypatch.setattr(sys, "stdout", terminal)

    with pytest.raises(RuntimeError, match="corrupt"):
        flash_and_boot(tmp_path / "image.img", URI, print)

    assert [call[0] for call in calls] == ["flash"]
    # The half-drawn bar is closed, so the traceback starts on its own line.
    assert terminal.getvalue().endswith("deck memory\x1b[K\n")


def test_flash_and_boot_failed_power_cycle_raises(flash_parts, tmp_path):
    _, state = flash_parts
    state["power_cycle"] = False

    with pytest.raises(BenchError, match="power cycle"):
        flash_and_boot(tmp_path / "image.img", URI, print)


# --- RadioConsole --------------------------------------------------------------------------------

class FakeLink:
    def __init__(self, packets):
        self.packets = list(packets)
        self.closed = threading.Event()

    def receive_packet(self, wait):
        if self.packets:
            return self.packets.pop(0)
        time.sleep(0.002)
        return None

    def close(self):
        self.closed.set()


@pytest.fixture
def radio(monkeypatch):
    import cflib.crtp

    def install(link=None, error=None):
        def get_link_driver(uri):
            if error is not None:
                raise error
            return link

        monkeypatch.setattr(cflib.crtp, "init_drivers", lambda: None)
        monkeypatch.setattr(cflib.crtp, "get_link_driver", get_link_driver)
    return install


def packet(port, data: bytes):
    from cflib.crtp.crtpstack import CRTPPacket

    pk = CRTPPacket()
    pk.port = port
    pk.data = data
    return pk


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


def test_radio_console_reassembles_lines_across_packets(radio, tmp_path):
    from cflib.crtp.crtpstack import CRTPPort

    link = FakeLink([
        packet(CRTPPort.CONSOLE, b"CPX: GAP8: fr"),
        packet(CRTPPort.LOGGING, b"not console\n"),
        packet(CRTPPort.CONSOLE, b"ame 12\nSYS: ok\n\xffbad"),
        packet(CRTPPort.CONSOLE, b"\n"),
    ])
    radio(link=link)
    path = tmp_path / "console.log"

    console = RadioConsole(URI, path)
    assert wait_for(lambda: path.read_text().count("\n") == 3)
    console.close()
    console.close()

    lines = [line.split(" ", 1)[1] for line in path.read_text().splitlines()]
    assert lines == ["CPX: GAP8: frame 12", "SYS: ok", "�bad"]
    assert link.closed.is_set()


def test_radio_console_logs_an_unavailable_radio(radio, tmp_path):
    radio(error=Exception("No Crazyradio dongle found"))
    path = tmp_path / "console.log"

    console = RadioConsole(URI, path)
    assert wait_for(lambda: "\n" in path.read_text())
    console.close()

    assert "[bench] radio console unavailable: No Crazyradio dongle found" in path.read_text()


def test_radio_console_logs_a_uri_no_driver_accepts(radio, tmp_path):
    radio(link=None)
    path = tmp_path / "console.log"

    console = RadioConsole("radio:/0/80/2M", path)
    console._thread.join(1.0)
    console.close()

    assert "radio console unavailable" in path.read_text()
