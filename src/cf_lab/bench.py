"""What the hardware benches share: joining the deck's access point from this
computer, putting a build on the drone, listening to its radio console, and
running a bench's jobs.

Used by examples/camera_bench, wifi_bench, nn_bench, kernel_bench, jpeg_cluster_bench,
hyperbus_bench and scheduler_bench.
"""

from __future__ import annotations

import contextlib
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


class BenchError(RuntimeError):
    """A bench step could not be completed."""


def _nmcli(*args: str, timeout_s: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(["nmcli", *args], capture_output=True, text=True, timeout=timeout_s)


@dataclass(frozen=True)
class AccessPoint:
    """The deck's access point, joined through a NetworkManager profile of our own.

    The profile binds to the first WiFi device and never becomes the default
    route, so a wired connection keeps the internet while a bench is on the
    deck's network. Power saving is off: a sleeping host radio stalls a stream.
    """

    profile: str
    ssid: str
    key: str = ""

    def ensure(self, log) -> None:
        """Create the profile, or point an existing one at this SSID and key."""
        names = _nmcli("-t", "-f", "NAME", "connection", "show").stdout.splitlines()
        if self.profile in names:
            ssid = _nmcli("-g", "802-11-wireless.ssid", "connection", "show", self.profile).stdout.strip()
            # Rewrite SSID and security every time: a profile left on an old SSID
            # or key never completes the join, and a new key, or a switch between
            # an open and a secured access point, can keep the same SSID.
            security = (["wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", self.key] if self.key
                        else ["remove", "802-11-wireless-security"])
            result = _nmcli("connection", "modify", self.profile, "802-11-wireless.ssid", self.ssid,
                            *security)
            if result.returncode != 0:
                raise BenchError(f"could not update NetworkManager profile {self.profile}: "
                                 f"{result.stderr.strip()}")
            if ssid != self.ssid:
                log(f"NetworkManager profile {self.profile!r}: SSID {ssid!r} -> {self.ssid!r}")
            return
        devices = _nmcli("-t", "-f", "DEVICE,TYPE", "device", "status").stdout.splitlines()
        wifi = [line.split(":")[0] for line in devices if line.endswith(":wifi")]
        if not wifi:
            raise BenchError("no WiFi device on this computer")
        security = ["wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", self.key] if self.key else []
        result = _nmcli("connection", "add", "type", "wifi", "ifname", wifi[0],
                        "con-name", self.profile, "ssid", self.ssid,
                        "connection.autoconnect", "no", "ipv4.never-default", "yes",
                        "ipv4.ignore-auto-dns", "yes", "ipv6.method", "ignore",
                        "802-11-wireless.powersave", "2", *security)
        if result.returncode != 0:
            raise BenchError(f"could not create NetworkManager profile {self.profile}: "
                             f"{result.stderr.strip()}")
        log(f"Created NetworkManager profile {self.profile!r} on {wifi[0]}")

    def is_up(self) -> bool:
        # --active also lists a connection that is still activating.
        active = _nmcli("-t", "-f", "NAME,STATE", "connection", "show", "--active").stdout.splitlines()
        return f"{self.profile}:activated" in active

    def up(self, deadline: float) -> None:
        """Join, retrying while the deck boots, until the monotonic deadline."""
        while True:
            if self.is_up():
                return
            _nmcli("device", "wifi", "rescan")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BenchError(f"could not join {self.ssid!r}: the deck's access point never appeared")
            _nmcli("--wait", str(int(min(20, max(remaining, 1)))), "connection", "up", self.profile,
                   timeout_s=25)

    def down(self) -> None:
        _nmcli("connection", "down", self.profile)


def flash_idle_controller(uri: str, log_file, log) -> None:
    """Put the idle controller on the STM32, so nothing else talks to the deck."""
    from cf_lab.firmware import build_controller, controller_artifact
    from cf_lab.sdk import sdk_root
    from cf_lab.stm32_flash import flash_stm32

    app = sdk_root() / "examples" / "idle" / "controller"
    log("STM32: building and flashing the idle controller")
    with contextlib.redirect_stdout(log_file):
        build_controller(app, output=log_file)
        flash_stm32(artifact=controller_artifact(app), uri=uri)


def flash_and_boot(image: Path, uri: str, log) -> None:
    """Flash a GAP8 image and cold-boot the drone, so the build starts from power-up."""
    from cf_lab.aideck.flash import flash_deck
    from cf_lab.backends.crazyflie import power_cycle
    from cf_lab.progress import FlashProgress

    progress = FlashProgress()
    try:
        flash_deck(artifact=image, uri=uri, processor="gap8", progress_cb=progress)
    finally:
        progress.end()
    # The ESP32 restarts its access point and the deck starts fresh.
    if not power_cycle(uri, log=log):
        raise BenchError("power cycle failed")


class RadioConsole:
    """The drone's radio console, written line by line to a file.

    A bare radio link that only listens, so it attaches at once, even while the
    drone is still booting, and never competes with a cflib connection. An
    active radio link disturbs the deck's WiFi, so a bench that measures WiFi
    throughput closes it before measuring.
    """

    def __init__(self, uri: str, path: Path) -> None:
        self._uri = uri
        self._file = path.open("a", buffering=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="radio-console", daemon=True)
        self._thread.start()

    def _write(self, line: str) -> None:
        self._file.write(f"{time.strftime('%H:%M:%S')} {line}\n")

    def _run(self) -> None:
        import cflib.crtp
        from cflib.crtp.crtpstack import CRTPPort

        cflib.crtp.init_drivers()
        try:
            link = cflib.crtp.get_link_driver(self._uri)
            reason = f"no driver for {self._uri}"
        except Exception as exc:
            link, reason = None, str(exc)
        if link is None:
            self._write(f"[bench] radio console unavailable: {reason}")
            return
        text = ""
        try:
            while not self._stop.is_set():
                packet = link.receive_packet(0.1)
                if packet is None or packet.port != CRTPPort.CONSOLE:
                    continue
                text += bytes(packet.data).decode("utf-8", "replace")
                while "\n" in text:
                    line, text = text.split("\n", 1)
                    self._write(line)
        finally:
            link.close()

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=5)
        self._file.close()


@dataclass
class Job:
    """One step of a bench: commands run in order, their output in one log.

    A job that uses the radio runs alone; the others (GVSOC simulations, host
    tools) run side by side.
    """

    name: str
    commands: list[list[str]]
    radio: bool = False
    codes: list[int] = field(default_factory=list)
    minutes: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.codes) and not any(self.codes)


def idle_controller_job() -> Job:
    """A radio job that puts the idle controller on the STM32 (see
    flash_idle_controller), with the cf-lab installed beside this Python, which
    need not be on PATH."""
    from cf_lab.sdk import sdk_root

    cf_lab = Path(sys.executable).with_name("cf-lab")
    app = sdk_root() / "examples" / "idle" / "controller"
    return Job("idle-controller", [[str(cf_lab), "flash", "controller", "--app-dir", str(app)]], radio=True)


def run_jobs(jobs: list[Job], logs: Path, slots: int, cwd: Path,
             on_done: Callable[[Job], None]) -> None:
    """Run the radio jobs one at a time and the rest ``slots`` at once (at least
    one), both lanes together. Each job's output goes to ``logs/<name>.log``;
    ``on_done`` is called as each job finishes, one call at a time. An error from
    ``on_done`` does not stop the jobs after it: the first is raised at the end."""
    logs.mkdir(parents=True, exist_ok=True)
    done_lock = threading.Lock()
    errors: list[Exception] = []

    def run(job: Job) -> None:
        started = time.monotonic()
        with (logs / f"{job.name}.log").open("w") as out:
            for command in job.commands:
                out.write(f"$ {shlex.join(command)}\n")
                out.flush()
                try:
                    job.codes.append(subprocess.run(command, cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
                                                    stdin=subprocess.DEVNULL).returncode)
                except OSError as exc:  # a missing program fails the job, not the lane
                    out.write(f"{exc}\n")
                    job.codes.append(127)
        job.minutes = (time.monotonic() - started) / 60
        with done_lock:
            try:
                on_done(job)
            except Exception as exc:
                errors.append(exc)

    def parallel(batch: list[Job]) -> None:
        free = threading.Semaphore(max(1, slots))
        threads = []
        for job in batch:
            free.acquire()

            def work(job: Job = job) -> None:
                try:
                    run(job)
                finally:
                    free.release()

            thread = threading.Thread(target=work, daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

    lanes = [threading.Thread(target=parallel, args=([j for j in jobs if not j.radio],), daemon=True),
             threading.Thread(target=lambda: [run(j) for j in jobs if j.radio], daemon=True)]
    for lane in lanes:
        lane.start()
    for lane in lanes:
        lane.join()
    if errors:
        raise errors[0]
