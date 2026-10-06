"""cf_lab.cli: command dispatch, radio URI precedence, and the fail-closed preflight command."""
import json

import pytest

from cf_lab import cli, firmware
from cf_lab.aideck import dory_pipeline
from cf_lab.camera import find_deck, viewer
from cf_lab.preflight.core import PreflightResult
from cf_lab.sdk import SdkError

CONFIG_URI = "radio://0/80/2M/E7E7E7E7E7"
OTHER_URI = "radio://0/90/2M/E7E7E7E7E9"


@pytest.fixture
def calls(monkeypatch):
    """Replace a command's function with a recorder of the parsed arguments."""
    seen = []

    def replace(module, name):
        # Qualified, since dory_pipeline, viewer and find_deck all name theirs run.
        target = f"{module.__name__.rsplit('.', 1)[-1]}.{name}"

        def record(args):
            seen.append((target, args))
            return 0
        monkeypatch.setattr(module, name, record)

    for module, name in [
        (firmware, "_build_controller"), (firmware, "_build_gap8"), (firmware, "_build_esp"),
        (firmware, "_flash_controller"), (firmware, "_flash_gap8"), (firmware, "_flash_esp"),
        (dory_pipeline, "run"), (viewer, "run"), (find_deck, "run"),
        (cli, "_reboot"), (cli, "_shutdown"),
    ]:
        replace(module, name)
    return seen


DORY = ["--model-module", "m.py", "--checkpoint", "c.pt", "--input-shape", "1,1,1",
        "--calibration", "c.npy", "--integrate-into", "gap8/net"]


@pytest.mark.parametrize(("argv", "target", "expected"), [
    (["build", "controller", "--example", "lqr"], "firmware._build_controller", {"example": "lqr"}),
    (["build", "gap8", "--app-dir", "app", "--wifi-mode", "station"], "firmware._build_gap8",
     {"wifi_mode": "station"}),
    (["build", "esp"], "firmware._build_esp", {}),
    (["flash", "controller", "--example", "lqr", "--uri", OTHER_URI, "--no-build"],
     "firmware._flash_controller", {"uri": OTHER_URI, "no_build": True}),
    (["flash", "gap8", "--example", "camera_stream", "--cold-boot"], "firmware._flash_gap8",
     {"cold_boot": True, "uri": None}),
    (["flash", "esp", "--uri", OTHER_URI], "firmware._flash_esp", {"uri": OTHER_URI, "cold_boot": False}),
    (["compile-model", *DORY], "dory_pipeline.run", {"input_shape": "1,1,1"}),
    (["view", "--transport", "udp"], "viewer.run", {"transport": "udp"}),
    (["find-deck", "--uri", OTHER_URI], "find_deck.run", {"uri": OTHER_URI}),
    (["reboot", "--uri", OTHER_URI], "cli._reboot", {"uri": OTHER_URI}),
    (["shutdown"], "cli._shutdown", {"uri": None}),
])
def test_main_dispatches_each_command_to_its_function(calls, argv, target, expected):
    assert cli.main(argv) == 0

    ((name, args),) = calls
    assert name == target
    assert {key: getattr(args, key) for key in expected} == expected


@pytest.mark.parametrize("argv", [
    ["flash", "gap8", "--example", "a", "--app-dir", "b"],
    ["build", "controller", "--uri", OTHER_URI],
    ["flash", "esp", "--example", "a"],
    ["flash"],
])
def test_main_rejects_arguments_a_command_does_not_take(calls, argv):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == 2
    assert calls == []


def test_main_reports_an_sdk_error_as_exit_1(monkeypatch, capsys):
    def fail(args):
        raise SdkError("Docker is not running")

    monkeypatch.setattr(firmware, "_build_esp", fail)

    assert cli.main(["build", "esp"]) == 1
    assert capsys.readouterr().err == "error: Docker is not running\n"


# --- find-deck, reboot, shutdown: which drone ---------------------------------------------------

@pytest.fixture
def resolved(monkeypatch):
    uris = []

    def resolve_deck(uri, timeout_s=0.0):
        uris.append(uri)
        return "192.0.2.7"

    monkeypatch.setattr(find_deck, "resolve_deck", resolve_deck)
    return uris


def test_find_deck_uri_beats_a_missing_config(resolved, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert cli.main(["find-deck", "--uri", OTHER_URI]) == 0

    assert resolved == [OTHER_URI]
    assert capsys.readouterr().out.startswith("cf-90-2m-e7e7e7e7e9.local (90/2M/E7E7E7E7E9) is 192.0.2.7")


@pytest.mark.parametrize("command", ["find-deck", "reboot", "shutdown"])
def test_a_missing_config_without_uri_names_both_options(command, tmp_path, capsys):
    missing = tmp_path / "drone.yaml"

    assert cli.main([command, "--config", str(missing)]) == 1

    assert capsys.readouterr().err == f"error: {missing} not found; pass --uri or --config\n"


def test_find_deck_uses_radio_uri_of_the_config(resolved, lab_yaml, capsys):
    assert cli.main(["find-deck", "--config", str(lab_yaml)]) == 0

    assert resolved == [CONFIG_URI]
    assert "is 192.0.2.7" in capsys.readouterr().out


def test_find_deck_lookup_failure_exits_1(monkeypatch, lab_yaml, capsys):
    def resolve_deck(uri, timeout_s=0.0):
        raise OSError("no deck answers to cf-80-2m-e7e7e7e7e7.local")

    monkeypatch.setattr(find_deck, "resolve_deck", resolve_deck)

    assert cli.main(["find-deck", "--config", str(lab_yaml)]) == 1
    assert capsys.readouterr().err == "error: no deck answers to cf-80-2m-e7e7e7e7e7.local\n"


@pytest.mark.parametrize(("command", "function"), [("reboot", "power_cycle_until_healthy"),
                                                   ("shutdown", "power_off")])
@pytest.mark.parametrize(("uri_args", "expected_uri", "ok"), [
    ([], CONFIG_URI, True), (["--uri", OTHER_URI], OTHER_URI, True), ([], CONFIG_URI, False),
])
def test_power_commands_target_the_right_drone_and_report_failure(
    monkeypatch, lab_yaml, command, function, uri_args, expected_uri, ok
):
    from cf_lab.backends import crazyflie

    uris = []

    def power(uri, **_):
        uris.append(uri)
        return ok

    monkeypatch.setattr(crazyflie, function, power)

    assert cli.main([command, "--config", str(lab_yaml), *uri_args]) == (0 if ok else 1)
    assert uris == [expected_uri]


# --- preflight ------------------------------------------------------------------------------------

class Ran(list):
    """The checks that ran; ``replace(name, ...)`` swaps in a failing or raising one."""


class Radio:
    """A stand-in for CrazyflieRadioBackend that records how it was used."""

    def __init__(self, open_error=None):
        self.open_error = open_error
        self.events = []
        self.crazyflie = object()

    def __call__(self, uri):
        self.events.append(("new", uri))
        return self

    def open(self):
        self.events.append("open")
        if self.open_error:
            raise self.open_error

    def close(self):
        self.events.append("close")


@pytest.fixture
def radio(monkeypatch):
    def install(open_error=None):
        backend = Radio(open_error)
        monkeypatch.setattr(cli, "CrazyflieRadioBackend", backend)
        return backend
    return install


@pytest.fixture
def checks(monkeypatch):
    """Replace the radio checks with ones that pass and record their calls."""
    ran = Ran()

    def fake(name, passed=True, error=None):
        def check(cf, *args, **kwargs):
            ran.append((name, cf, args, kwargs))
            if error:
                raise error
            return PreflightResult(check_id=name, passed=passed, summary=f"{name} ran")
        monkeypatch.setattr(cli, f"check_{name}", check)

    for name in ("battery", "supervisor_ready", "expected_parameters", "required_decks"):
        fake(name)
    ran.replace = fake
    return ran


def preflight(capsys, *argv):
    code = cli.main(["preflight", "--json", *argv])
    return code, json.loads(capsys.readouterr().out)


def test_preflight_defaults_to_the_configuration_check(radio, lab_yaml, capsys):
    backend = radio()

    code, report = preflight(capsys, "--config", str(lab_yaml))

    assert code == 0 and report["passed"]
    assert [r["check_id"] for r in report["results"]] == ["configuration"]
    assert backend.events == []


@pytest.mark.parametrize("text", ["camera:\n  mode: hotspot\n", "- not a mapping\n", None])
def test_preflight_unloadable_config_fails_without_touching_the_radio(
    radio, checks, write_config, tmp_path, capsys, text
):
    backend = radio()
    path = write_config(text) if text is not None else tmp_path / "missing.yaml"

    code, report = preflight(capsys, "--config", str(path), "--checks", "battery")

    assert code == 1 and not report["passed"]
    (result,) = report["results"]
    assert result["check_id"] == "configuration_load" and not result["passed"]
    assert backend.events == [] and checks == []


def test_preflight_runs_named_checks_once_in_order_on_one_radio_link(radio, checks, lab_yaml, capsys):
    backend = radio()

    code, report = preflight(capsys, "--config", str(lab_yaml),
                             "--checks", " Supervisor-Ready,battery", "--checks", "battery,supervisor_ready")

    assert code == 0
    assert [name for name, *_ in checks] == ["supervisor_ready", "battery"]
    assert all(cf is backend.crazyflie for _, cf, *_ in checks)
    assert backend.events == [("new", CONFIG_URI), "open", "close"]


def test_preflight_closes_the_radio_when_a_check_raises(radio, checks, lab_yaml, capsys):
    backend = radio()
    checks.replace("battery", error=RuntimeError("log block refused"))

    code, report = preflight(capsys, "--config", str(lab_yaml), "--checks", "battery")

    assert code == 1
    assert "log block refused" in report["results"][0]["error"]
    assert backend.events[-1] == "close"


def test_preflight_unavailable_radio_fails_every_radio_check(radio, checks, lab_yaml, capsys):
    backend = radio(open_error=OSError("Cannot find a Crazyradio Dongle"))

    code, report = preflight(capsys, "--config", str(lab_yaml),
                             "--checks", "configuration,battery,supervisor_ready")

    assert code == 1
    passed = {r["check_id"]: r["passed"] for r in report["results"]}
    assert passed == {"configuration": True, "battery": False, "supervisor_ready": False}
    assert "Cannot find a Crazyradio Dongle" in report["results"][1]["error"]
    # Opened once, closed after the failed open, never handed to a check.
    assert backend.events == [("new", CONFIG_URI), "open", "close"]
    assert checks == []


@pytest.mark.parametrize(("stop", "expected"), [
    (True, ["configuration"]), (False, ["configuration", "battery"]),
])
def test_preflight_stop_on_failure_never_opens_the_radio(
    radio, checks, write_config, capsys, stop, expected
):
    backend = radio()
    path = write_config("mocap:\n  transport: carrier-pigeon\n")

    code, report = preflight(capsys, "--config", str(path), "--checks", "configuration,battery",
                             *(["--stop-on-failure"] if stop else []))

    assert code == 1
    assert [r["check_id"] for r in report["results"]] == expected
    assert (backend.events == []) is stop


@pytest.mark.parametrize("source_fails", [True, False])
def test_preflight_unavailable_mocap_fails_its_check_and_closes_the_source(
    monkeypatch, lab_yaml, capsys, source_fails
):
    closed = []

    class Source:
        def close(self):
            closed.append(True)

    def build_live_pose_source(cfg):
        if source_fails:
            raise ConnectionRefusedError("Motive is not streaming")
        return Source()

    class Streamer:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError("Motive is not streaming")

    monkeypatch.setattr(cli, "build_live_pose_source", build_live_pose_source)
    monkeypatch.setattr(cli, "ExtposeStreamer", Streamer)

    code, report = preflight(capsys, "--config", str(lab_yaml), "--checks", "mocap-quality")

    assert code == 1
    (result,) = report["results"]
    assert result["summary"] == "Mocap resource is unavailable"
    assert "Motive is not streaming" in result["error"]
    assert closed == ([] if source_fails else [True])


def test_preflight_expected_parameters_are_typed_yaml_scalars(radio, checks, lab_yaml, capsys):
    radio()

    code, _ = preflight(
        capsys, "--config", str(lab_yaml), "--checks", "expected-parameters",
        "--expect-parameter", "stabilizer.controller=2",
        "--expect-parameter", "kalman.robustTdoa = true",
        "--expect-parameter", "deck.name=bcAI",
        "--expect-parameter", "pid_rate.roll_kp=250.5",
        "--parameter-tolerance", "pid_rate.roll_kp=0.5",
    )

    assert code == 0
    ((name, _, (expected,), kwargs),) = checks
    assert expected == {"stabilizer.controller": 2, "kalman.robustTdoa": True,
                        "deck.name": "bcAI", "pid_rate.roll_kp": 250.5}
    assert kwargs["tolerances"] == {"pid_rate.roll_kp": 0.5}


def test_preflight_expected_parameter_in_exponent_form_is_numeric(radio, checks, lab_yaml, capsys):
    radio()

    preflight(capsys, "--config", str(lab_yaml), "--checks", "expected_parameters",
              "--expect-parameter", "pid_rate.roll_ki=5e-1")

    ((_, _, (expected,), _),) = checks
    assert expected == {"pid_rate.roll_ki": 0.5}


@pytest.mark.parametrize("argv", [
    ["--checks", "bogus"],
    ["--checks", "battery,,static_imu"],
    ["--checks", "expected_parameters"],
    ["--checks", "required_decks"],
    ["--checks", "expected_parameters", "--expect-parameter", "stabilizer.controller"],
    ["--checks", "expected_parameters", "--expect-parameter", "a=1", "--expect-parameter", "a=2"],
    ["--checks", "expected_parameters", "--expect-parameter", "a=[1, 2]"],
    ["--checks", "expected_parameters", "--expect-parameter", "a=.nan"],
    ["--checks", "expected_parameters", "--expect-parameter", "a=1", "--parameter-tolerance", "b=1"],
    ["--checks", "expected_parameters", "--expect-parameter", "a=x", "--parameter-tolerance", "a=1"],
    ["--checks", "expected_parameters", "--expect-parameter", "a=1", "--parameter-tolerance", "a=-1"],
])
def test_preflight_rejects_bad_selections_before_any_hardware(radio, checks, lab_yaml, capsys, argv):
    backend = radio()

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["preflight", "--config", str(lab_yaml), *argv])

    assert exit_info.value.code == 2
    assert backend.events == [] and checks == []
