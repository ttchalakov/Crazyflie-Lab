"""Finding the cf-lab checkout, running toolchain commands, and Docker mounts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from cf_lab import sdk
from cf_lab.sdk import SdkError, docker_mounts, sdk_root

REPO = Path(__file__).resolve().parents[2]


def test_sdk_root_is_cf_lab_env_when_set(cf_lab_root):
    assert sdk_root() == cf_lab_root


def test_sdk_root_expands_a_home_relative_cf_lab(cf_lab_root, monkeypatch):
    monkeypatch.setenv("HOME", str(cf_lab_root.parent))
    monkeypatch.setenv("CF_LAB", "~/cf-lab")

    assert sdk_root() == cf_lab_root


def test_sdk_root_rejects_a_directory_that_is_not_a_checkout(tmp_path, monkeypatch):
    (tmp_path / "lib").mkdir()
    monkeypatch.setenv("CF_LAB", str(tmp_path))

    with pytest.raises(SdkError, match="no cf-lab checkout"):
        sdk_root()


def test_sdk_root_defaults_to_the_checkout_the_package_runs_from(monkeypatch):
    monkeypatch.delenv("CF_LAB", raising=False)

    assert sdk_root() == REPO


def test_run_reports_a_missing_tool_as_sdk_error():
    with pytest.raises(SdkError, match="not found: cf-lab-no-such-tool"):
        sdk.run(["cf-lab-no-such-tool"])


def test_run_reports_the_exit_status_of_a_failed_command():
    with pytest.raises(SdkError, match="exit 3"):
        sdk.run([sys.executable, "-c", "raise SystemExit(3)"])


@pytest.mark.parametrize("which, outcome, message", [
    (None, None, "not installed"),
    ("/usr/bin/docker", subprocess.CalledProcessError(1, ["docker", "info"]), "not running"),
    ("/usr/bin/docker", subprocess.TimeoutExpired(["docker", "info"], 15), "not running"),
])
def test_require_docker_explains_what_is_missing(monkeypatch, which, outcome, message):
    def docker_info(*args, **kwargs):
        raise outcome

    monkeypatch.setattr(sdk.shutil, "which", lambda name: which)
    monkeypatch.setattr(sdk.subprocess, "run", docker_info)

    with pytest.raises(SdkError, match=message):
        sdk.require_docker()


def mounted(args: list[str]) -> list[str]:
    assert args[::2] == ["-v"] * (len(args) // 2)
    return [spec.split(":")[0] for spec in args[1::2]]


def test_docker_mounts_skip_directories_inside_another_mount(tmp_path):
    (tmp_path / "repo" / "src").mkdir(parents=True)
    (tmp_path / "repo" / "model.py").write_text("")

    args = docker_mounts(tmp_path / "repo" / "src", tmp_path / "repo", tmp_path / "repo" / "model.py")

    assert args == ["-v", f"{tmp_path / 'repo'}:{tmp_path / 'repo'}"]


def test_docker_mounts_keep_siblings_that_share_a_name_prefix(tmp_path):
    # app is shallower, so it is kept first and app2/src is then checked against it.
    (tmp_path / "app").mkdir()
    (tmp_path / "app2" / "src").mkdir(parents=True)

    assert sorted(mounted(docker_mounts(tmp_path / "app2" / "src", tmp_path / "app"))) == [
        str(tmp_path / "app"), str(tmp_path / "app2" / "src")]


def test_docker_mounts_mount_the_parent_of_a_file_or_a_path_not_yet_made(tmp_path):
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "policy.pt").write_bytes(b"")

    args = docker_mounts(tmp_path / "ckpt" / "policy.pt", tmp_path / "out" / "model.onnx")

    assert sorted(mounted(args)) == [str(tmp_path / "ckpt"), str(tmp_path / "out")]


def test_docker_mounts_resolve_relative_paths(tmp_path, monkeypatch):
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path)

    assert mounted(docker_mounts(Path("work"))) == [str(tmp_path / "work")]
