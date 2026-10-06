"""Find the cf-lab checkout and run its Docker toolchains.

Building firmware needs the checkout itself (lib/, external/, docker/), not only the
installed Python package. It is ``$CF_LAB`` when set, else the checkout this package
is installed from (``pip install -e path/to/cf-lab``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

DOCKER_PLATFORM = "linux/amd64"


class SdkError(RuntimeError):
    """A build or flash step cannot run; the message says what to do instead."""


def sdk_root() -> Path:
    env = os.environ.get("CF_LAB")
    root = Path(env).expanduser().resolve() if env else Path(__file__).resolve().parents[2]
    if not (root / "lib").is_dir() or not (root / "external").is_dir():
        raise SdkError(
            f"no cf-lab checkout at {root}. Building needs one: "
            "`pip install -e path/to/cf-lab`, or set CF_LAB=path/to/cf-lab."
        )
    return root


def submodule_hint(*paths: str) -> str:
    return f"git -C {sdk_root()} submodule update --init --recursive {' '.join(paths)}"


def run(command: list[str], **kwargs) -> None:
    try:
        subprocess.run(command, check=True, **kwargs)
    except FileNotFoundError as exc:
        raise SdkError(f"required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise SdkError(f"command failed (exit {exc.returncode}): {' '.join(command)}") from exc


def require_docker() -> None:
    if shutil.which("docker") is None:
        raise SdkError("Docker is not installed.")
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=15)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise SdkError("Docker is not running. Start it and retry.") from exc


def docker_mounts(*paths: Path) -> list[str]:
    """``-v`` arguments mounting each path's directory at its own host path.

    With identical paths nothing needs translating: a path that works on the host (in
    a Makefile, an argument, a generated file) works unchanged in the container. A file
    or a not-yet-created path mounts its parent; directories already inside another
    mount are skipped.
    """
    dirs = {p.resolve() if p.is_dir() else p.resolve().parent for p in map(Path, paths)}
    kept: list[Path] = []
    for directory in sorted(dirs, key=lambda d: len(d.parts)):
        if not any(directory.is_relative_to(parent) for parent in kept):
            kept.append(directory)
    return [arg for directory in kept for arg in ("-v", f"{directory}:{directory}")]
