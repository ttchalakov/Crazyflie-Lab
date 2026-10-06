"""Shared fixtures for the core tests: config files written per test."""
import textwrap
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture
def lab_yaml() -> Path:
    """The lab's drone.yaml, with dummy credentials."""
    return DATA / "drone.yaml"


@pytest.fixture
def write_config(tmp_path):
    """Write YAML text (dedented) to a fresh file and return its path."""
    count = 0

    def write(text: str) -> Path:
        nonlocal count
        count += 1
        path = tmp_path / f"config_{count}.yaml"
        path.write_text(textwrap.dedent(text), encoding="utf-8")
        return path

    return write
