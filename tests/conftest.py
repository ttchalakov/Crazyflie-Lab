"""Marks every test with the folder it lives in, so `pytest -m camera` runs one group."""
from pathlib import Path

import pytest

TESTS = Path(__file__).parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        parts = item.path.relative_to(TESTS).parts
        if len(parts) > 1:
            item.add_marker(parts[0])
