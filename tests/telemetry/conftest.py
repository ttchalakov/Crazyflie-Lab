"""Shared helpers for the telemetry tests: write a run directory through the real RunWriter."""
import shutil
import time
import warnings
from pathlib import Path

import pytest

from cf_lab.telemetry.writer import RunWriter

DATA = Path(__file__).parent / "data"


def _write_run(run_dir, specs, rows, **writer_kwargs):
    """Record ``rows`` = {stream: [(host_ns, values), ...]} and close the run cleanly."""
    # No drain until close, so the queue must hold the whole run.
    writer_kwargs.setdefault("queue_size", 1 << 20)
    writer = RunWriter(run_dir, specs, flush_interval_s=60.0, **writer_kwargs)
    writer.start()
    for stream, records in rows.items():
        for host_ns, values in records:
            writer.append_at(stream, tuple(values), host_ns=host_ns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # empty-stream warnings are tested on their own
        writer.close()
    return Path(run_dir)


def _wait_for(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.002)


@pytest.fixture
def bench_run(tmp_path):
    """A private copy of the trimmed real telemetry_bench run (safe to mutate)."""
    return Path(shutil.copytree(DATA / "bench_run", tmp_path / "bench_run"))


@pytest.fixture
def write_run():
    return _write_run


@pytest.fixture
def wait_for():
    return _wait_for
