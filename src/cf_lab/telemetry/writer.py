"""Non-blocking multi-stream recorder writing one Parquet file per stream.

Producers (the cflib RX thread, the mocap streamer thread, the main thread)
only ever ``append``: a ``put_nowait`` into a bounded per-stream queue plus a
drop counter when it's full. A single background thread drains the queues and
does all disk I/O, so telemetry can never stall a control loop or wedge a
radio callback — records are dropped, visibly, instead.

During the run each stream is an Arrow IPC *stream* file (``<name>.arrow``),
which stays readable up to the last flushed batch even if the process dies.
``close()`` converts each to Parquet and deletes the ``.arrow`` on success;
``manifest.json`` carries the run metadata, per-stream row/drop counts and a
``status`` that only becomes ``"clean"`` on a completed close.

pyarrow objects are touched only (a) in ``start()`` before the writer thread
spawns, (b) on the writer thread, (c) in ``close()`` after joining it — never
concurrently.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_lab.telemetry.schema import StreamSpec

MANIFEST_FORMAT_VERSION = 1


class _StreamState:
    """Queue + counters + arrow writer handles for one stream."""

    __slots__ = ("spec", "queue", "lock", "seq", "dropped", "rows", "schema", "writer", "file")

    def __init__(self, spec: StreamSpec, queue_size: int) -> None:
        self.spec = spec
        self.queue: queue.Queue[tuple] = queue.Queue(maxsize=queue_size)
        # Held while taking seq and queueing, so a stream with two producers
        # (events) queues its records in seq order.
        self.lock = threading.Lock()
        self.seq = 0  # incremented per append, even when the record is dropped
        self.dropped = 0
        self.rows = 0
        self.schema = None
        self.writer = None
        self.file = None


class RunWriter:
    """Owns one run directory: bounded ingest queues, writer thread, manifest."""

    def __init__(
        self,
        run_dir: str | Path,
        specs: Sequence[StreamSpec],
        *,
        queue_size: int = 1024,
        flush_interval_s: float = 0.25,
        manifest_extra: Mapping[str, Any] | None = None,
    ) -> None:
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate stream names in {names}.")
        self._run_dir = Path(run_dir)
        self._streams = {spec.name: _StreamState(spec, queue_size) for spec in specs}
        self._flush_interval_s = flush_interval_s
        self._manifest_extra = dict(manifest_extra or {})
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._empty_streams: list[str] = []
        self._created_wall_ns = 0
        self._created_monotonic_ns = 0
        self.error: Exception | None = None

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def start(self) -> None:
        import pyarrow as pa

        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._created_wall_ns = time.time_ns()
        self._created_monotonic_ns = time.monotonic_ns()
        for st in self._streams.values():
            st.schema = pa.schema(
                [("seq", pa.int64()), ("host_ns", pa.int64())]
                + [(name, getattr(pa, type_name)()) for name, type_name in st.spec.fields]
            )
            st.file = (self._run_dir / f"{st.spec.name}.arrow").open("wb")
            st.writer = pa.ipc.new_stream(st.file, st.schema)
            # Arrow writes the schema with the first batch; an empty one puts it
            # on disk now, so a crashed run's still-empty stream loads as empty.
            st.writer.write_batch(pa.RecordBatch.from_pylist([], schema=st.schema))
            st.file.flush()
        self._write_manifest(status="recording")
        self._thread = threading.Thread(target=self._run, daemon=True, name="telemetry-writer")
        self._thread.start()

    def append(self, stream: str, values: tuple) -> None:
        """Enqueue one record. Never blocks, never raises: full queue = drop."""
        self.append_at(stream, values, host_ns=time.monotonic_ns())

    def append_at(self, stream: str, values: tuple, *, host_ns: int) -> None:
        """Enqueue a replay/simulation record with an explicit host timestamp."""
        st = self._streams[stream]
        with st.lock:
            st.seq += 1
            try:
                st.queue.put_nowait((st.seq, int(host_ns), *values))
            except queue.Full:
                st.dropped += 1

    def event(self, kind: str, message: str = "", data: Mapping[str, Any] | None = None) -> None:
        self.append("events", (kind, message, json.dumps(data or {}, default=str)))

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            name: {"rows": st.rows, "dropped": st.dropped, "queued": st.queue.qsize()}
            for name, st in self._streams.items()
        }

    def close(self, *, status: str = "clean") -> None:
        """Drain, finalize arrow files, convert to parquet, seal the manifest.

        Idempotent and safe in ``finally`` blocks; only SIGKILL/power loss can
        skip it, which is what the crash-readable ``.arrow`` files cover.
        """
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is None:  # start() never ran; nothing on disk
            return
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            # Closing the files under a writer that is still busy could corrupt
            # them; left as .arrow, they load like a crashed run's.
            self.error = self.error or RuntimeError("telemetry writer still busy 5 s into close()")
            self._write_manifest(status="error")
            return
        try:
            self._drain_all()  # safe: writer thread has exited
            self._report_empty_streams()  # only meaningful once the drain is done
        except Exception as exc:  # a record that will not convert ends the recording, as mid-run
            self.error = self.error or exc
        for st in self._streams.values():
            try:
                st.writer.close()
                st.file.close()
            except Exception:
                status = "error"
        self._finalize_parquet()
        if self.error is not None:
            status = "error"
        self._write_manifest(status=status)

    def _report_empty_streams(self) -> None:
        """Say so when a stream was asked for and nothing ever arrived.

        Enabling a stream is a request for data; recording none of it is a soft
        failure, not an error -- a run that flies fine without mocap should not
        be stopped by its absence, and anyone actually depending on mocap has
        their own reason not to fly without it. But it must not be *silent*: an
        empty mocap.parquet looks identical whether the rig was off, the rigid
        body was misnamed, or the runner simply never wrapped its pose source.

        The list also lands in the manifest, so post-hoc tooling sees it without
        having to have been watching the console.
        """
        self._empty_streams = sorted(
            name
            for name, st in self._streams.items()
            # events is the mechanism, not a data stream; it is empty only when
            # nothing at all happened, which the other entries already say.
            if st.rows == 0 and name != "events"
        )
        if not self._empty_streams:
            return
        message = (
            f"telemetry recorded no rows for: {', '.join(self._empty_streams)}. "
            f"These streams were enabled but nothing was ever appended to them "
            f"(source not connected, or the runner never tapped it)."
        )
        warnings.warn(message, stacklevel=3)
        self.event("streams_empty", message, {"streams": self._empty_streams})
        self._drain_all()  # the event above is still queued

    # -- writer thread ------------------------------------------------------

    def _run(self) -> None:
        try:
            while not self._stop.wait(self._flush_interval_s):
                self._drain_all()
        except Exception as exc:  # surfaced via close(); recording just stops
            self.error = exc

    def _drain_all(self) -> None:
        import pyarrow as pa

        for st in self._streams.values():
            rows = []
            while True:
                try:
                    rows.append(st.queue.get_nowait())
                except queue.Empty:
                    break
            if not rows:
                continue
            columns = list(zip(*rows))
            batch = pa.record_batch(
                [pa.array(column, type=field.type) for column, field in zip(columns, st.schema)],
                schema=st.schema,
            )
            st.writer.write_batch(batch)
            st.file.flush()
            st.rows += len(rows)

    # -- finalize ------------------------------------------------------------

    def _finalize_parquet(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for st in self._streams.values():
            arrow_path = self._run_dir / f"{st.spec.name}.arrow"
            try:
                with pa.ipc.open_stream(arrow_path) as reader:
                    table = reader.read_all()
                pq.write_table(table, self._run_dir / f"{st.spec.name}.parquet")
                arrow_path.unlink()
            except Exception as exc:
                # Keep the .arrow so no data is lost; the manifest points at it.
                self.error = self.error or exc

    def _write_manifest(self, *, status: str) -> None:
        streams = {}
        for name, st in self._streams.items():
            parquet = self._run_dir / f"{name}.parquet"
            streams[name] = {
                "file": parquet.name if parquet.exists() else f"{name}.arrow",
                "fields": ["seq", "host_ns"] + [field for field, _ in st.spec.fields],
                "clock": st.spec.clock,
                "rows": st.rows,
                "dropped": st.dropped,
                "meta": {key: value for key, value in st.spec.meta},
            }
        manifest = {
            "format_version": MANIFEST_FORMAT_VERSION,
            "created_wall_ns": self._created_wall_ns,
            "created_monotonic_ns": self._created_monotonic_ns,
            "closed_wall_ns": time.time_ns() if status != "recording" else None,
            "status": status,
            "error": str(self.error) if self.error else None,
            # Enabled but never written to. Not an error -- see
            # _report_empty_streams -- but never leave it to be inferred.
            "empty_streams": self._empty_streams,
            "streams": streams,
            **self._manifest_extra,
        }
        tmp = self._run_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self._run_dir / "manifest.json")
