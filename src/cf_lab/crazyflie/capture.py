"""Read firmware log variables for a short window, for checks you write yourself."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence


def capture_log_window(
    cf,
    variables: Sequence[tuple[str, str]],
    *,
    seconds: float,
    period_ms: int,
    name: str = "cf_lab_capture",
) -> tuple[dict[str, object], ...]:
    """Sample ``variables`` for ``seconds`` and return one dict per sample.

    ``variables`` are ``(toc name, cflib type)`` pairs, such as
    ``("pm.vbat", "float")``, in one firmware log block: at most 26 bytes, sampled
    every ``period_ms`` on the firmware's 10 ms grid. ``cf`` is a connected cflib
    Crazyflie. The rows carry the values only, not the firmware timestamp.

    The block is always stopped and deleted from the drone, even when starting
    it fails, so a capture never uses up one of the firmware's few log-block slots.
    """
    from cflib.crazyflie.log import LogConfig

    rows: list[dict[str, object]] = []
    lock = threading.Lock()
    variable_names = tuple(variable for variable, _ in variables)
    config = LogConfig(name=name, period_in_ms=period_ms)
    for variable, cflib_type in variables:
        config.add_variable(variable, cflib_type)

    def on_sample(_timestamp_ms, data, _config) -> None:
        row = {variable: data[variable] for variable in variable_names}
        with lock:
            rows.append(row)

    config.data_received_cb.add_callback(on_sample)
    registered = False
    try:
        cf.log.add_config(config)
        registered = True
        config.start()
        time.sleep(seconds)
    finally:
        if registered:
            try:
                config.stop()
            finally:
                config.delete()
    with lock:
        return tuple(rows)
