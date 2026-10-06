"""STM32 telemetry via the stock CRTP log framework.

The log framework is already source-driven: the host registers each block once
and the firmware's log task samples and pushes packets at the block period —
no polling. One ``LogConfig`` per stream; the callback records the onboard ms
timestamp (cflib's first callback argument, a 24-bit counter) so post-hoc clock
sync can reconstruct measurement time.
"""

from __future__ import annotations

from cf_lab.telemetry.schema import CLOCK_STM32_MS, StreamSpec
from cf_lab.telemetry.writer import RunWriter


class Stm32LogTap:
    """Start/stop the LogConfigs for every STM32-clocked stream spec."""

    def __init__(self, writer: RunWriter, specs: list[StreamSpec]) -> None:
        self._writer = writer
        self._specs = [spec for spec in specs if spec.clock == CLOCK_STM32_MS]
        self._log_configs = []

    def start(self, cf) -> None:
        """Register and start one log block per stream on a connected cf.

        A variable name missing from the firmware TOC raises out of
        ``add_config`` — loud and pre-flight, aborting through the runner's
        normal teardown path.
        """
        from cflib.crazyflie.log import LogConfig

        for spec in self._specs:
            period_ms = int(dict(spec.meta)["period_ms"])
            lc = LogConfig(name=f"tel_{spec.name}", period_in_ms=period_ms)
            for var, cflib_type in spec.log_variables:
                lc.add_variable(var, cflib_type)
            lc.data_received_cb.add_callback(self._make_callback(spec))
            cf.log.add_config(lc)
            lc.start()
            self._log_configs.append(lc)

    def _make_callback(self, spec: StreamSpec):
        writer = self._writer
        stream = spec.name
        var_names = tuple(var for var, _ in spec.log_variables)

        def on_sample(ts_ms, data, _cfg) -> None:
            writer.append(stream, (ts_ms, *(data[name] for name in var_names)))

        return on_sample

    def stop(self) -> None:
        for lc in self._log_configs:
            try:
                lc.stop()
            except Exception:
                pass  # link may already be down; recording simply ends
        self._log_configs = []
