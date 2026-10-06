"""Trim pad_imu.csv from a telemetry_bench recording of the drone resting on the pad, disarmed."""
import sys
from pathlib import Path

import pyarrow.parquet as pq

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/telemetry_bench/artifacts/run_20261001T165053")
COLUMNS = ["stm32_ms", "acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]

table = pq.read_table(RUN / "sysid_imu.parquet", columns=COLUMNS).to_pandas()
window = table.iloc[5000:5250]  # 250 rows at 50 Hz: a 5 s window mid-run
window.to_csv(Path(__file__).with_name("pad_imu.csv"), index=False, float_format="%.6f")
