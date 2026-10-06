"""Trims real recorded runs into the IMU and mocap windows the preflight tests replay.

Run from the repo root; the source runs live in gitignored examples/*/artifacts.
"""
import csv
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).parent
REPO = HERE.parents[2]
BENCH = REPO / "examples/telemetry_bench/artifacts/run_20261001T154518"
FLIGHT = REPO / "examples/mocap_waypoint/artifacts/run_20260717T164919"
IMU_COLUMNS = ["acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"]
MOCAP_COLUMNS = ["source_seq", "x", "y", "z", "qx", "qy", "qz", "qw"]


def window(table, start_ns, seconds):
    return table[(table.host_ns >= start_ns) & (table.host_ns < start_ns + seconds * 1e9)]


def write(path, rows, columns, *, host_ns_origin=None):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((["host_ms"] if host_ns_origin is not None else []) + columns)
        for host_ns, values in zip(rows.host_ns, rows[columns].itertuples(index=False, name=None)):
            prefix = [] if host_ns_origin is None else [f"{(host_ns - host_ns_origin) / 1e6:.3f}"]
            writer.writerow(prefix + [f"{v:.7g}" if isinstance(v, float) else v for v in values])


def main():
    still = pq.read_table(BENCH / "sysid_imu.parquet").to_pandas()
    write(HERE / "imu_still_bench.csv", window(still, still.host_ns.iloc[0], 1.5), IMU_COLUMNS)

    imu = pq.read_table(FLIGHT / "sysid_imu.parquet").to_pandas()
    write(HERE / "imu_in_flight.csv", window(imu, imu.host_ns.iloc[0] + 5e9, 1.5), IMU_COLUMNS)

    events = pq.read_table(FLIGHT / "events.parquet").to_pandas()
    armed_ns = int(events[events.kind == "armed"].host_ns.iloc[0])
    mocap = pq.read_table(FLIGHT / "mocap.parquet").to_pandas()
    pad = window(mocap, armed_ns - int(2e9), 1.0)
    write(HERE / "mocap_pad_natnet.csv", pad, MOCAP_COLUMNS, host_ns_origin=int(pad.host_ns.iloc[0]))
    flight = window(mocap, armed_ns + int(6e9), 1.0)
    write(HERE / "mocap_flight_natnet.csv", flight, MOCAP_COLUMNS,
          host_ns_origin=int(flight.host_ns.iloc[0]))


if __name__ == "__main__":
    main()
