#!/usr/bin/env python3
"""Create Plotly summaries for a mocap_waypoint telemetry run."""

from __future__ import annotations

import argparse
from pathlib import Path

from cf_lab.analysis import (
    TelemetryDataset,
    load_dataset,
    plot_timeseries,
    plot_xy_trajectory,
)
from cf_lab.telemetry import load_run


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
IMAGES_DIR = ARTIFACTS_DIR / "images"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Recorded run folder; omitted selects the newest run in artifacts/",
    )
    parser.add_argument("--output-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--show", action="store_true", help="Open figures in a browser")
    args = parser.parse_args(argv)

    run_dir = args.run_dir or latest_run(ARTIFACTS_DIR)
    if run_dir is None:
        parser.error(f"no recorded runs found below {ARTIFACTS_DIR}")
    data = load_flight_dataset(run_dir)
    source_name = run_dir.name

    figures = flight_figures(data, source_name=source_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        image_path = args.output_dir / f"{source_name}_{name}.png"
        width = figure.layout.width or 1600
        figure.write_image(image_path, width=width, height=figure.layout.height, scale=2.0)
        print(image_path)
        if args.show:
            figure.show()
    return 0


def latest_run(root: Path) -> Path | None:
    """Return the most recently modified run folder below ``root``."""
    manifests = tuple(root.rglob("manifest.json")) if root.is_dir() else ()
    return max((path.parent for path in manifests), key=lambda path: path.stat().st_mtime, default=None)


def load_flight_dataset(run_dir: Path) -> TelemetryDataset:
    """Load the complete 12D state and whichever comparison signals were recorded."""
    run = load_run(run_dir)
    return load_dataset(
        run,
        state="full_12d",
        include_mocap_position=True,
        include_setpoints=True,
    )


def flight_figures(data: TelemetryDataset, *, source_name: str) -> dict[str, object]:
    """Build the standard trajectory, translation, and rotation figures."""
    position_groups = {
        f"{axis.upper()} position": _present(
            data, axis, f"mocap_{axis}", f"setpoint_{axis}"
        )
        for axis in ("x", "y", "z")
    }
    return {
        "trajectory_xy": plot_xy_trajectory(
            data, title=f"{source_name}: XY trajectory"
        ),
        "position": plot_timeseries(
            data,
            position_groups,
            title=f"{source_name}: position tracking",
        ),
        "velocity": plot_timeseries(
            data,
            {"Linear velocity": ("vx", "vy", "vz")},
            title=f"{source_name}: linear velocity",
        ),
        "attitude": plot_timeseries(
            data,
            {
                "Attitude": ("roll", "pitch", "yaw"),
                "Body rates": ("p", "q", "r"),
            },
            title=f"{source_name}: attitude and body rates",
        ),
    }


def _present(data: TelemetryDataset, *names: str) -> tuple[str, ...]:
    return tuple(name for name in names if name in data.signals)


if __name__ == "__main__":
    raise SystemExit(main())
