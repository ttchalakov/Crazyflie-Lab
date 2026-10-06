"""Generate a flyable 3-D figure-eight as timed absolute setpoints.

The output is a list of ``(duration_s, x, y, z, yaw_deg)`` tuples in exactly the
shape ``examples/mocap_waypoint`` streams as ``TIMED_SETPOINTS``: a straight-up
takeoff over the box centre, the figure-eight itself, then a return to the centre
at a low height for the runner's straight-down land. Every point lies inside the
requested box by construction (each coordinate is a sinusoid mapped onto its
range, then clamped against float error), so the whole path stays in the volume.

Shape. The horizontal path is the classic 1:2 Lissajous figure-eight
``x = sin(t)``, ``y = sin(2t)``; height is ``z = cos(t)``, which lifts the eight's
two centre-crossings to opposite ends of the z range -- a genuine space curve,
not a flat eight tilted in a plane. Yaw sweeps its entire range as ``yaw = sin(t)``.

You pick the box, the number of loops and how long they take; the eights are cut
into one setpoint every ``SETPOINT_SPACING_S``, so a longer ``duration_s`` flies
the same geometry slower in more, smaller steps.

    from cf_lab.trajectory import figure_eight_setpoints
    TIMED_SETPOINTS = figure_eight_setpoints(z_range=(0.5, 2.0), loops=2, duration_s=40.0)
"""
from __future__ import annotations

import math

Setpoint = tuple[float, float, float, float, float]
Range = tuple[float, float]

# Time between successive setpoints along the eight (five frames at 50 Hz).
SETPOINT_SPACING_S = 0.1
# Straight-up climb over the box centre to where the eight starts.
TAKEOFF_S = 5.0
# A long, low hover back over the centre: gives the position controller time
# to null the lateral error left by the eight, so the drone reconverges on its
# takeoff point before the runner's straight-down land to the floor.
LAND_S = 8.0
LAND_HEIGHT_M = 0.6  # clamped into z_range


def figure_eight_setpoints(
    *,
    x_range: Range = (-1.0, 1.0),
    y_range: Range = (-1.0, 1.0),
    z_range: Range = (0.5, 2.0),
    yaw_range_deg: Range = (-180.0, 180.0),
    loops: int = 1,
    duration_s: float = 20.0,
) -> list[Setpoint]:
    """``loops`` figure-eights over ``[x|y|z]_range`` flown in ``duration_s``.

    ``duration_s`` is the eights alone; the ``TAKEOFF_S`` climb before them and
    the ``LAND_S`` hover at ``LAND_HEIGHT_M``, below where they end, come on top.
    The setpoint count is ``duration_s / SETPOINT_SPACING_S`` rounded, each held
    for an equal share of ``duration_s``.

    Raises ``ValueError`` on an empty/backwards range, ``loops < 1``, or a
    ``duration_s`` too short to give each eight at least four setpoints.
    """
    _check_range("x_range", x_range)
    _check_range("y_range", y_range)
    _check_range("z_range", z_range)
    _check_range("yaw_range_deg", yaw_range_deg)
    if loops < 1:
        raise ValueError(f"loops must be >= 1, got {loops}")
    if not math.isfinite(duration_s) or round(duration_s / SETPOINT_SPACING_S) < 4 * loops:
        raise ValueError(
            f"duration_s must give each of the {loops} eight(s) at least four "
            f"{SETPOINT_SPACING_S} s setpoints, got {duration_s}"
        )

    xc, xa = _center_amp(x_range)
    yc, ya = _center_amp(y_range)
    zc, za = _center_amp(z_range)
    yawc, yawa = _center_amp(yaw_range_deg)

    def position(t: float) -> tuple[float, float, float, float]:
        x = _clamp(xc + xa * math.sin(t), x_range)
        y = _clamp(yc + ya * math.sin(2.0 * t), y_range)
        z = _clamp(zc + za * math.cos(t), z_range)
        yaw = _clamp(yawc + yawa * math.sin(t), yaw_range_deg)
        return x, y, z, yaw

    # t=0 puts us at (xc, yc, ...) -- the eight always starts over the centre, so
    # the takeoff is a clean vertical climb to it with no lateral component.
    traj: list[Setpoint] = [(TAKEOFF_S, *position(0.0))]

    segments = round(duration_s / SETPOINT_SPACING_S)
    dwell = duration_s / segments
    span = 2.0 * math.pi * loops
    for k in range(1, segments + 1):
        traj.append((dwell, *position(span * k / segments)))

    # Straight down from where the eights end: over the centre for whole loops,
    # with no sideways step into the landing for a fractional count.
    _, x_end, y_end, _, yaw_end = traj[-1]
    traj.append((LAND_S, x_end, y_end, _clamp(LAND_HEIGHT_M, z_range), yaw_end))
    return traj


def _center_amp(r: Range) -> tuple[float, float]:
    lo, hi = r
    return (lo + hi) / 2.0, (hi - lo) / 2.0


def _clamp(value: float, r: Range) -> float:
    lo, hi = r
    return max(lo, min(hi, value))


def _check_range(name: str, r: Range) -> None:
    lo, hi = r
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"{name} bounds must be finite, got {r}")
    if lo >= hi:
        raise ValueError(f"{name} lower bound must be < upper bound, got {r}")
