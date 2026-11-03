"""Align a Motive rigid body with the drone's IMU, from bead geometry and hand motion.

The estimator flies in the IMU's axes and treats extpose as the pose of the IMU, so the
asset's pivot belongs on the IMU, with the same axes. Each unknown comes from the
comparison that can actually see it:

- Asset-to-IMU rotation: gyro rates against the mocap rotation rates (Wahba's problem).
  Neither gravity nor the floor enters.
- IMU position, accelerometer bias and the tilt of Motive's vertical from true gravity:
  one least-squares fit of the accelerometer against the mocap motion. Bias is fixed to
  the drone and Motive's tilt to the room, so turning the drone separates them.
- Mechanical front and centre: the deck beads against their known holes. These are a
  cross-check on the rotation and the prior for the IMU position.

The accelerometer is low-passed on the drone before it is logged, so the mocap side of
the fit gets the same filter; then both sides get one identical low-pass.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, filtfilt, lfilter, savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from cf_lab.mocap.types import Quaternion

GRAVITY_MPS2 = 9.81  # what the firmware scales acc.x/y/z by
ACCEL_LPF_HZ = 30.0  # ACCEL_LPF_CUTOFF_FREQ, 2nd order, in sensors_bmi088_bmp3xx.c
FORCE_LPF_HZ = 10.0  # the identical low-pass on both sides of the accelerometer fit
MOCAP_GRID_HZ = 360.0
DERIVATIVE_WINDOW = 15  # Savitzky-Golay samples on the mocap grid, about 40 ms
MAX_MOCAP_GAP_S = 0.02  # longer gaps (occlusion) are excluded, with this margin:
MOCAP_GAP_MARGIN_S = 0.15
LAG_SEARCH_S = 0.15
ROTATING_RAD_S = 0.5  # samples turning slower than this do not constrain the rotation
CHUNKS = 5  # consistency: the fits repeated on this many consecutive parts of the motion

# Pass/fail limits.
MIN_RATE_CORRELATION = 0.95
# Good runs on cf_teo (2026-11-03) reached 0.39 deg and 11.4 mg: Motive's orientation
# itself varies by about 0.3 deg across the volume.
MAX_ROTATION_SPREAD_DEG = 0.5
MAX_DECK_RMS_M = 0.001  # cf_teo's beads fit to 0.57 mm; a bead one hole off, 2.3 mm
MAX_DECK_YAW_DIFF_DEG = 2.0
MAX_CLOSURE_MG = 15.0
MAX_BIAS_TILT_CORRELATION = 0.9
# The IMU position comes from the data only when it clears all three of these.
MAX_POSITION_CHANNEL_GAP_M = 0.005
MAX_POSITION_SPREAD_M = 0.003
MAX_POSITION_PRIOR_GAP_M = 0.008


@dataclass(frozen=True)
class MarkerGeometry:
    """What is known about the drone, in body axes (x front, y left, z up), metres."""

    deck_holes_m: np.ndarray  # (N, 2): the holes holding deck beads, from the deck centre
    bead_radius_m: float  # beads sit flush on the deck, so their centres are this high
    imu_xy_m: np.ndarray  # the IMU from the deck centre
    imu_z_from_deck_top_m: float  # negative: below the deck's top surface


def load_geometry(path: str | Path) -> MarkerGeometry:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    holes = np.array(data["deck_holes_mm"], float) / 1e3
    if holes.ndim != 2 or holes.shape[1] != 2 or len(holes) < 3:
        raise ValueError(f"{path}: deck_holes_mm must list at least three [x, y] holes.")
    return MarkerGeometry(
        deck_holes_m=holes,
        bead_radius_m=float(data["bead_diameter_mm"]) / 2e3,
        imu_xy_m=np.array(data["imu_xy_mm"], float) / 1e3,
        imu_z_from_deck_top_m=float(data["imu_z_from_deck_top_mm"]) / 1e3,
    )


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The proper rotation Q and translation t minimising |Q source + t - target|."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    U, _, Vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    Q = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return Q, target_mean - Q @ source_mean


@dataclass(frozen=True)
class DeckFit:
    rotation: np.ndarray  # deck axes as columns, in asset axes
    center_m: np.ndarray  # the deck centre at bead-centre height, in asset axes
    marker_indices: tuple[int, ...]  # which asset marker sits in each hole
    rms_m: float
    runner_up_rms_m: float  # the best fit with a different choice of markers


def fit_deck(markers_m: np.ndarray, geometry: MarkerGeometry) -> DeckFit:
    """Find which asset markers are the deck beads, and the deck's pose in the asset.

    Every assignment of markers to holes is tried. The deck must face up the asset's
    z, which rejects a mirrored triangle.
    """
    holes = np.c_[geometry.deck_holes_m, np.zeros(len(geometry.deck_holes_m))]
    fits = []
    for chosen in itertools.permutations(range(len(markers_m)), len(holes)):
        target = markers_m[list(chosen)]
        Q, t = _rigid_fit(holes, target)
        if Q[2, 2] > 0.0:
            rms = math.sqrt(np.mean(np.sum((holes @ Q.T + t - target) ** 2, axis=1)))
            fits.append((rms, chosen, Q, t))
    if not fits:
        raise ValueError(
            f"The asset's {len(markers_m)} markers cannot fill {len(holes)} deck holes."
        )
    fits.sort(key=lambda fit: fit[0])
    rms, chosen, Q, t = fits[0]
    runner_up = next((fit[0] for fit in fits[1:] if set(fit[1]) != set(chosen)), math.inf)
    return DeckFit(
        rotation=Q, center_m=t, marker_indices=chosen, rms_m=rms, runner_up_rms_m=runner_up
    )


# -- pivot edits ----------------------------------------------------------------------


def pivot_edit(
    rotation_asset_to_imu: np.ndarray, origin_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(translation in asset axes, intrinsic XYZ euler degrees) moving the pivot to the IMU."""
    return origin_m, Rotation.from_matrix(rotation_asset_to_imu.T).as_euler("XYZ", degrees=True)


def markers_after_edit(
    markers_m: np.ndarray, rotation_asset_to_imu: np.ndarray, origin_m: np.ndarray
) -> np.ndarray:
    """The asset's marker offsets once its pivot is on the IMU, with the IMU's axes."""
    return (markers_m - origin_m) @ rotation_asset_to_imu.T


def measure_pivot_edit(
    before_m: np.ndarray, after_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """What an edit did, from the marker offsets before and after it:
    (translation in the old axes, intrinsic XYZ euler degrees, fit rms in metres)."""
    Q, s = _rigid_fit(before_m, after_m)  # after = Q before + s
    rms = math.sqrt(np.mean(np.sum((before_m @ Q.T + s - after_m) ** 2, axis=1)))
    return -Q.T @ s, Rotation.from_matrix(Q.T).as_euler("XYZ", degrees=True), rms


# -- live coverage --------------------------------------------------------------------


class MotionCoverage:
    """Has the hand motion turned the drone enough to separate the unknowns?

    Rotation is summed per asset axis, over steps long enough that pose noise on a
    still drone does not count as turning.
    """

    STEP_S = 0.05
    TURNING_RAD_S = 0.3
    MIN_AXIS_ROTATION_RAD = 15.0
    MIN_TILT_DEG = 60.0
    MIN_TURNING_S = 20.0

    def __init__(self) -> None:
        self.axes_rad = [0.0, 0.0, 0.0]
        self.max_tilt_deg = 0.0
        self.turning_s = 0.0
        self.last_turning_s: float | None = None
        self._last: Quaternion | None = None
        self._last_s = 0.0

    def update(self, q: Quaternion, now_s: float) -> None:
        if self._last is None:
            self._last, self._last_s = q, now_s
            return
        dt = now_s - self._last_s
        if dt < self.STEP_S:
            return
        last = self._last
        rel = Quaternion(-last.qx, -last.qy, -last.qz, last.qw).multiply(q)
        sign = 1.0 if rel.qw >= 0.0 else -1.0
        step = [2.0 * sign * value for value in (rel.qx, rel.qy, rel.qz)]
        if math.hypot(*step) / dt > self.TURNING_RAD_S:
            self.axes_rad = [total + abs(value) for total, value in zip(self.axes_rad, step)]
            self.turning_s += dt
            self.last_turning_s = now_s
        up = 1.0 - 2.0 * (q.qx * q.qx + q.qy * q.qy)
        self.max_tilt_deg = max(self.max_tilt_deg, math.degrees(math.acos(max(-1.0, min(1.0, up)))))
        self._last, self._last_s = q, now_s

    @property
    def fraction(self) -> float:
        """Progress towards complete, 0 to 1: the least-covered requirement."""
        return min(
            1.0,
            min(self.axes_rad) / self.MIN_AXIS_ROTATION_RAD,
            self.max_tilt_deg / self.MIN_TILT_DEG,
            self.turning_s / self.MIN_TURNING_S,
        )

    @property
    def complete(self) -> bool:
        return self.fraction >= 1.0


# -- the motion fit -------------------------------------------------------------------


@dataclass(frozen=True)
class Motion:
    """A recording on one clock: mocap poses in Motive's world, IMU samples on the drone."""

    mocap_t_s: np.ndarray
    quaternion_xyzw: np.ndarray
    position_m: np.ndarray
    imu_t_s: np.ndarray
    gyro_rad_s: np.ndarray
    acc_mps2: np.ndarray


def load_motion(run_dir: str | Path, *, imu_stream: str = "imu") -> Motion:
    """A telemetry run's raw ``mocap`` stream and an IMU stream of acc.x/y/z, gyro.x/y/z."""
    from cf_lab.telemetry.sync import load_run

    run = load_run(run_dir)

    def columns(table, names):
        return np.column_stack(
            [np.asarray(table.column(name).to_pylist(), float) for name in names]
        )

    mocap, imu = run.streams["mocap"], run.streams[imu_stream]
    t = np.asarray(mocap.column("t_s").to_pylist(), float)
    order = np.argsort(t)
    order = order[np.r_[True, np.diff(t[order]) > 0.0]]
    return Motion(
        mocap_t_s=t[order],
        quaternion_xyzw=columns(mocap, ("qx", "qy", "qz", "qw"))[order],
        position_m=columns(mocap, ("x", "y", "z"))[order],
        imu_t_s=np.asarray(imu.column("t_s").to_pylist(), float),
        gyro_rad_s=np.radians(columns(imu, ("gyro.x", "gyro.y", "gyro.z"))),
        acc_mps2=GRAVITY_MPS2 * columns(imu, ("acc.x", "acc.y", "acc.z")),
    )


def _interp(t: np.ndarray, grid: np.ndarray, series: np.ndarray) -> np.ndarray:
    flat = series.reshape(len(grid), -1)
    out = np.column_stack([np.interp(t, grid, flat[:, j]) for j in range(flat.shape[1])])
    return out.reshape((len(t), *series.shape[1:]))


def _skew(v: np.ndarray) -> np.ndarray:
    out = np.zeros((len(v), 3, 3))
    out[:, 0, 1], out[:, 0, 2] = -v[:, 2], v[:, 1]
    out[:, 1, 0], out[:, 1, 2] = v[:, 2], -v[:, 0]
    out[:, 2, 0], out[:, 2, 1] = -v[:, 1], v[:, 0]
    return out


def _accel_filter(series: np.ndarray, fs_hz: float) -> np.ndarray:
    """The drone's accelerometer low-pass, for the mocap side of the fit."""
    b, a = butter(2, ACCEL_LPF_HZ / (fs_hz / 2.0))
    return lfilter(b, a, series, axis=0)


class _ForceFit:
    """The accelerometer model, linear in the unknowns, in asset axes:

    f = R_WA^T (a_W - g_W - dg) + alpha x r + w x (w x r) + b
    """

    def __init__(self, grid, slerp, w, alpha, acc_w, ts, measured_asset) -> None:
        rotations = slerp(grid)
        fs_hz = 1.0 / (grid[1] - grid[0])
        n = len(grid)
        known = rotations.inv().apply(acc_w - np.array([0.0, 0.0, -GRAVITY_MPS2]))
        lever = _skew(alpha) + _skew(w) @ _skew(w)
        tilt = np.stack([rotations.inv().apply(axis) for axis in np.eye(3)[:2]], axis=2)
        self._grid = grid
        self._known = _accel_filter(known, fs_hz)
        self._lever = _accel_filter(lever.reshape(n, 9), fs_hz).reshape(n, 3, 3)
        self._tilt = _accel_filter(tilt.reshape(n, 6), fs_hz).reshape(n, 3, 2)
        self._ts = ts
        imu_hz = 1.0 / np.median(np.diff(ts))
        self._lowpass = butter(2, FORCE_LPF_HZ / (imu_hz / 2.0))
        self._measured = self._smooth(measured_asset)

    def _smooth(self, series: np.ndarray) -> np.ndarray:
        return filtfilt(*self._lowpass, series.reshape(len(series), -1), axis=0).reshape(
            series.shape
        )

    def solve(self, lag_s: float, select: np.ndarray, channels=(0, 1, 2)):
        """(r, bias, dg_xy, rms, covariance) from the selected samples and accelerometer axes."""
        t = self._ts + lag_s
        known = self._smooth(_interp(t, self._grid, self._known))[select]
        lever = self._smooth(_interp(t, self._grid, self._lever))[select]
        tilt = self._smooth(_interp(t, self._grid, self._tilt))[select]
        measured = self._measured[select]
        rows, rhs = [], []
        for c in channels:
            bias = np.zeros((len(measured), 3))
            bias[:, c] = 1.0
            rows.append(np.hstack([lever[:, c, :], bias, -tilt[:, c, :]]))
            rhs.append(measured[:, c] - known[:, c])
        A, y = np.vstack(rows), np.concatenate(rhs)
        solution, *_ = np.linalg.lstsq(A, y, rcond=None)
        residual = y - A @ solution
        covariance = np.linalg.pinv(A.T @ A) * np.mean(residual**2)
        return (
            solution[:3],
            solution[3:6],
            solution[6:8],
            math.sqrt(np.mean(residual**2)),
            covariance,
        )

    def rest_error(
        self, lag_s: float, select: np.ndarray, bias: np.ndarray, dg_xy: np.ndarray
    ) -> np.ndarray:
        """Measured minus predicted specific force, for samples at rest (no lever term)."""
        t = self._ts + lag_s
        known = self._smooth(_interp(t, self._grid, self._known))
        tilt = self._smooth(_interp(t, self._grid, self._tilt))
        predicted = known + bias - tilt @ dg_xy
        return (self._measured - predicted)[select]


def _robust_spread(values: np.ndarray) -> np.ndarray:
    return 1.4826 * np.median(np.abs(values - np.median(values, axis=0)), axis=0)


def _wahba(mocap: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """The rotation R taking mocap rates to gyro rates: gyro = R mocap."""
    U, _, Vt = np.linalg.svd(mocap.T @ gyro)
    return Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class AlignmentResult:
    rotation_asset_to_imu: np.ndarray  # v_imu = R v_asset
    rotation_spread_deg: np.ndarray  # per euler axis, between chunks of the motion
    gyro_bias_rad_s: np.ndarray
    gyro_rate_ratio: float
    gyro_lag_s: float  # IMU time + lag = mocap time
    rate_correlation: float
    accel_lag_s: float
    accel_bias_mps2: np.ndarray  # in IMU axes
    motive_tilt_deg: float
    true_up_in_motive: np.ndarray  # true vertical, in Motive's world axes
    force_rms_mps2: float
    closure_mg: float | None  # None when the recording has no rest
    imu_position_m: np.ndarray  # the motion fit, asset axes
    imu_position_spread_m: np.ndarray
    imu_z_by_channel_m: tuple[float, float]  # from the z accelerometer; from x and y
    prior_position_m: np.ndarray  # from the deck and the entered IMU offset
    deck: DeckFit
    deck_yaw_from_imu_deg: float
    origin_m: np.ndarray  # where the pivot should go
    origin_source: str  # "motion fit" or "geometry"
    checks: tuple[Check, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def analyze_motion(
    motion: Motion, markers_m: np.ndarray, geometry: MarkerGeometry
) -> AlignmentResult:
    """Solve for every unknown in the module docstring; raises if nothing moved."""
    t = motion.mocap_t_s
    dt = 1.0 / MOCAP_GRID_HZ
    # Whole steps only: np.arange can round its last point past t[-1], which Slerp refuses.
    grid = t[0] + dt * np.arange(int((t[-1] - t[0]) / dt))
    slerp = Slerp(t, Rotation.from_quat(motion.quaternion_xyzw))
    rotations = slerp(grid)
    position = np.column_stack([np.interp(grid, t, motion.position_m[:, j]) for j in range(3)])
    steps = (rotations[:-1].inv() * rotations[1:]).as_rotvec() / dt
    w = savgol_filter(np.vstack([steps, steps[-1]]), DERIVATIVE_WINDOW, 3, axis=0)
    alpha = savgol_filter(w, DERIVATIVE_WINDOW, 3, deriv=1, delta=dt, axis=0)
    acc_w = savgol_filter(position, DERIVATIVE_WINDOW, 3, deriv=2, delta=dt, axis=0)

    valid_grid = np.ones(len(grid), bool)
    for i in np.flatnonzero(np.diff(t) > MAX_MOCAP_GAP_S):
        valid_grid[(grid > t[i] - MOCAP_GAP_MARGIN_S) & (grid < t[i + 1] + MOCAP_GAP_MARGIN_S)] = (
            False
        )

    inside = (motion.imu_t_s > grid[0] + 0.5) & (motion.imu_t_s < grid[-1] - 0.5)
    ts, gyro, acc = motion.imu_t_s[inside], motion.gyro_rad_s[inside], motion.acc_mps2[inside]

    # Time: the rotation-rate magnitude does not depend on the unknown rotation.
    gyro_mag = np.linalg.norm(gyro, axis=1)
    mocap_mag = np.linalg.norm(w, axis=1)
    lags = np.arange(-LAG_SEARCH_S, LAG_SEARCH_S, 0.0005)
    correlations = [
        np.corrcoef(gyro_mag, np.interp(ts + lag, grid, mocap_mag))[0, 1] for lag in lags
    ]
    gyro_lag = float(lags[int(np.argmax(correlations))])
    correlation = float(max(correlations))

    # Rotation, with the gyro bias.
    valid = np.interp(ts + gyro_lag, grid, valid_grid.astype(float)) > 0.999
    rates = _interp(ts + gyro_lag, grid, w)
    rotating = valid & (np.linalg.norm(rates, axis=1) > ROTATING_RAD_S)
    if rotating.sum() < 100:
        raise ValueError(
            f"Only {rotating.sum()} IMU samples show the drone turning; move it by hand."
        )
    bias_g = np.zeros(3)
    for _ in range(10):
        R = _wahba(rates[rotating], gyro[rotating] - bias_g)
        bias_g = np.mean(gyro[valid] - rates[valid] @ R.T, axis=0)
    ratio = float(
        np.linalg.norm(gyro[rotating] - bias_g, axis=1).sum()
        / np.linalg.norm(rates[rotating], axis=1).sum()
    )
    eulers = [
        Rotation.from_matrix(_wahba(rates[chunk], gyro[chunk] - bias_g)).as_euler(
            "xyz", degrees=True
        )
        for chunk in np.array_split(np.flatnonzero(rotating), CHUNKS)
    ]
    rotation_spread = _robust_spread(np.array(eulers))

    # Accelerometer: IMU position, bias and Motive's tilt, with the accelerometer's own lag.
    # The samples at rest stay out of the fit, for the closure test below.
    still = maximum_filter1d(np.linalg.norm(gyro - bias_g, axis=1), size=51) < math.radians(3.0)
    fit = _ForceFit(grid, slerp, w, alpha, acc_w, ts, acc @ R)
    moving = valid & ~still
    accel_lag = min(
        np.arange(gyro_lag - 0.03, gyro_lag + 0.03, 0.001),
        key=lambda lag: fit.solve(lag, moving)[3],
    )
    r, bias_a, dg, rms, covariance = fit.solve(accel_lag, moving)
    corr = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    bias_tilt_corr = float(np.max(np.abs(corr[3:5, 6:8])))
    r_chunks = []
    for chunk in np.array_split(np.flatnonzero(moving), CHUNKS):
        select = np.zeros(len(ts), bool)
        select[chunk] = True
        r_chunks.append(fit.solve(accel_lag, select)[0])
    r_spread = _robust_spread(np.array(r_chunks))
    z_from_z = float(fit.solve(accel_lag, moving, channels=(2,))[0][2])
    z_from_xy = float(fit.solve(accel_lag, moving, channels=(0, 1))[0][2])

    true_up = np.array([-dg[0], -dg[1], GRAVITY_MPS2])
    true_up /= np.linalg.norm(true_up)
    tilt_deg = math.degrees(math.acos(true_up[2]))

    closure_mg = None
    rest = valid & still
    if rest.sum() >= 100:
        error = fit.rest_error(accel_lag, rest, bias_a, dg)
        closure_mg = float(np.median(np.linalg.norm(error, axis=1)) / GRAVITY_MPS2 * 1e3)

    # Geometry: the deck, and the prior for the IMU position. "Down" is the IMU's, since
    # the plane through three beads tilts a lot on a small height error.
    deck = fit_deck(markers_m, geometry)
    deck_x = R @ deck.rotation[:, 0]
    deck_yaw = math.degrees(math.atan2(deck_x[1], deck_x[0]))
    below_centers = geometry.imu_z_from_deck_top_m - geometry.bead_radius_m
    prior = deck.center_m + R.T @ np.array([*geometry.imu_xy_m, below_centers])

    channel_gap = abs(z_from_z - z_from_xy)
    prior_gap = float(np.linalg.norm(r - prior))
    use_fit = (
        channel_gap <= MAX_POSITION_CHANNEL_GAP_M
        and float(r_spread.max()) <= MAX_POSITION_SPREAD_M
        and prior_gap <= MAX_POSITION_PRIOR_GAP_M
    )
    checks = (
        Check(
            "time sync",
            correlation >= MIN_RATE_CORRELATION,
            f"gyro and mocap rotation rates correlate at {correlation:.3f}",
        ),
        Check(
            "rotation repeatable",
            float(rotation_spread.max()) <= MAX_ROTATION_SPREAD_DEG,
            f"chunks agree within {rotation_spread.max():.2f} deg",
        ),
        Check(
            "deck beads fit their holes",
            deck.rms_m <= MAX_DECK_RMS_M,
            f"{deck.rms_m * 1e3:.2f} mm rms; next-best markers {deck.runner_up_rms_m * 1e3:.2f} mm",
        ),
        Check(
            "deck yaw matches the IMU",
            abs(deck_yaw) <= MAX_DECK_YAW_DIFF_DEG,
            f"{deck_yaw:+.2f} deg apart",
        ),
        Check(
            "bias and Motive tilt separated",
            bias_tilt_corr <= MAX_BIAS_TILT_CORRELATION,
            f"correlation {bias_tilt_corr:.2f}; turn the drone through more orientations if high",
        ),
        Check(
            "closure at rest",
            closure_mg is not None and closure_mg <= MAX_CLOSURE_MG,
            "no rest in the recording"
            if closure_mg is None
            else f"accelerometer predicted to {closure_mg:.1f} mg",
        ),
    )
    return AlignmentResult(
        rotation_asset_to_imu=R,
        rotation_spread_deg=rotation_spread,
        gyro_bias_rad_s=bias_g,
        gyro_rate_ratio=ratio,
        gyro_lag_s=gyro_lag,
        rate_correlation=correlation,
        accel_lag_s=float(accel_lag),
        accel_bias_mps2=R @ bias_a,
        motive_tilt_deg=tilt_deg,
        true_up_in_motive=true_up,
        force_rms_mps2=rms,
        closure_mg=closure_mg,
        imu_position_m=r,
        imu_position_spread_m=r_spread,
        imu_z_by_channel_m=(z_from_z, z_from_xy),
        prior_position_m=prior,
        deck=deck,
        deck_yaw_from_imu_deg=deck_yaw,
        origin_m=r if use_fit else prior,
        origin_source="motion fit" if use_fit else "geometry",
        checks=checks,
    )
