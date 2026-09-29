"""
Evaluation Metrics Module for OdysseyV1.
Computes rigorous dead-reckoning drift, position, velocity, yaw, and recovery metrics
during and after GNSS outages against ground truth.

CRITICAL INVARIANT:
Ground truth is strictly for post-processing evaluation.
Ground truth must NEVER enter the filter or influence navigation state.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, List, Optional
import numpy as np


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle in radians to [-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


@dataclass
class TrajectorySample:
    """Single time-step record of estimated state and evaluation ground truth."""
    timestamp_ns: int
    pos_est: np.ndarray          # [E, N, U] in meters
    pos_gt: Optional[np.ndarray] # [E, N, U] in meters (None if GT unavailable)
    vel_est: np.ndarray          # [vE, vN, vU] in m/s
    vel_gt: Optional[np.ndarray] # [vE, vN, vU] in m/s (None if GT unavailable)
    yaw_est: float               # Estimated yaw (radians True North)
    yaw_gt: Optional[float]      # Ground truth yaw (radians True North)
    navigation_mode: str
    gnss_available: bool
    ai_velocity: Optional[float] = None
    nhc_active: bool = False
    zupt_active: bool = False
    speed_est_mps: float = 0.0
    speed_gt_mps: Optional[float] = None


@dataclass
class OutageEvaluationMetrics:
    """Comprehensive evaluation metrics for a single GNSS outage experiment."""
    # Metadata
    dataset: str = "UNKNOWN"
    sequence: str = "UNKNOWN"
    split: str = "UNKNOWN"
    configuration: str = "UNKNOWN"
    outage_start_s: float = 0.0
    outage_duration_s: float = 0.0
    outage_samples: int = 0
    ground_truth_available: bool = True

    # Trajectory Distances (during outage)
    ground_truth_distance_m: Optional[float] = None
    estimated_distance_m: float = 0.0

    # Core Position Error Metrics (APE during outage)
    end_position_error_m: Optional[float] = None
    mean_position_error_m: Optional[float] = None
    median_position_error_m: Optional[float] = None
    p95_position_error_m: Optional[float] = None
    max_position_error_m: Optional[float] = None
    position_rmse_m: Optional[float] = None

    # Drift Percentages (relative to GT traveled distance during outage)
    # Definition: drift_percent = 100 * end_position_error_m / ground_truth_distance_m
    # Definition: max_drift_percent = 100 * max_position_error_m / ground_truth_distance_m
    drift_percent: Optional[float] = None
    max_drift_percent: Optional[float] = None

    # Relative Trajectory Error (RPE during outage, relative to outage start)
    # RPE(t) = || (p_est(t) - p_est(t0)) - (p_gt(t) - p_gt(t0)) ||
    rpe_end_m: Optional[float] = None
    rpe_max_m: Optional[float] = None
    rpe_mean_m: Optional[float] = None
    rpe_rmse_m: Optional[float] = None

    # Velocity Error Metrics (during outage)
    velocity_rmse_mps: Optional[float] = None
    velocity_mae_mps: Optional[float] = None
    velocity_median_mps: Optional[float] = None
    velocity_p95_mps: Optional[float] = None
    velocity_error_end_mps: Optional[float] = None
    forward_velocity_rmse_mps: Optional[float] = None

    # Attitude / Yaw Metrics (during outage)
    yaw_rmse_deg: Optional[float] = None
    yaw_mean_deg: Optional[float] = None
    yaw_median_deg: Optional[float] = None
    yaw_p95_deg: Optional[float] = None
    yaw_max_deg: Optional[float] = None

    # GNSS Recovery Metrics (after outage ends)
    recovery_evaluated: bool = False
    recovery_time_s: Optional[float] = None
    accepted_recovery_updates: int = 0
    rejected_recovery_updates: int = 0
    pos_error_pre_recovery_m: Optional[float] = None
    pos_error_post_first_gnss_m: Optional[float] = None
    pos_error_1s_recovery_m: Optional[float] = None
    pos_error_5s_recovery_m: Optional[float] = None
    recovery_position_delta_m: Optional[float] = None
    recovery_velocity_delta_mps: Optional[float] = None

    # Feature Usage Summary
    ai_velocity_active_samples: int = 0
    nhc_active_samples: int = 0
    zupt_active_samples: int = 0
    engine_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_position_errors(
    pos_est: np.ndarray,
    pos_gt: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Compute absolute position error vector and summary statistics.
    pos_est: [N, 2] or [N, 3]
    pos_gt:  [N, 2] or [N, 3]
    """
    if len(pos_est) == 0 or len(pos_gt) == 0:
        return np.array([]), {
            "end": 0.0, "mean": 0.0, "median": 0.0,
            "p95": 0.0, "max": 0.0, "rmse": 0.0,
        }

    diffs = pos_est - pos_gt
    ape = np.linalg.norm(diffs, axis=1)

    stats = {
        "end": float(ape[-1]),
        "mean": float(np.mean(ape)),
        "median": float(np.median(ape)),
        "p95": float(np.percentile(ape, 95)),
        "max": float(np.max(ape)),
        "rmse": float(math.sqrt(np.mean(ape ** 2))),
    }
    return ape, stats


def calculate_relative_position_error(
    pos_est: np.ndarray,
    pos_gt: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Compute Relative Position Error (RPE) during outage:
    RPE(t) = || (p_est(t) - p_est(t0)) - (p_gt(t) - p_gt(t0)) ||
    """
    if len(pos_est) == 0 or len(pos_gt) == 0:
        return np.array([]), {"end": 0.0, "max": 0.0, "mean": 0.0, "rmse": 0.0}

    disp_est = pos_est - pos_est[0]
    disp_gt = pos_gt - pos_gt[0]
    rpe = np.linalg.norm(disp_est - disp_gt, axis=1)

    stats = {
        "end": float(rpe[-1]),
        "max": float(np.max(rpe)),
        "mean": float(np.mean(rpe)),
        "rmse": float(math.sqrt(np.mean(rpe ** 2))),
    }
    return rpe, stats


def calculate_traveled_distance(positions: np.ndarray) -> float:
    """Accumulate step-by-step Euclidean distance between consecutive points."""
    if len(positions) < 2:
        return 0.0
    step_vectors = np.diff(positions, axis=0)
    step_distances = np.linalg.norm(step_vectors, axis=1)
    return float(np.sum(step_distances))


def calculate_velocity_errors(
    vel_est: np.ndarray,
    vel_gt: np.ndarray,
) -> dict[str, float]:
    """Compute 3D / 2D velocity errors norm, RMSE, MAE, median, P95, and end error."""
    if len(vel_est) == 0 or len(vel_gt) == 0:
        return {
            "rmse": 0.0, "mae": 0.0, "median": 0.0, "p95": 0.0, "end": 0.0
        }

    diffs = vel_est - vel_gt
    v_norm_err = np.linalg.norm(diffs, axis=1)

    return {
        "rmse": float(math.sqrt(np.mean(v_norm_err ** 2))),
        "mae": float(np.mean(v_norm_err)),
        "median": float(np.median(v_norm_err)),
        "p95": float(np.percentile(v_norm_err, 95)),
        "end": float(v_norm_err[-1]),
    }


def calculate_yaw_errors(
    yaw_est: np.ndarray,
    yaw_gt: np.ndarray,
) -> dict[str, float]:
    """
    Compute yaw angle errors in degrees with proper wrapping to [-pi, pi].
    error = wrap_to_pi(yaw_est - yaw_gt)
    """
    if len(yaw_est) == 0 or len(yaw_gt) == 0:
        return {
            "rmse": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0
        }

    raw_diff = yaw_est - yaw_gt
    wrapped_rad = np.arctan2(np.sin(raw_diff), np.cos(raw_diff))
    abs_err_deg = np.degrees(np.abs(wrapped_rad))

    return {
        "rmse": float(math.sqrt(np.mean(abs_err_deg ** 2))),
        "mean": float(np.mean(abs_err_deg)),
        "median": float(np.median(abs_err_deg)),
        "p95": float(np.percentile(abs_err_deg, 95)),
        "max": float(np.max(abs_err_deg)),
    }


def compute_outage_metrics(
    trajectory: List[TrajectorySample],
    outage_start_ns: int,
    outage_end_ns: int,
    recovery_end_ns: Optional[int] = None,
    dataset: str = "UNKNOWN",
    sequence: str = "UNKNOWN",
    split: str = "UNKNOWN",
    configuration: str = "UNKNOWN",
) -> OutageEvaluationMetrics:
    """
    Compute the complete set of dead-reckoning and recovery evaluation metrics.
    Outage window: [outage_start_ns, outage_end_ns]
    Recovery window: (outage_end_ns, recovery_end_ns]
    """
    metrics = OutageEvaluationMetrics(
        dataset=dataset,
        sequence=sequence,
        split=split,
        configuration=configuration,
        outage_start_s=outage_start_ns * 1e-9,
        outage_duration_s=(outage_end_ns - outage_start_ns) * 1e-9,
    )

    outage_samples = [
        s for s in trajectory
        if outage_start_ns <= s.timestamp_ns <= outage_end_ns
    ]
    metrics.outage_samples = len(outage_samples)

    if not outage_samples:
        return metrics

    has_gt_pos = all(s.pos_gt is not None for s in outage_samples)
    metrics.ground_truth_available = has_gt_pos

    # 1. Trajectory Positions
    pos_est = np.array([s.pos_est for s in outage_samples], dtype=float)
    metrics.estimated_distance_m = calculate_traveled_distance(pos_est)

    metrics.ai_velocity_active_samples = sum(1 for s in outage_samples if s.ai_velocity is not None)
    metrics.nhc_active_samples = sum(1 for s in outage_samples if s.nhc_active)
    metrics.zupt_active_samples = sum(1 for s in outage_samples if s.zupt_active)

    if has_gt_pos:
        pos_gt = np.array([s.pos_gt for s in outage_samples], dtype=float)
        gt_dist = calculate_traveled_distance(pos_gt)
        metrics.ground_truth_distance_m = gt_dist

        # Absolute Position Error
        _, pos_stats = calculate_position_errors(pos_est, pos_gt)
        metrics.end_position_error_m = pos_stats["end"]
        metrics.mean_position_error_m = pos_stats["mean"]
        metrics.median_position_error_m = pos_stats["median"]
        metrics.p95_position_error_m = pos_stats["p95"]
        metrics.max_position_error_m = pos_stats["max"]
        metrics.position_rmse_m = pos_stats["rmse"]

        # Drift Percentages (require GT distance > 1.0m to avoid divide-by-zero on stationary)
        if gt_dist >= 1.0:
            metrics.drift_percent = float(100.0 * pos_stats["end"] / gt_dist)
            metrics.max_drift_percent = float(100.0 * pos_stats["max"] / gt_dist)
        else:
            metrics.drift_percent = None
            metrics.max_drift_percent = None

        # Relative Trajectory Error (RPE)
        _, rpe_stats = calculate_relative_position_error(pos_est, pos_gt)
        metrics.rpe_end_m = rpe_stats["end"]
        metrics.rpe_max_m = rpe_stats["max"]
        metrics.rpe_mean_m = rpe_stats["mean"]
        metrics.rpe_rmse_m = rpe_stats["rmse"]

    # 2. Velocity Metrics
    has_gt_vel = all(s.vel_gt is not None for s in outage_samples)
    if has_gt_vel:
        vel_est = np.array([s.vel_est for s in outage_samples], dtype=float)
        vel_gt = np.array([s.vel_gt for s in outage_samples], dtype=float)
        v_stats = calculate_velocity_errors(vel_est, vel_gt)
        metrics.velocity_rmse_mps = v_stats["rmse"]
        metrics.velocity_mae_mps = v_stats["mae"]
        metrics.velocity_median_mps = v_stats["median"]
        metrics.velocity_p95_mps = v_stats["p95"]
        metrics.velocity_error_end_mps = v_stats["end"]

        speeds_est = np.array([s.speed_est_mps for s in outage_samples], dtype=float)
        speeds_gt = [s.speed_gt_mps for s in outage_samples if s.speed_gt_mps is not None]
        if len(speeds_gt) == len(outage_samples):
            fwd_diff = np.abs(speeds_est - np.array(speeds_gt, dtype=float))
            metrics.forward_velocity_rmse_mps = float(math.sqrt(np.mean(fwd_diff ** 2)))

    # 3. Attitude / Yaw Metrics
    has_gt_yaw = all(s.yaw_gt is not None for s in outage_samples)
    if has_gt_yaw:
        yaw_est = np.array([s.yaw_est for s in outage_samples], dtype=float)
        yaw_gt = np.array([s.yaw_gt for s in outage_samples], dtype=float)
        y_stats = calculate_yaw_errors(yaw_est, yaw_gt)
        metrics.yaw_rmse_deg = y_stats["rmse"]
        metrics.yaw_mean_deg = y_stats["mean"]
        metrics.yaw_median_deg = y_stats["median"]
        metrics.yaw_p95_deg = y_stats["p95"]
        metrics.yaw_max_deg = y_stats["max"]

    # 4. Recovery Metrics
    if recovery_end_ns is not None and recovery_end_ns > outage_end_ns:
        recovery_samples = [
            s for s in trajectory
            if outage_end_ns < s.timestamp_ns <= recovery_end_ns
        ]
        if recovery_samples:
            metrics.recovery_evaluated = True

            if has_gt_pos:
                last_outage_sample = outage_samples[-1]
                metrics.pos_error_pre_recovery_m = float(np.linalg.norm(
                    last_outage_sample.pos_est - last_outage_sample.pos_gt
                ))

            first_gnss_sample = next(
                (s for s in recovery_samples if s.gnss_available), None
            )
            if first_gnss_sample and has_gt_pos and first_gnss_sample.pos_gt is not None:
                metrics.pos_error_post_first_gnss_m = float(np.linalg.norm(
                    first_gnss_sample.pos_est - first_gnss_sample.pos_gt
                ))
                metrics.recovery_position_delta_m = float(np.linalg.norm(
                    first_gnss_sample.pos_est - outage_samples[-1].pos_est
                ))
                metrics.recovery_velocity_delta_mps = float(np.linalg.norm(
                    first_gnss_sample.vel_est - outage_samples[-1].vel_est
                ))

            t0_rec = outage_end_ns
            sample_1s = min(recovery_samples, key=lambda s: abs(s.timestamp_ns - (t0_rec + 1_000_000_000)))
            sample_5s = min(recovery_samples, key=lambda s: abs(s.timestamp_ns - (t0_rec + 5_000_000_000)))

            if has_gt_pos and sample_1s.pos_gt is not None:
                metrics.pos_error_1s_recovery_m = float(np.linalg.norm(
                    sample_1s.pos_est - sample_1s.pos_gt
                ))
            if has_gt_pos and sample_5s.pos_gt is not None:
                metrics.pos_error_5s_recovery_m = float(np.linalg.norm(
                    sample_5s.pos_est - sample_5s.pos_gt
                ))

            restored_sample = next(
                (s for s in recovery_samples if s.navigation_mode == "GNSS_INS"), None
            )
            if restored_sample:
                metrics.recovery_time_s = (restored_sample.timestamp_ns - t0_rec) * 1e-9

            metrics.accepted_recovery_updates = sum(1 for s in recovery_samples if s.gnss_available)

    return metrics
