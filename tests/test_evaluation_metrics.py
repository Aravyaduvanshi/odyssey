"""
Unit tests for edge/evaluation/metrics.py.
Verifies all mathematical definitions of position, drift, velocity, yaw, and recovery metrics.
"""

import math
import numpy as np
import pytest

from edge.evaluation.metrics import (
    TrajectorySample,
    calculate_position_errors,
    calculate_relative_position_error,
    calculate_traveled_distance,
    calculate_velocity_errors,
    calculate_yaw_errors,
    compute_outage_metrics,
    wrap_to_pi,
)


def test_wrap_to_pi():
    """Verify wrap_to_pi wraps any angle to [-pi, pi]."""
    assert math.isclose(wrap_to_pi(0.0), 0.0)
    assert math.isclose(wrap_to_pi(math.pi), math.pi, abs_tol=1e-6)
    assert math.isclose(wrap_to_pi(-math.pi), -math.pi, abs_tol=1e-6)
    assert math.isclose(wrap_to_pi(2 * math.pi), 0.0, abs_tol=1e-6)
    # 355 deg in rad is close to -5 deg
    deg355 = math.radians(355.0)
    assert math.isclose(wrap_to_pi(deg355), math.radians(-5.0), abs_tol=1e-6)


def test_zero_position_error():
    """Verify metrics when estimated equals ground truth."""
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 4.0, 0.0],
    ])
    ape, stats = calculate_position_errors(pts, pts)
    assert len(ape) == 3
    assert np.allclose(ape, 0.0)
    assert stats["end"] == 0.0
    assert stats["rmse"] == 0.0
    assert stats["max"] == 0.0


def test_known_position_error():
    """Verify metrics for known constant offset."""
    gt = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ])
    # Constant offset of (3, 4, 0) -> norm = 5.0
    est = gt + np.array([3.0, 4.0, 0.0])
    ape, stats = calculate_position_errors(est, gt)
    assert np.allclose(ape, 5.0)
    assert stats["end"] == 5.0
    assert stats["rmse"] == 5.0
    assert stats["mean"] == 5.0
    assert stats["median"] == 5.0
    assert stats["max"] == 5.0


def test_rmse_and_mae_calculation():
    """Verify RMSE and MAE formulas."""
    v_est = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    v_gt = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    stats = calculate_velocity_errors(v_est, v_gt)
    # norms are 1, 2, 3
    # MAE = (1 + 2 + 3) / 3 = 2.0
    # RMSE = sqrt((1 + 4 + 9) / 3) = sqrt(14 / 3) ≈ 2.1602
    assert math.isclose(stats["mae"], 2.0, abs_tol=1e-5)
    assert math.isclose(stats["rmse"], math.sqrt(14.0 / 3.0), abs_tol=1e-5)
    assert stats["median"] == 2.0
    assert stats["end"] == 3.0


def test_p95_percentile():
    """Verify 95th percentile error."""
    # 100 values from 1 to 100
    pos_gt = np.zeros((100, 3))
    pos_est = np.zeros((100, 3))
    for i in range(100):
        pos_est[i, 0] = float(i + 1)
    _, stats = calculate_position_errors(pos_est, pos_gt)
    # 95th percentile of 1..100
    expected_p95 = float(np.percentile(np.arange(1, 101), 95))
    assert math.isclose(stats["p95"], expected_p95, abs_tol=1e-3)


def test_traveled_distance():
    """Verify step-by-step accumulated trajectory distance."""
    # Move along East: 0 -> 10 -> 25 -> 25 (stop) -> 35
    pts = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [25.0, 0.0, 0.0],
        [25.0, 0.0, 0.0],
        [35.0, 0.0, 0.0],
    ])
    dist = calculate_traveled_distance(pts)
    assert math.isclose(dist, 35.0)


def test_drift_percentage_calculation():
    """Verify drift percentage = 100 * end_err / gt_dist."""
    # GT travels 100m
    gt_pts = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    # Est drifts 2.0m at the end
    est_pts = np.array([[0.0, 0.0, 0.0], [50.5, 0.0, 0.0], [102.0, 0.0, 0.0]])

    samples = [
        TrajectorySample(
            timestamp_ns=1_000_000_000 + i * 1_000_000_000,
            pos_est=est_pts[i],
            pos_gt=gt_pts[i],
            vel_est=np.array([10.0, 0.0, 0.0]),
            vel_gt=np.array([10.0, 0.0, 0.0]),
            yaw_est=0.0,
            yaw_gt=0.0,
            navigation_mode="DEAD_RECKONING",
            gnss_available=False,
        )
        for i in range(3)
    ]

    metrics = compute_outage_metrics(
        trajectory=samples,
        outage_start_ns=1_000_000_000,
        outage_end_ns=3_000_000_000,
    )

    # Traveled distance = 100m, end error = 2.0m -> drift = 2.0%
    assert math.isclose(metrics.ground_truth_distance_m, 100.0)
    assert math.isclose(metrics.end_position_error_m, 2.0)
    assert math.isclose(metrics.drift_percent, 2.0)


def test_zero_distance_handling():
    """Verify stationary zero-distance returns None for drift_percent without divide-by-zero."""
    # Stationary: 0 distance traveled
    gt_pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    est_pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])

    samples = [
        TrajectorySample(
            timestamp_ns=1_000_000_000 + i * 1_000_000_000,
            pos_est=est_pts[i],
            pos_gt=gt_pts[i],
            vel_est=np.zeros(3),
            vel_gt=np.zeros(3),
            yaw_est=0.0,
            yaw_gt=0.0,
            navigation_mode="DEAD_RECKONING",
            gnss_available=False,
        )
        for i in range(3)
    ]

    metrics = compute_outage_metrics(
        trajectory=samples,
        outage_start_ns=1_000_000_000,
        outage_end_ns=3_000_000_000,
    )

    assert metrics.ground_truth_distance_m == 0.0
    assert metrics.drift_percent is None
    assert metrics.max_drift_percent is None


def test_relative_trajectory_error():
    """Verify RPE isolates drift relative to outage start, cancelling initial offset."""
    # Initial offset of 50.0m before outage starts!
    init_offset = np.array([50.0, 0.0, 0.0])
    gt = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ])
    # Est maintains 50m offset plus 1m drift at end
    est = gt + init_offset
    est[-1] += np.array([1.0, 0.0, 0.0])

    rpe, stats = calculate_relative_position_error(est, gt)
    # Relative displacement error at t0 must be 0
    assert math.isclose(rpe[0], 0.0)
    assert math.isclose(rpe[1], 0.0)
    # End RPE is exactly 1.0m (ignoring the 50m initial bias)
    assert math.isclose(stats["end"], 1.0)


def test_yaw_wrapping_error():
    """Verify yaw error computes wrap_to_pi(yaw_est - yaw_gt) correctly."""
    # est is 1 deg (0.0174 rad), gt is 359 deg (6.2657 rad)
    # Error should be 2 deg, NOT 358 deg!
    yaw_est = np.array([math.radians(1.0)])
    yaw_gt = np.array([math.radians(359.0)])
    stats = calculate_yaw_errors(yaw_est, yaw_gt)
    assert math.isclose(stats["mean"], 2.0, abs_tol=1e-3)
    assert math.isclose(stats["max"], 2.0, abs_tol=1e-3)


def test_recovery_metrics():
    """Verify GNSS recovery behavior calculation."""
    samples = []
    # Outage: t=1s to t=3s
    for i in range(1, 4):
        t = i * 1_000_000_000
        samples.append(TrajectorySample(
            timestamp_ns=t,
            pos_est=np.array([float(i) * 2.0, 0.0, 0.0]),
            pos_gt=np.array([float(i) * 1.5, 0.0, 0.0]),
            vel_est=np.array([2.0, 0.0, 0.0]),
            vel_gt=np.array([1.5, 0.0, 0.0]),
            yaw_est=0.0,
            yaw_gt=0.0,
            navigation_mode="DEAD_RECKONING",
            gnss_available=False,
        ))

    # Recovery: t=4s to t=10s
    for i in range(4, 11):
        t = i * 1_000_000_000
        mode = "GNSS_INS" if i >= 8 else "GNSS_RECOVERY"
        samples.append(TrajectorySample(
            timestamp_ns=t,
            pos_est=np.array([float(i) * 1.5 + 0.2, 0.0, 0.0]),
            pos_gt=np.array([float(i) * 1.5, 0.0, 0.0]),
            vel_est=np.array([1.5, 0.0, 0.0]),
            vel_gt=np.array([1.5, 0.0, 0.0]),
            yaw_est=0.0,
            yaw_gt=0.0,
            navigation_mode=mode,
            gnss_available=True,
        ))

    metrics = compute_outage_metrics(
        trajectory=samples,
        outage_start_ns=1_000_000_000,
        outage_end_ns=3_000_000_000,
        recovery_end_ns=10_000_000_000,
    )

    assert metrics.recovery_evaluated is True
    # Outage end error at t=3s: est=6.0, gt=4.5 -> error=1.5m
    assert math.isclose(metrics.pos_error_pre_recovery_m, 1.5)
    # First GNSS sample at t=4s: est=6.2, gt=6.0 -> error=0.2m
    assert math.isclose(metrics.pos_error_post_first_gnss_m, 0.2)
    # Mode restored to GNSS_INS at t=8s -> recovery_time = 8s - 3s = 5s
    assert math.isclose(metrics.recovery_time_s, 5.0)
    assert metrics.accepted_recovery_updates == 7
