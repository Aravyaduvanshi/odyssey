"""
Regression tests for Step 9C: Real-Data Replay Preprocessing Before Vta2 Evaluation.

Covers:
1. repeated GPS fix is submitted once
2. repeated GPS fix does not collapse EKF covariance
3. Velocity TCN receives gravity-removed acceleration
4. stationary AI feature ACC_VEH_Z ≈ 0
5. evaluation origin alignment
6. ground truth never enters navigation state
"""

import math
from pathlib import Path
import numpy as np
import pytest

from edge.coordinate.frames import ImuVector, VehicleAlignedIMU, VehicleFrameTransform
from edge.evaluation.metrics import TrajectorySample, compute_outage_metrics
from edge.evaluation.replay import IOVNBDLoader
from edge.models.velocity.preprocessing import VelocityTCNPreprocessor
from edge.navigation.engine import NavigationEngine
from edge.sensors.buffer import RollingIMUBuffer
from edge.sensors.models import SensorSample, SensorSource


def make_test_sample(
    timestamp_ns: int,
    ax: float = 0.0,
    ay: float = 0.0,
    az: float = 9.80665,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
    latitude: float | None = None,
    longitude: float | None = None,
    altitude: float | None = None,
    gnss_velocity: float | None = None,
    gnss_heading: float | None = None,
    gnss_available: bool = False,
    gnss_fix_id: str | None = None,
    truth_local_x: float | None = None,
    truth_local_y: float | None = None,
    truth_yaw_rad: float | None = None,
) -> SensorSample:
    return SensorSample(
        timestamp_ns=timestamp_ns,
        ax=ax,
        ay=ay,
        az=az,
        gx=gx,
        gy=gy,
        gz=gz,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        gnss_velocity=gnss_velocity,
        gnss_heading=gnss_heading,
        gnss_accuracy=2.5 if gnss_available else None,
        gnss_available=gnss_available,
        gnss_fix_id=gnss_fix_id,
        sensor_source=SensorSource.IO_VNBD,
        truth_local_x=truth_local_x,
        truth_local_y=truth_local_y,
        truth_yaw_rad=truth_yaw_rad,
    )


def test_repeated_gps_fix_submitted_once():
    """Verify that 10 consecutive samples with identical GPS fix identity result in exactly ONE update."""
    engine = NavigationEngine()
    engine.ekf.max_dt_sec = 0.25
    engine.ins_mechanization.max_dt_sec = 0.25
    engine.start()

    # Feed 10 samples with identical GPS fix ID and coordinates at 10 Hz (every 100ms)
    t0 = 1_000_000_000
    for i in range(10):
        sample = make_test_sample(
            timestamp_ns=t0 + i * 100_000_000,
            latitude=52.760456,
            longitude=-1.679398,
            altitude=118.13,
            gnss_velocity=5.22 / 3.6,
            gnss_heading=math.radians(319.47),
            gnss_available=True,
            gnss_fix_id="fix_vta2_first",
        )
        engine.step(sample)

    assert engine._gnss_updates_attempted == 10
    assert engine._gnss_updates_accepted == 1
    assert engine._gnss_repeated_fixes_ignored == 9


def test_repeated_gps_fix_does_not_collapse_ekf_covariance():
    """
    Verify that repeated GPS fixes do NOT collapse the EKF covariance.
    Comparing:
    - Engine A: receives 1 genuine GPS fix, followed by 9 IMU-only samples.
    - Engine B: receives 1 genuine GPS fix, followed by 9 repeated GPS samples (with deduplication).
    Covariance in Engine B must match Engine A and must not collapse by 10x.
    """
    engine_a = NavigationEngine()
    engine_b = NavigationEngine()
    engine_a.ekf.max_dt_sec = 0.25
    engine_a.ins_mechanization.max_dt_sec = 0.25
    engine_b.ekf.max_dt_sec = 0.25
    engine_b.ins_mechanization.max_dt_sec = 0.25
    engine_a.start()
    engine_b.start()

    t0 = 1_000_000_000

    # Sample 0: genuine GPS fix for both
    s0 = make_test_sample(
        timestamp_ns=t0,
        latitude=52.760456,
        longitude=-1.679398,
        altitude=118.13,
        gnss_velocity=2.0,
        gnss_heading=0.0,
        gnss_available=True,
        gnss_fix_id="fix_test_0",
    )
    engine_a.step(s0)
    engine_b.step(s0)

    # Samples 1-9: Engine A has gnss_available=False, Engine B has repeated gnss_available=True with same fix_id
    for i in range(1, 10):
        t = t0 + i * 100_000_000
        sa = make_test_sample(
            timestamp_ns=t,
            latitude=None,
            longitude=None,
            gnss_available=False,
            gnss_fix_id=None,
        )
        sb = make_test_sample(
            timestamp_ns=t,
            latitude=52.760456,
            longitude=-1.679398,
            altitude=118.13,
            gnss_velocity=2.0,
            gnss_heading=0.0,
            gnss_available=True,
            gnss_fix_id="fix_test_0",
        )
        engine_a.step(sa)
        engine_b.step(sb)

    # Position covariance diagonal elements must be identical
    cov_a = engine_a.ekf.P[:3, :3]
    cov_b = engine_b.ekf.P[:3, :3]

    np.testing.assert_allclose(
        np.diag(cov_b),
        np.diag(cov_a),
        rtol=1e-5,
        err_msg="Deduplication failed: repeated GPS fixes altered EKF covariance!",
    )


def test_velocity_tcn_receives_gravity_removed_acceleration():
    """Verify that RollingIMUBuffer.get_vehicle_window produces linear acceleration with gravity removed on Z."""
    buf = RollingIMUBuffer(window_size=100)
    transform = VehicleFrameTransform()

    # Fill buffer with 105 samples at 10 Hz (every 100ms) with stationary level specific force (az = 9.80665)
    for i in range(105):
        s = make_test_sample(
            timestamp_ns=1_000_000_000 + i * 100_000_000,
            ax=0.0,
            ay=0.0,
            az=9.80665,
        )
        buf.add_sample(s)

    assert buf.is_ready

    # AI vehicle window with default linear_accel=True
    ai_win = buf.get_vehicle_window(transform, linear_accel=True)
    assert ai_win.shape == (100, 6)

    # Linear vertical acceleration on column 2 (ACC_VEH_Z) must be ~0 m/s^2
    np.testing.assert_allclose(
        ai_win[:, 2],
        0.0,
        atol=1e-4,
        err_msg="AI window ACC_VEH_Z did not have gravity removed!",
    )

    # Raw window with linear_accel=False maintains specific force (~9.80665 m/s^2)
    raw_win = buf.get_vehicle_window(transform, linear_accel=False)
    np.testing.assert_allclose(raw_win[:, 2], 9.80665, atol=1e-4)


def test_stationary_ai_feature_acc_veh_z_near_zero():
    """Verify that stationary level data produces ACC_VEH_Z ≈ 0 m/s² for the AI input branch."""
    # 1. Direct VehicleAlignedIMU check
    veh_imu = VehicleAlignedIMU(
        acc_veh_x=0.02,
        acc_veh_y=-0.01,
        acc_veh_z=9.81,
        gyro_yaw_corrected=0.0,
        gyro_pitch_corrected=0.0,
        gyro_roll_corrected=0.0,
    )
    ai_feats = veh_imu.as_ai_features()
    assert abs(ai_feats[2]) < 0.05, f"Expected ACC_VEH_Z ≈ 0, got {ai_feats[2]}"

    # 2. Preprocessor transformation with contract safeguard (no double removal)
    prep = VelocityTCNPreprocessor()

    # Stationary linear acceleration window (already gravity-compensated, ~0 m/s^2)
    win_linear = np.zeros((100, 6), dtype=np.float32)
    win_linear[:, 2] = -0.018169  # exact mean from training dataset
    norm_linear = prep.transform(win_linear)
    # Channel 2 should normalize to ~0.0 std deviations
    np.testing.assert_allclose(norm_linear[:, :, 2], 0.0, atol=1e-3)

    # Raw specific force window (+9.80665 m/s^2) should be automatically compensated once
    win_raw = np.zeros((100, 6), dtype=np.float32)
    win_raw[:, 2] = 9.80665 - 0.018169
    norm_raw = prep.transform(win_raw)
    np.testing.assert_allclose(norm_raw[:, :, 2], 0.0, atol=1e-3)


def test_evaluation_origin_alignment():
    """Verify that evaluation origin alignment aligns reference trajectory with estimated trajectory at t0."""
    t0_ns = 1_000_000_000

    # Synthetic trajectory with artificial 25m offset between reference and estimated at start
    traj = []
    for i in range(10):
        t = t0_ns + i * 100_000_000
        p_est = np.array([float(i), 0.0, 0.0])
        # Ground truth starts at [25.0, 10.0, 0.0] and moves at same speed
        p_gt = np.array([25.0 + float(i), 10.0, 0.0])
        rec = TrajectorySample(
            timestamp_ns=t,
            pos_est=p_est,
            pos_gt=p_gt.copy(),
            vel_est=np.array([1.0, 0.0, 0.0]),
            vel_gt=np.array([1.0, 0.0, 0.0]),
            yaw_est=0.0,
            yaw_gt=0.0,
            navigation_mode="GNSS_INS",
            gnss_available=True,
        )
        traj.append(rec)

    # Perform evaluation origin alignment
    valid_indices = [i for i, r in enumerate(traj) if r.pos_gt is not None and r.pos_est is not None]
    assert valid_indices
    i0 = valid_indices[0]
    origin_offset = traj[i0].pos_gt - traj[i0].pos_est
    for i in valid_indices:
        traj[i].pos_gt = traj[i].pos_gt - origin_offset

    # Trajectory ground truth at t0 must match estimated position exactly
    np.testing.assert_allclose(traj[0].pos_gt, traj[0].pos_est, atol=1e-6)

    # Position error across the entire trajectory must be 0 (since motion was identical)
    for rec in traj:
        np.testing.assert_allclose(rec.pos_gt, rec.pos_est, atol=1e-6)


def test_ground_truth_never_enters_navigation_state():
    """
    CRITICAL INVARIANT TEST:
    Verify that NavigationEngine never inspects, uses, or depends on ground truth fields.
    Running two identical sensor streams—one with ground truth populated, one with None—
    must produce 100% numerically identical navigation states and filter covariances.
    """
    engine_clean = NavigationEngine()
    engine_tainted = NavigationEngine()
    engine_clean.ekf.max_dt_sec = 0.25
    engine_clean.ins_mechanization.max_dt_sec = 0.25
    engine_tainted.ekf.max_dt_sec = 0.25
    engine_tainted.ins_mechanization.max_dt_sec = 0.25
    engine_clean.start()
    engine_tainted.start()

    t0 = 1_000_000_000
    for i in range(25):
        t = t0 + i * 100_000_000
        # Common sensor inputs
        ax = 0.5 * math.sin(i * 0.1)
        ay = 0.1 * math.cos(i * 0.1)
        az = 9.80665
        gx = 0.01
        gy = -0.01
        gz = 0.02
        lat = 52.7604 + i * 1e-5
        lon = -1.6793 + i * 1e-5
        spd = 5.0

        sample_clean = make_test_sample(
            timestamp_ns=t,
            ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz,
            latitude=lat, longitude=lon, altitude=100.0,
            gnss_velocity=spd, gnss_heading=0.0,
            gnss_available=True,
            gnss_fix_id=f"fix_{i}",
            truth_local_x=None,
            truth_local_y=None,
            truth_yaw_rad=None,
        )

        sample_tainted = make_test_sample(
            timestamp_ns=t,
            ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz,
            latitude=lat, longitude=lon, altitude=100.0,
            gnss_velocity=spd, gnss_heading=0.0,
            gnss_available=True,
            gnss_fix_id=f"fix_{i}",
            truth_local_x=99999.99,   # Poison values
            truth_local_y=-88888.88,
            truth_yaw_rad=2.71828,
        )

        state_clean = engine_clean.step(sample_clean)
        state_tainted = engine_tainted.step(sample_tainted)

        assert state_clean is not None
        assert state_tainted is not None

        assert state_clean.local_x == state_tainted.local_x
        assert state_clean.local_y == state_tainted.local_y
        assert state_clean.local_z == state_tainted.local_z
        assert state_clean.velocity_mps == state_tainted.velocity_mps
        assert state_clean.heading_rad == state_tainted.heading_rad

    # EKF state vectors and covariance matrices must be bit-for-bit identical
    np.testing.assert_allclose(
        engine_clean.ekf.state.position_enu,
        engine_tainted.ekf.state.position_enu,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        engine_clean.ekf.state.velocity_enu,
        engine_tainted.ekf.state.velocity_enu,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        engine_clean.ekf.P,
        engine_tainted.ekf.P,
        atol=1e-12,
    )
