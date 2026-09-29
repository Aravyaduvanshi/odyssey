import math
import numpy as np
import pytest

from edge.coordinate.quaternion import Quaternion, from_euler_rad
from edge.fusion.ekf import ErrorStateEKF
from edge.fusion.measurements import NHCMeasurement, ZUPTMeasurement
from edge.ins.state import INSState
from edge.motion.constraints import NHCValidator, ZUPTValidator
from edge.motion.detector import MotionDetector, MotionThresholds, VehicleMotionState
from edge.navigation.engine import NavigationEngine
from edge.navigation.models import MotionState
from edge.sensors.models import SensorSample, SensorSource


# ============================================================================
# 1. Motion Detector Tests
# ============================================================================

def test_motion_detector_stationary():
    detector = MotionDetector()
    accel = np.array([0.02, -0.01, 9.81])
    gyro = np.array([0.005, 0.002, -0.001])
    
    state, score = detector.detect(accel, gyro, estimated_speed_mps=0.05)
    assert state == VehicleMotionState.STATIONARY
    assert score < 1.0


def test_motion_detector_normal_driving():
    detector = MotionDetector()
    accel = np.array([0.1, 0.05, 9.81])
    gyro = np.array([0.01, 0.01, 0.01])
    
    state, score = detector.detect(accel, gyro, estimated_speed_mps=15.0)
    assert state == VehicleMotionState.NORMAL_DRIVING


def test_motion_detector_acceleration():
    detector = MotionDetector()
    accel = np.array([1.5, 0.1, 9.81])  # Strong forward acceleration
    gyro = np.array([0.01, 0.01, 0.01])
    
    state, score = detector.detect(accel, gyro, estimated_speed_mps=5.0)
    assert state == VehicleMotionState.ACCELERATION


def test_motion_detector_braking():
    detector = MotionDetector()
    accel = np.array([-1.8, -0.05, 9.81])  # Strong braking deceleration
    gyro = np.array([0.01, 0.01, 0.01])
    
    state, score = detector.detect(accel, gyro, estimated_speed_mps=12.0)
    assert state == VehicleMotionState.BRAKING


def test_motion_detector_turning():
    detector = MotionDetector()
    # High yaw rate
    accel = np.array([0.1, 0.2, 9.81])
    gyro = np.array([0.01, 0.01, 0.25])  # 0.25 rad/s yaw rate
    
    state, _ = detector.detect(accel, gyro, estimated_speed_mps=10.0)
    assert state == VehicleMotionState.TURNING

    # Or high lateral acceleration
    accel_lat = np.array([0.1, 1.8, 9.81])  # 1.8 m/s^2 lateral
    gyro_calm = np.array([0.01, 0.01, 0.05])
    state_lat, _ = detector.detect(accel_lat, gyro_calm, estimated_speed_mps=10.0)
    assert state_lat == VehicleMotionState.TURNING


def test_motion_detector_high_vibration():
    detector = MotionDetector()
    # High vibration score: abs(ax) + abs(ay) + max(0, abs(az - g)) > 3.5
    accel = np.array([2.0, 1.8, 12.0])
    gyro = np.array([0.02, 0.02, 0.02])
    
    state, score = detector.detect(accel, gyro, estimated_speed_mps=10.0)
    assert state == VehicleMotionState.HIGH_VIBRATION
    assert score > 3.5


def test_motion_detector_phone_movement():
    detector = MotionDetector()
    # High angular rate exceeding vehicle chassis physical capabilities (> 2.0 rad/s)
    accel = np.array([0.2, 0.2, 9.81])
    gyro = np.array([2.5, 0.5, 0.3])
    
    state, _ = detector.detect(accel, gyro, estimated_speed_mps=5.0)
    assert state == VehicleMotionState.PHONE_MOVEMENT


def test_motion_thresholds_from_config():
    cfg = {
        "motion": {
            "stationary": {"speed_max_mps": 0.25, "gyro_max_rads": 0.04},
            "turning": {"gyro_entry_rads": 0.18},
            "vibration": {"score_threshold": 4.0},
        }
    }
    thresholds = MotionThresholds.from_config(cfg)
    assert thresholds.stationary_speed_max_mps == 0.25
    assert thresholds.stationary_gyro_max_rads == 0.04
    assert thresholds.turning_gyro_min_rads == 0.18
    assert thresholds.vibration_min_score == 4.0


# ============================================================================
# 2. Constraint Validator Tests
# ============================================================================

def test_nhc_validator():
    validator = NHCValidator(max_yaw_rate_rads=0.12, max_vibration_score=3.0)

    # Valid during smooth driving
    assert validator.is_valid(VehicleMotionState.NORMAL_DRIVING, vibration_score=1.2, yaw_rate_rads=0.03)

    # Invalid during turning state
    assert not validator.is_valid(VehicleMotionState.TURNING, vibration_score=1.2, yaw_rate_rads=0.03)

    # Invalid during excessive yaw rate
    assert not validator.is_valid(VehicleMotionState.NORMAL_DRIVING, vibration_score=1.2, yaw_rate_rads=0.20)

    # Invalid during high vibration
    assert not validator.is_valid(VehicleMotionState.HIGH_VIBRATION, vibration_score=3.8, yaw_rate_rads=0.03)
    assert not validator.is_valid(VehicleMotionState.NORMAL_DRIVING, vibration_score=3.5, yaw_rate_rads=0.03)

    # Invalid during phone movement
    assert not validator.is_valid(VehicleMotionState.PHONE_MOVEMENT, vibration_score=1.0, yaw_rate_rads=0.02)


def test_zupt_validator():
    validator = ZUPTValidator(required_consecutive_samples=5)

    # Must require 5 consecutive stationary steps
    for _ in range(4):
        valid = validator.evaluate(
            VehicleMotionState.STATIONARY,
            gyro_norm=0.01,
            accel_norm=9.81,
            ai_velocity_mps=0.05,
            gnss_velocity_mps=0.02,
        )
        assert not valid

    # 5th step satisfies criteria
    valid_5th = validator.evaluate(
        VehicleMotionState.STATIONARY,
        gyro_norm=0.01,
        accel_norm=9.81,
        ai_velocity_mps=0.05,
        gnss_velocity_mps=0.02,
    )
    assert valid_5th
    assert validator.consecutive_stationary_count == 5

    # Any non-stationary sample resets the counter
    valid_moving = validator.evaluate(
        VehicleMotionState.NORMAL_DRIVING,
        gyro_norm=0.05,
        accel_norm=9.81,
    )
    assert not valid_moving
    assert validator.consecutive_stationary_count == 0

    # Inconsistent AI velocity rejects ZUPT even if motion state is stationary
    for _ in range(5):
        valid_bad_ai = validator.evaluate(
            VehicleMotionState.STATIONARY,
            gyro_norm=0.01,
            accel_norm=9.81,
            ai_velocity_mps=1.5,  # Speed inconsistent with stationary
        )
    assert not valid_bad_ai


# ============================================================================
# 3. NHC Measurement & EKF Update Tests
# ============================================================================

def test_nhc_measurement_evaluation():
    # Facing North (heading = 90 deg counter-clockwise from East in ENU):
    # R_nav_veh translates vehicle X to North [0, 1, 0], vehicle Y to West [-1, 0, 0], vehicle Z to Up [0, 0, 1].
    q_north = from_euler_rad(0.0, 0.0, math.pi / 2.0)
    
    # State moving purely North at 10 m/s with small lateral slip (v_E = 0.5 m/s)
    # v_veh = R_v_n @ v_nav:
    # v_veh_x (forward) = 10 m/s
    # v_veh_y (lateral) = -0.5 m/s
    # v_veh_z (vertical) = 0.0 m/s
    state = INSState(
        position_enu=np.zeros(3),
        velocity_enu=np.array([0.5, 10.0, 0.0]),
        quaternion=q_north,
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        timestamp_ns=0,
    )

    nhc = NHCMeasurement(lateral_std_mps=0.1, vertical_std_mps=0.15)
    assert nhc.measurement_type == "NHC"

    innovation, H, R = nhc.evaluate(state)
    assert innovation.shape == (2,)
    # Target lateral and vertical velocity is 0:
    # innovation = 0 - v_veh[1:3]
    assert np.isclose(innovation[0], 0.5, atol=1e-3)
    assert np.isclose(innovation[1], 0.0, atol=1e-3)
    assert H.shape == (2, 15)
    assert R.shape == (2, 2)


def test_nhc_ekf_filter_update():
    ekf = ErrorStateEKF()
    # Vehicle moving forward along East (v_E = 10 m/s) but with unwanted lateral drift (v_N = 1.0 m/s)
    q_east = Quaternion(1.0, 0.0, 0.0, 0.0)  # vehicle X aligns with ENU East, Y with ENU North
    ekf.state.quaternion = q_east
    ekf.state.velocity_enu = np.array([10.0, 1.0, 0.0])

    res = ekf.update_nhc(lateral_std_mps=0.1, vertical_std_mps=0.1)
    assert res.accepted
    assert res.measurement_type == "NHC"

    # After update, lateral velocity in vehicle frame (v_veh_y) must be corrected towards zero
    # while forward velocity (v_veh_x) remains virtually unaffected
    v_veh_after = ekf.state.rotation_matrix.T @ ekf.state.velocity_enu
    assert abs(v_veh_after[1]) < 0.05
    assert np.isclose(v_veh_after[0], 10.0, atol=0.2)


# ============================================================================
# 4. ZUPT Measurement & EKF Update Tests
# ============================================================================

def test_zupt_measurement_evaluation():
    state = INSState(
        position_enu=np.array([1.0, 2.0, 0.0]),
        velocity_enu=np.array([0.1, -0.05, 0.02]),
        quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        timestamp_ns=0,
    )

    zupt = ZUPTMeasurement(noise_std_mps=0.05)
    assert zupt.measurement_type == "ZUPT"

    innovation, H, R = zupt.evaluate(state)
    assert innovation.shape == (3,)
    assert np.allclose(innovation, -state.velocity_enu)
    assert H.shape == (3, 15)
    # H on velocity block (cols 3:6) is I_3
    assert np.allclose(H[:, 3:6], np.eye(3))
    assert np.allclose(H[:, 0:3], np.zeros((3, 3)))
    assert R.shape == (3, 3)


def test_zupt_ekf_filter_update():
    ekf = ErrorStateEKF()
    # Stationary vehicle with residual velocity drift
    ekf.state.velocity_enu = np.array([0.2, -0.15, 0.05])
    initial_vel_cov = np.diag(ekf.velocity_covariance).copy()

    res = ekf.update_zupt(noise_std_mps=0.02)
    assert res.accepted
    assert res.measurement_type == "ZUPT"

    # Velocity must be pulled to zero
    assert np.linalg.norm(ekf.state.velocity_enu) < 0.05
    # Velocity covariance must be reduced
    updated_vel_cov = np.diag(ekf.velocity_covariance)
    assert np.all(updated_vel_cov < initial_vel_cov)


# ============================================================================
# 5. NavigationEngine Integration Tests
# ============================================================================

def test_navigation_engine_motion_constraints_integration():
    engine = NavigationEngine()
    engine.enable_gnss_position_update = False
    engine.enable_gnss_velocity_update = False
    engine.enable_ai_velocity_update = False
    engine.enable_nhc_update = True
    engine.enable_zupt_update = True

    # 1. Drive straight at 10 m/s for 10 samples (should activate NHC and NOT ZUPT)
    # Forward along body X: ax ~ 0, ay ~ 0, az ~ 9.81
    sample_idx = 0
    for _ in range(10):
        t_ns = 1_000_000_000 + sample_idx * 10_000_000  # 100 Hz step (10ms)
        sample = SensorSample(
            timestamp_ns=t_ns,
            ax=0.02,
            ay=0.01,
            az=9.80665,
            gx=0.001,
            gy=0.001,
            gz=0.001,
            sensor_source=SensorSource.SYNTHETIC,
        )
        # Seed velocity on state
        engine.ekf.state.velocity_enu = np.array([10.0, 0.1, 0.0])
        state = engine._sample_to_state(sample)
        sample_idx += 1

    assert state.nhc_active
    assert not state.zupt_active
    assert state.motion_state in (MotionState.NORMAL_DRIVING, MotionState.STATIONARY)

    # 2. Stop the vehicle completely for 10 samples (should activate ZUPT and deactivate NHC)
    engine.ekf.state.velocity_enu = np.array([0.01, 0.01, 0.0])
    for _ in range(10):
        t_ns = 1_000_000_000 + sample_idx * 10_000_000
        sample = SensorSample(
            timestamp_ns=t_ns,
            ax=0.0,
            ay=0.0,
            az=9.80665,
            gx=0.0001,
            gy=0.0001,
            gz=0.0001,
            sensor_source=SensorSource.SYNTHETIC,
        )
        state = engine._sample_to_state(sample)
        sample_idx += 1

    assert state.zupt_active
    # Redundant NHC must NOT be active when ZUPT is active
    assert not state.nhc_active
    assert state.motion_state == MotionState.STATIONARY
