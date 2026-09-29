import math
import unittest
import numpy as np

from edge.coordinate.quaternion import Quaternion, from_euler_rad
from edge.fusion.ekf import ErrorStateEKF
from edge.fusion.error_state import (
    STATE_DIM, POSITION, VELOCITY, ATTITUDE, ACCEL_BIAS, GYRO_BIAS,
    POS_IDX, VEL_IDX, ATT_IDX, ACC_BIAS_IDX, GYRO_BIAS_IDX,
)
from edge.fusion.measurements import GNSSVelocityMeasurement
from edge.ins.state import INSState


class GNSSVelocityUpdateTests(unittest.TestCase):
    """
    Rigorously validate GNSS Velocity Measurement Update in the 15-State ES-EKF (Step 5).
    """

    def setUp(self) -> None:
        self.ekf = ErrorStateEKF()

    def test_test1_measurement_dimensions(self) -> None:
        """
        TEST 1 — Correct measurement dimensions
        Verify:
            H.shape == (3, 15)
            R.shape == (3, 3)
            S.shape == (3, 3)
        """
        z_vel = np.array([5.0, -2.0, 0.1])
        meas = GNSSVelocityMeasurement(z_vel_enu=z_vel)

        y, H, R = meas.evaluate(self.ekf.state)
        S = H @ self.ekf.P @ H.T + R

        self.assertEqual(H.shape, (3, 15))
        self.assertEqual(R.shape, (3, 3))
        self.assertEqual(S.shape, (3, 3))
        self.assertEqual(y.shape, (3,))

        # Verify H matrix has identity in velocity block and zeros elsewhere
        np.testing.assert_allclose(H[:, POS_IDX], np.zeros((3, 3)), atol=1e-12)
        np.testing.assert_allclose(H[:, VEL_IDX], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(H[:, ATT_IDX], np.zeros((3, 3)), atol=1e-12)
        np.testing.assert_allclose(H[:, ACC_BIAS_IDX], np.zeros((3, 3)), atol=1e-12)
        np.testing.assert_allclose(H[:, GYRO_BIAS_IDX], np.zeros((3, 3)), atol=1e-12)

        # Measurement type check
        self.assertEqual(meas.measurement_type, "GNSS_VELOCITY")

    def test_test2_zero_innovation(self) -> None:
        """
        TEST 2 — Zero innovation
        Set GNSS velocity = current EKF velocity.
        Expected:
            innovation ≈ [0, 0, 0]
            No meaningful state correction.
        """
        self.ekf.state.velocity_enu = np.array([14.2, -8.5, 0.3])
        p_before = self.ekf.state.position_enu.copy()
        v_before = self.ekf.state.velocity_enu.copy()
        q_before = self.ekf.state.quaternion.as_array().copy()

        # Measure exactly current velocity
        res = self.ekf.update_gnss_velocity(z_vel_enu=v_before)

        self.assertTrue(res.accepted)
        self.assertEqual(res.measurement_type, "GNSS_VELOCITY")
        np.testing.assert_allclose(res.innovation, np.zeros(3), atol=1e-12)
        self.assertAlmostEqual(res.innovation_norm, 0.0, places=10)
        self.assertAlmostEqual(res.mahalanobis_distance, 0.0, places=10)

        # State should not have moved
        np.testing.assert_allclose(self.ekf.state.position_enu, p_before, atol=1e-10)
        np.testing.assert_allclose(self.ekf.state.velocity_enu, v_before, atol=1e-10)
        np.testing.assert_allclose(self.ekf.state.quaternion.as_array(), q_before, atol=1e-10)

    def test_test3_velocity_correction(self) -> None:
        """
        TEST 3 — Velocity correction
        INS/EKF velocity: [10, 5, 0] m/s
        GNSS velocity:    [12, 4, 0] m/s
        Apply update.
        Verify corrected velocity moves toward GNSS.
        Do NOT require corrected velocity == GNSS velocity.
        """
        # Prior velocity variance = 1.0 (m/s)^2 to reflect uncertainty when a 2 m/s drift has accumulated
        self.ekf.P[VELOCITY, VELOCITY] = np.eye(3) * 1.0
        self.ekf.state.velocity_enu = np.array([10.0, 5.0, 0.0])
        z_gnss_vel = np.array([12.0, 4.0, 0.0])

        res = self.ekf.update_gnss_velocity(z_vel_enu=z_gnss_vel)

        self.assertTrue(res.accepted)
        v_corrected = self.ekf.state.velocity_enu

        # In X: started at 10, GNSS is 12 -> must move towards 12: 10 < v_corr_x < 12
        self.assertGreater(v_corrected[0], 10.0)
        self.assertLess(v_corrected[0], 12.0)

        # In Y: started at 5, GNSS is 4 -> must move towards 4: 4 < v_corr_y < 5
        self.assertLess(v_corrected[1], 5.0)
        self.assertGreater(v_corrected[1], 4.0)

        # In Z: started at 0, GNSS is 0 -> remains 0
        self.assertAlmostEqual(v_corrected[2], 0.0, places=6)

    def test_test4_no_hard_snapping(self) -> None:
        """
        TEST 4 — No hard snapping
        INS velocity = [0, 0, 0]
        GNSS velocity = [30, 0, 0]
        After update:
            0 < corrected_velocity_x < 30
        assuming realistic covariance values.
        This proves the measurement is fused through the EKF and does NOT overwrite state.
        """
        self.ekf.state.velocity_enu = np.array([0.0, 0.0, 0.0])
        z_gnss_vel = np.array([30.0, 0.0, 0.0])

        # Prior variance = 0.5 (m/s)^2, GNSS variance = 0.5 (m/s)^2
        self.ekf.P[VELOCITY, VELOCITY] = np.eye(3) * 0.5
        cov_gnss = np.eye(3) * 0.5

        # Permissive gating threshold to isolate Kalman gain evaluation
        meas = GNSSVelocityMeasurement(z_vel_enu=z_gnss_vel, cov_vel_enu=cov_gnss, gating_threshold=1000.0)
        res = self.ekf.update(meas)

        self.assertTrue(res.accepted)
        vel_x = self.ekf.state.velocity_enu[0]

        # Theoretical Kalman gain K = P / (P + R) = 0.5 / (0.5 + 0.5) = 0.5
        # Expected vel_x = 0 + 0.5 * 30 = 15 m/s
        expected_vel_x = (0.5 / (0.5 + 0.5)) * 30.0
        self.assertAlmostEqual(vel_x, expected_vel_x, places=4)

        # Explicitly proves GNSS does not snap to 30.0
        self.assertGreater(vel_x, 0.0)
        self.assertLess(vel_x, 30.0)

    def test_test5_outlier_rejection(self) -> None:
        """
        TEST 5 — Outlier rejection
        Provide an unrealistic measurement:
            GNSS velocity = [1000, 1000, 1000] m/s
        Expected:
            measurement rejected
            position unchanged
            velocity unchanged
            attitude unchanged
            biases unchanged
            covariance unchanged
        """
        self.ekf.state.position_enu = np.array([10.0, 20.0, 1.0])
        self.ekf.state.velocity_enu = np.array([5.0, 2.0, 0.0])
        self.ekf.state.quaternion = from_euler_rad(0.02, -0.01, 0.3)
        self.ekf.state.accel_bias = np.array([0.02, -0.01, 0.01])
        self.ekf.state.gyro_bias = np.array([0.001, 0.002, -0.001])

        p_before = self.ekf.state.position_enu.copy()
        v_before = self.ekf.state.velocity_enu.copy()
        q_before = self.ekf.state.quaternion.as_array().copy()
        ba_before = self.ekf.state.accel_bias.copy()
        bg_before = self.ekf.state.gyro_bias.copy()
        P_before = self.ekf.P.copy()

        outlier_gnss_vel = np.array([1000.0, 1000.0, 1000.0])
        res = self.ekf.update_gnss_velocity(z_vel_enu=outlier_gnss_vel)

        # 1. Rejected
        self.assertFalse(res.accepted)
        self.assertEqual(res.measurement_type, "GNSS_VELOCITY")
        self.assertIsNotNone(res.rejection_reason)
        self.assertGreater(res.mahalanobis_dist_sq, 16.27)

        # 2. State completely unchanged
        np.testing.assert_allclose(self.ekf.state.position_enu, p_before, atol=1e-12)
        np.testing.assert_allclose(self.ekf.state.velocity_enu, v_before, atol=1e-12)
        np.testing.assert_allclose(self.ekf.state.quaternion.as_array(), q_before, atol=1e-12)
        np.testing.assert_allclose(self.ekf.state.accel_bias, ba_before, atol=1e-12)
        np.testing.assert_allclose(self.ekf.state.gyro_bias, bg_before, atol=1e-12)

        # 3. Covariance completely unchanged
        np.testing.assert_allclose(self.ekf.P, P_before, atol=1e-12)

    def test_test6_covariance_reduction(self) -> None:
        """
        TEST 6 — Covariance reduction
        Create a velocity uncertainty larger than GNSS velocity measurement uncertainty.
        After an accepted update:
            relevant velocity covariance should decrease.
        """
        # Set large prior velocity variance: var = 1.0 (m/s)^2
        self.ekf.P[VELOCITY, VELOCITY] = np.eye(3) * 1.0
        v_var_before = np.diag(self.ekf.P[VELOCITY, VELOCITY]).copy()

        # GNSS velocity measurement with small variance = 0.04 (m/s)^2 (sigma = 0.2 m/s)
        gnss_vel_cov = np.eye(3) * 0.04
        z_gnss_vel = self.ekf.state.velocity_enu + np.array([0.1, -0.1, 0.05])

        meas = GNSSVelocityMeasurement(z_vel_enu=z_gnss_vel, cov_vel_enu=gnss_vel_cov)
        res = self.ekf.update(meas)

        self.assertTrue(res.accepted)
        v_var_after = np.diag(self.ekf.P[VELOCITY, VELOCITY])

        # Covariance must decrease substantially in all 3 axes
        for i in range(3):
            self.assertLess(v_var_after[i], v_var_before[i])
            # Theoretical posterior variance P_post = (1.0 * 0.04) / (1.0 + 0.04) ≈ 0.03846
            expected_var = (1.0 * 0.04) / (1.0 + 0.04)
            self.assertAlmostEqual(v_var_after[i], expected_var, places=4)

    def test_test7_coupled_correction(self) -> None:
        """
        TEST 7 — Coupled correction
        Create a covariance matrix with cross-correlations between velocity and:
        position, attitude, accelerometer bias, gyro bias.
        Verify that velocity innovation produces corrections to all coupled states through K.
        """
        # Cross-covariance between velocity and position
        self.ekf.P[VELOCITY, POSITION] = np.eye(3) * 0.4
        self.ekf.P[POSITION, VELOCITY] = np.eye(3) * 0.4

        # Cross-covariance between velocity and attitude
        self.ekf.P[VELOCITY, ATTITUDE] = np.eye(3) * 0.03
        self.ekf.P[ATTITUDE, VELOCITY] = np.eye(3) * 0.03

        # Cross-covariance between velocity and accel bias
        self.ekf.P[VELOCITY, ACCEL_BIAS] = np.eye(3) * 0.02
        self.ekf.P[ACCEL_BIAS, VELOCITY] = np.eye(3) * 0.02

        # Cross-covariance between velocity and gyro bias
        self.ekf.P[VELOCITY, GYRO_BIAS] = np.eye(3) * 0.002
        self.ekf.P[GYRO_BIAS, VELOCITY] = np.eye(3) * 0.002

        # Ensure P remains positive definite
        self.ekf.P += np.eye(STATE_DIM) * 0.2
        self.ekf.P = 0.5 * (self.ekf.P + self.ekf.P.T)

        p_before = self.ekf.state.position_enu.copy()
        q_before = self.ekf.state.quaternion.as_array().copy()
        ba_before = self.ekf.state.accel_bias.copy()
        bg_before = self.ekf.state.gyro_bias.copy()

        # Apply velocity innovation
        z_gnss_vel = self.ekf.state.velocity_enu + np.array([0.5, 0.5, 0.2])
        meas = GNSSVelocityMeasurement(z_vel_enu=z_gnss_vel, gating_threshold=50.0)
        res = self.ekf.update(meas)

        self.assertTrue(res.accepted)
        self.assertIsNotNone(res.kalman_gain)

        # Verify Kalman gain has non-zero blocks for all coupled states
        k = res.kalman_gain
        self.assertGreater(np.linalg.norm(k[POSITION, :]), 1e-4)
        self.assertGreater(np.linalg.norm(k[ATTITUDE, :]), 1e-4)
        self.assertGreater(np.linalg.norm(k[ACCEL_BIAS, :]), 1e-4)
        self.assertGreater(np.linalg.norm(k[GYRO_BIAS, :]), 1e-5)

        # Verify coupled states actually received corrections
        self.assertFalse(np.allclose(self.ekf.state.position_enu, p_before, atol=1e-4))
        self.assertFalse(np.allclose(self.ekf.state.quaternion.as_array(), q_before, atol=1e-5))
        self.assertFalse(np.allclose(self.ekf.state.accel_bias, ba_before, atol=1e-5))
        self.assertFalse(np.allclose(self.ekf.state.gyro_bias, bg_before, atol=1e-6))

    def test_test8_joseph_covariance(self) -> None:
        """
        TEST 8 — Joseph covariance
        After an accepted update verify:
            P ≈ P.T
            P remains finite
            P remains positive semi-definite within numerical tolerance.
        """
        z_gnss_vel = self.ekf.state.velocity_enu + np.array([0.2, -0.15, 0.05])
        res = self.ekf.update_gnss_velocity(z_vel_enu=z_gnss_vel)

        self.assertTrue(res.accepted)

        # 1. P is finite
        self.assertTrue(np.all(np.isfinite(self.ekf.P)))

        # 2. Symmetry
        np.testing.assert_allclose(self.ekf.P, self.ekf.P.T, atol=1e-10)

        # 3. Positive semi-definiteness
        eigvals = np.linalg.eigvalsh(self.ekf.P)
        self.assertGreaterEqual(np.min(eigvals), -1e-9)

        # 4. Strictly positive diagonal variances
        diag = np.diag(self.ekf.P)
        self.assertTrue(np.all(diag > 0.0))

    def test_test9_enu_velocity(self) -> None:
        """
        TEST 9 — ENU velocity conversion
        Speed and course (clockwise from North) conversion to ENU:
            v_E = speed * sin(course)
            v_N = speed * cos(course)
            v_U = vertical_mps
        Known cases:
            speed = 10 m/s, course = 0°   -> v_E = 0,  v_N = 10
            speed = 10 m/s, course = 90°  -> v_E = 10, v_N = 0
            speed = 10 m/s, course = 180° -> v_E = 0,  v_N = -10
            speed = 10 m/s, course = 270° -> v_E = -10, v_N = 0
        """
        # Case 1: Heading North (course = 0 deg)
        v1 = GNSSVelocityMeasurement.speed_course_to_enu(speed_mps=10.0, course_rad=0.0)
        np.testing.assert_allclose(v1, np.array([0.0, 10.0, 0.0]), atol=1e-12)

        # Case 2: Heading East (course = 90 deg)
        v2 = GNSSVelocityMeasurement.speed_course_to_enu(speed_mps=10.0, course_rad=math.pi / 2.0)
        np.testing.assert_allclose(v2, np.array([10.0, 0.0, 0.0]), atol=1e-12)

        # Case 3: Heading South (course = 180 deg)
        v3 = GNSSVelocityMeasurement.speed_course_to_enu(speed_mps=10.0, course_rad=math.pi)
        np.testing.assert_allclose(v3, np.array([0.0, -10.0, 0.0]), atol=1e-12)

        # Case 4: Heading West (course = 270 deg)
        v4 = GNSSVelocityMeasurement.speed_course_to_enu(speed_mps=10.0, course_rad=3.0 * math.pi / 2.0)
        np.testing.assert_allclose(v4, np.array([-10.0, 0.0, 0.0]), atol=1e-12)

        # Case 5: 45 degrees North-East with climb
        v5 = GNSSVelocityMeasurement.speed_course_to_enu(speed_mps=10.0, course_rad=math.pi / 4.0, vertical_mps=1.5)
        expected_comp = 10.0 * math.sqrt(0.5)
        np.testing.assert_allclose(v5, np.array([expected_comp, expected_comp, 1.5]), atol=1e-12)

        # Test from_speed_course constructor
        meas = GNSSVelocityMeasurement.from_speed_course(speed_mps=10.0, course_rad=math.pi / 2.0)
        np.testing.assert_allclose(meas.z, np.array([10.0, 0.0, 0.0]), atol=1e-12)

    def test_test10_repeated_velocity_updates(self) -> None:
        """
        TEST 10 — Repeated velocity updates
        Run a synthetic trajectory with repeated GNSS velocity measurements.
        Verify:
        - no NaN
        - no Inf
        - covariance remains symmetric
        - velocity error decreases
        - trajectory remains continuous
        """
        dt = 0.01  # 100 Hz IMU
        gnss_period = 10  # 10 Hz GNSS velocity (every 10 IMU steps)

        # True velocity is constant 10 m/s North
        true_vel = np.array([0.0, 10.0, 0.0])
        self.ekf.state.velocity_enu = np.array([0.0, 10.0, 0.0])

        # Accelerometer has a small constant bias of 0.05 m/s^2 causing open-loop velocity drift
        accel_raw = np.array([0.0, 0.05, 9.80665])
        gyro_raw = np.zeros(3)

        vel_errors = []

        for step in range(200):  # 2.0 seconds
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)

            # 1. High-rate prediction step
            self.ekf.predict(accel_raw, gyro_raw, timestamp_ns=t_ns)

            # 2. GNSS velocity update at 10 Hz
            if (step + 1) % gnss_period == 0:
                # GNSS measures true velocity with zero-mean noise
                gnss_noise = np.array([0.02, -0.01, 0.0])
                z_gnss_vel = true_vel + gnss_noise
                res = self.ekf.update_gnss_velocity(z_vel_enu=z_gnss_vel)
                self.assertTrue(res.accepted)

                # Record velocity error
                err = float(np.linalg.norm(self.ekf.state.velocity_enu - true_vel))
                vel_errors.append(err)

            # Check health at every step
            self.assertTrue(np.all(np.isfinite(self.ekf.P)))
            self.assertTrue(np.all(np.isfinite(self.ekf.state.velocity_enu)))
            np.testing.assert_allclose(self.ekf.P, self.ekf.P.T, atol=1e-8)

        # With 10 Hz GNSS velocity updates, velocity error is bounded tightly (< 0.1 m/s)
        self.assertLess(vel_errors[-1], 0.1)
        mean_error = float(np.mean(vel_errors[5:]))
        self.assertLess(mean_error, 0.05)


if __name__ == "__main__":
    unittest.main()
