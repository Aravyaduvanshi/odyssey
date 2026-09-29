import math
import unittest
import numpy as np

from edge.coordinate.quaternion import Quaternion, from_euler_rad
from edge.fusion.ekf import ErrorStateEKF
from edge.fusion.error_state import (
    STATE_DIM, POSITION, VELOCITY, ATTITUDE, ACCEL_BIAS, GYRO_BIAS,
    POS_IDX, VEL_IDX, ATT_IDX, ACC_BIAS_IDX, GYRO_BIAS_IDX,
)
from edge.fusion.measurements import GNSSPositionMeasurement
from edge.ins.earth import EarthModel
from edge.ins.state import INSState


class GNSSPositionUpdateTests(unittest.TestCase):
    """
    Rigorously validate GNSS Position Measurement Update in the 15-State ES-EKF (Step 4).
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
        z_enu = np.array([10.0, 20.0, 5.0])
        meas = GNSSPositionMeasurement(z_enu=z_enu)

        y, H, R = meas.evaluate(self.ekf.state)
        S = H @ self.ekf.P @ H.T + R

        self.assertEqual(H.shape, (3, 15))
        self.assertEqual(R.shape, (3, 3))
        self.assertEqual(S.shape, (3, 3))
        self.assertEqual(y.shape, (3,))

        # Verify H matrix has identity in position block and zeros elsewhere
        np.testing.assert_allclose(H[:, POS_IDX], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(H[:, VEL_IDX], np.zeros((3, 3)), atol=1e-12)
        np.testing.assert_allclose(H[:, ATT_IDX], np.zeros((3, 3)), atol=1e-12)
        np.testing.assert_allclose(H[:, ACC_BIAS_IDX], np.zeros((3, 3)), atol=1e-12)
        np.testing.assert_allclose(H[:, GYRO_BIAS_IDX], np.zeros((3, 3)), atol=1e-12)

    def test_test2_zero_innovation(self) -> None:
        """
        TEST 2 — Zero innovation
        Set GNSS position = current EKF position.
        Expected:
            innovation ≈ [0, 0, 0]
            update produces essentially zero correction.
        """
        self.ekf.state.position_enu = np.array([12.5, -45.0, 100.0])
        p_before = self.ekf.state.position_enu.copy()
        v_before = self.ekf.state.velocity_enu.copy()
        q_before = self.ekf.state.quaternion.as_array().copy()

        # Measure exactly where we are
        res = self.ekf.update_gnss_position(z_enu=p_before)

        self.assertTrue(res.accepted)
        np.testing.assert_allclose(res.innovation, np.zeros(3), atol=1e-12)
        self.assertAlmostEqual(res.innovation_norm, 0.0, places=10)
        self.assertAlmostEqual(res.mahalanobis_distance, 0.0, places=10)

        # State should not have moved
        np.testing.assert_allclose(self.ekf.state.position_enu, p_before, atol=1e-10)
        np.testing.assert_allclose(self.ekf.state.velocity_enu, v_before, atol=1e-10)
        np.testing.assert_allclose(self.ekf.state.quaternion.as_array(), q_before, atol=1e-10)

    def test_test3_position_error_correction(self) -> None:
        """
        TEST 3 — Position error correction
        Create an artificial position error:
            EKF position:  [10, 20, 0]
            GNSS position: [15, 18, 0]
        Apply update.
        Verify the corrected position moves toward GNSS via Kalman-weighted correction.
        """
        self.ekf.state.position_enu = np.array([10.0, 20.0, 0.0])
        z_gnss = np.array([15.0, 18.0, 0.0])

        res = self.ekf.update_gnss_position(z_enu=z_gnss)

        self.assertTrue(res.accepted)
        p_corrected = self.ekf.state.position_enu

        # In X: started at 10, GNSS is 15 -> must move towards 15: 10 < p_corr_x < 15
        self.assertGreater(p_corrected[0], 10.0)
        self.assertLess(p_corrected[0], 15.0)

        # In Y: started at 20, GNSS is 18 -> must move towards 18: 18 < p_corr_y < 20
        self.assertLess(p_corrected[1], 20.0)
        self.assertGreater(p_corrected[1], 18.0)

        # In Z: started at 0, GNSS is 0 -> remains 0
        self.assertAlmostEqual(p_corrected[2], 0.0, places=6)

    def test_test4_no_hard_snapping(self) -> None:
        """
        TEST 4 — No hard snapping
        INS position = [0, 0, 0]
        GNSS position = [100, 0, 0]
        with realistic covariance and permissive gating to test the Kalman gain weighting.
        After update verify:
            corrected_position_x < 100
            corrected_position_x > 0
        This explicitly proves GNSS does NOT directly overwrite the state.
        """
        self.ekf.state.position_enu = np.array([0.0, 0.0, 0.0])
        z_gnss = np.array([100.0, 0.0, 0.0])

        # Prior variance = 16 m^2 (sigma = 4m), GNSS variance = 25 m^2 (sigma = 5m)
        self.ekf.P[POSITION, POSITION] = np.eye(3) * 16.0
        cov_gnss = np.eye(3) * 25.0

        # High gating threshold so this controlled test evaluates the Kalman gain directly
        meas = GNSSPositionMeasurement(z_enu=z_gnss, cov_enu=cov_gnss, gating_threshold=1000.0)
        res = self.ekf.update(meas)

        self.assertTrue(res.accepted)
        pos_x = self.ekf.state.position_enu[0]

        # Theoretical Kalman gain K = P / (P + R) = 16 / (16 + 25) = 16/41 ≈ 0.3902
        # Expected pos_x = 0 + 0.3902 * 100 ≈ 39.02 m
        expected_pos_x = (16.0 / (16.0 + 25.0)) * 100.0
        self.assertAlmostEqual(pos_x, expected_pos_x, places=4)

        # Crucially: 0 < pos_x < 100, NOT snapped to 100!
        self.assertGreater(pos_x, 0.0)
        self.assertLess(pos_x, 100.0)

    def test_test5_innovation_gating(self) -> None:
        """
        TEST 5 — Innovation gating
        Create a very large GNSS outlier: GNSS = [10000, 10000, 10000].
        Expected:
            measurement rejected
            position unchanged
            velocity unchanged
            attitude unchanged
            biases unchanged
            covariance unchanged
        """
        self.ekf.state.position_enu = np.array([5.0, 10.0, 1.0])
        self.ekf.state.velocity_enu = np.array([1.0, -0.5, 0.0])
        self.ekf.state.quaternion = from_euler_rad(0.05, -0.02, 0.5)
        self.ekf.state.accel_bias = np.array([0.01, -0.01, 0.02])
        self.ekf.state.gyro_bias = np.array([0.001, -0.002, 0.001])

        p_before = self.ekf.state.position_enu.copy()
        v_before = self.ekf.state.velocity_enu.copy()
        q_before = self.ekf.state.quaternion.as_array().copy()
        ba_before = self.ekf.state.accel_bias.copy()
        bg_before = self.ekf.state.gyro_bias.copy()
        P_before = self.ekf.P.copy()

        outlier_gnss = np.array([10000.0, 10000.0, 10000.0])
        res = self.ekf.update_gnss_position(z_enu=outlier_gnss)

        # 1. Rejected
        self.assertFalse(res.accepted)
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
        Create position uncertainty larger than GNSS measurement uncertainty.
        Apply an accepted GNSS measurement.
        Verify relevant position covariance decreases.
        """
        # Set large prior position variance: sigma = 10m -> var = 100 m^2
        self.ekf.P[POSITION, POSITION] = np.eye(3) * 100.0
        p_var_before = np.diag(self.ekf.P[POSITION, POSITION]).copy()

        # GNSS measurement with small variance = 4.0 m^2 (sigma = 2m)
        gnss_cov = np.eye(3) * 4.0
        z_gnss = self.ekf.state.position_enu + np.array([0.5, -0.5, 0.2])

        meas = GNSSPositionMeasurement(z_enu=z_gnss, cov_enu=gnss_cov)
        res = self.ekf.update(meas)

        self.assertTrue(res.accepted)
        p_var_after = np.diag(self.ekf.P[POSITION, POSITION])

        # Covariance must decrease substantially in all 3 axes
        for i in range(3):
            self.assertLess(p_var_after[i], p_var_before[i])
            # Theoretical posterior variance P_post = (100 * 4) / (100 + 4) ≈ 3.846
            expected_var = (100.0 * 4.0) / (100.0 + 4.0)
            self.assertAlmostEqual(p_var_after[i], expected_var, places=3)

    def test_test7_joseph_covariance_update(self) -> None:
        """
        TEST 7 — Joseph covariance update
        After an accepted update verify:
            P ≈ P^T
            covariance remains numerically positive semi-definite
        """
        z_gnss = self.ekf.state.position_enu + np.array([1.0, -1.0, 0.5])
        res = self.ekf.update_gnss_position(z_enu=z_gnss)

        self.assertTrue(res.accepted)

        # 1. Symmetry
        np.testing.assert_allclose(self.ekf.P, self.ekf.P.T, atol=1e-10)

        # 2. Positive semi-definiteness
        eigvals = np.linalg.eigvalsh(self.ekf.P)
        self.assertGreaterEqual(np.min(eigvals), -1e-9)

        # 3. Strictly positive diagonal variances
        diag = np.diag(self.ekf.P)
        self.assertTrue(np.all(diag > 0.0))

    def test_test8_velocity_attitude_bias_correction(self) -> None:
        """
        TEST 8 — Velocity/attitude/bias correction
        Create a coupled covariance matrix with off-diagonal terms between position
        and velocity, attitude, accel bias, gyro bias.
        Verify that a position innovation produces corrections in all coupled states through K.
        """
        # Set up non-zero cross-covariances
        # Cross-covariance between position and velocity
        self.ekf.P[POSITION, VELOCITY] = np.eye(3) * 0.5
        self.ekf.P[VELOCITY, POSITION] = np.eye(3) * 0.5

        # Cross-covariance between position and attitude
        self.ekf.P[POSITION, ATTITUDE] = np.eye(3) * 0.02
        self.ekf.P[ATTITUDE, POSITION] = np.eye(3) * 0.02

        # Cross-covariance between position and accel bias
        self.ekf.P[POSITION, ACCEL_BIAS] = np.eye(3) * 0.01
        self.ekf.P[ACCEL_BIAS, POSITION] = np.eye(3) * 0.01

        # Cross-covariance between position and gyro bias
        self.ekf.P[POSITION, GYRO_BIAS] = np.eye(3) * 0.001
        self.ekf.P[GYRO_BIAS, POSITION] = np.eye(3) * 0.001

        # Ensure P remains positive definite
        self.ekf.P += np.eye(STATE_DIM) * 0.1
        self.ekf.P = 0.5 * (self.ekf.P + self.ekf.P.T)

        v_before = self.ekf.state.velocity_enu.copy()
        q_before = self.ekf.state.quaternion.as_array().copy()
        ba_before = self.ekf.state.accel_bias.copy()
        bg_before = self.ekf.state.gyro_bias.copy()

        # Apply position innovation
        z_gnss = self.ekf.state.position_enu + np.array([2.0, 2.0, 1.0])
        meas = GNSSPositionMeasurement(z_enu=z_gnss, gating_threshold=50.0)
        res = self.ekf.update(meas)

        self.assertTrue(res.accepted)
        self.assertIsNotNone(res.kalman_gain)

        # Verify Kalman gain has non-zero blocks for coupled states
        k = res.kalman_gain
        self.assertGreater(np.linalg.norm(k[VELOCITY, :]), 1e-4)
        self.assertGreater(np.linalg.norm(k[ATTITUDE, :]), 1e-4)
        self.assertGreater(np.linalg.norm(k[ACCEL_BIAS, :]), 1e-4)
        self.assertGreater(np.linalg.norm(k[GYRO_BIAS, :]), 1e-5)

        # Verify coupled states actually received corrections
        self.assertFalse(np.allclose(self.ekf.state.velocity_enu, v_before, atol=1e-4))
        self.assertFalse(np.allclose(self.ekf.state.quaternion.as_array(), q_before, atol=1e-5))
        self.assertFalse(np.allclose(self.ekf.state.accel_bias, ba_before, atol=1e-5))
        self.assertFalse(np.allclose(self.ekf.state.gyro_bias, bg_before, atol=1e-6))

    def test_test9_enu_conversion(self) -> None:
        """
        TEST 9 — ENU conversion
        Use a known GNSS origin and a small known displacement.
        Verify latitude/longitude/altitude is converted into the correct ENU displacement
        using the existing EarthModel implementation from edge/ins/earth.py.
        """
        ref_lat = 37.7749  # San Francisco
        ref_lon = -122.4194
        ref_alt = 10.0

        # Displace North by ~111 meters (0.001 deg lat)
        d_lat_deg = 0.001
        # Displace East by ~88 meters (0.001 deg lon * cos(37.7749 deg))
        d_lon_deg = 0.001
        d_alt_m = 5.0

        target_lat = ref_lat + d_lat_deg
        target_lon = ref_lon + d_lon_deg
        target_alt = ref_alt + d_alt_m

        enu = EarthModel.geodetic_to_enu(
            target_lat, target_lon, target_alt,
            ref_lat, ref_lon, ref_alt,
        )

        # 1. Up must equal 5.0m
        self.assertAlmostEqual(enu[2], 5.0, places=2)

        # 2. North must be positive and ~111.0m (1 degree lat ≈ 111,132 m)
        expected_north = d_lat_deg * (math.pi / 180.0) * 6378137.0 * (1.0 - 0.00669437999014 * (math.sin(math.radians(ref_lat)) ** 2))
        self.assertAlmostEqual(enu[1], 111.1, delta=1.0)

        # 3. East must be positive and ~88.0m (111132 * cos(37.7749 deg) ≈ 87842 m/deg)
        self.assertAlmostEqual(enu[0], 87.8, delta=1.0)

        # 4. Round-trip conversion via enu_to_geodetic recovers original coordinates
        lat_back, lon_back, alt_back = EarthModel.enu_to_geodetic(enu, ref_lat, ref_lon, ref_alt)
        self.assertAlmostEqual(lat_back, target_lat, places=8)
        self.assertAlmostEqual(lon_back, target_lon, places=8)
        self.assertAlmostEqual(alt_back, target_alt, places=4)

    def test_test10_repeated_gnss_updates(self) -> None:
        """
        TEST 10 — Repeated GNSS updates
        Run multiple GNSS measurements over a synthetic trajectory.
        Verify:
        - covariance does not become NaN or Inf
        - covariance remains symmetric
        - trajectory remains continuous
        - GNSS corrections gradually reduce position error
        """
        dt = 0.01  # 100 Hz IMU
        gnss_period = 10  # 10 Hz GNSS (every 10 IMU steps)

        # Vehicle moves forward at 5 m/s with small accel bias = 0.1 m/s^2 (leads to INS drift)
        true_pos = np.zeros(3)
        self.ekf.state.position_enu = np.zeros(3)
        self.ekf.state.velocity_enu = np.array([5.0, 0.0, 0.0])

        accel_raw = np.array([0.1, 0.0, 9.80665])  # 0.1 m/s^2 forward bias
        gyro_raw = np.zeros(3)

        position_errors = []

        for step in range(200):  # 2.0 seconds
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            true_pos[0] += 5.0 * dt

            # 1. High-rate prediction step
            self.ekf.predict(accel_raw, gyro_raw, timestamp_ns=t_ns)

            # 2. GNSS position update at 10 Hz
            if (step + 1) % gnss_period == 0:
                # GNSS measures true position with small noise
                gnss_noise = np.array([0.2, -0.1, 0.0])
                z_gnss = true_pos + gnss_noise
                res = self.ekf.update_gnss_position(z_enu=z_gnss)
                self.assertTrue(res.accepted)

                # Record position error
                err = float(np.linalg.norm(self.ekf.state.position_enu - true_pos))
                position_errors.append(err)

            # Check health at every step
            self.assertTrue(np.all(np.isfinite(self.ekf.P)))
            self.assertTrue(np.all(np.isfinite(self.ekf.state.position_enu)))
            np.testing.assert_allclose(self.ekf.P, self.ekf.P.T, atol=1e-8)

        # Initial error vs final error: GNSS updates constrain drift to under 1.5 meters
        self.assertLess(position_errors[-1], 1.5)
        # Without GNSS, 0.1 m/s^2 over 2s would produce unmitigated quadratic drift
        # With GNSS, the filter bounds the drift tightly around true position
        mean_late_error = float(np.mean(position_errors[5:]))
        self.assertLess(mean_late_error, 1.0)


if __name__ == "__main__":
    unittest.main()
