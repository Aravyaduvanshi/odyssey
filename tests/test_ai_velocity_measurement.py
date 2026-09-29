import math
import unittest
import numpy as np

from edge.coordinate.quaternion import Quaternion, from_rotation_vector, from_euler_rad
from edge.fusion.ekf import ErrorStateEKF
from edge.fusion.error_state import STATE_DIM, VELOCITY, ATTITUDE
from edge.fusion.measurements import AIVelocityMeasurement
from edge.ins.state import INSState
from edge.models.velocity.preprocessing import (
    VelocityTCNPreprocessor, VELOCITY_FEATURES, TARGET_MEAN, TARGET_STD,
)
from edge.sensors.buffer import RollingIMUBuffer
from edge.sensors.models import SensorSample


class TestAIVelocityMeasurement(unittest.TestCase):
    """Rigorous validation of Velocity TCN as an EKF forward-velocity measurement update (Step 6)."""

    def setUp(self):
        self.state = INSState(
            position_enu=np.array([0.0, 0.0, 0.0], dtype=float),
            velocity_enu=np.array([10.0, 0.0, 0.0], dtype=float),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3, dtype=float),
            gyro_bias=np.zeros(3, dtype=float),
            timestamp_ns=1_000_000_000,
        )

    # -------------------------------------------------------------------------
    # TEST 1 — Measurement dimensions
    # -------------------------------------------------------------------------
    def test_01_measurement_dimensions(self):
        """Verify H.shape == (1, 15) and R.shape == (1, 1)."""
        meas = AIVelocityMeasurement(v_forward_mps=10.0, variance_mps2=0.25)
        innov, H, R = meas.evaluate(self.state)

        self.assertEqual(H.shape, (1, STATE_DIM))
        self.assertEqual(R.shape, (1, 1))
        self.assertEqual(innov.shape, (1,))
        self.assertEqual(H.shape[1], 15)
        self.assertEqual(R[0, 0], 0.25)

    # -------------------------------------------------------------------------
    # TEST 2 — Vehicle-frame velocity
    # -------------------------------------------------------------------------
    def test_02_vehicle_frame_velocity(self):
        """
        Use a known orientation (e.g. 90 deg yaw: vehicle facing North).
        v_nav = [0, 15, 0] (moving North).
        Verify v_vehicle = R_vehicle_nav @ v_nav and v_forward = v_vehicle[0] == 15 m/s.
        """
        # Yaw = 90 deg (+pi/2 around Up):
        # East (nav X) -> Vehicle -Y (right)
        # North (nav Y) -> Vehicle +X (forward)
        # Up (nav Z) -> Vehicle +Z (up)
        q_yaw90 = from_euler_rad(roll=0.0, pitch=0.0, yaw=math.pi / 2.0)
        state_turned = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.array([0.0, 15.0, 0.0]),  # Moving North in ENU
            quaternion=q_yaw90,
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )

        r_n_v = state_turned.rotation_matrix
        r_v_n = r_n_v.T
        v_veh = r_v_n @ state_turned.velocity_enu
        v_forward = v_veh[0]

        np.testing.assert_allclose(v_veh, [15.0, 0.0, 0.0], atol=1e-5)
        self.assertAlmostEqual(v_forward, 15.0, places=5)

        meas = AIVelocityMeasurement(v_forward_mps=15.0, variance_mps2=0.2)
        innov, H, R = meas.evaluate(state_turned)
        self.assertAlmostEqual(float(innov[0]), 0.0, places=5)

    # -------------------------------------------------------------------------
    # TEST 3 — Zero innovation
    # -------------------------------------------------------------------------
    def test_03_zero_innovation(self):
        """Set TCN velocity = predicted forward velocity -> innovation ≈ 0."""
        # Vehicle moving at 12.34 m/s in ENU East with identity quaternion (X_veh = East)
        self.state.velocity_enu = np.array([12.34, 0.0, 0.0])
        meas = AIVelocityMeasurement(v_forward_mps=12.34, variance_mps2=0.5)
        innov, _, _ = meas.evaluate(self.state)

        self.assertAlmostEqual(float(innov[0]), 0.0, places=7)

    # -------------------------------------------------------------------------
    # TEST 4 — Forward velocity correction
    # -------------------------------------------------------------------------
    def test_04_forward_velocity_correction(self):
        """
        predicted forward velocity = 10 m/s
        AI velocity = 12 m/s
        Apply update -> verify EKF forward velocity moves toward 12 m/s.
        """
        ekf = ErrorStateEKF()
        ekf.state = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.array([10.0, 0.0, 0.0]),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        # Set realistic velocity prior variance
        ekf.P[VELOCITY, VELOCITY] = np.diag([1.0, 1.0, 1.0])

        res = ekf.update_ai_velocity(v_forward_mps=12.0, variance_mps2=0.5)
        self.assertTrue(res.accepted)

        v_corr = ekf.state.velocity_enu[0]
        # Should have moved from 10.0 toward 12.0 (strictly > 10.0 and < 12.0)
        self.assertGreater(v_corr, 10.0)
        self.assertLess(v_corr, 12.0)
        self.assertAlmostEqual(v_corr, 10.0 + (1.0 / (1.0 + 0.5)) * 2.0, places=3)

    # -------------------------------------------------------------------------
    # TEST 5 — No hard overwrite
    # -------------------------------------------------------------------------
    def test_05_no_hard_overwrite(self):
        """
        predicted forward velocity = 0 m/s
        AI velocity = 30 m/s
        Verify corrected forward velocity is between 0 and 30 m/s (not overwritten to 30).
        """
        ekf = ErrorStateEKF()
        ekf.state = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.zeros(3),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        ekf.P[VELOCITY, VELOCITY] = np.diag([2.0, 2.0, 2.0])

        # Use relaxed gating threshold for test purposes to inspect fusion behavior
        res = ekf.update_ai_velocity(v_forward_mps=30.0, variance_mps2=1.0, gating_threshold=1000.0)
        self.assertTrue(res.accepted)

        v_corr = ekf.state.velocity_enu[0]
        self.assertGreater(v_corr, 0.0)
        self.assertLess(v_corr, 30.0)
        self.assertNotEqual(v_corr, 30.0)

    # -------------------------------------------------------------------------
    # TEST 6 — AI outlier rejection
    # -------------------------------------------------------------------------
    def test_06_ai_outlier_rejection(self):
        """
        Provide an obviously inconsistent AI velocity.
        Verify measurement is rejected, nominal state unchanged, covariance unchanged.
        """
        ekf = ErrorStateEKF()
        ekf.state = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.array([5.0, 0.0, 0.0]),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        P_before = ekf.P.copy()
        state_before_vel = ekf.state.velocity_enu.copy()

        # Inconsistent velocity: 45 m/s when predicted is 5 m/s with small variance 0.1
        # Innovation = 40 m/s, S ≈ 0.35, d^2 ≈ 4500 >> 10.83 threshold
        res = ekf.update_ai_velocity(v_forward_mps=45.0, variance_mps2=0.1)

        self.assertFalse(res.accepted)
        self.assertIn("Outlier rejected", res.rejection_reason)
        np.testing.assert_array_equal(ekf.state.velocity_enu, state_before_vel)
        np.testing.assert_array_equal(ekf.P, P_before)

    # -------------------------------------------------------------------------
    # TEST 7 — Uncertainty effect
    # -------------------------------------------------------------------------
    def test_07_uncertainty_effect(self):
        """
        Case A: small AI uncertainty -> strong correction
        Case B: large AI uncertainty -> weaker correction
        """
        # Case A: R = 0.1
        ekf_a = ErrorStateEKF()
        ekf_a.state.velocity_enu = np.array([10.0, 0.0, 0.0])
        ekf_a.P[VELOCITY, VELOCITY] = np.diag([1.0, 1.0, 1.0])
        ekf_a.update_ai_velocity(v_forward_mps=13.0, variance_mps2=0.1)
        correction_a = ekf_a.state.velocity_enu[0] - 10.0

        # Case B: R = 5.0
        ekf_b = ErrorStateEKF()
        ekf_b.state.velocity_enu = np.array([10.0, 0.0, 0.0])
        ekf_b.P[VELOCITY, VELOCITY] = np.diag([1.0, 1.0, 1.0])
        ekf_b.update_ai_velocity(v_forward_mps=13.0, variance_mps2=5.0)
        correction_b = ekf_b.state.velocity_enu[0] - 10.0

        self.assertGreater(correction_a, correction_b)
        self.assertGreater(correction_b, 0.0)

    # -------------------------------------------------------------------------
    # TEST 8 — Numerical uncertainty
    # -------------------------------------------------------------------------
    def test_08_numerical_uncertainty_safety(self):
        """
        Verify invalid model uncertainty (NaN, Inf, negative, zero)
        does not crash EKF and uses fallback uncertainty.
        """
        for bad_var in [float("nan"), float("inf"), -1.5, 0.0, None]:
            meas = AIVelocityMeasurement(
                v_forward_mps=10.0,
                variance_mps2=bad_var,
                fallback_variance_mps2=0.5,
            )
            self.assertTrue(meas.used_fallback or meas.was_clamped)
            self.assertTrue(np.isfinite(meas.r[0, 0]))
            self.assertGreater(meas.r[0, 0], 0.0)

            ekf = ErrorStateEKF()
            ekf.state.velocity_enu = np.array([10.0, 0.0, 0.0])
            res = ekf.update_ai_velocity(v_forward_mps=10.5, variance_mps2=bad_var)
            self.assertTrue(res.accepted)
            self.assertTrue(np.all(np.isfinite(ekf.P)))
            self.assertTrue(np.all(np.isfinite(ekf.state.velocity_enu)))

    # -------------------------------------------------------------------------
    # TEST 9 — Analytical H validation via finite differences
    # -------------------------------------------------------------------------
    def test_09_analytical_attitude_jacobian_validation(self):
        """
        Rigorously test attitude Jacobian against numerical finite differences
        using the project's right-multiplicative body-frame convention:
        q_perturbed = q_nominal ⊗ Quaternion.from_rotation_vector(delta_theta)
        Compare analytical H[0, ATT_IDX] with numerical derivative.
        """
        # Test across arbitrary 3D orientations and non-trivial 3D velocities
        test_cases = [
            (from_euler_rad(0.1, -0.2, 0.7), np.array([12.0, -4.0, 1.5])),
            (from_euler_rad(-0.3, 0.4, -1.2), np.array([-8.0, 14.0, -2.5])),
            (from_euler_rad(0.0, 0.0, 0.0), np.array([10.0, 5.0, -3.0])),
            (from_euler_rad(0.5, 0.5, 1.0), np.array([20.0, 0.0, 0.0])),
        ]

        eps = 1e-7

        for q_nom, v_nav in test_cases:
            state = INSState(
                position_enu=np.zeros(3),
                velocity_enu=v_nav,
                quaternion=q_nom,
                accel_bias=np.zeros(3),
                gyro_bias=np.zeros(3),
                timestamp_ns=1_000_000_000,
            )

            meas = AIVelocityMeasurement(v_forward_mps=0.0, variance_mps2=0.5)
            _, H_analytical, _ = meas.evaluate(state)
            H_att_analytical = H_analytical[0, ATTITUDE]

            # Numerical derivative w.r.t delta_theta (3x1)
            H_att_numerical = np.zeros(3)

            for i in range(3):
                delta_vec_pos = np.zeros(3)
                delta_vec_pos[i] = eps
                q_pert_pos = (q_nom * from_rotation_vector(delta_vec_pos)).normalized()
                r_v_n_pos = q_pert_pos.rotation_matrix.T
                h_pos = (r_v_n_pos @ v_nav)[0]

                delta_vec_neg = np.zeros(3)
                delta_vec_neg[i] = -eps
                q_pert_neg = (q_nom * from_rotation_vector(delta_vec_neg)).normalized()
                r_v_n_neg = q_pert_neg.rotation_matrix.T
                h_neg = (r_v_n_neg @ v_nav)[0]

                H_att_numerical[i] = (h_pos - h_neg) / (2.0 * eps)

            # Assert analytical and numerical Jacobians match with tight tolerance
            np.testing.assert_allclose(
                H_att_analytical,
                H_att_numerical,
                atol=1e-5,
                rtol=1e-5,
                err_msg="Analytical attitude Jacobian mismatch with right-multiplicative perturbation!",
            )

    # -------------------------------------------------------------------------
    # TEST 10 — Normalization parameters
    # -------------------------------------------------------------------------
    def test_10_normalization_parameters(self):
        """Verify the actual TCN preprocessor uses the exact stored means and stds."""
        prep = VelocityTCNPreprocessor()

        expected_params = {
            "ACC_VEH_X": (-0.038078, 1.156829),
            "ACC_VEH_Y": (0.020610, 1.104612),
            "ACC_VEH_Z": (-0.018169, 0.628754),
            "GYRO_Yaw_corrected": (0.000854, 0.169824),
            "GYRO_Pitch_corrected": (-0.007635, 0.144094),
            "GYRO_Roll_corrected": (0.002120, 0.080973),
        }

        for i, feat in enumerate(VELOCITY_FEATURES):
            exp_mean, exp_std = expected_params[feat]
            self.assertAlmostEqual(prep.feature_means[i], exp_mean, places=5, msg=f"Mean mismatch for {feat}")
            self.assertAlmostEqual(prep.feature_stds[i], exp_std, places=5, msg=f"Std mismatch for {feat}")

        # Verify normalization transform on known vector
        test_window = np.zeros((100, 6), dtype=np.float32)
        norm_window = prep.transform(test_window)
        for i in range(6):
            expected_norm_val = (0.0 - prep.feature_means[i]) / prep.feature_stds[i]
            np.testing.assert_allclose(norm_window[0, :, i], expected_norm_val, atol=1e-5)

    # -------------------------------------------------------------------------
    # TEST 11 — Target denormalization
    # -------------------------------------------------------------------------
    def test_11_target_denormalization(self):
        """Verify conversion using exact target_mean and target_std."""
        prep = VelocityTCNPreprocessor()

        self.assertAlmostEqual(prep.target_mean, 26.57159258, places=4)
        self.assertAlmostEqual(prep.target_std, 20.84706572, places=4)

        # Test velocity denormalization: vel = vel_norm * target_std + target_mean
        vel_norm = 0.5
        exp_vel_mps = vel_norm * TARGET_STD + TARGET_MEAN
        self.assertAlmostEqual(prep.denormalize_velocity(vel_norm), exp_vel_mps, places=5)

        # Test uncertainty denormalization:
        # sigma_mps = exp(0.5 * log_var) * target_std
        # variance_mps2 = exp(log_var) * target_std^2
        log_var = -1.0
        sigma_mps, var_mps2 = prep.denormalize_uncertainty(log_var)
        exp_sigma = math.exp(0.5 * log_var) * TARGET_STD
        exp_var = math.exp(log_var) * (TARGET_STD ** 2)

        self.assertAlmostEqual(sigma_mps, exp_sigma, places=5)
        self.assertAlmostEqual(var_mps2, exp_var, places=4)

    # -------------------------------------------------------------------------
    # TEST 12 — Model warmup
    # -------------------------------------------------------------------------
    def test_12_model_warmup(self):
        """Verify fewer than 100 samples in RollingIMUBuffer does not trigger inference."""
        buf = RollingIMUBuffer(window_size=100, resample_rate_hz=10.0, velocity_stride=10)

        from edge.sensors.models import SensorSource
        # Add 99 samples at 100 ms intervals
        ts = 1_000_000_000
        for i in range(99):
            s = SensorSample(
                timestamp_ns=ts + i * 100_000_000,
                ax=0.0, ay=0.0, az=9.81,
                gx=0.0, gy=0.0, gz=0.0,
                sensor_source=SensorSource.SYNTHETIC,
            )
            buf.add_sample(s)

        self.assertFalse(buf.is_ready)
        self.assertFalse(buf.should_run_velocity())
        self.assertEqual(buf.sample_count, 99)

        # 100th sample completes warmup
        s100 = SensorSample(
            timestamp_ns=ts + 99 * 100_000_000,
            ax=0.0, ay=0.0, az=9.81,
            gx=0.0, gy=0.0, gz=0.0,
            sensor_source=SensorSource.SYNTHETIC,
        )
        buf.add_sample(s100)

        self.assertTrue(buf.is_ready)
        self.assertEqual(buf.sample_count, 100)
        self.assertTrue(buf.should_run_velocity())

    # -------------------------------------------------------------------------
    # TEST 13 — Feature ordering
    # -------------------------------------------------------------------------
    def test_13_feature_ordering(self):
        """
        Verify the model receives exactly:
        [ACC_VEH_X, ACC_VEH_Y, ACC_VEH_Z, GYRO_Yaw_corrected, GYRO_Pitch_corrected, GYRO_Roll_corrected]
        """
        expected_order = [
            "ACC_VEH_X",
            "ACC_VEH_Y",
            "ACC_VEH_Z",
            "GYRO_Yaw_corrected",
            "GYRO_Pitch_corrected",
            "GYRO_Roll_corrected",
        ]
        self.assertEqual(VELOCITY_FEATURES, expected_order)

        # Check that buffer extracts features in exact order
        from edge.coordinate.frames import VehicleAlignedIMU, VehicleFrameTransform
        veh_imu = VehicleAlignedIMU(
            acc_veh_x=1.1,
            acc_veh_y=2.2,
            acc_veh_z=3.3,
            gyro_yaw_corrected=4.4,
            gyro_pitch_corrected=5.5,
            gyro_roll_corrected=6.6,
        )
        feat = veh_imu.as_features()
        np.testing.assert_allclose(feat, [1.1, 2.2, 3.3, 4.4, 5.5, 6.6])


if __name__ == "__main__":
    unittest.main()
