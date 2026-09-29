import math
import unittest
import numpy as np

from edge.coordinate.quaternion import Quaternion, from_euler_rad
from edge.fusion.covariance import CovariancePropagator
from edge.fusion.ekf import ErrorStateEKF
from edge.fusion.error_state import (
    STATE_DIM, POSITION, VELOCITY, ATTITUDE, ACCEL_BIAS, GYRO_BIAS,
    skew,
)
from edge.ins.mechanization import StrapdownMechanization
from edge.ins.state import INSState


class EKFPredictionTests(unittest.TestCase):
    """
    Validation tests for 15-State Error-State EKF Prediction Layer (Step 3).
    """

    def setUp(self) -> None:
        self.ekf = ErrorStateEKF()
        self.propagator = self.ekf.propagator

    def test_test1_error_state_dimensions(self) -> None:
        """
        TEST 1 — Error-state dimensions
        Verify dimensions of P, continuous F, and discrete Qd are all 15x15.
        """
        r_nav_veh = np.eye(3)
        f_veh = np.array([0.0, 0.0, 9.80665])
        omega_veh = np.zeros(3)
        dt = 0.01

        f_matrix = self.propagator.compute_continuous_f(r_nav_veh, f_veh, omega_veh)
        q_d = self.propagator.compute_discrete_q(r_nav_veh, dt)

        self.assertEqual(self.ekf.P.shape, (15, 15))
        self.assertEqual(f_matrix.shape, (15, 15))
        self.assertEqual(q_d.shape, (15, 15))

    def test_test2_initial_covariance(self) -> None:
        """
        TEST 2 — Initial covariance
        Verify P matches the configured initial variances in configs/ekf.yaml.
        """
        init_vars = self.ekf.config.get("initial_variances", {})
        expected_pos = np.asarray(init_vars.get("position", [1.0, 1.0, 4.0]))
        expected_vel = np.asarray(init_vars.get("velocity", [0.25, 0.25, 0.25]))
        expected_att = np.asarray(init_vars.get("attitude", [0.01, 0.01, 0.04]))
        expected_acc_b = np.asarray(init_vars.get("accel_bias", [0.0025, 0.0025, 0.0025]))
        expected_gyro_b = np.asarray(init_vars.get("gyro_bias", [0.000025, 0.000025, 0.000025]))

        p = self.ekf.P

        np.testing.assert_allclose(np.diag(p[POSITION, POSITION]), expected_pos, rtol=1e-5)
        np.testing.assert_allclose(np.diag(p[VELOCITY, VELOCITY]), expected_vel, rtol=1e-5)
        np.testing.assert_allclose(np.diag(p[ATTITUDE, ATTITUDE]), expected_att, rtol=1e-5)
        np.testing.assert_allclose(np.diag(p[ACCEL_BIAS, ACCEL_BIAS]), expected_acc_b, rtol=1e-5)
        np.testing.assert_allclose(np.diag(p[GYRO_BIAS, GYRO_BIAS]), expected_gyro_b, rtol=1e-5)

        # Off-diagonal blocks should be initially zero
        mask_diag = np.zeros((15, 15), dtype=bool)
        for s in [POSITION, VELOCITY, ATTITUDE, ACCEL_BIAS, GYRO_BIAS]:
            mask_diag[s, s] = True
        off_diag = p[~mask_diag]
        np.testing.assert_allclose(off_diag, np.zeros_like(off_diag), atol=1e-12)

    def test_test3_skew_matrix(self) -> None:
        """
        TEST 3 — Skew matrix
        For random 3D vectors a and b, verify:
            skew(a) @ b == np.cross(a, b)
        within numerical tolerance.
        """
        np.random.seed(42)
        for _ in range(50):
            a = np.random.randn(3)
            b = np.random.randn(3)
            expected_cross = np.cross(a, b)
            computed_skew = skew(a) @ b
            np.testing.assert_allclose(computed_skew, expected_cross, atol=1e-12)

    def test_test4_f_matrix_structure(self) -> None:
        """
        TEST 4 — F matrix structure
        Use a simple known orientation (R = I) and verify each expected F block:
        F =
        [ 0     I       0                  0              0
          0     0   -R[f]x             -R                0
          0     0   -[ω]x                0              -I
          0     0     0                  0              0
          0     0     0                  0              0 ]
        """
        r_nav_veh = np.eye(3)
        f_veh = np.array([1.5, -2.0, 9.80665])
        omega_veh = np.array([0.05, -0.1, 0.2])

        f_matrix = self.propagator.compute_continuous_f(r_nav_veh, f_veh, omega_veh)

        # Block (0, 1): Position by velocity
        np.testing.assert_allclose(f_matrix[POSITION, VELOCITY], np.eye(3), atol=1e-12)

        # Block (1, 2): Velocity by attitude error = -R @ skew(f_veh)
        expected_block_1_2 = -r_nav_veh @ skew(f_veh)
        np.testing.assert_allclose(f_matrix[VELOCITY, ATTITUDE], expected_block_1_2, atol=1e-12)

        # Block (1, 3): Velocity by accel bias = -R
        np.testing.assert_allclose(f_matrix[VELOCITY, ACCEL_BIAS], -r_nav_veh, atol=1e-12)

        # Block (2, 2): Attitude by attitude error = -skew(omega_veh)
        expected_block_2_2 = -skew(omega_veh)
        np.testing.assert_allclose(f_matrix[ATTITUDE, ATTITUDE], expected_block_2_2, atol=1e-12)

        # Block (2, 4): Attitude by gyro bias = -I (body-frame attitude and gyro bias)
        np.testing.assert_allclose(f_matrix[ATTITUDE, GYRO_BIAS], -np.eye(3), atol=1e-12)

        # Blocks (0, 0), (0, 2), (0, 3), (0, 4) should be 0
        np.testing.assert_allclose(f_matrix[POSITION, POSITION], np.zeros((3, 3)), atol=1e-12)
        np.testing.assert_allclose(f_matrix[POSITION, ATTITUDE], np.zeros((3, 3)), atol=1e-12)
        np.testing.assert_allclose(f_matrix[POSITION, ACCEL_BIAS], np.zeros((3, 3)), atol=1e-12)
        np.testing.assert_allclose(f_matrix[POSITION, GYRO_BIAS], np.zeros((3, 3)), atol=1e-12)

        # Rows 3 and 4 (bias random walks) should be completely 0
        np.testing.assert_allclose(f_matrix[ACCEL_BIAS, :], np.zeros((3, 15)), atol=1e-12)
        np.testing.assert_allclose(f_matrix[GYRO_BIAS, :], np.zeros((3, 15)), atol=1e-12)

    def test_test5_stationary_covariance_propagation(self) -> None:
        """
        TEST 5 — Stationary covariance propagation
        Stationary vehicle: gyro = [0, 0, 0], accel = [0, 0, +g].
        Run multiple prediction steps.
        Verify:
        - no NaN, no Inf
        - P remains symmetric
        - P remains finite
        - Covariance remains physically valid (positive semi-definite)
        """
        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.zeros(3)
        dt = 0.01

        for step in range(200):
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            self.ekf.predict(accel, gyro, timestamp_ns=t_ns)

            # Check finite
            self.assertTrue(np.all(np.isfinite(self.ekf.P)))
            # Check symmetric
            np.testing.assert_allclose(self.ekf.P, self.ekf.P.T, atol=1e-9)
            # Check positive semi-definiteness
            eigvals = np.linalg.eigvalsh(self.ekf.P)
            self.assertGreaterEqual(np.min(eigvals), -1e-9)

    def test_test6_constant_acceleration_prediction(self) -> None:
        """
        TEST 6 — Constant acceleration prediction
        Use the same controlled scenario from the INS tests (forward accel = 1.0 m/s^2, 10 seconds).
        Verify that the nominal INS state continues to produce:
            v ≈ 10 m/s after 10 seconds
            p ≈ 50 m after 10 seconds
        The EKF must NOT change the nominal state during prediction.
        """
        state = INSState.zero(timestamp_ns=1_000_000_000)
        self.ekf.state = state.copy()

        accel = np.array([1.0, 0.0, 9.80665])
        gyro = np.zeros(3)
        dt = 0.01
        total_steps = 1000  # 10.0 seconds

        for step in range(total_steps):
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            self.ekf.predict(accel, gyro, timestamp_ns=t_ns)

        # Verify nominal INS state is preserved exactly
        self.assertAlmostEqual(self.ekf.state.velocity_enu[0], 10.0, places=4)
        self.assertAlmostEqual(self.ekf.state.position_enu[0], 50.0, places=3)
        self.assertAlmostEqual(self.ekf.state.velocity_enu[1], 0.0, places=6)
        self.assertAlmostEqual(self.ekf.state.velocity_enu[2], 0.0, places=6)

    def test_test7_covariance_growth(self) -> None:
        """
        TEST 7 — Covariance growth
        Run prediction without measurement updates.
        Verify uncertainty increases appropriately in position and velocity components.
        """
        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.zeros(3)
        dt = 0.01

        p0_pos_trace = np.trace(self.ekf.P[POSITION, POSITION])
        p0_vel_trace = np.trace(self.ekf.P[VELOCITY, VELOCITY])

        for step in range(500):  # 5.0 seconds
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            self.ekf.predict(accel, gyro, timestamp_ns=t_ns)

        p_final_pos_trace = np.trace(self.ekf.P[POSITION, POSITION])
        p_final_vel_trace = np.trace(self.ekf.P[VELOCITY, VELOCITY])

        # Covariance in position and velocity MUST grow over open-loop integration
        self.assertGreater(p_final_pos_trace, p0_pos_trace)
        self.assertGreater(p_final_vel_trace, p0_vel_trace)

    def test_test8_zero_process_noise(self) -> None:
        """
        TEST 8 — Zero process noise
        Temporarily configure process noise to zero.
        Verify covariance follows deterministic transition:
            P_new = Phi @ P @ Phi^T
        """
        zero_noise_propagator = CovariancePropagator(
            accel_noise_density=0.0,
            gyro_noise_density=0.0,
            accel_bias_random_walk=0.0,
            gyro_bias_random_walk=0.0,
        )
        ekf = ErrorStateEKF(
            initial_state=INSState.zero(timestamp_ns=1_000_000_000),
            propagator=zero_noise_propagator,
        )
        r_nav_veh = ekf.state.rotation_matrix
        f_veh = np.array([0.0, 0.0, 9.80665])
        omega_veh = np.zeros(3)
        dt = 0.01

        f_cont = zero_noise_propagator.compute_continuous_f(r_nav_veh, f_veh, omega_veh)
        phi = zero_noise_propagator.compute_discrete_transition(f_cont, dt)
        expected_p_new = phi @ ekf.P @ phi.T
        expected_p_new = 0.5 * (expected_p_new + expected_p_new.T)

        ekf.predict(f_veh, omega_veh, dt=dt)

        np.testing.assert_allclose(ekf.P, expected_p_new, atol=1e-10)

    def test_test9_orientation_dependence(self) -> None:
        """
        TEST 9 — Orientation dependence
        Repeat covariance prediction with two different vehicle attitudes (Yaw = 0 vs Yaw = 90 deg).
        Verify F changes consistently with R_nav_vehicle.
        """
        f_veh = np.array([2.0, 0.0, 9.80665])
        omega_veh = np.zeros(3)

        # Orientation 1: Yaw = 0 (vehicle X = East)
        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
        r1 = q1.rotation_matrix
        f_mat1 = self.propagator.compute_continuous_f(r1, f_veh, omega_veh)

        # Orientation 2: Yaw = 90 deg (vehicle X = North)
        q2 = from_euler_rad(0.0, 0.0, math.pi / 2.0)
        r2 = q2.rotation_matrix
        f_mat2 = self.propagator.compute_continuous_f(r2, f_veh, omega_veh)

        # The rotation matrices R1 and R2 must differ
        self.assertFalse(np.allclose(r1, r2))

        # Block (1, 2) is -R @ skew(f_veh) and must differ between orientations
        block1_1_2 = f_mat1[VELOCITY, ATTITUDE]
        block2_1_2 = f_mat2[VELOCITY, ATTITUDE]
        self.assertFalse(np.allclose(block1_1_2, block2_1_2))

        # Check exact transformation: block2_1_2 should equal -R2 @ skew(f_veh)
        np.testing.assert_allclose(block2_1_2, -r2 @ skew(f_veh), atol=1e-12)

    def test_test10_long_run_numerical_stability(self) -> None:
        """
        TEST 10 — Long-run numerical stability
        Run 3,000 prediction steps (30 seconds of multi-axis dynamic motion).
        Verify:
        - P is finite
        - P is symmetric
        - Quaternion is finite and normalized
        - No exploding numerical values or negative diagonal variances
        """
        accel = np.array([0.5, -0.2, 9.80665])
        gyro = np.array([0.05, -0.02, 0.1])
        dt = 0.01

        for step in range(3000):
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            self.ekf.predict(accel, gyro, timestamp_ns=t_ns)

        # 1. P is finite
        self.assertTrue(np.all(np.isfinite(self.ekf.P)))
        # 2. P is symmetric
        np.testing.assert_allclose(self.ekf.P, self.ekf.P.T, atol=1e-8)
        # 3. Quaternion is finite and normalized
        q_norm = float(np.linalg.norm(self.ekf.state.quaternion.as_array()))
        self.assertAlmostEqual(q_norm, 1.0, places=9)
        # 4. No negative variances on diagonal
        diag = np.diag(self.ekf.P)
        self.assertTrue(np.all(diag > 0.0))
        # 5. Positive semi-definiteness
        eigvals = np.linalg.eigvalsh(self.ekf.P)
        self.assertGreaterEqual(np.min(eigvals), -1e-8)

    def test_test11_attitude_error_convention_right_multiplication(self) -> None:
        """
        TEST 11 (TEST A & TEST B) — Verify right-multiplicative body-frame attitude error convention.

        TEST A: Small body-frame roll error with identity nominal attitude.
        Verify:
            q_corrected = q_nominal * delta_q
        matches the expected body-frame error injection.

        TEST B: Non-identity nominal attitude (Yaw = 90 deg, vehicle X points North).
        Inject small body-frame roll error delta_theta = [0.05, 0, 0] rad (roll around vehicle X axis).
        Verify that q_nominal * delta_q produces the expected navigation orientation:
          - Vehicle X (forward) still points North in navigation frame (roll does not rotate forward axis)
          - Left multiplication (delta_q * q_nominal) would rotate about ENU East axis, causing pitch tilt.
        Assert that q_corrected equals q_nominal * delta_q and does NOT equal delta_q * q_nominal.
        """
        # --- TEST A: Identity nominal attitude ---
        q_nominal_a = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.ekf.state.quaternion = q_nominal_a
        delta_theta_a = np.array([0.05, 0.0, 0.0])
        delta_x_a = np.zeros(STATE_DIM)
        delta_x_a[ATTITUDE] = delta_theta_a
        delta_q_a = Quaternion.from_rotation_vector(delta_theta_a)

        self.ekf._inject_error_state(delta_x_a)

        expected_q_a = (q_nominal_a * delta_q_a).normalized()
        np.testing.assert_allclose(
            self.ekf.state.quaternion.as_array(),
            expected_q_a.as_array(),
            atol=1e-12,
        )

        # --- TEST B: Non-identity nominal attitude (Yaw = 90 deg) ---
        # Vehicle forward (X_v) points North: R_nominal @ [1, 0, 0] == [0, 1, 0]
        q_nominal_b = from_euler_rad(0.0, 0.0, math.pi / 2.0)
        self.ekf.state.quaternion = q_nominal_b
        delta_theta_b = np.array([0.05, 0.0, 0.0])  # Body-frame roll error
        delta_x_b = np.zeros(STATE_DIM)
        delta_x_b[ATTITUDE] = delta_theta_b
        delta_q_b = Quaternion.from_rotation_vector(delta_theta_b)

        self.ekf._inject_error_state(delta_x_b)

        # Expected right-multiplication: q_nominal * delta_q
        q_right = (q_nominal_b * delta_q_b).normalized()
        # Erroneous left-multiplication: delta_q * q_nominal
        q_left = (delta_q_b * q_nominal_b).normalized()

        # 1. State must match RIGHT multiplication
        np.testing.assert_allclose(
            self.ekf.state.quaternion.as_array(),
            q_right.as_array(),
            atol=1e-12,
        )

        # 2. State must NOT match LEFT multiplication
        self.assertFalse(
            np.allclose(self.ekf.state.quaternion.as_array(), q_left.as_array(), atol=1e-4),
            "Attitude injection matched left multiplication; must be right multiplication!",
        )

        # 3. Physical orientation verification:
        # Since roll is around vehicle X axis (forward), the vehicle forward direction
        # in navigation frame must remain strictly along North [0, 1, 0].
        r_right = q_right.rotation_matrix
        v_fwd_nav_right = r_right @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v_fwd_nav_right, np.array([0.0, 1.0, 0.0]), atol=1e-6)

        # In contrast, left multiplication rotates around navigation X (East), tilting
        # the North-pointing forward vector out of the horizontal plane into Z (Pitch):
        r_left = q_left.rotation_matrix
        v_fwd_nav_left = r_left @ np.array([1.0, 0.0, 0.0])
        self.assertFalse(np.allclose(v_fwd_nav_left, np.array([0.0, 1.0, 0.0]), atol=1e-4))
        self.assertGreater(abs(v_fwd_nav_left[2]), 0.04)  # Significant pitch tilt in left-multiplication

    def test_test12_attitude_error_dynamics_gyro_bias_identity(self) -> None:
        """
        TEST 12 — Attitude error dynamics test with non-zero rate, non-zero bias, and non-identity R.
        Verify that in continuous F:
            F[ATTITUDE, ATTITUDE] == -skew(omega_vehicle)
            F[ATTITUDE, GYRO_BIAS] == -I  (NOT -R)
        and in noise coupling G:
            G[ATTITUDE, 3:6] == I  (NOT R)
        """
        # Non-identity attitude (Yaw = 45 deg, Pitch = 15 deg, Roll = -10 deg)
        q = from_euler_rad(-0.1745, 0.2618, 0.7854)
        r_nav_veh = q.rotation_matrix

        f_veh = np.array([1.2, -0.8, 9.7])
        omega_veh = np.array([0.15, -0.25, 0.35])

        f_matrix = self.propagator.compute_continuous_f(r_nav_veh, f_veh, omega_veh)
        g_matrix = self.propagator.compute_noise_matrix_g(r_nav_veh)

        # 1. F[ATTITUDE, ATTITUDE] == -skew(omega_veh)
        np.testing.assert_allclose(
            f_matrix[ATTITUDE, ATTITUDE],
            -skew(omega_veh),
            atol=1e-12,
        )

        # 2. F[ATTITUDE, GYRO_BIAS] == -I (identity, not -R)
        np.testing.assert_allclose(
            f_matrix[ATTITUDE, GYRO_BIAS],
            -np.eye(3),
            atol=1e-12,
        )

        # 3. Verify it is NOT -R
        self.assertFalse(
            np.allclose(f_matrix[ATTITUDE, GYRO_BIAS], -r_nav_veh, atol=1e-3),
            "F[ATTITUDE, GYRO_BIAS] must be -I, not -R",
        )

        # 4. G[ATTITUDE, 3:6] == I (NOT R)
        np.testing.assert_allclose(
            g_matrix[ATTITUDE, 3:6],
            np.eye(3),
            atol=1e-12,
        )
        self.assertFalse(
            np.allclose(g_matrix[ATTITUDE, 3:6], r_nav_veh, atol=1e-3),
            "G[ATTITUDE, 3:6] must be I, not R",
        )

    def test_test13_velocity_error_dynamics_body_frame(self) -> None:
        """
        TEST 13 — Velocity error dynamics test.
        Under body-frame / right-multiplicative attitude error convention:
            d(delta_v)/dt = -R @ [f_vehicle]_x @ delta_theta - R @ delta_b_a
        Verify:
            F[VELOCITY, ATTITUDE] == -R @ skew(f_vehicle)
        and NOT:
            -skew(R @ f_vehicle)
        which corresponds to a navigation-frame (left-error) convention.
        """
        # Non-identity attitude (Yaw = 60 deg, Pitch = 20 deg)
        q = from_euler_rad(0.0, 0.349, 1.047)
        r_nav_veh = q.rotation_matrix
        f_veh = np.array([2.5, -1.8, 9.2])
        omega_veh = np.array([0.05, 0.1, -0.05])

        f_matrix = self.propagator.compute_continuous_f(r_nav_veh, f_veh, omega_veh)

        expected_body_block = -r_nav_veh @ skew(f_veh)
        nav_error_block = -skew(r_nav_veh @ f_veh)

        # 1. Matches -R @ skew(f_veh)
        np.testing.assert_allclose(
            f_matrix[VELOCITY, ATTITUDE],
            expected_body_block,
            atol=1e-12,
        )

        # 2. Does NOT match -skew(R @ f_veh)
        self.assertFalse(
            np.allclose(f_matrix[VELOCITY, ATTITUDE], nav_error_block, atol=1e-3),
            "F[VELOCITY, ATTITUDE] matched nav-frame error -skew(R@f); must be body-frame -R@skew(f)",
        )

    def test_test14_production_dt_limit_enforcement(self) -> None:
        """
        TEST 14 — Verify production max_valid_dt_sec = 0.05 s (50 ms) is loaded and enforced.
        """
        # Confirm loaded max_dt_sec is 0.05 s from configs/ekf.yaml
        self.assertEqual(self.ekf.max_dt_sec, 0.05)

        # Propagation with dt <= 0.05 s succeeds and returns covariance P
        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.zeros(3)
        p_res = self.ekf.predict(accel, gyro, dt=0.04)
        self.assertEqual(p_res.shape, (15, 15))

        # Propagation with dt > 0.05 s (e.g. 0.06 s gap) raises ValueError
        with self.assertRaises(ValueError):
            self.ekf.predict(accel, gyro, dt=0.06)


if __name__ == "__main__":
    unittest.main()
