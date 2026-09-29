import math
import unittest
import numpy as np

from edge.coordinate.quaternion import Quaternion, from_euler_rad, to_euler_rad
from edge.ins.gravity import GravityModel
from edge.ins.mechanization import StrapdownMechanization
from edge.ins.state import INSState


class StrapdownINSTests(unittest.TestCase):
    """
    Validation test suite for Strapdown INS Mechanization (Step 2).
    """

    def setUp(self) -> None:
        self.gravity = GravityModel(9.80665)
        self.ins = StrapdownMechanization(
            gravity_model=self.gravity,
            max_dt_sec=1.0,
            min_dt_sec=0.00001,
        )

    def test_test1_stationary_imu(self) -> None:
        """
        TEST 1 — Stationary IMU
        Vehicle is stationary and aligned with navigation frame (ENU).
        Accelerometer measures reaction to gravity [0, 0, +9.80665] m/s^2.
        Gyro measures [0, 0, 0] rad/s.
        Expected after multiple time steps:
        - velocity ≈ [0, 0, 0]
        - position ≈ constant [0, 0, 0]
        - attitude ≈ constant [0, 0, 0]
        """
        state = INSState.zero(timestamp_ns=1_000_000_000)
        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.array([0.0, 0.0, 0.0])
        dt = 0.01  # 100 Hz

        # Propagate for 500 steps (5.0 seconds)
        for step in range(500):
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            state = self.ins.propagate(state, accel, gyro, timestamp_ns=t_ns)

        # Velocity must remain zero within floating-point tolerance
        np.testing.assert_allclose(state.velocity_enu, np.zeros(3), atol=1e-10)
        # Position must remain constant (0, 0, 0)
        np.testing.assert_allclose(state.position_enu, np.zeros(3), atol=1e-10)
        # Attitude must remain at initial orientation
        roll, pitch, yaw = state.euler_rad
        self.assertAlmostEqual(roll, 0.0, places=7)
        self.assertAlmostEqual(pitch, 0.0, places=7)
        self.assertAlmostEqual(yaw, 0.0, places=7)

    def test_test2_constant_forward_acceleration(self) -> None:
        """
        TEST 2 — Constant Forward Acceleration
        Vehicle aligned with East (X_veh = East, Y_veh = North, Z_veh = Up).
        Linear acceleration = [1, 0, 0] m/s^2.
        Accelerometer measurement in vehicle frame = [1, 0, 9.80665] m/s^2 (specific force).
        Gyro = [0, 0, 0].
        Run for 10 seconds.
        Expected:
        - velocity_x ≈ 10 m/s
        - position_x ≈ 50 m  (p = 0.5 * a * t^2 = 0.5 * 1 * 100 = 50 m)
        """
        state = INSState.zero(timestamp_ns=1_000_000_000)
        accel = np.array([1.0, 0.0, 9.80665])
        gyro = np.array([0.0, 0.0, 0.0])
        dt = 0.01  # 100 Hz
        total_steps = 1000  # 10.0 seconds

        for step in range(total_steps):
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            state = self.ins.propagate(state, accel, gyro, timestamp_ns=t_ns)

        # Verify velocity: v = a * t = 1.0 * 10.0 = 10.0 m/s
        self.assertAlmostEqual(state.velocity_enu[0], 10.0, places=5)
        self.assertAlmostEqual(state.velocity_enu[1], 0.0, places=6)
        self.assertAlmostEqual(state.velocity_enu[2], 0.0, places=6)

        # Verify position: p = 0.5 * a * t^2 = 0.5 * 1.0 * 100.0 = 50.0 m
        self.assertAlmostEqual(state.position_enu[0], 50.0, places=4)
        self.assertAlmostEqual(state.position_enu[1], 0.0, places=6)
        self.assertAlmostEqual(state.position_enu[2], 0.0, places=6)

    def test_test3_constant_velocity(self) -> None:
        """
        TEST 3 — Constant Velocity
        Initial velocity = [10, 0, 0] m/s.
        Linear acceleration = [0, 0, 0] -> Accelerometer = [0, 0, 9.80665] m/s^2.
        Gyro = [0, 0, 0].
        Run for 10 seconds.
        Expected:
        - velocity ≈ 10 m/s (constant)
        - position displacement ≈ 100 m (p = v * t = 10 * 10 = 100 m)
        """
        state = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.array([10.0, 0.0, 0.0]),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.array([0.0, 0.0, 0.0])
        dt = 0.01  # 100 Hz
        total_steps = 1000  # 10.0 seconds

        for step in range(total_steps):
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            state = self.ins.propagate(state, accel, gyro, timestamp_ns=t_ns)

        # Velocity remains exactly 10 m/s
        self.assertAlmostEqual(state.velocity_enu[0], 10.0, places=6)
        self.assertAlmostEqual(state.velocity_enu[1], 0.0, places=6)
        self.assertAlmostEqual(state.velocity_enu[2], 0.0, places=6)

        # Position displacement = 100 m
        self.assertAlmostEqual(state.position_enu[0], 100.0, places=5)
        self.assertAlmostEqual(state.position_enu[1], 0.0, places=6)
        self.assertAlmostEqual(state.position_enu[2], 0.0, places=6)

    def test_test4_known_yaw_rotation(self) -> None:
        """
        TEST 4 — Known Yaw Rotation
        Apply constant yaw rate = pi/2 rad/s (90 deg/s) around vertical Z-axis.
        Duration = 1.0 second.
        Expected:
        - Yaw change ≈ +90 degrees (pi/2 radians)
        - Quaternion remains normalized: abs(norm(q) - 1) < 1e-10
        - Orientation changes in expected direction (positive yaw)
        - No unexpected roll or pitch appears (< 1e-6)
        """
        state = INSState.zero(timestamp_ns=1_000_000_000)
        yaw_rate = math.pi / 2.0
        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.array([0.0, 0.0, yaw_rate])
        dt = 0.01  # 100 Hz
        total_steps = 100  # 1.0 second

        for step in range(total_steps):
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            state = self.ins.propagate(state, accel, gyro, timestamp_ns=t_ns)

        roll, pitch, yaw = state.euler_rad
        # Yaw change must be 90 degrees (+pi/2 rad)
        self.assertAlmostEqual(yaw, math.pi / 2.0, places=4)
        # Roll and pitch must remain near zero
        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        # Quaternion normalization check
        norm_q = float(np.linalg.norm(state.quaternion.as_array()))
        self.assertAlmostEqual(norm_q, 1.0, places=10)

    def test_test5_gravity_compensation(self) -> None:
        """
        TEST 5 — Gravity Compensation
        Verify that a stationary IMU with:
        accelerometer = [0, 0, +9.80665] (specific force pointing UP)
        and gravity:
        g_nav = [0, 0, -9.80665] (pointing DOWN)
        produces:
        a_nav ≈ [0, 0, 0]
        """
        accel = np.array([0.0, 0.0, 9.80665])
        g_nav = self.gravity.vector_enu()  # [0, 0, -9.80665]

        # In vehicle frame aligned with navigation frame (R = I)
        R_nav_vehicle = np.eye(3)
        f_nav = R_nav_vehicle @ accel
        a_nav = f_nav + g_nav

        np.testing.assert_allclose(a_nav, np.zeros(3), atol=1e-12)

        # Propagate one step through mechanization
        state = INSState.zero(timestamp_ns=1_000_000_000)
        gyro = np.zeros(3)
        next_state = self.ins.propagate(state, accel, gyro, dt=0.01)

        np.testing.assert_allclose(next_state.velocity_enu, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(next_state.position_enu, np.zeros(3), atol=1e-12)

    def test_test6_timestamp_validation(self) -> None:
        """
        TEST 6 — Timestamp Validation
        Verify rejection (raising ValueError) of:
        - dt = 0
        - dt < 0
        - NaN dt
        - Inf dt
        - dt greater than configured maximum
        """
        state = INSState.zero(timestamp_ns=1_000_000_000)
        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.zeros(3)

        # 1. dt = 0
        with self.assertRaises(ValueError):
            self.ins.propagate(state, accel, gyro, dt=0.0)

        with self.assertRaises(ValueError):
            self.ins.propagate(state, accel, gyro, timestamp_ns=1_000_000_000)

        # 2. dt < 0
        with self.assertRaises(ValueError):
            self.ins.propagate(state, accel, gyro, dt=-0.01)

        with self.assertRaises(ValueError):
            self.ins.propagate(state, accel, gyro, timestamp_ns=990_000_000)

        # 3. NaN dt
        with self.assertRaises(ValueError):
            self.ins.propagate(state, accel, gyro, dt=float("nan"))

        # 4. Inf dt
        with self.assertRaises(ValueError):
            self.ins.propagate(state, accel, gyro, dt=float("inf"))

        with self.assertRaises(ValueError):
            self.ins.propagate(state, accel, gyro, dt=float("-inf"))

        # 5. dt greater than configured maximum
        custom_ins = StrapdownMechanization(max_dt_sec=0.05)  # 50 ms ceiling
        with self.assertRaises(ValueError):
            custom_ins.propagate(state, accel, gyro, dt=0.06)

    def test_test7_bias_compensation(self) -> None:
        """
        TEST 7 — Bias Compensation
        Provide known accelerometer and gyroscope biases.
        Verify that after subtracting the configured biases, the expected
        inertial values and kinematics are recovered.
        """
        # Accelerometer bias test:
        # Bias = [0.5, -0.2, 0.1] m/s^2.
        # Desired true specific force = [1.0, 0.0, 9.80665].
        # Measured sensor output = true + bias = [1.5, -0.2, 9.90665].
        state = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.zeros(3),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.array([0.5, -0.2, 0.1]),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        meas_accel = np.array([1.5, -0.2, 9.90665])
        gyro = np.zeros(3)

        # Propagate 1 second at 100 Hz
        for step in range(100):
            t_ns = 1_000_000_000 + int((step + 1) * 0.01 * 1e9)
            state = self.ins.propagate(state, meas_accel, gyro, timestamp_ns=t_ns)

        # Should recover 1.0 m/s^2 forward acceleration:
        # v_x = 1.0 m/s, p_x = 0.5 m, other components zero
        self.assertAlmostEqual(state.velocity_enu[0], 1.0, places=4)
        self.assertAlmostEqual(state.velocity_enu[1], 0.0, places=6)
        self.assertAlmostEqual(state.velocity_enu[2], 0.0, places=6)
        self.assertAlmostEqual(state.position_enu[0], 0.5, places=4)

        # Gyroscope bias test:
        # Bias = [0.05, -0.02, 0.1] rad/s.
        # Sensor measures exactly the bias values (vehicle is NOT rotating).
        state_gyro = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.zeros(3),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3),
            gyro_bias=np.array([0.05, -0.02, 0.1]),
            timestamp_ns=1_000_000_000,
        )
        meas_gyro = np.array([0.05, -0.02, 0.1])
        accel = np.array([0.0, 0.0, 9.80665])

        for step in range(100):
            t_ns = 1_000_000_000 + int((step + 1) * 0.01 * 1e9)
            state_gyro = self.ins.propagate(state_gyro, accel, meas_gyro, timestamp_ns=t_ns)

        # Attitude must remain completely unchanged because measured gyro equals bias
        roll, pitch, yaw = state_gyro.euler_rad
        self.assertAlmostEqual(roll, 0.0, places=7)
        self.assertAlmostEqual(pitch, 0.0, places=7)
        self.assertAlmostEqual(yaw, 0.0, places=7)

    def test_test8_quaternion_normalization(self) -> None:
        """
        TEST 8 — Quaternion Normalization
        Run 2,000 continuous propagation steps with multi-axis rotations
        (simultaneous roll, pitch, and yaw rates).
        Verify at every step:
        abs(norm(q) - 1) < 1e-10
        """
        state = INSState.zero(timestamp_ns=1_000_000_000)
        dt = 0.01  # 10 ms
        accel = np.array([0.5, -0.3, 9.80665])
        # Dynamic rates: 0.1 rad/s roll, 0.2 rad/s pitch, 0.5 rad/s yaw
        gyro = np.array([0.1, 0.2, 0.5])

        for step in range(2000):  # 20 seconds of vigorous 3D rotation
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            state = self.ins.propagate(state, accel, gyro, timestamp_ns=t_ns)

            q_norm = float(np.linalg.norm(state.quaternion.as_array()))
            self.assertAlmostEqual(q_norm, 1.0, places=10)

        # Final check
        final_norm = float(np.linalg.norm(state.quaternion.as_array()))
        self.assertTrue(abs(final_norm - 1.0) < 1e-12)


if __name__ == "__main__":
    unittest.main()
