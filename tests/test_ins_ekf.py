import math
import unittest
import numpy as np

from edge.coordinate.quaternion import Quaternion, from_euler_rad, to_euler_rad
from edge.fusion.ekf import ErrorStateEKF
from edge.ins.gravity import GravityModel
from edge.ins.mechanization import StrapdownMechanization
from edge.ins.state import INSState


class StrapdownMechanizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gravity = GravityModel(9.80665)
        self.ins = StrapdownMechanization(gravity_model=self.gravity)

    def test_step1_quaternion_propagation(self) -> None:
        """Verify quaternion propagation for pure rotation around axes."""
        # 90 degrees/sec pure yaw rotation
        state = INSState.zero(timestamp_ns=1_000_000_000)
        dt = 0.01  # 10 ms (100 Hz)
        yaw_rate = math.pi / 2.0  # 90 deg/s
        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.array([0.0, 0.0, yaw_rate])

        for step in range(100):
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            state = self.ins.propagate(state, accel, gyro, t_ns)

        # After 1.0s of pi/2 rad/s yaw rate, yaw should be pi/2 rad (90 degrees)
        roll, pitch, yaw = state.euler_rad
        self.assertAlmostEqual(yaw, math.pi / 2.0, places=3)
        self.assertAlmostEqual(roll, 0.0, places=4)
        self.assertAlmostEqual(pitch, 0.0, places=4)
        self.assertAlmostEqual(float(np.linalg.norm(state.quaternion.as_array())), 1.0, places=6)

    def test_test1_stationary_imu(self) -> None:
        """
        TEST 1: Stationary IMU
        Accelerometer measures reaction to gravity [0, 0, +g] in vehicle frame.
        Gyro measures zero.
        Result: Nav acceleration = 0. Velocity stays ~0, position stays ~constant.
        """
        state = INSState.zero(timestamp_ns=1_000_000_000)
        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.array([0.0, 0.0, 0.0])
        dt = 0.01

        for step in range(100):  # 1 second at 100 Hz
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            state = self.ins.propagate(state, accel, gyro, t_ns)

        np.testing.assert_allclose(state.velocity_enu, np.zeros(3), atol=1e-5)
        np.testing.assert_allclose(state.position_enu, np.zeros(3), atol=1e-5)

    def test_test2_constant_acceleration(self) -> None:
        """
        TEST 2: Constant Forward Acceleration
        Vehicle pointing East (yaw = 0):
        Forward accel = 2.0 m/s^2, vertical accel = +g.
        Result: East velocity increases at 2.0 m/s^2, East position increases quadratically.
        """
        state = INSState.zero(timestamp_ns=1_000_000_000)
        fwd_accel = 2.0
        accel = np.array([fwd_accel, 0.0, 9.80665])
        gyro = np.array([0.0, 0.0, 0.0])
        dt = 0.01

        for step in range(100):  # 1.0 second
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            state = self.ins.propagate(state, accel, gyro, t_ns)

        # In 1.0s: v = a * t = 2.0 m/s, p = 0.5 * a * t^2 = 1.0 m
        self.assertAlmostEqual(state.velocity_enu[0], 2.0, places=3)
        self.assertAlmostEqual(state.position_enu[0], 1.0, places=3)
        self.assertAlmostEqual(state.velocity_enu[1], 0.0, places=4)
        self.assertAlmostEqual(state.velocity_enu[2], 0.0, places=4)

    def test_test3_constant_velocity(self) -> None:
        """
        TEST 3: Constant Velocity
        Initial velocity = [10.0, 5.0, 0.0] m/s. Accel = [0, 0, g], gyro = 0.
        Result: Velocity remains constant, position grows linearly p = v * t.
        """
        state = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.array([10.0, 5.0, 0.0]),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.array([0.0, 0.0, 0.0])
        dt = 0.01

        for step in range(100):  # 1.0 second
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            state = self.ins.propagate(state, accel, gyro, t_ns)

        np.testing.assert_allclose(state.velocity_enu, np.array([10.0, 5.0, 0.0]), atol=1e-4)
        np.testing.assert_allclose(state.position_enu, np.array([10.0, 5.0, 0.0]), atol=1e-4)

    def test_test4_known_yaw_rotation(self) -> None:
        """
        TEST 4: Known Yaw Rotation
        Initial vehicle pointing North: yaw = 90 deg (pi/2 rad).
        Vehicle frame X (forward) points North (Y in ENU).
        Forward acceleration of 3.0 m/s^2 should accelerate along North (Y in ENU).
        """
        # Vehicle pointing North: yaw = pi/2
        q_north = from_euler_rad(0.0, 0.0, math.pi / 2.0)
        state = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.zeros(3),
            quaternion=q_north,
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        fwd_accel = 3.0
        accel = np.array([fwd_accel, 0.0, 9.80665])
        gyro = np.array([0.0, 0.0, 0.0])
        dt = 0.01

        for step in range(100):  # 1.0 second
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            state = self.ins.propagate(state, accel, gyro, t_ns)

        # Forward in vehicle is North in ENU (index 1)
        self.assertAlmostEqual(state.velocity_enu[0], 0.0, places=3) # East is 0
        self.assertAlmostEqual(state.velocity_enu[1], 3.0, places=3) # North is 3.0 m/s
        self.assertAlmostEqual(state.velocity_enu[2], 0.0, places=3) # Up is 0
        self.assertAlmostEqual(state.position_enu[1], 1.5, places=3) # 0.5 * 3 * 1^2 = 1.5 m


class ErrorStateEKFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gravity = GravityModel(9.80665)
        self.ins = StrapdownMechanization(gravity_model=self.gravity)

    def test_step3_ekf_prediction_covariance_grows(self) -> None:
        """Verify that open-loop EKF prediction causes position uncertainty to grow."""
        state = INSState.zero(timestamp_ns=1_000_000_000)
        ekf = ErrorStateEKF(initial_state=state, mechanization=self.ins)
        initial_unc = ekf.position_uncertainty_1sigma_m

        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.array([0.0, 0.0, 0.0])
        dt = 0.01

        for step in range(50):
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            ekf.predict(accel, gyro, t_ns)

        # Covariance grows in open-loop integration
        self.assertGreater(ekf.position_uncertainty_1sigma_m, initial_unc)

    def test_test5_gnss_position_correction(self) -> None:
        """
        TEST 5: GNSS Position Correction
        Inject position error (true=0, state=[10, -5, 2]).
        Provide accurate GNSS measurement [0, 0, 0].
        Verify EKF reduces position error significantly.
        """
        from edge.fusion.measurements import GNSSPositionMeasurement

        state = INSState(
            position_enu=np.array([10.0, -5.0, 2.0]),
            velocity_enu=np.zeros(3),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        ekf = ErrorStateEKF(initial_state=state, mechanization=self.ins)
        # Set large prior position variance
        ekf.P[0:3, 0:3] = np.eye(3) * 100.0

        gnss_meas = GNSSPositionMeasurement(
            z_enu=np.array([0.0, 0.0, 0.0]),
            cov_enu=np.eye(3) * 4.0,  # 2m 1-sigma
            gating_threshold=50.0,
        )

        res = ekf.update(gnss_meas)
        self.assertTrue(res.accepted)
        # Position error should be significantly reduced toward zero
        self.assertLess(np.linalg.norm(ekf.state.position_enu), 3.0)
        # Position covariance should decrease
        self.assertLess(ekf.P[0, 0], 10.0)

    def test_step5_gnss_velocity_correction(self) -> None:
        """Verify GNSS velocity measurement corrects velocity error."""
        from edge.fusion.measurements import GNSSVelocityMeasurement

        state = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.array([15.0, 0.0, 0.0]), # Has 5 m/s error vs true 10 m/s
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        ekf = ErrorStateEKF(initial_state=state, mechanization=self.ins)
        ekf.P[3:6, 3:6] = np.eye(3) * 4.0

        meas = GNSSVelocityMeasurement(
            z_vel_enu=np.array([10.0, 0.0, 0.0]),
            cov_vel_enu=np.eye(3) * 0.04, # 0.2 m/s noise
            gating_threshold=50.0,
        )
        res = ekf.update(meas)
        self.assertTrue(res.accepted)
        # Velocity should be pulled close to 10 m/s
        self.assertAlmostEqual(ekf.state.velocity_enu[0], 10.0, delta=0.5)

    def test_test6_ai_velocity_correction(self) -> None:
        """
        TEST 6: AI Velocity Correction
        Vehicle heading East (yaw = 0): vehicle forward axis is East.
        INS velocity is 20.0 m/s, AI predicts 12.0 m/s forward speed.
        Verify EKF moves forward velocity toward AI prediction.
        """
        from edge.fusion.measurements import AIVelocityMeasurement

        state = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.array([20.0, 0.0, 0.0]),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        ekf = ErrorStateEKF(initial_state=state, mechanization=self.ins)
        ekf.P[3:6, 3:6] = np.eye(3) * 16.0 # High velocity uncertainty

        ai_meas = AIVelocityMeasurement(
            v_forward_mps=12.0,
            variance_mps2=0.5, # 0.7 m/s uncertainty
            gating_threshold=50.0,
        )

        res = ekf.update(ai_meas)
        self.assertTrue(res.accepted)
        # Forward velocity moved substantially toward 12.0 m/s
        self.assertLess(ekf.state.velocity_enu[0], 15.0)
        self.assertGreater(ekf.state.velocity_enu[0], 11.5)

    def test_test7_ai_outlier(self) -> None:
        """
        TEST 7: AI Outlier Rejection
        Vehicle moving at 10.0 m/s forward.
        AI predicts unrealistic 200.0 m/s (anomalous inference).
        Verify innovation gating rejects the measurement and state does not jump.
        """
        from edge.fusion.measurements import AIVelocityMeasurement

        state = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.array([10.0, 0.0, 0.0]),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        ekf = ErrorStateEKF(initial_state=state, mechanization=self.ins)
        ekf.P[3:6, 3:6] = np.eye(3) * 1.0 # Realistic 1 m/s uncertainty

        # Outlier: 200 m/s
        ai_outlier = AIVelocityMeasurement(
            v_forward_mps=200.0,
            variance_mps2=1.0,
            gating_threshold=10.83, # 1-DOF @ 99.9%
        )

        res = ekf.update(ai_outlier)
        self.assertFalse(res.accepted)
        # State velocity must NOT jump
        self.assertAlmostEqual(ekf.state.velocity_enu[0], 10.0, places=4)

    def test_test8_nhc_constraint(self) -> None:
        """
        TEST 8: Non-Holonomic Constraint (NHC)
        Vehicle pointing East (yaw = 0): vehicle forward axis is East (0), lateral is North (1).
        Inject lateral velocity of 4.0 m/s (drift).
        Apply NHC measurement (target: v_lat=0, v_vert=0).
        Verify EKF reduces lateral velocity toward zero without harming forward velocity.
        """
        from edge.fusion.measurements import NHCMeasurement

        state = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.array([12.0, 4.0, 0.0]), # 12 fwd, 4 lat
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        ekf = ErrorStateEKF(initial_state=state, mechanization=self.ins)
        ekf.P[3:6, 3:6] = np.eye(3) * 4.0 # 2 m/s velocity uncertainty

        nhc_meas = NHCMeasurement(
            cov_2x2=np.diag([0.01, 0.01]), # 0.1 m/s constraint noise
            gating_threshold=50.0,
        )

        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.zeros(3)
        dt = 0.01
        for step in range(20): # 0.2s of driving with active NHC
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            ekf.predict(accel, gyro, t_ns)
            res = ekf.update(nhc_meas)
            self.assertTrue(res.accepted)

        # Evaluate vehicle-frame velocity: v_veh = R_v_n @ v_nav
        v_veh = ekf.state.rotation_matrix.T @ ekf.state.velocity_enu
        # Lateral velocity in vehicle chassis frame must be constrained down near zero
        self.assertLess(abs(v_veh[1]), 0.5)
        self.assertLess(abs(v_veh[2]), 0.5)
        # Forward speed in vehicle chassis frame remains ~12 m/s
        self.assertAlmostEqual(v_veh[0], 12.0, delta=1.0)

    def test_step8_zupt_constraint(self) -> None:
        """Verify ZUPT clamps residual velocities when stationary."""
        from edge.fusion.measurements import ZUPTMeasurement

        state = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.array([0.6, -0.4, 0.1]),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        ekf = ErrorStateEKF(initial_state=state, mechanization=self.ins)
        ekf.P[3:6, 3:6] = np.eye(3) * 0.25

        zupt_meas = ZUPTMeasurement(cov_3x3=np.eye(3) * 0.0025, gating_threshold=50.0)
        res = ekf.update(zupt_meas)
        self.assertTrue(res.accepted)
        np.testing.assert_allclose(ekf.state.velocity_enu, np.zeros(3), atol=0.1)

    def test_test11_bias_estimation(self) -> None:
        """
        TEST 11: Bias Estimation
        Inject constant gyro yaw bias = 0.02 rad/s.
        During regular updates with heading/position observations,
        verify EKF bias state begins estimating the injected bias.
        """
        from edge.fusion.measurements import (
            GNSSPositionMeasurement,
            GNSSVelocityMeasurement,
            HeadingMeasurement,
        )

        state = INSState.zero(timestamp_ns=1_000_000_000)
        ekf = ErrorStateEKF(initial_state=state, mechanization=self.ins)
        ekf.P[12:15, 12:15] = np.eye(3) * 0.001

        true_gyro_bias_z = 0.02  # rad/s injected bias
        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.array([0.0, 0.0, true_gyro_bias_z]) # Stationary sensor, but gyro reads bias
        dt = 0.01

        # Run 2 seconds of stationary filter with GNSS position, velocity & heading updates
        for step in range(200):
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            ekf.predict(accel, gyro, t_ns)

            # Updates at 10 Hz
            if step % 10 == 0:
                pos_meas = GNSSPositionMeasurement(
                    z_enu=np.zeros(3),
                    cov_enu=np.eye(3) * 0.25,
                    gating_threshold=50.0,
                )
                vel_meas = GNSSVelocityMeasurement(
                    z_vel_enu=np.zeros(3),
                    cov_vel_enu=np.eye(3) * 0.01,
                    gating_threshold=50.0,
                )
                head_meas = HeadingMeasurement(
                    heading_rad=0.0,
                    variance_rad2=0.001,
                    gating_threshold=50.0,
                )
                ekf.update(pos_meas)
                ekf.update(vel_meas)
                ekf.update(head_meas)

        # The estimated gyro bias should be positive and converging toward 0.02 rad/s
        self.assertGreater(ekf.state.gyro_bias[2], 0.001)

    def test_test9_gnss_blackout(self) -> None:
        """
        TEST 9: GNSS Blackout
        Verify:
        - When GNSS is cut off, GNSS updates stop entering EKF.
        - INS propagation continues.
        - AI forward velocity updates continue.
        - Outage distance accumulates.
        """
        from edge.fusion.outage import GNSSOutageManager, GNSSOutageState
        from edge.fusion.measurements import AIVelocityMeasurement

        state = INSState(
            position_enu=np.zeros(3),
            velocity_enu=np.array([10.0, 0.0, 0.0]),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        ekf = ErrorStateEKF(initial_state=state, mechanization=self.ins)
        outage_mgr = GNSSOutageManager()

        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.zeros(3)
        dt = 0.01

        # 1.0 second of blackout driving
        gnss_updates_count = 0
        ai_updates_count = 0

        for step in range(100):
            t_ns = 1_000_000_000 + int((step + 1) * dt * 1e9)
            ekf.predict(accel, gyro, t_ns)

            outage_state, should_process_gnss = outage_mgr.on_sample(
                gnss_available=False, # Blackout!
                gnss_accuracy_m=10.0,
                timestamp_ns=t_ns,
                current_pos_enu=ekf.state.position_enu,
            )
            if should_process_gnss:
                gnss_updates_count += 1

            # AI forward velocity updates continue at 10 Hz
            if step % 10 == 0:
                ai_meas = AIVelocityMeasurement(v_forward_mps=10.0, variance_mps2=0.5)
                res = ekf.update(ai_meas)
                if res.accepted:
                    ai_updates_count += 1

        self.assertEqual(gnss_updates_count, 0)
        self.assertGreater(ai_updates_count, 5)
        self.assertEqual(outage_mgr.state, GNSSOutageState.GNSS_LOST)
        # Position propagated forward by ~10m
        self.assertAlmostEqual(ekf.state.position_enu[0], 10.0, delta=1.0)
        self.assertGreater(outage_mgr.drift_tracker.outage_distance_m, 8.0)

    def test_test10_gnss_recovery(self) -> None:
        """
        TEST 10: GNSS Recovery
        Verify:
        - When GNSS returns, position does NOT jump/snap instantly.
        - State smoothly converges toward GNSS over updates.
        """
        from edge.fusion.outage import GNSSOutageManager, GNSSOutageState
        from edge.fusion.measurements import GNSSPositionMeasurement

        # Start with drifted position [15, 0, 0], while true GNSS is [10, 0, 0]
        state = INSState(
            position_enu=np.array([15.0, 0.0, 0.0]),
            velocity_enu=np.array([10.0, 0.0, 0.0]),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3),
            gyro_bias=np.zeros(3),
            timestamp_ns=1_000_000_000,
        )
        ekf = ErrorStateEKF(initial_state=state, mechanization=self.ins)
        outage_mgr = GNSSOutageManager(recovery_required_samples=3)
        outage_mgr.state = GNSSOutageState.GNSS_LOST # currently in outage

        pos_before_recovery = ekf.state.position_enu.copy()

        # GNSS reacquired at t=1.01s, position is [10, 0, 0]
        gnss_pos = np.array([10.0, 0.0, 0.0])
        t_ns = 1_010_000_000

        outage_state, should_process = outage_mgr.on_sample(
            gnss_available=True,
            gnss_accuracy_m=2.0,
            timestamp_ns=t_ns,
            current_pos_enu=ekf.state.position_enu,
        )
        self.assertEqual(outage_state, GNSSOutageState.GNSS_RECOVERY)
        self.assertTrue(should_process)

        # Single measurement update
        meas = GNSSPositionMeasurement(z_enu=gnss_pos, cov_enu=np.eye(3) * 4.0, gating_threshold=50.0)
        res = ekf.update(meas)
        self.assertTrue(res.accepted)
        outage_mgr.on_gnss_measurement_result(res.accepted)

        # Position must NOT snap directly to gnss_pos [10, 0, 0] in one step
        # It must be between pos_before_recovery and gnss_pos (smooth Kalman gain transition)
        self.assertNotEqual(ekf.state.position_enu[0], gnss_pos[0])
        self.assertLess(ekf.state.position_enu[0], pos_before_recovery[0])
        self.assertGreater(ekf.state.position_enu[0], gnss_pos[0])


if __name__ == "__main__":
    unittest.main()



