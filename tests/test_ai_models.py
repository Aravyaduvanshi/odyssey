import math
import unittest
import numpy as np

from edge.coordinate.frames import ImuVector, VehicleAlignedIMU, VehicleFrameTransform
from edge.coordinate.quaternion import from_euler_rad
from edge.models.base import ModelStatus
from edge.models.manager import ModelManager
from edge.models.velocity.preprocessing import VELOCITY_FEATURES, VelocityTCNPreprocessor
from edge.models.yaw.preprocessing import MOUNTNET_FEATURES, MountNetPreprocessor
from edge.sensors.buffer import RollingIMUBuffer
from edge.sensors.models import SensorSample, SensorSource


class AIModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manager = ModelManager(auto_load=True)

    def test_model_contracts_and_startup_status(self) -> None:
        """Verify models loaded and marked READY with correct input dimensions."""
        statuses = self.manager.statuses()
        self.assertEqual(statuses["yaw"], ModelStatus.READY.value)
        self.assertEqual(statuses["velocity"], ModelStatus.READY.value)

        # MountNet contract check: (None, 100, 6)
        self.assertEqual(self.manager.mountnet._model.input_shape[1:], (100, 6))
        # Velocity contract check: (None, 100, 6)
        self.assertEqual(self.manager.velocity._model.input_shape[1:], (100, 6))

    def test_mountnet_normalization(self) -> None:
        """Verify MountNet normalization uses exact baseline mean/std."""
        prep = MountNetPreprocessor()
        self.assertTrue(prep.is_ready)
        self.assertEqual(prep._mean.shape, (6,))
        self.assertEqual(prep._std.shape, (6,))

        # Test normalization on sample identical to mean -> should yield zeros
        dummy = np.tile(prep._mean, (100, 1))
        norm = prep.transform(dummy)
        np.testing.assert_allclose(norm, 0.0, atol=1e-5)

    def test_mountnet_output_conversion_and_vector_norm(self) -> None:
        """Verify atan2 conversion from [sin_yaw, cos_yaw]."""
        # Test synthetic forward pass with known sin/cos
        window = np.zeros((100, 6), dtype=np.float32)
        window[:, 2] = 9.80665

        pred = self.manager.predict_yaw(window, timestamp_ns=1_000_000)
        self.assertIsNotNone(pred)
        self.assertTrue(math.isfinite(pred.mounting_yaw_rad))
        self.assertTrue(math.isfinite(pred.mounting_yaw_deg))
        # Verify 2D vector normalization: sin^2 + cos^2 ≈ 1
        vec_norm = math.sqrt(pred.sin_yaw ** 2 + pred.cos_yaw ** 2)
        self.assertAlmostEqual(vec_norm, 1.0, places=4)
        # Verify atan2 consistency
        expected_rad = math.atan2(pred.sin_yaw, pred.cos_yaw)
        self.assertAlmostEqual(pred.mounting_yaw_rad, expected_rad, places=5)
        self.assertAlmostEqual(pred.mounting_yaw_deg, math.degrees(expected_rad), places=4)

    def test_critical_frame_validation_synthetic_rotation(self) -> None:
        """
        Critical Frame Validation:
        Synthetically rotate phone by 90 deg around Z relative to vehicle.
        Verify phone_to_vehicle transforms acceleration and gyros into:
        ACC_VEH_X, ACC_VEH_Y, ACC_VEH_Z, GYRO_Yaw_corrected, GYRO_Pitch_corrected, GYRO_Roll_corrected
        """
        # Known rotation: phone yaw = +90 degrees relative to vehicle
        # phone X points vehicle lateral-left (+Y_veh)
        # phone Y points vehicle backward (-X_veh)
        # phone Z points vehicle vertical (+Z_veh)
        yaw_rad = math.pi / 2.0
        q = from_euler_rad(0.0, 0.0, yaw_rad)
        transform = VehicleFrameTransform(q)

        # Phone accelerates 2.0 m/s^2 along phone forward (phone X)
        phone_imu = ImuVector(ax=2.0, ay=0.0, az=9.80665, gx=0.05, gy=0.0, gz=0.10)
        veh_imu: VehicleAlignedIMU = transform.to_vehicle_aligned_imu(phone_imu)

        feature_dict = veh_imu.as_dict()
        expected_keys = [
            "ACC_VEH_X",
            "ACC_VEH_Y",
            "ACC_VEH_Z",
            "GYRO_Yaw_corrected",
            "GYRO_Pitch_corrected",
            "GYRO_Roll_corrected",
        ]
        self.assertEqual(list(feature_dict.keys()), expected_keys)
        self.assertEqual(list(veh_imu.as_features().shape), [6])

        # Under +90 deg Z rotation, rotated X is 0, rotated Y is +2.0
        self.assertAlmostEqual(veh_imu.acc_veh_x, 0.0, places=4)
        self.assertAlmostEqual(veh_imu.acc_veh_y, 2.0, places=4)
        self.assertAlmostEqual(veh_imu.acc_veh_z, 9.80665, places=4)
        # Z gyro (yaw rate) preserved along vertical
        self.assertAlmostEqual(veh_imu.gyro_yaw_corrected, 0.10, places=4)

    def test_velocity_normalization_and_target_denormalization(self) -> None:
        """Verify Velocity TCN input normalization and target denormalization formulas."""
        prep = VelocityTCNPreprocessor()
        self.assertEqual(list(VELOCITY_FEATURES), [
            "ACC_VEH_X", "ACC_VEH_Y", "ACC_VEH_Z",
            "GYRO_Yaw_corrected", "GYRO_Pitch_corrected", "GYRO_Roll_corrected"
        ])

        # Input matching mean -> normalized to 0
        mean_window = np.tile(prep.feature_means, (100, 1))
        norm_win = prep.transform(mean_window)
        np.testing.assert_allclose(norm_win, 0.0, atol=1e-5)

        # Target denormalization: velocity_mps = norm * 20.847066 + 26.571593
        # When norm = 0 -> velocity_mps = 26.571593
        vel_mps = prep.denormalize_velocity(0.0)
        self.assertAlmostEqual(vel_mps, 26.57159258, places=4)

        # Uncertainty: sigma_mps = exp(0.5 * log_var) * target_std
        # When log_var = 0.0 -> sigma_mps = 1.0 * 20.847066
        sigma, var = prep.denormalize_uncertainty(0.0)
        self.assertAlmostEqual(sigma, 20.8470657, places=4)
        self.assertAlmostEqual(var, 20.8470657 ** 2, places=3)

    def test_rolling_buffer_and_10hz_resampling(self) -> None:
        """Verify RollingIMUBuffer 10 Hz resampling, size limit, and stride triggers."""
        buffer = RollingIMUBuffer(window_size=100, resample_rate_hz=10.0, mountnet_stride=50, velocity_stride=10)
        self.assertFalse(buffer.is_ready)

        # Simulate 100 Hz input (step = 10 ms = 10_000_000 ns) for 15 seconds (1500 samples)
        t_ns = 1_000_000_000
        committed_count = 0
        for i in range(1500):
            sample = SensorSample(
                timestamp_ns=t_ns,
                ax=0.1,
                ay=0.0,
                az=9.8,
                gx=0.01,
                gy=0.0,
                gz=0.02,
                sensor_source=SensorSource.SYNTHETIC,
            )
            if buffer.add_sample(sample):
                committed_count += 1
            t_ns += 10_000_000  # 100 Hz step

        # At 10 Hz over 15 seconds, we expect ~150 resampled epochs committed
        self.assertGreaterEqual(committed_count, 140)
        self.assertTrue(buffer.is_ready)
        self.assertEqual(buffer.sample_count, 100)

        # Test window shapes
        phone_win = buffer.get_phone_window()
        self.assertEqual(phone_win.shape, (100, 6))

        tf = VehicleFrameTransform()
        veh_win = buffer.get_vehicle_window(tf, (0.0, 0.0, 0.0))
        self.assertEqual(veh_win.shape, (100, 6))

        # Stride checks
        self.assertTrue(buffer.should_run_mountnet())
        buffer.mark_mountnet_run()
        self.assertFalse(buffer.should_run_mountnet())

        self.assertTrue(buffer.should_run_velocity())
        buffer.mark_velocity_run()
        self.assertFalse(buffer.should_run_velocity())

    def test_nan_and_inf_rejection(self) -> None:
        """Verify preprocessors strictly reject NaN and Inf inputs."""
        prep_yaw = MountNetPreprocessor()
        nan_win = np.zeros((100, 6), dtype=np.float32)
        nan_win[5, 2] = float("nan")
        with self.assertRaises(ValueError):
            prep_yaw.transform(nan_win)

        prep_vel = VelocityTCNPreprocessor()
        inf_win = np.zeros((100, 6), dtype=np.float32)
        inf_win[10, 0] = float("inf")
        with self.assertRaises(ValueError):
            prep_vel.transform(inf_win)

    def test_incorrect_window_dimensions_rejected(self) -> None:
        """Verify preprocessors reject windows with invalid shape."""
        prep = MountNetPreprocessor()
        with self.assertRaises(ValueError):
            prep.transform(np.zeros((50, 6), dtype=np.float32))
        with self.assertRaises(ValueError):
            prep.transform(np.zeros((100, 5), dtype=np.float32))

    def test_velocity_inference_and_latency(self) -> None:
        """Verify Velocity TCN runs forward pass and records latency."""
        veh_window = np.zeros((100, 6), dtype=np.float32)
        veh_window[:, 0] = 0.5  # forward accel
        veh_window[:, 2] = 9.80665

        pred = self.manager.predict_velocity(veh_window, timestamp_ns=2_000_000_000)
        self.assertIsNotNone(pred)
        self.assertTrue(math.isfinite(pred.velocity_mps))
        self.assertTrue(math.isfinite(pred.velocity_kmh))
        self.assertTrue(math.isfinite(pred.log_variance))
        self.assertGreater(pred.inference_latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
