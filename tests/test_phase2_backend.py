import math
import unittest
import numpy as np

from edge.api.main import app
from edge.calibration.engine import CalibrationEngine, CalibrationStatus
from edge.coordinate.frames import ImuVector, VehicleFrameTransform
from edge.coordinate.quaternion import Quaternion, from_euler_rad, from_rotation_vector, to_euler_rad
from edge.models.adapters import ModelManager, ModelStatus
from edge.navigation.engine import NavigationEngine
from edge.navigation.models import MotionState, NavigationMode, NavigationState
from edge.sensors.models import SensorSample, SensorSource
from edge.sensors.scenario import GRAVITY_MPS2, MockVehicleScenario
from edge.sensors.time_sync import TimeSynchronizer
from starlette.testclient import TestClient


class BackendPhase2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_root_endpoint_payload(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["application"], "OdysseyV1")
        self.assertEqual(data["service"], "Edge Navigation Engine")
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["version"], "1.0.0")

    def test_health_endpoint_payload(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["edge_engine"], "online")
        self.assertEqual(data["sensor_provider"], "MOCK")
        self.assertIn(data["models"]["velocity"], ["READY", "NOT_LOADED"])
        self.assertIn(data["models"]["yaw"], ["READY", "NOT_LOADED"])

    def test_navigation_state_rejects_nan_and_inf(self) -> None:
        with self.assertRaises(ValueError):
            SensorSample(
                timestamp_ns=100,
                ax=float("nan"),
                ay=0.0,
                az=9.80665,
                gx=0.0,
                gy=0.0,
                gz=0.0,
                sensor_source=SensorSource.SYNTHETIC,
            )
        with self.assertRaises(ValueError):
            SensorSample(
                timestamp_ns=100,
                ax=float("inf"),
                ay=0.0,
                az=9.80665,
                gx=0.0,
                gy=0.0,
                gz=0.0,
                sensor_source=SensorSource.SYNTHETIC,
            )

    def test_mock_scenario_deterministic_and_phases(self) -> None:
        scenario = MockVehicleScenario(rate_hz=100.0, seed=42, noise_std=0.0)
        self.assertGreater(scenario.length, 0)

        # Verify 7 distinct phases exist in sequence
        phases_seen: list[str] = []
        for i in range(scenario.length):
            truth = scenario.truth_at(i, start_ns=0)
            if not phases_seen or phases_seen[-1] != truth.phase:
                phases_seen.append(truth.phase)

        expected_phases = [
            "stationary",
            "acceleration",
            "straight",
            "gentle_left_turn",
            "constant_speed",
            "gentle_right_turn",
            "braking",
        ]
        self.assertEqual(phases_seen, expected_phases)

        # Monotonicity test
        last_t = -1
        for i in range(200):
            sample = scenario.sample_at(i, start_ns=1_000_000)
            self.assertGreater(sample.timestamp_ns, last_t)
            last_t = sample.timestamp_ns

    def test_time_synchronizer(self) -> None:
        sync = TimeSynchronizer(target_rate_hz=100.0)
        s1 = SensorSample(timestamp_ns=100, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0, sensor_source=SensorSource.SYNTHETIC)
        self.assertTrue(sync.accept(s1))

        # Duplicate timestamp rejected
        s_dup = SensorSample(timestamp_ns=100, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0, sensor_source=SensorSource.SYNTHETIC)
        self.assertFalse(sync.accept(s_dup))
        self.assertEqual(sync.status.duplicate_timestamps, 1)

        # Timestamp reversal rejected
        s_rev = SensorSample(timestamp_ns=50, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0, sensor_source=SensorSource.SYNTHETIC)
        self.assertFalse(sync.accept(s_rev))
        self.assertEqual(sync.status.timestamp_reversals, 1)

        # Monotonic forward accepted
        s2 = SensorSample(timestamp_ns=10_000_100, ax=0, ay=0, az=9.8, gx=0, gy=0, gz=0, sensor_source=SensorSource.SYNTHETIC)
        self.assertTrue(sync.accept(s2))
        self.assertIsNotNone(sync.status.estimated_rate_hz)

    def test_quaternion_operations(self) -> None:
        q_ident = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(q_ident.as_array())), 1.0)

        # Inverse of identity is identity
        q_inv = q_ident.inverse()
        self.assertAlmostEqual(q_inv.w, 1.0)
        self.assertAlmostEqual(q_inv.x, 0.0)

        # Multiplication
        q1 = from_euler_rad(0.1, 0.2, 0.3)
        q1_inv = q1.inverse()
        prod = q1.multiply(q1_inv)
        self.assertAlmostEqual(prod.w, 1.0, places=5)
        self.assertAlmostEqual(prod.x, 0.0, places=5)
        self.assertAlmostEqual(prod.y, 0.0, places=5)
        self.assertAlmostEqual(prod.z, 0.0, places=5)

        # Vector rotation 90 deg around Z
        q_rot_z90 = from_euler_rad(0.0, 0.0, math.pi / 2.0)
        vec_x = np.array([1.0, 0.0, 0.0])
        rotated = q_rot_z90.rotate_vector(vec_x)
        self.assertAlmostEqual(rotated[0], 0.0, places=5)
        self.assertAlmostEqual(rotated[1], 1.0, places=5)
        self.assertAlmostEqual(rotated[2], 0.0, places=5)

        # Euler conversion roundtrip
        r, p, y = 0.15, -0.25, 0.85
        q_euler = from_euler_rad(r, p, y)
        r_out, p_out, y_out = to_euler_rad(q_euler)
        self.assertAlmostEqual(r, r_out, places=5)
        self.assertAlmostEqual(p, p_out, places=5)
        self.assertAlmostEqual(y, y_out, places=5)

    def test_vehicle_frame_transform(self) -> None:
        # Phone aligned with vehicle (identity)
        tf = VehicleFrameTransform()
        raw_imu = ImuVector(ax=0.5, ay=0.2, az=9.8, gx=0.01, gy=0.02, gz=0.03)
        v_imu = tf.phone_to_vehicle(raw_imu)
        self.assertAlmostEqual(v_imu.ax, 0.5)
        self.assertAlmostEqual(v_imu.ay, 0.2)
        self.assertAlmostEqual(v_imu.az, 9.8)

    def test_calibration_engine_pipeline(self) -> None:
        cal = CalibrationEngine(gravity_threshold_mps2=0.35, stable_samples_required=40)

        # 1. Collecting stable samples -> ESTIMATING_GRAVITY
        for i in range(15):
            s = SensorSample(
                timestamp_ns=i * 10_000_000,
                ax=0.0,
                ay=0.0,
                az=GRAVITY_MPS2,
                gx=0.01,
                gy=0.02,
                gz=0.005,
                sensor_source=SensorSource.SYNTHETIC,
                gnss_velocity=0.0,
            )
            state = cal.update(s)
        self.assertEqual(state.stage, CalibrationStatus.ESTIMATING_GRAVITY)

        # 2. More stable samples -> ESTIMATING_BIAS
        for i in range(15, 30):
            s = SensorSample(
                timestamp_ns=i * 10_000_000,
                ax=0.0,
                ay=0.0,
                az=GRAVITY_MPS2,
                gx=0.01,
                gy=0.02,
                gz=0.005,
                sensor_source=SensorSource.SYNTHETIC,
                gnss_velocity=0.0,
            )
            state = cal.update(s)
        self.assertEqual(state.stage, CalibrationStatus.ESTIMATING_BIAS)

        # 3. Complete stable buffer -> ESTIMATING_HEADING
        for i in range(30, 45):
            s = SensorSample(
                timestamp_ns=i * 10_000_000,
                ax=0.0,
                ay=0.0,
                az=GRAVITY_MPS2,
                gx=0.01,
                gy=0.02,
                gz=0.005,
                sensor_source=SensorSource.SYNTHETIC,
                gnss_velocity=0.0,
            )
            state = cal.update(s)
        self.assertEqual(state.stage, CalibrationStatus.ESTIMATING_HEADING)
        self.assertAlmostEqual(state.alignment.gyro_bias_x, 0.01, places=3)
        self.assertAlmostEqual(state.alignment.gyro_bias_y, 0.02, places=3)

        # 4. Vehicle moves with GNSS heading -> COMPLETE
        moving_sample = SensorSample(
            timestamp_ns=45 * 10_000_000,
            ax=1.2,
            ay=0.0,
            az=GRAVITY_MPS2,
            gx=0.0,
            gy=0.0,
            gz=0.0,
            mx=math.cos(1.57),
            my=math.sin(1.57),
            gnss_available=True,
            gnss_heading=1.57,
            gnss_velocity=8.5,
            sensor_source=SensorSource.SYNTHETIC,
        )
        final_state = cal.update(moving_sample)
        self.assertEqual(final_state.stage, CalibrationStatus.COMPLETE)
        self.assertAlmostEqual(final_state.alignment.yaw_rad, 1.57, places=2)
        self.assertGreaterEqual(final_state.alignment.confidence, 0.8)

    def test_ai_models_strictly_unloaded(self) -> None:
        manager = ModelManager(auto_load=False)
        statuses = manager.statuses()
        self.assertEqual(statuses["velocity"], "NOT_LOADED")
        self.assertEqual(statuses["yaw"], "NOT_LOADED")
        self.assertIsNone(manager.velocity.predict([]))
        self.assertIsNone(manager.yaw.predict([]))


if __name__ == "__main__":
    unittest.main()
