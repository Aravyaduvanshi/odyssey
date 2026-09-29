import unittest

from edge.api.main import app
from edge.navigation.engine import NavigationEngine
from edge.navigation.models import NavigationMode
from edge.sensors.models import SensorSample, SensorSource


class Phase1Tests(unittest.TestCase):
    def test_sensor_sample_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            SensorSample(
                timestamp_ns=1,
                ax=float("nan"),
                ay=0.0,
                az=9.80665,
                gx=0.0,
                gy=0.0,
                gz=0.0,
                gnss_available=False,
                sensor_source=SensorSource.SYNTHETIC,
            )

    def test_navigation_state_is_synthetic_and_blackout_removes_gnss(self) -> None:
        engine = NavigationEngine()
        sample = SensorSample(
            timestamp_ns=1_000,
            ax=0.0,
            ay=0.0,
            az=9.80665,
            gx=0.0,
            gy=0.0,
            gz=0.0,
            latitude=12.0,
            longitude=77.0,
            altitude=10.0,
            gnss_velocity=5.0,
            gnss_heading=0.5,
            gnss_accuracy=3.0,
            gnss_available=True,
            sensor_source=SensorSource.SYNTHETIC,
        )
        engine.set_gnss_blackout(True)
        state = engine._sample_to_state(sample)
        self.assertEqual(state.sensor_source, "SYNTHETIC")
        self.assertEqual(state.navigation_mode, NavigationMode.DEAD_RECKONING)
        self.assertIsNone(state.latitude)
        self.assertIsNone(state.ai_velocity)
        self.assertEqual(state.ai_velocity_confidence, 0.0)

    def test_health_status_and_models_routes_exist(self) -> None:
        paths = {route.path for route in app.routes}
        self.assertIn("/health", paths)
        self.assertIn("/status", paths)
        self.assertIn("/models", paths)
        self.assertIn("/ws/navigation", paths)


if __name__ == "__main__":
    unittest.main()
