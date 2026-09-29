import asyncio
import math
import unittest

from edge.api.main import app
from edge.navigation.engine import NavigationEngine
from edge.navigation.models import NavigationMode
from edge.sensors.providers import MockSensorProvider
from starlette.testclient import TestClient


class EndToEndScenarioTests(unittest.TestCase):
    def test_e2e_full_scenario_pipeline(self) -> None:
        """
        Scenario:
        Start edge -> Start mock sensor -> Stream 10 Hz NavigationState ->
        Validate calibration completes -> vehicle-frame values appear ->
        Verify no NaN, no invalid timestamps, no fake AI values.
        """
        # 1. Start edge engine with mock provider
        provider = MockSensorProvider(rate_hz=100.0, seed=123, noise_std=0.0)
        engine = NavigationEngine(provider=provider)
        engine.start()

        self.assertTrue(engine.running)

        # 2. Run simulation generator for 250 steps (~2.5s simulated at 100 Hz = 25 published navigation states)
        async def run_pipeline() -> list:
            collected_states = []
            count = 0
            async for state in engine.states(publish_rate_hz=10.0):
                collected_states.append(state)
                count += 1
                if count >= 30:  # 3 seconds simulated at 10 Hz navigation rate
                    break
            return collected_states

        states = asyncio.run(run_pipeline())
        self.assertGreaterEqual(len(states), 25)

        # 3. Verify each NavigationState conforms strictly to specs
        last_t = -1
        for s in states:
            # No NaN or Inf anywhere
            self.assertFalse(math.isnan(s.local_x))
            self.assertFalse(math.isnan(s.local_y))
            self.assertFalse(math.isnan(s.velocity_mps))
            self.assertFalse(math.isnan(s.heading_rad))
            self.assertFalse(math.isinf(s.local_x))
            self.assertFalse(math.isinf(s.velocity_mps))

            # Timestamps are monotonic
            self.assertGreater(s.timestamp, last_t)
            last_t = s.timestamp

            # AI models status and validation
            self.assertIn(s.model_status["velocity"], ["READY", "LOADED", "NOT_LOADED"])
            self.assertIn(s.model_status["yaw"], ["READY", "LOADED", "NOT_LOADED"])
            if s.ai_velocity is not None:
                self.assertFalse(math.isnan(s.ai_velocity))
                self.assertFalse(math.isinf(s.ai_velocity))
            if s.ai_yaw is not None:
                self.assertFalse(math.isnan(s.ai_yaw))
                self.assertFalse(math.isinf(s.ai_yaw))

            # Vehicle frame and calibrated IMU exist
            self.assertIn("ax", s.raw_imu)
            self.assertIn("ax", s.calibrated_imu)
            self.assertIn("ax", s.vehicle_frame_imu)
            self.assertIsNotNone(s.vehicle_frame_imu["az"])

        # 4. Check WebSocket client connection via FastAPI test client
        with TestClient(app) as client:
            with client.websocket_connect("/ws/navigation") as ws:
                data = ws.receive_json()
                self.assertIn("timestamp", data)
                self.assertIn("velocity_mps", data)
                self.assertIn("heading_rad", data)
                self.assertIn("navigation_mode", data)
                self.assertIn("vehicle_frame_imu", data)
                self.assertIsNone(data["ai_velocity"])
                self.assertEqual(data["ai_velocity_confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
