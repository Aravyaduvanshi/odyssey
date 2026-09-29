import time
from typing import Any, Dict, Optional
import numpy as np

from edge.models.base import ModelStatus, VelocityPrediction, YawPrediction
from edge.models.velocity.adapter import VelocityTCNAdapter
from edge.models.yaw.adapter import MountNetAdapter

# Stale prediction thresholds in nanoseconds
YAW_STALE_THRESHOLD_NS = 15_000_000_000       # 15 seconds
VELOCITY_STALE_THRESHOLD_NS = 5_000_000_000    # 5 seconds


class ModelManager:
    """Central lifecycle and execution manager for AI models."""

    def __init__(
        self,
        mountnet_path: str = "models/yaw/mountnet_v2_baseline.keras",
        velocity_path: str = "models/velocity/S2_velocity_TCN_uncertainty_v2.keras",
        auto_load: bool = True,
    ) -> None:
        self.mountnet = MountNetAdapter(model_path=mountnet_path)
        self.velocity = VelocityTCNAdapter(model_path=velocity_path)
        self._last_yaw_prediction: Optional[YawPrediction] = None
        self._last_velocity_prediction: Optional[VelocityPrediction] = None
        self._total_pipeline_latency_ms: float = 0.0

        if auto_load:
            self.load_all()

    def load_all(self) -> Dict[str, bool]:
        """Load and warm up all available models once."""
        res_yaw = self.mountnet.load()
        res_vel = self.velocity.load()
        return {"yaw": res_yaw, "velocity": res_vel}

    def statuses(self) -> Dict[str, str]:
        """Return current status strings for all models."""
        return {
            "yaw": self.mountnet.status.value,
            "velocity": self.velocity.status.value,
        }

    def predict_yaw(self, phone_window: np.ndarray, timestamp_ns: int) -> Optional[YawPrediction]:
        """Execute MountNet inference on 100x6 phone-frame IMU window."""
        t0 = time.perf_counter()
        pred = self.mountnet.predict(phone_window, timestamp_ns)
        dt = (time.perf_counter() - t0) * 1000.0
        if pred:
            self._last_yaw_prediction = pred
        return pred

    def predict_velocity(self, vehicle_window: np.ndarray, timestamp_ns: int) -> Optional[VelocityPrediction]:
        """Execute Velocity TCN inference on 100x6 vehicle-aligned IMU window."""
        t0 = time.perf_counter()
        pred = self.velocity.predict(vehicle_window, timestamp_ns)
        dt = (time.perf_counter() - t0) * 1000.0
        if pred:
            self._last_velocity_prediction = pred
        return pred

    def get_effective_yaw_prediction(self, current_timestamp_ns: int) -> Optional[YawPrediction]:
        """Return the latest mounting yaw prediction, checking for staleness."""
        if self._last_yaw_prediction is None:
            return None

        age_ns = current_timestamp_ns - self._last_yaw_prediction.timestamp_ns
        is_stale = age_ns > YAW_STALE_THRESHOLD_NS
        if self._last_yaw_prediction.is_stale != is_stale:
            self._last_yaw_prediction = self._last_yaw_prediction.model_copy(update={"is_stale": is_stale})
        return self._last_yaw_prediction

    def get_effective_velocity_prediction(self, current_timestamp_ns: int) -> Optional[VelocityPrediction]:
        """Return the latest velocity prediction, checking for staleness."""
        if self._last_velocity_prediction is None:
            return None

        age_ns = current_timestamp_ns - self._last_velocity_prediction.timestamp_ns
        is_stale = age_ns > VELOCITY_STALE_THRESHOLD_NS
        if self._last_velocity_prediction.is_stale != is_stale:
            self._last_velocity_prediction = self._last_velocity_prediction.model_copy(update={"is_stale": is_stale})
        return self._last_velocity_prediction

    def diagnostics(self, current_timestamp_ns: int = 0) -> Dict[str, Any]:
        """Return detailed diagnostics for edge logging and UI inspection."""
        yaw_pred = self.get_effective_yaw_prediction(current_timestamp_ns)
        vel_pred = self.get_effective_velocity_prediction(current_timestamp_ns)

        return {
            "statuses": self.statuses(),
            "mountnet": {
                "status": self.mountnet.status.value,
                "load_latency_ms": round(self.mountnet.load_latency_ms, 2),
                "warmup_latency_ms": round(self.mountnet.warmup_latency_ms, 2),
                "last_inference_ms": round(self.mountnet.last_inference_latency_ms, 2),
                "last_yaw_deg": round(yaw_pred.mounting_yaw_deg, 2) if yaw_pred else None,
                "is_stale": yaw_pred.is_stale if yaw_pred else True,
                "error": self.mountnet.error_message,
            },
            "velocity": {
                "status": self.velocity.status.value,
                "load_latency_ms": round(self.velocity.load_latency_ms, 2),
                "warmup_latency_ms": round(self.velocity.warmup_latency_ms, 2),
                "last_inference_ms": round(self.velocity.last_inference_latency_ms, 2),
                "velocity_mps": round(vel_pred.velocity_mps, 3) if vel_pred else None,
                "velocity_kmh": round(vel_pred.velocity_kmh, 2) if vel_pred else None,
                "sigma_mps": round(vel_pred.sigma_mps, 3) if vel_pred and vel_pred.sigma_mps is not None else None,
                "is_stale": vel_pred.is_stale if vel_pred else True,
                "error": self.velocity.error_message,
            },
        }
