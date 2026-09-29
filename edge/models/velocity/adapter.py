import os
import time
from pathlib import Path
from typing import Optional
import numpy as np

os.environ["KERAS_BACKEND"] = "torch"
import keras

from edge.models.base import BaseModelAdapter, ModelStatus, VelocityPrediction
from edge.models.velocity.preprocessing import VELOCITY_FEATURES, VelocityTCNPreprocessor


def _patch_keras_lambda_output_shape():
    """Ensure Keras 3 Lambda layer shape inference works with torch backend."""
    orig_compute = getattr(keras.layers.Lambda, "_odyssey_patched", None)
    if orig_compute is not None:
        return
    old_compute = keras.layers.Lambda.compute_output_shape

    def patched_compute(self, input_shape):
        if self.name == "last_timestep" or (len(input_shape) == 3 and input_shape[1] == 100):
            return (input_shape[0], input_shape[-1])
        return old_compute(self, input_shape)

    keras.layers.Lambda.compute_output_shape = patched_compute
    keras.layers.Lambda._odyssey_patched = True


class VelocityTCNAdapter(BaseModelAdapter):
    name: str = "S2_velocity_TCN_uncertainty_v2"

    def __init__(
        self,
        model_path: str = "models/velocity/S2_velocity_TCN_uncertainty_v2.keras",
        norm_path: str = "models/velocity/S2_velocity_TCN_uncertainty_v2_normalization.csv",
        target_norm_path: str = "models/velocity/S2_velocity_TCN_uncertainty_v2_target_normalization.csv",
    ) -> None:
        self.model_path = Path(model_path)
        self.norm_path = norm_path
        self.target_norm_path = target_norm_path
        self.preprocessor: Optional[VelocityTCNPreprocessor] = None
        self._model = None
        self.status = ModelStatus.NOT_LOADED
        self.last_prediction: Optional[VelocityPrediction] = None

    def load(self) -> bool:
        """Load Velocity TCN model weights and normalization parameters once."""
        if self.status == ModelStatus.READY and self._model is not None:
            return True

        self.status = ModelStatus.LOADING
        t0 = time.perf_counter()
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Velocity model file not found at {self.model_path}")

            _patch_keras_lambda_output_shape()

            self.preprocessor = VelocityTCNPreprocessor(
                norm_path=self.norm_path,
                target_norm_path=self.target_norm_path,
            )

            # Load model with safe_mode=False and compile=False
            self._model = keras.models.load_model(
                str(self.model_path),
                compile=False,
                safe_mode=False,
            )

            # Patch lambda layer function in memory for cross-version bytecode compatibility
            for layer in self._model.layers:
                if layer.name == "last_timestep":
                    layer.function = lambda x: x[:, -1, :]

            # Verify contract: input shape must be (None, 100, 6)
            input_shape = self._model.input_shape
            if input_shape[1:] != (100, 6):
                raise ValueError(
                    f"MODEL CONTRACT ERROR: Expected input (None, 100, 6), got {input_shape}"
                )

            self.load_latency_ms = (time.perf_counter() - t0) * 1000.0
            self.status = ModelStatus.READY
            self.error_message = None
            self.warmup()
            return True
        except Exception as e:
            self.status = ModelStatus.ERROR
            self.error_message = str(e)
            self._model = None
            return False

    def warmup(self) -> bool:
        """Perform warmup forward pass."""
        if self._model is None or self.preprocessor is None:
            return False
        t0 = time.perf_counter()
        try:
            dummy = np.zeros((1, 100, 6), dtype=np.float32)
            norm_dummy = self.preprocessor.transform(dummy)
            _ = self._model(norm_dummy)
            self.warmup_latency_ms = (time.perf_counter() - t0) * 1000.0
            return True
        except Exception as e:
            self.status = ModelStatus.ERROR
            self.error_message = f"Warmup failed: {e}"
            return False

    def predict(self, vehicle_aligned_window: np.ndarray, timestamp_ns: int = 0) -> Optional[VelocityPrediction]:
        """
        Run Velocity TCN inference on 100x6 VEHICLE-ALIGNED IMU window.
        Returns VelocityPrediction with denormalized velocity (m/s and km/h) and uncertainty.
        """
        if self.status != ModelStatus.READY or self._model is None or self.preprocessor is None:
            return None

        t0 = time.perf_counter()
        try:
            # 1. Dimensions, ordering, and finite validation + normalization
            normalized = self.preprocessor.transform(vehicle_aligned_window)

            # 2. Forward pass
            out = self._model(normalized)
            if hasattr(out, "detach"):
                out = out.detach().cpu().numpy()
            elif hasattr(out, "numpy"):
                out = out.numpy()
            else:
                out = np.asarray(out)

            # 3. Extract [velocity_normalized, log_variance]
            vel_norm = float(out[0, 0])
            log_var = float(out[0, 1])

            # 4. Denormalize velocity to physical m/s
            velocity_mps = max(0.0, self.preprocessor.denormalize_velocity(vel_norm))
            velocity_kmh = velocity_mps * 3.6

            # 5. Convert uncertainty log_variance to physical sigma_mps
            sigma_mps, variance_mps2 = self.preprocessor.denormalize_uncertainty(log_var)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.last_inference_latency_ms = dt_ms
            self.last_inference_timestamp_ns = timestamp_ns

            prediction = VelocityPrediction(
                velocity_mps=velocity_mps,
                velocity_kmh=velocity_kmh,
                velocity_normalized=vel_norm,
                log_variance=log_var,
                sigma_mps=sigma_mps,
                variance_mps2=variance_mps2,
                model_name=self.name,
                timestamp_ns=timestamp_ns,
                inference_latency_ms=dt_ms,
                quality=None,  # Preserved as null until verified
                is_stale=False,
            )
            self.last_prediction = prediction
            return prediction
        except Exception as e:
            self.status = ModelStatus.ERROR
            self.error_message = f"Inference failure: {e}"
            return None
