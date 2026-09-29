import math
import os
import time
from pathlib import Path
from typing import Optional
import numpy as np

os.environ["KERAS_BACKEND"] = "torch"
import keras

from edge.models.base import BaseModelAdapter, ModelStatus, YawPrediction
from edge.models.yaw.preprocessing import MOUNTNET_FEATURES, MountNetPreprocessor


class MountNetAdapter(BaseModelAdapter):
    name: str = "MountNet_V2_CNN"

    def __init__(
        self,
        model_path: str = "models/yaw/mountnet_v2_baseline.keras",
        mean_path: str = "models/yaw/baseline_mean.npy",
        std_path: str = "models/yaw/baseline_std.npy",
    ) -> None:
        self.model_path = Path(model_path)
        self.mean_path = mean_path
        self.std_path = std_path
        self.preprocessor: Optional[MountNetPreprocessor] = None
        self._model = None
        self.status = ModelStatus.NOT_LOADED
        self.last_prediction: Optional[YawPrediction] = None

    def load(self) -> bool:
        """Load MountNet model weights and baseline preprocessing statistics once."""
        if self.status == ModelStatus.READY and self._model is not None:
            return True

        self.status = ModelStatus.LOADING
        t0 = time.perf_counter()
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"MountNet model file not found at {self.model_path}")

            self.preprocessor = MountNetPreprocessor(
                mean_path=self.mean_path,
                std_path=self.std_path,
            )

            # Load model without compilation (inference only)
            self._model = keras.models.load_model(
                str(self.model_path),
                compile=False,
                safe_mode=False,
            )

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
        """Perform warmup forward pass with dummy 100x6 input."""
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

    def predict(self, window: np.ndarray, timestamp_ns: int = 0) -> Optional[YawPrediction]:
        """
        Run MountNet inference on 100x6 phone-frame IMU window.
        Returns YawPrediction with mounting_yaw_rad and mounting_yaw_deg.
        """
        if self.status != ModelStatus.READY or self._model is None or self.preprocessor is None:
            return None

        t0 = time.perf_counter()
        try:
            # 1. Dimensions, ordering, and finite validation + normalization
            normalized = self.preprocessor.transform(window)

            # 2. Forward pass
            out = self._model(normalized)
            if hasattr(out, "detach"):
                out = out.detach().cpu().numpy()
            elif hasattr(out, "numpy"):
                out = out.numpy()
            else:
                out = np.asarray(out)

            # 3. Output shape (batch, 2): [sin_yaw, cos_yaw]
            vec = out[0]
            sin_yaw = float(vec[0])
            cos_yaw = float(vec[1])

            # 4. Normalize 2D vector if necessary
            norm = math.sqrt(sin_yaw * sin_yaw + cos_yaw * cos_yaw)
            if norm > 1e-6:
                sin_yaw /= norm
                cos_yaw /= norm

            # 5. Extract mounting yaw angle via atan2
            mounting_yaw_rad = math.atan2(sin_yaw, cos_yaw)
            mounting_yaw_deg = math.degrees(mounting_yaw_rad)

            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.last_inference_latency_ms = dt_ms
            self.last_inference_timestamp_ns = timestamp_ns

            prediction = YawPrediction(
                mounting_yaw_rad=mounting_yaw_rad,
                mounting_yaw_deg=mounting_yaw_deg,
                sin_yaw=sin_yaw,
                cos_yaw=cos_yaw,
                model_name=self.name,
                timestamp_ns=timestamp_ns,
                inference_latency_ms=dt_ms,
                quality=None,  # Not statistically calibrated yet
                is_stale=False,
            )
            self.last_prediction = prediction
            return prediction
        except Exception as e:
            self.status = ModelStatus.ERROR
            self.error_message = f"Inference failure: {e}"
            return None
