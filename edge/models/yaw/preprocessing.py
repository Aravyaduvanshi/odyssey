import os
from pathlib import Path
import numpy as np

# Exact feature names expected by MountNet
MOUNTNET_FEATURES = [
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
]


class MountNetPreprocessor:
    def __init__(
        self,
        mean_path: str = "models/yaw/baseline_mean.npy",
        std_path: str = "models/yaw/baseline_std.npy",
    ) -> None:
        self.mean_path = Path(mean_path)
        self.std_path = Path(std_path)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.load()

    def load(self) -> None:
        if not self.mean_path.exists() or not self.std_path.exists():
            raise FileNotFoundError(
                f"MountNet baseline artifacts missing: {self.mean_path}, {self.std_path}"
            )
        self._mean = np.load(str(self.mean_path)).astype(np.float32)
        self._std = np.load(str(self.std_path)).astype(np.float32)

        if self._mean.shape != (6,) or self._std.shape != (6,):
            raise ValueError(
                f"Invalid MountNet baseline shapes: mean={self._mean.shape}, std={self._std.shape}, expected (6,)"
            )
        # Avoid division by zero
        self._std[self._std == 0.0] = 1.0

    @property
    def is_ready(self) -> bool:
        return self._mean is not None and self._std is not None

    def transform(self, window: np.ndarray) -> np.ndarray:
        """
        Validate and normalize phone-frame IMU window [100, 6].
        Formula: (x - baseline_mean) / baseline_std
        """
        if not self.is_ready:
            raise RuntimeError("MountNetPreprocessor not loaded.")

        arr = np.asarray(window, dtype=np.float32)

        if arr.ndim == 2:
            if arr.shape != (100, 6):
                raise ValueError(f"MountNet input window must be (100, 6), got {arr.shape}")
            arr_batch = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3:
            if arr.shape[1:] != (100, 6):
                raise ValueError(f"MountNet batched window must be (B, 100, 6), got {arr.shape}")
            arr_batch = arr
        else:
            raise ValueError(f"Invalid MountNet input dimensions: ndim={arr.ndim}")

        if not np.all(np.isfinite(arr_batch)):
            raise ValueError("MountNet input contains NaN or Infinite values.")

        # Vectorized normalization
        normalized = (arr_batch - self._mean) / self._std
        return normalized
