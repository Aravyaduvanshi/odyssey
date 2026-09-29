import csv
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

# Exact feature names in exact model input order
VELOCITY_FEATURES = [
    "ACC_VEH_X",
    "ACC_VEH_Y",
    "ACC_VEH_Z",
    "GYRO_Yaw_corrected",
    "GYRO_Pitch_corrected",
    "GYRO_Roll_corrected",
]

TARGET_MEAN = 26.57159258033888
TARGET_STD = 20.84706571807423


class VelocityTCNPreprocessor:
    def __init__(
        self,
        norm_path: str = "models/velocity/S2_velocity_TCN_uncertainty_v2_normalization.csv",
        target_norm_path: str = "models/velocity/S2_velocity_TCN_uncertainty_v2_target_normalization.csv",
    ) -> None:
        self.norm_path = Path(norm_path)
        self.target_norm_path = Path(target_norm_path)
        self.feature_means = np.zeros(6, dtype=np.float32)
        self.feature_stds = np.ones(6, dtype=np.float32)
        self.target_mean = TARGET_MEAN
        self.target_std = TARGET_STD
        self.load()

    def load(self) -> None:
        """Load normalization CSVs and verify against model contract."""
        if not self.norm_path.exists():
            raise FileNotFoundError(f"Velocity normalization file missing: {self.norm_path}")

        # Read input normalization CSV
        raw_params: Dict[str, Tuple[float, float]] = {}
        with open(self.norm_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feat = row["feature"].strip()
                mean_val = float(row["mean"])
                std_val = float(row["std"])
                raw_params[feat] = (mean_val, std_val)

        # Populate in EXACT VELOCITY_FEATURES order
        for i, feat in enumerate(VELOCITY_FEATURES):
            if feat not in raw_params:
                raise ValueError(f"Missing required velocity feature in normalization CSV: {feat}")
            m, s = raw_params[feat]
            self.feature_means[i] = m
            self.feature_stds[i] = s if s != 0.0 else 1.0

        # Read target normalization CSV if exists
        if self.target_norm_path.exists():
            with open(self.target_norm_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    param = row["parameter"].strip()
                    val = float(row["value"])
                    if param == "target_mean":
                        self.target_mean = val
                    elif param == "target_std":
                        self.target_std = val

    def transform(self, vehicle_window: np.ndarray, auto_gravity_compensate: bool = True) -> np.ndarray:
        """
        Validate and normalize vehicle-aligned IMU window [100, 6].
        Formula: (x - mean) / std in exact feature order.

        Transformation Pipeline:
        specific force
        → gravity compensation
        → linear acceleration
        → Velocity TCN

        Contract Safeguard:
        The Velocity TCN normalization model was trained on linear acceleration
        where ACC_VEH_Z is centered near 0 m/s² (mean = -0.018 m/s²).
        If the input contains specific force including Earth's gravity (mean ACC_VEH_Z > 5.0 m/s²),
        gravity compensation is applied (subtracting 9.80665 m/s²).
        If the input is already linear acceleration (mean ACC_VEH_Z <= 5.0 m/s²),
        gravity removal is NOT performed again to prevent double-subtraction.
        """
        arr = np.asarray(vehicle_window, dtype=np.float32)

        if arr.ndim == 2:
            if arr.shape != (100, 6):
                raise ValueError(f"Velocity TCN input window must be (100, 6), got {arr.shape}")
            arr_batch = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3:
            if arr.shape[1:] != (100, 6):
                raise ValueError(f"Velocity TCN batched window must be (B, 100, 6), got {arr.shape}")
            arr_batch = arr
        else:
            raise ValueError(f"Invalid Velocity TCN input dimensions: ndim={arr.ndim}")

        if not np.all(np.isfinite(arr_batch)):
            raise ValueError("Velocity TCN input contains NaN or Infinite values.")

        # Guard against double gravity removal
        arr_proc = arr_batch.copy()
        if auto_gravity_compensate and np.mean(arr_proc[:, :, 2]) > 5.0:
            arr_proc[:, :, 2] -= 9.80665

        normalized = (arr_proc - self.feature_means) / self.feature_stds
        return normalized

    def denormalize_velocity(self, normalized_vel: float) -> float:
        """
        Convert normalized target output back to physical forward velocity (m/s).
        velocity_mps = normalized * target_std + target_mean
        """
        return float(normalized_vel * self.target_std + self.target_mean)

    def denormalize_uncertainty(self, log_variance: float) -> Tuple[float, float]:
        """
        Convert normalized target log_variance to physical sigma (m/s) and variance (m/s)².
        sigma_mps = exp(0.5 * log_var) * target_std
        variance_mps2 = exp(log_var) * (target_std ** 2)
        """
        # Clamp log_variance to prevent overflow/underflow
        clamped_log_var = max(-20.0, min(10.0, log_variance))
        variance_norm = np.exp(clamped_log_var)
        sigma_norm = np.exp(0.5 * clamped_log_var)
        sigma_mps = float(sigma_norm * self.target_std)
        variance_mps2 = float(variance_norm * (self.target_std ** 2))
        return sigma_mps, variance_mps2
