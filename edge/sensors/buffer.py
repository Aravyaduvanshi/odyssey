from collections import deque
from typing import Optional, Tuple
import numpy as np

from edge.coordinate.frames import ImuVector, VehicleFrameTransform
from edge.sensors.models import SensorSample


class RollingIMUBuffer:
    """
    Fixed-size 100-sample rolling buffer resampled to exactly 10 Hz.
    Provides verified [100, 6] phone-frame and vehicle-aligned arrays for model inference.
    """

    def __init__(
        self,
        window_size: int = 100,
        resample_rate_hz: float = 10.0,
        mountnet_stride: int = 50,
        velocity_stride: int = 10,
    ) -> None:
        self.window_size = window_size
        self.resample_rate_hz = resample_rate_hz
        self.resample_interval_ns = int(1_000_000_000 / resample_rate_hz)
        self.mountnet_stride = mountnet_stride
        self.velocity_stride = velocity_stride

        self._buffer: deque[SensorSample] = deque(maxlen=window_size)
        self._last_raw_timestamp_ns: Optional[int] = None
        self._next_resample_timestamp_ns: Optional[int] = None

        self._samples_since_mountnet: int = 0
        self._samples_since_velocity: int = 0
        self._total_resampled_samples: int = 0

    @property
    def is_ready(self) -> bool:
        return len(self._buffer) >= self.window_size

    @property
    def sample_count(self) -> int:
        return len(self._buffer)

    def add_sample(self, sample: SensorSample) -> bool:
        """
        Accept an incoming high-rate or variable-rate sensor sample.
        Resamples down to exact 10 Hz epochs.
        Returns True if a new 10 Hz sample was committed to the rolling window.
        """
        t = sample.timestamp_ns

        # 1. Monotonicity & finite check
        if self._last_raw_timestamp_ns is not None:
            if t <= self._last_raw_timestamp_ns:
                # Reject duplicate or out-of-order timestamps
                return False

        self._last_raw_timestamp_ns = t

        # 2. Resample to 10 Hz epochs (100 ms step)
        if self._next_resample_timestamp_ns is None:
            self._next_resample_timestamp_ns = t
            self._buffer.append(sample)
            self._samples_since_mountnet += 1
            self._samples_since_velocity += 1
            self._total_resampled_samples += 1
            return True

        if t >= self._next_resample_timestamp_ns:
            self._buffer.append(sample)
            self._next_resample_timestamp_ns += self.resample_interval_ns
            # Protect against huge pauses or clock jumps
            if t > self._next_resample_timestamp_ns + 5 * self.resample_interval_ns:
                self._next_resample_timestamp_ns = t + self.resample_interval_ns

            self._samples_since_mountnet += 1
            self._samples_since_velocity += 1
            self._total_resampled_samples += 1
            return True

        return False

    def should_run_mountnet(self) -> bool:
        """Check if 50 samples (5.0s at 10 Hz) have passed since last MountNet inference."""
        return self.is_ready and self._samples_since_mountnet >= self.mountnet_stride

    def mark_mountnet_run(self) -> None:
        self._samples_since_mountnet = 0

    def should_run_velocity(self) -> bool:
        """Check if 10 samples (1.0s at 10 Hz) have passed since last Velocity TCN inference."""
        return self.is_ready and self._samples_since_velocity >= self.velocity_stride

    def mark_velocity_run(self) -> None:
        self._samples_since_velocity = 0

    def get_phone_window(self) -> np.ndarray:
        """
        Extract 100x6 phone-frame IMU window in MountNet order:
        [ax, ay, az, gx, gy, gz]
        """
        if not self.is_ready:
            raise RuntimeError(f"RollingIMUBuffer not ready: {len(self._buffer)}/{self.window_size}")

        window = np.zeros((self.window_size, 6), dtype=np.float32)
        for i, s in enumerate(self._buffer):
            window[i, 0] = s.ax
            window[i, 1] = s.ay
            window[i, 2] = s.az
            window[i, 3] = s.gx
            window[i, 4] = s.gy
            window[i, 5] = s.gz
        return window

    def get_vehicle_window(
        self,
        transform: VehicleFrameTransform,
        gyro_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        linear_accel: bool = True,
        gravity_mps2: float = 9.80665,
    ) -> np.ndarray:
        """
        Extract 100x6 vehicle-aligned IMU window in Velocity TCN order:
        [ACC_VEH_X, ACC_VEH_Y, ACC_VEH_Z, GYRO_Yaw_corrected, GYRO_Pitch_corrected, GYRO_Roll_corrected]

        When linear_accel=True (default), gravity is compensated on vehicle Z so that
        the window strictly conforms to the Velocity TCN training distribution
        (ACC_VEH_Z ≈ 0 m/s^2 for level stationary data).
        """
        if not self.is_ready:
            raise RuntimeError(f"RollingIMUBuffer not ready: {len(self._buffer)}/{self.window_size}")

        bx, by, bz = gyro_bias
        window = np.zeros((self.window_size, 6), dtype=np.float32)

        for i, s in enumerate(self._buffer):
            # 1. Bias-corrected IMU
            calibrated_imu = ImuVector(
                ax=s.ax,
                ay=s.ay,
                az=s.az,
                gx=s.gx - bx,
                gy=s.gy - by,
                gz=s.gz - bz,
            )
            # 2. Transform to vehicle-aligned features
            veh_imu = transform.to_vehicle_aligned_imu(calibrated_imu)
            if linear_accel:
                window[i, :] = veh_imu.as_ai_features(gravity_mps2=gravity_mps2)
            else:
                window[i, :] = veh_imu.as_features()

        return window

    def latest_timestamp_ns(self) -> int:
        if self._buffer:
            return self._buffer[-1].timestamp_ns
        return 0
