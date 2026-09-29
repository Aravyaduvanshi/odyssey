from typing import Optional
import numpy as np

from edge.motion.detector import VehicleMotionState


class NHCValidator:
    """
    Evaluates whether Non-Holonomic Constraints (NHC) are valid to apply.
    NHC assumes zero lateral and vertical velocity in vehicle chassis frame.
    Must be disabled during high turning, high vibration, potholes, or phone movement.
    """

    def __init__(
        self,
        max_yaw_rate_rads: float = 0.12,
        max_vibration_score: float = 3.0,
    ) -> None:
        self.max_yaw_rate_rads = max_yaw_rate_rads
        self.max_vibration_score = max_vibration_score

    def is_valid(
        self,
        motion_state: VehicleMotionState,
        vibration_score: float,
        yaw_rate_rads: float,
    ) -> bool:
        if motion_state in (
            VehicleMotionState.TURNING,
            VehicleMotionState.HIGH_VIBRATION,
            VehicleMotionState.PHONE_MOVEMENT,
            VehicleMotionState.POTHOLE,
            VehicleMotionState.BUMP,
        ):
            return False

        if vibration_score > self.max_vibration_score:
            return False

        # Suppress NHC if yaw rate is high (centripetal slip angle)
        if abs(yaw_rate_rads) > self.max_yaw_rate_rads:
            return False

        return True


class ZUPTValidator:
    """
    Evaluates whether Zero-Velocity Updates (ZUPT) are valid to apply.
    Only enabled when vehicle is stationary for consecutive steps,
    gyro is low, accel is close to gravity, and AI / GNSS speeds are consistent.
    """

    def __init__(
        self,
        required_consecutive_samples: int = 5,
        gyro_max_rads: float = 0.06,
        accel_diff_max_mps2: float = 0.4,
        max_speed_tolerance_mps: float = 0.4,
    ) -> None:
        self.required_consecutive_samples = required_consecutive_samples
        self.gyro_max_rads = gyro_max_rads
        self.accel_diff_max_mps2 = accel_diff_max_mps2
        self.max_speed_tolerance_mps = max_speed_tolerance_mps
        self._consecutive_stationary_count = 0

    @property
    def consecutive_stationary_count(self) -> int:
        return self._consecutive_stationary_count

    def evaluate(
        self,
        motion_state: VehicleMotionState,
        gyro_norm: float,
        accel_norm: float,
        gravity_mps2: float = 9.80665,
        ai_velocity_mps: Optional[float] = None,
        gnss_velocity_mps: Optional[float] = None,
    ) -> bool:
        if motion_state == VehicleMotionState.STATIONARY:
            self._consecutive_stationary_count += 1
        else:
            self._consecutive_stationary_count = 0
            return False

        if self._consecutive_stationary_count < self.required_consecutive_samples:
            return False

        # Gyro must be calm
        if gyro_norm > self.gyro_max_rads:
            return False

        # Accel must be near 1g
        if abs(accel_norm - gravity_mps2) > self.accel_diff_max_mps2:
            return False

        # If AI velocity is available, it must be near zero
        if ai_velocity_mps is not None and abs(ai_velocity_mps) > self.max_speed_tolerance_mps:
            return False

        # If GNSS velocity is available, it must be near zero
        if gnss_velocity_mps is not None and abs(gnss_velocity_mps) > self.max_speed_tolerance_mps:
            return False

        return True

    def reset(self) -> None:
        self._consecutive_stationary_count = 0
