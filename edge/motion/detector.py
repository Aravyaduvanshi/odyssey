from dataclasses import dataclass
from enum import Enum
import math
import numpy as np

DEFAULT_GRAVITY = 9.80665


class VehicleMotionState(str, Enum):
    STATIONARY = "STATIONARY"
    NORMAL_DRIVING = "NORMAL_DRIVING"
    ACCELERATION = "ACCELERATION"
    BRAKING = "BRAKING"
    TURNING = "TURNING"
    POTHOLE = "POTHOLE"
    BUMP = "BUMP"
    HIGH_VIBRATION = "HIGH_VIBRATION"
    PHONE_MOVEMENT = "PHONE_MOVEMENT"


@dataclass
class MotionThresholds:
    stationary_speed_max_mps: float = 0.3
    stationary_gyro_max_rads: float = 0.05
    stationary_accel_diff_max_mps2: float = 0.4
    stationary_variance_max: float = 0.04
    acceleration_min_mps2: float = 0.6
    braking_min_mps2: float = -0.6
    turning_gyro_min_rads: float = 0.15
    turning_lat_accel_min_mps2: float = 1.0
    vibration_min_score: float = 3.5
    vibration_variance_max: float = 2.0
    phone_movement_gyro_max_rads: float = 2.0
    phone_movement_rot_accel_max_rads2: float = 25.0
    phone_movement_roll_pitch_rate_max_rads: float = 1.5

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "MotionThresholds":
        if cfg is None:
            return cls()
        mot = cfg.get("motion", {})
        stat = mot.get("stationary", {})
        acc = mot.get("acceleration", {})
        brk = mot.get("braking", {})
        turn = mot.get("turning", {})
        vib = mot.get("vibration", {})
        phone = mot.get("phone_movement", {})

        return cls(
            stationary_speed_max_mps=float(stat.get("speed_max_mps", 0.3)),
            stationary_gyro_max_rads=float(stat.get("gyro_max_rads", 0.05)),
            stationary_accel_diff_max_mps2=float(stat.get("accel_diff_max_mps2", 0.4)),
            stationary_variance_max=float(stat.get("accel_variance_max", 0.04)),
            acceleration_min_mps2=float(acc.get("min_mps2", 0.6)),
            braking_min_mps2=float(brk.get("min_mps2", -0.6)),
            turning_gyro_min_rads=float(turn.get("gyro_entry_rads", 0.15)),
            turning_lat_accel_min_mps2=float(turn.get("lat_accel_min_mps2", 1.0)),
            vibration_min_score=float(vib.get("score_threshold", 3.5)),
            vibration_variance_max=float(vib.get("variance_threshold", 2.0)),
            phone_movement_gyro_max_rads=float(phone.get("gyro_max_rads", 2.0)),
            phone_movement_rot_accel_max_rads2=float(phone.get("rot_accel_max_rads2", 25.0)),
            phone_movement_roll_pitch_rate_max_rads=float(phone.get("roll_pitch_rate_max_rads", 1.5)),
        )


class MotionDetector:
    """Deterministic, threshold-based vehicle and phone motion classifier."""

    def __init__(self, thresholds: MotionThresholds | None = None) -> None:
        self.cfg = thresholds or MotionThresholds()
        self._recent_accel_norms: list[float] = []
        self._last_gyro: np.ndarray | None = None
        self._last_timestamp_ns: int | None = None
        self._max_history = 20

    def detect(
        self,
        accel_veh: np.ndarray,
        gyro_veh: np.ndarray,
        estimated_speed_mps: float = 0.0,
        timestamp_ns: int | None = None,
    ) -> tuple[VehicleMotionState, float]:
        """
        Classify vehicle motion state and calculate vibration score.
        Returns (motion_state, vibration_score).
        """
        a = np.asarray(accel_veh, dtype=float).reshape(3)
        w = np.asarray(gyro_veh, dtype=float).reshape(3)

        accel_norm = float(np.linalg.norm(a))
        gyro_norm = float(np.linalg.norm(w))

        # Check rotational acceleration for phone movement
        rot_accel_norm = 0.0
        if self._last_gyro is not None and self._last_timestamp_ns is not None and timestamp_ns is not None:
            dt = (timestamp_ns - self._last_timestamp_ns) * 1e-9
            if dt > 1e-5:
                rot_accel_norm = float(np.linalg.norm((w - self._last_gyro) / dt))
        self._last_gyro = w.copy()
        self._last_timestamp_ns = timestamp_ns

        self._recent_accel_norms.append(accel_norm)
        if len(self._recent_accel_norms) > self._max_history:
            self._recent_accel_norms.pop(0)

        accel_var = float(np.var(self._recent_accel_norms)) if len(self._recent_accel_norms) >= 5 else 0.0
        vibration_score = abs(a[0]) + abs(a[1]) + max(0.0, abs(a[2] - DEFAULT_GRAVITY))

        # 1. Check phone movement (high angular velocity beyond vehicle chassis capabilities or high rot accel)
        if (
            gyro_norm > self.cfg.phone_movement_gyro_max_rads
            or abs(w[0]) > self.cfg.phone_movement_roll_pitch_rate_max_rads
            or abs(w[1]) > self.cfg.phone_movement_roll_pitch_rate_max_rads
            or rot_accel_norm > self.cfg.phone_movement_rot_accel_max_rads2
        ):
            return VehicleMotionState.PHONE_MOVEMENT, vibration_score

        # 2. Check high vibration / bump
        if vibration_score > self.cfg.vibration_min_score or accel_var > self.cfg.vibration_variance_max:
            return VehicleMotionState.HIGH_VIBRATION, vibration_score

        # 3. Check stationary condition:
        # Gyro low, accel norm near gravity, low variance, low speed
        is_stationary = (
            gyro_norm < self.cfg.stationary_gyro_max_rads
            and abs(accel_norm - DEFAULT_GRAVITY) < self.cfg.stationary_accel_diff_max_mps2
            and accel_var < self.cfg.stationary_variance_max
            and abs(estimated_speed_mps) < self.cfg.stationary_speed_max_mps
        )
        if is_stationary:
            return VehicleMotionState.STATIONARY, vibration_score

        # 4. Check turning:
        # High yaw rate or high lateral acceleration
        if abs(w[2]) > self.cfg.turning_gyro_min_rads or abs(a[1]) > self.cfg.turning_lat_accel_min_mps2:
            return VehicleMotionState.TURNING, vibration_score

        # 5. Check longitudinal acceleration / braking
        fwd_accel = a[0]
        if fwd_accel > self.cfg.acceleration_min_mps2:
            return VehicleMotionState.ACCELERATION, vibration_score
        if fwd_accel < self.cfg.braking_min_mps2:
            return VehicleMotionState.BRAKING, vibration_score

        # 6. Default smooth motion
        return VehicleMotionState.NORMAL_DRIVING, vibration_score
