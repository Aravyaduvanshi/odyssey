import math
from collections import deque
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field

from edge.sensors.models import SensorSample
from edge.sensors.scenario import GRAVITY_MPS2


class CalibrationStatus(StrEnum):
    INITIALIZING = "INITIALIZING"
    ESTIMATING_GRAVITY = "ESTIMATING_GRAVITY"
    ESTIMATING_BIAS = "ESTIMATING_BIAS"
    ESTIMATING_HEADING = "ESTIMATING_HEADING"
    VALIDATING = "VALIDATING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class AlignmentState(BaseModel):
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: CalibrationStatus = CalibrationStatus.INITIALIZING
    timestamp_ns: int = 0
    gyro_bias_x: float = 0.0
    gyro_bias_y: float = 0.0
    gyro_bias_z: float = 0.0
    magnetometer_consistent: bool | None = None
    accepted_samples: int = 0


class CalibrationState(BaseModel):
    alignment: AlignmentState
    stage: CalibrationStatus
    message: str


class CalibrationEngine:
    def __init__(
        self,
        gravity_threshold_mps2: float = 0.35,
        stable_samples_required: int = 80,
        moving_speed_threshold_mps: float = 2.0,
    ) -> None:
        self.gravity_threshold_mps2 = gravity_threshold_mps2
        self.stable_samples_required = stable_samples_required
        self.moving_speed_threshold_mps = moving_speed_threshold_mps
        self._accel_window: deque[np.ndarray] = deque(maxlen=stable_samples_required)
        self._gyro_window: deque[np.ndarray] = deque(maxlen=stable_samples_required)
        self._alignment = AlignmentState()

    @property
    def alignment(self) -> AlignmentState:
        return self._alignment

    def reset(self) -> None:
        """Reset calibration engine to initial state."""
        self._accel_window.clear()
        self._gyro_window.clear()
        self._alignment = AlignmentState()

    def update(self, sample: SensorSample) -> CalibrationState:
        accel = np.array([sample.ax, sample.ay, sample.az], dtype=float)
        gyro = np.array([sample.gx, sample.gy, sample.gz], dtype=float)
        accel_norm = float(np.linalg.norm(accel))

        # Reject samples if acceleration magnitude deviates from gravity or vehicle is accelerating
        stable = abs(accel_norm - GRAVITY_MPS2) <= self.gravity_threshold_mps2 and (sample.gnss_velocity or 0.0) < 0.5

        if stable:
            self._accel_window.append(accel)
            self._gyro_window.append(gyro)

        # Stage 1: Initialization / Gravity estimation (collecting stable samples)
        if len(self._accel_window) < self.stable_samples_required // 2:
            self._alignment = self._alignment.model_copy(
                update={
                    "status": CalibrationStatus.ESTIMATING_GRAVITY,
                    "timestamp_ns": sample.timestamp_ns,
                    "accepted_samples": len(self._accel_window),
                    "confidence": min(0.30, (len(self._accel_window) / self.stable_samples_required) * 0.3),
                }
            )
            return CalibrationState(alignment=self._alignment, stage=CalibrationStatus.ESTIMATING_GRAVITY, message="Collecting stable gravity samples for roll/pitch estimation")

        # Stage 2: Gyro bias estimation (window between 50% and 100%)
        if len(self._accel_window) < self.stable_samples_required:
            mean_gyro = np.mean(np.array(self._gyro_window), axis=0)
            self._alignment = self._alignment.model_copy(
                update={
                    "status": CalibrationStatus.ESTIMATING_BIAS,
                    "timestamp_ns": sample.timestamp_ns,
                    "accepted_samples": len(self._accel_window),
                    "gyro_bias_x": float(mean_gyro[0]),
                    "gyro_bias_y": float(mean_gyro[1]),
                    "gyro_bias_z": float(mean_gyro[2]),
                    "confidence": min(0.55, 0.30 + (len(self._accel_window) / self.stable_samples_required) * 0.25),
                }
            )
            return CalibrationState(alignment=self._alignment, stage=CalibrationStatus.ESTIMATING_BIAS, message="Estimating stationary gyro bias offsets")

        # Compute gravity-based roll and pitch from averaged stable accelerometer samples
        mean_accel = np.mean(np.array(self._accel_window), axis=0)
        mean_gyro = np.mean(np.array(self._gyro_window), axis=0)
        roll = math.atan2(mean_accel[1], mean_accel[2])
        pitch = math.atan2(-mean_accel[0], math.sqrt(mean_accel[1] ** 2 + mean_accel[2] ** 2))
        yaw = self._alignment.yaw_rad
        status = CalibrationStatus.ESTIMATING_HEADING
        confidence = 0.65
        message = "Gravity and gyro bias estimated; waiting for vehicle motion to establish GNSS course heading"

        # Stage 3: Vehicle direction estimation from GNSS course while vehicle is moving
        if sample.gnss_available and sample.gnss_heading is not None and (sample.gnss_velocity or 0.0) >= self.moving_speed_threshold_mps:
            yaw = sample.gnss_heading
            status = CalibrationStatus.VALIDATING
            confidence = 0.82
            message = "Vehicle forward heading estimated from reliable GNSS track"

        # Stage 4: Magnetometer consistency check
        mag_consistent = self._magnetometer_consistent(sample, yaw)
        if status == CalibrationStatus.VALIDATING:
            if mag_consistent is not False:
                status = CalibrationStatus.COMPLETE
                confidence = 0.92 if mag_consistent else 0.85
                message = "Phone-to-vehicle alignment complete (gravity + bias + GNSS course validated)"
            else:
                message = "Validating alignment: magnetometer shows local magnetic distortion"

        if self._alignment.status == CalibrationStatus.COMPLETE:
            # Maintain complete status once achieved unless reset
            status = CalibrationStatus.COMPLETE
            confidence = self._alignment.confidence
            message = "Phone-to-vehicle alignment complete"

        self._alignment = AlignmentState(
            roll_rad=roll,
            pitch_rad=pitch,
            yaw_rad=yaw,
            confidence=confidence,
            status=status,
            timestamp_ns=sample.timestamp_ns,
            gyro_bias_x=float(mean_gyro[0]),
            gyro_bias_y=float(mean_gyro[1]),
            gyro_bias_z=float(mean_gyro[2]),
            magnetometer_consistent=mag_consistent,
            accepted_samples=len(self._accel_window),
        )
        return CalibrationState(alignment=self._alignment, stage=status, message=message)

    def _magnetometer_consistent(self, sample: SensorSample, yaw_rad: float) -> bool | None:
        if sample.mx is None or sample.my is None:
            return None
        mag_yaw = math.atan2(sample.my, sample.mx)
        diff = math.atan2(math.sin(mag_yaw - yaw_rad), math.cos(mag_yaw - yaw_rad))
        return abs(diff) < math.radians(35.0)

