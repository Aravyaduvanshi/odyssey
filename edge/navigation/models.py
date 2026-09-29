from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class NavigationMode(StrEnum):
    GNSS_INS = "GNSS_INS"
    DEGRADED_GNSS_INS = "DEGRADED_GNSS_INS"
    DEAD_RECKONING = "DEAD_RECKONING"
    GNSS_RECOVERY = "GNSS_RECOVERY"
    SIMULATION = "SIMULATION"


class MotionState(StrEnum):
    STATIONARY = "STATIONARY"
    NORMAL_DRIVING = "NORMAL_DRIVING"
    ACCELERATION = "ACCELERATION"
    BRAKING = "BRAKING"
    TURNING = "TURNING"
    POTHOLE = "POTHOLE"
    BUMP = "BUMP"
    IDLE_VIBRATION = "IDLE_VIBRATION"
    HIGH_VIBRATION = "HIGH_VIBRATION"
    PHONE_MOVEMENT = "PHONE_MOVEMENT"
    PHONE_REORIENTATION = "PHONE_REORIENTATION"


class NavigationState(BaseModel):
    timestamp: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    local_x: float
    local_y: float
    local_z: float
    velocity_mps: float
    velocity_kmph: float
    heading_rad: float  # Vehicle navigation heading (from GNSS/INS)
    heading_deg: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    velocity: float
    heading: float
    roll: float
    pitch: float
    yaw: float
    gnss_health: float = Field(..., ge=0.0, le=1.0)

    # Legacy AI fields (for backwards compatibility)
    ai_velocity: Optional[float] = None
    ai_velocity_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ai_yaw: Optional[float] = None
    ai_yaw_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Actual AI model integration fields
    mounting_yaw_rad: Optional[float] = None  # From MountNet V2 (phone-to-vehicle alignment)
    mounting_yaw_deg: Optional[float] = None
    ai_velocity_mps: Optional[float] = None    # From Velocity TCN (denormalized forward velocity)
    ai_velocity_kmh: Optional[float] = None
    ai_velocity_normalized: Optional[float] = None
    ai_velocity_log_variance: Optional[float] = None
    ai_velocity_sigma_mps: Optional[float] = None
    ai_velocity_variance_mps2: Optional[float] = None
    ai_velocity_quality: Optional[float] = None
    mountnet_status: str = "NOT_LOADED"
    velocity_model_status: str = "NOT_LOADED"
    mountnet_latency_ms: Optional[float] = None
    velocity_model_latency_ms: Optional[float] = None

    motion_state: MotionState
    vibration_score: float = Field(..., ge=0.0)
    map_match_confidence: float = Field(..., ge=0.0, le=1.0)
    position_uncertainty: float = Field(..., ge=0.0)
    navigation_confidence: float = Field(..., ge=0.0, le=1.0)
    navigation_mode: NavigationMode
    sensor_source: str
    scenario_phase: Optional[str] = None

    # Real Strapdown INS & Error-State EKF fields
    ins_position: Optional[dict[str, float]] = None       # {E, N, U}
    ins_velocity: Optional[dict[str, float]] = None       # {vE, vN, vU}
    ins_orientation: Optional[dict[str, float]] = None    # {roll_rad, pitch_rad, yaw_rad}
    ekf_position: Optional[dict[str, float]] = None       # {E, N, U}
    ekf_velocity: Optional[dict[str, float]] = None       # {vE, vN, vU}
    ekf_attitude: Optional[dict[str, float]] = None       # {roll_rad, pitch_rad, yaw_rad}
    position_covariance: Optional[list[float]] = None     # diag(P_pos)
    velocity_covariance: Optional[list[float]] = None     # diag(P_vel)
    attitude_covariance: Optional[list[float]] = None     # diag(P_att)
    gyro_bias: Optional[dict[str, float]] = None          # {bx, by, bz}
    accelerometer_bias: Optional[dict[str, float]] = None # {bx, by, bz}
    gnss_innovation: Optional[list[float]] = None
    gnss_measurement_accepted: bool = False
    gnss_position_innovation: Optional[list[float]] = None
    gnss_position_mahalanobis_distance: Optional[float] = None
    gnss_position_accepted: bool = False
    gnss_position_rejection_reason: Optional[str] = None
    gnss_velocity_innovation: Optional[list[float]] = None
    gnss_velocity_mahalanobis_distance: Optional[float] = None
    gnss_velocity_accepted: bool = False
    ai_velocity_innovation: Optional[float] = None
    ai_velocity_mahalanobis_distance: Optional[float] = None
    ai_velocity_accepted: bool = False
    ai_velocity_rejection_reason: Optional[str] = None
    nhc_active: bool = False
    zupt_active: bool = False
    dead_reckoning_distance: float = 0.0
    gnss_outage_duration: float = 0.0
    gnss_available: bool = True
    gnss_last_seen_timestamp: Optional[int] = None
    gnss_recovery_active: bool = False
    gnss_recovery_measurements: int = 0
    gnss_position_updates_enabled: bool = True
    gnss_velocity_updates_enabled: bool = True
    position_error_m: Optional[float] = None              # Only computed if ground truth available
    drift_percent: Optional[float] = None                 # Only computed if ground truth available

    raw_imu: dict[str, Optional[float]]
    calibrated_imu: dict[str, Optional[float]]
    vehicle_frame_imu: dict[str, Optional[float]]
    calibration: dict[str, object]
    model_status: dict[str, str]
    rates: dict[str, Optional[float]]
    ground_truth: Optional[dict[str, float]] = None
    gnss_position: Optional[dict[str, float]] = None
    map_label: str = "LOCAL TEST MAP"
