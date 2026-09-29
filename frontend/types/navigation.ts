export type NavigationMode = "GNSS_INS" | "DEGRADED_GNSS_INS" | "DEAD_RECKONING" | "GNSS_RECOVERY" | "SIMULATION";

export type MotionState =
  | "STATIONARY"
  | "NORMAL_DRIVING"
  | "ACCELERATION"
  | "BRAKING"
  | "TURNING"
  | "POTHOLE"
  | "BUMP"
  | "IDLE_VIBRATION"
  | "PHONE_MOVEMENT"
  | "PHONE_REORIENTATION";

export interface NavigationState {
  timestamp: number;
  latitude: number | null;
  longitude: number | null;
  altitude: number | null;
  local_x: number;
  local_y: number;
  local_z: number;
  velocity_mps: number;
  velocity_kmph: number;
  heading_rad: number;
  heading_deg: number;
  roll_rad: number;
  pitch_rad: number;
  yaw_rad: number;
  velocity: number;
  heading: number;
  roll: number;
  pitch: number;
  yaw: number;
  gnss_health: number;
  ai_velocity: number | null;
  ai_velocity_confidence: number;
  ai_yaw: number | null;
  ai_yaw_confidence: number;
  mounting_yaw_rad?: number | null;
  mounting_yaw_deg?: number | null;
  ai_velocity_mps?: number | null;
  ai_velocity_kmh?: number | null;
  ai_velocity_normalized?: number | null;
  ai_velocity_log_variance?: number | null;
  ai_velocity_sigma_mps?: number | null;
  ai_velocity_variance_mps2?: number | null;
  ai_velocity_quality?: number | null;
  mountnet_status?: string;
  velocity_model_status?: string;
  mountnet_latency_ms?: number | null;
  velocity_model_latency_ms?: number | null;
  motion_state: MotionState;
  vibration_score: number;
  map_match_confidence: number;
  position_uncertainty: number;
  navigation_confidence: number;
  navigation_mode: NavigationMode;
  sensor_source: string;
  // Real Strapdown INS & Error-State EKF fields
  ins_position?: { E: number; N: number; U: number } | null;
  ins_velocity?: { vE: number; vN: number; vU: number } | null;
  ins_orientation?: { roll_rad: number; pitch_rad: number; yaw_rad: number } | null;
  ekf_position?: { E: number; N: number; U: number } | null;
  ekf_velocity?: { vE: number; vN: number; vU: number } | null;
  ekf_attitude?: { roll_rad: number; pitch_rad: number; yaw_rad: number } | null;
  position_covariance?: number[] | null;
  velocity_covariance?: number[] | null;
  attitude_covariance?: number[] | null;
  gyro_bias?: { bx: number; by: number; bz: number } | null;
  accelerometer_bias?: { bx: number; by: number; bz: number } | null;
  gnss_innovation?: number[] | null;
  gnss_measurement_accepted?: boolean;
  gnss_position_innovation?: number[] | null;
  gnss_position_mahalanobis_distance?: number | null;
  gnss_position_accepted?: boolean;
  gnss_position_rejection_reason?: string | null;
  gnss_velocity_innovation?: number[] | null;
  gnss_velocity_mahalanobis_distance?: number | null;
  gnss_velocity_accepted?: boolean;
  ai_velocity_innovation?: number | null;
  ai_velocity_mahalanobis_distance?: number | null;
  ai_velocity_accepted?: boolean;
  ai_velocity_rejection_reason?: string | null;
  nhc_active?: boolean;
  zupt_active?: boolean;
  dead_reckoning_distance?: number;
  gnss_outage_duration?: number;
  position_error_m?: number | null;
  drift_percent?: number | null;
  gnss_available?: boolean;
  gnss_last_seen_timestamp?: number | null;
  gnss_recovery_active?: boolean;
  gnss_recovery_measurements?: number;
  gnss_position_updates_enabled?: boolean;
  gnss_velocity_updates_enabled?: boolean;
  scenario_phase: string | null;
  raw_imu: ImuPacket;
  calibrated_imu: ImuPacket;
  vehicle_frame_imu: ImuPacket;
  calibration: CalibrationPacket;
  model_status: Record<string, string>;
  rates: Record<string, number | null>;
  ground_truth: { local_x: number; local_y: number; yaw_rad: number } | null;
  gnss_position: { latitude: number; longitude: number; accuracy_m: number } | null;
  map_label: string;
}

export interface ImuPacket {
  ax: number | null;
  ay: number | null;
  az: number | null;
  gx: number | null;
  gy: number | null;
  gz: number | null;
  mx?: number | null;
  my?: number | null;
  mz?: number | null;
  ACC_VEH_X?: number | null;
  ACC_VEH_Y?: number | null;
  ACC_VEH_Z?: number | null;
  GYRO_Yaw_corrected?: number | null;
  GYRO_Pitch_corrected?: number | null;
  GYRO_Roll_corrected?: number | null;
}

export interface CalibrationPacket {
  stage: string;
  message: string;
  alignment: {
    roll_rad: number;
    pitch_rad: number;
    yaw_rad: number;
    confidence: number;
    status: string;
    timestamp_ns: number;
    gyro_bias_x: number;
    gyro_bias_y: number;
    gyro_bias_z: number;
    magnetometer_consistent: boolean | null;
    accepted_samples: number;
  };
}

export interface OdysseyEvent {
  type: string;
  message: string;
  timestamp: string;
  payload: Record<string, unknown>;
}
