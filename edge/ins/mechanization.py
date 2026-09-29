import math
from typing import Optional
import numpy as np

from edge.coordinate.quaternion import Quaternion
from edge.ins.gravity import GravityModel
from edge.ins.state import INSState

DEFAULT_MAX_DT_SEC = 1.0     # 1.0 second max valid step
DEFAULT_MIN_DT_SEC = 0.00001 # 10 microseconds min valid step
DEFAULT_MAX_ACCEL_MPS2 = 500.0  # 500 m/s^2 (~50g) physical ceiling
DEFAULT_MAX_GYRO_RADPS = 100.0  # 100 rad/s (~5700 deg/s) physical ceiling


class StrapdownMechanization:
    """
    Strapdown Inertial Navigation System (INS) Mechanization in Local Tangent ENU.
    
    Frames:
    - Vehicle Body Frame: X = Forward, Y = Lateral-Left, Z = Vertical-Up
    - Navigation Frame: Local Tangent ENU (X = East, Y = North, Z = Up)
    - Gravity: g_nav = [0, 0, -9.80665] m/s^2 (downward in ENU)
    
    Equations of Motion:
    1. Bias Compensation:
       f_vehicle = accel_measurement - accel_bias
       omega_vehicle = gyro_measurement - gyro_bias
       (The accelerometer measures specific force, NOT directly linear acceleration)
    2. Attitude Quaternion Propagation:
       delta_theta = omega_vehicle * dt
       delta_q = Quaternion.from_rotation_vector(delta_theta)
       q_new = (q_old * delta_q).normalized()
    3. Specific Force Rotation:
       f_nav = R_nav_vehicle @ f_vehicle
    4. Gravity Addition:
       a_nav = f_nav + g_nav
    5. Velocity Propagation:
       v_new = v_old + a_nav * dt
    6. Position Propagation (using previous velocity):
       p_new = p_old + v_old * dt + 0.5 * a_nav * dt^2
    """

    def __init__(
        self,
        gravity_model: Optional[GravityModel] = None,
        max_dt_sec: float = DEFAULT_MAX_DT_SEC,
        min_dt_sec: float = DEFAULT_MIN_DT_SEC,
        max_accel_mps2: float = DEFAULT_MAX_ACCEL_MPS2,
        max_gyro_radps: float = DEFAULT_MAX_GYRO_RADPS,
    ) -> None:
        self.gravity = gravity_model or GravityModel()
        self.max_dt_sec = float(max_dt_sec)
        self.min_dt_sec = float(min_dt_sec)
        self.max_accel_limit = float(max_accel_mps2)
        self.max_gyro_limit = float(max_gyro_radps)

    def propagate(
        self,
        state: INSState,
        accel_veh: np.ndarray,
        gyro_veh: np.ndarray,
        timestamp_ns: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> INSState:
        """
        Execute high-rate strapdown propagation from previous state to next epoch.
        Accepts either timestamp_ns or explicit dt.
        Returns the updated INSState.
        """
        # Validate state quaternion
        norm_q = float(np.linalg.norm(state.quaternion.as_array()))
        if not math.isfinite(norm_q) or norm_q <= 1e-12:
            raise ValueError(f"Invalid state quaternion norm: {norm_q}")

        # Determine and validate dt
        if dt is not None:
            if not math.isfinite(dt):
                raise ValueError(f"Invalid dt (NaN or Inf): {dt}")
            if dt <= 0.0:
                raise ValueError(f"dt must be strictly positive, got {dt}")
            if dt > self.max_dt_sec:
                raise ValueError(f"dt {dt}s exceeds maximum allowed {self.max_dt_sec}s")
            if dt < self.min_dt_sec:
                raise ValueError(f"dt {dt}s is below minimum allowed {self.min_dt_sec}s")
            
            if timestamp_ns is None:
                new_timestamp_ns = state.timestamp_ns + int(round(dt * 1e9))
            else:
                new_timestamp_ns = timestamp_ns
        else:
            if timestamp_ns is None:
                raise ValueError("Either timestamp_ns or dt must be provided.")
            if not math.isfinite(timestamp_ns):
                raise ValueError(f"Invalid timestamp_ns: {timestamp_ns}")
            
            if state.timestamp_ns == 0:
                # Initialization step: stamp time, retain nominal state
                new_state = state.copy()
                new_state.timestamp_ns = int(timestamp_ns)
                return new_state

            dt = (timestamp_ns - state.timestamp_ns) * 1e-9
            if not math.isfinite(dt):
                raise ValueError(f"Invalid computed dt (NaN or Inf): {dt}")
            if dt <= 0.0:
                raise ValueError(f"Non-positive dt ({dt}s): timestamp_ns ({timestamp_ns}) <= state.timestamp_ns ({state.timestamp_ns})")
            if dt > self.max_dt_sec:
                raise ValueError(f"dt {dt}s exceeds maximum allowed {self.max_dt_sec}s")
            if dt < self.min_dt_sec:
                raise ValueError(f"dt {dt}s is below minimum allowed {self.min_dt_sec}s")
            new_timestamp_ns = int(timestamp_ns)

        # Numerical safety: validate sensor readings
        accel_raw = np.asarray(accel_veh, dtype=float).reshape(3)
        gyro_raw = np.asarray(gyro_veh, dtype=float).reshape(3)

        if not np.all(np.isfinite(accel_raw)):
            raise ValueError(f"Accelerometer measurement contains NaN or Inf: {accel_veh}")
        if not np.all(np.isfinite(gyro_raw)):
            raise ValueError(f"Gyroscope measurement contains NaN or Inf: {gyro_veh}")

        if np.any(np.abs(accel_raw) > self.max_accel_limit):
            raise ValueError(f"Accelerometer measurement exceeds physical ceiling ({self.max_accel_limit} m/s^2): {accel_raw}")
        if np.any(np.abs(gyro_raw) > self.max_gyro_limit):
            raise ValueError(f"Gyroscope measurement exceeds physical ceiling ({self.max_gyro_limit} rad/s): {gyro_raw}")

        # 1. Bias compensation in vehicle body frame
        f_vehicle = accel_raw - state.accel_bias
        omega_vehicle = gyro_raw - state.gyro_bias

        # 2. Attitude propagation using quaternion
        delta_theta = omega_vehicle * dt
        delta_q = Quaternion.from_rotation_vector(delta_theta)
        q_new = (state.quaternion * delta_q).normalized()

        # 3. Rotate specific force from vehicle frame into navigation frame
        R_nav_vehicle = q_new.rotation_matrix
        f_nav = R_nav_vehicle @ f_vehicle

        # 4. Add gravity vector (downward in ENU: [0, 0, -9.80665])
        g_nav = self.gravity.vector_enu()
        a_nav = f_nav + g_nav

        # 5. Propagate velocity: v_new = v_old + a_nav * dt
        v_new = state.velocity_enu + a_nav * dt

        # 6. Propagate position: p_new = p_old + v_old * dt + 0.5 * a_nav * dt^2
        # (IMPORTANT: uses previous velocity v_old)
        p_new = state.position_enu + state.velocity_enu * dt + 0.5 * a_nav * (dt ** 2)

        # Check numerical sanity of output
        if not np.all(np.isfinite(p_new)) or not np.all(np.isfinite(v_new)):
            raise FloatingPointError("Numerical overflow during INS state propagation.")

        return INSState(
            position_enu=p_new,
            velocity_enu=v_new,
            quaternion=q_new,
            accel_bias=state.accel_bias.copy(),
            gyro_bias=state.gyro_bias.copy(),
            timestamp_ns=new_timestamp_ns,
        )
