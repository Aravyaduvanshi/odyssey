import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from edge.fusion.error_state import (
    STATE_DIM, POS_IDX, VEL_IDX, ATT_IDX, skew_symmetric
)
from edge.ins.state import INSState


@dataclass
class MeasurementUpdateResult:
    """Structured result of an ES-EKF measurement update."""
    accepted: bool
    innovation: np.ndarray
    innovation_cov: np.ndarray
    mahalanobis_dist_sq: float
    measurement_type: str = "gnss_position"
    rejection_reason: Optional[str] = None
    kalman_gain: Optional[np.ndarray] = None

    @property
    def mahalanobis_distance(self) -> float:
        return float(math.sqrt(max(0.0, float(self.mahalanobis_dist_sq))))

    @property
    def innovation_norm(self) -> float:
        return float(np.linalg.norm(self.innovation))


class BaseMeasurement(ABC):
    """Abstract base class for all ES-EKF measurement models."""

    @abstractmethod
    def evaluate(self, state: INSState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate predicted measurement h(x), innovation y = z - h(x),
        measurement Jacobian H (m x 15), and covariance R (m x m).
        Returns (innovation, H, R).
        """


class GNSSPositionMeasurement(BaseMeasurement):
    """
    3D GNSS Position in Local Tangent ENU frame.
    
    Measurement Model:
        z = p_nav + v_gnss
        h(x) = p_nominal = [p_E, p_N, p_U]^T
        y = z - h(x)
        H = [I_3  0_3  0_3  0_3  0_3]  (3x15)
        R = diag(sigma_E^2, sigma_N^2, sigma_U^2)  (3x3)
    """

    def __init__(
        self,
        z_enu: np.ndarray,
        cov_enu: Optional[np.ndarray] = None,
        horizontal_std_m: Optional[float] = None,
        vertical_std_m: Optional[float] = None,
        gating_threshold: float = 16.27,  # Chi-square 3-DOF @ 99.9%
    ) -> None:
        self.z = np.asarray(z_enu, dtype=float).reshape(3)
        if cov_enu is not None:
            self.r = np.asarray(cov_enu, dtype=float).reshape((3, 3))
        else:
            h_std = float(horizontal_std_m if horizontal_std_m is not None else 2.5)
            v_std = float(vertical_std_m if vertical_std_m is not None else 4.5)
            self.r = np.diag([h_std ** 2, h_std ** 2, v_std ** 2])
        self.gating_threshold = gating_threshold
        self.measurement_type = "gnss_position"

    def evaluate(self, state: INSState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = state.position_enu
        innovation = self.z - h
        h_matrix = np.zeros((3, STATE_DIM), dtype=float)
        h_matrix[:, POS_IDX] = np.eye(3)
        return innovation, h_matrix, self.r


class GNSSVelocityMeasurement(BaseMeasurement):
    """
    3D GNSS Velocity in Local Tangent ENU frame.
    
    Measurement Model:
        z = v_nav + v_noise
        h(x) = v_nominal = [v_E, v_N, v_U]^T
        y = z - h(x)
        H = [0_3  I_3  0_3  0_3  0_3]  (3x15)
        R = diag(sigma_E^2, sigma_N^2, sigma_U^2)  (3x3)
    """

    def __init__(
        self,
        z_vel_enu: np.ndarray,
        cov_vel_enu: Optional[np.ndarray] = None,
        horizontal_std_mps: Optional[float] = None,
        vertical_std_mps: Optional[float] = None,
        gating_threshold: float = 16.27,  # Chi-square 3-DOF @ 99.9%
    ) -> None:
        self.z = np.asarray(z_vel_enu, dtype=float).reshape(3)
        if cov_vel_enu is not None:
            self.r = np.asarray(cov_vel_enu, dtype=float).reshape((3, 3))
        else:
            h_std = float(horizontal_std_mps if horizontal_std_mps is not None else 0.15)
            v_std = float(vertical_std_mps if vertical_std_mps is not None else 0.3)
            self.r = np.diag([h_std ** 2, h_std ** 2, v_std ** 2])
        self.gating_threshold = gating_threshold
        self.measurement_type = "GNSS_VELOCITY"

    @staticmethod
    def speed_course_to_enu(speed_mps: float, course_rad: float, vertical_mps: float = 0.0) -> np.ndarray:
        """
        Convert ground speed (m/s) and course/heading (radians, clockwise from North)
        to navigation-frame ENU velocity [v_East, v_North, v_Up].
        
        v_East  = speed * sin(course)
        v_North = speed * cos(course)
        v_Up    = vertical_mps
        """
        v_e = float(speed_mps) * math.sin(course_rad)
        v_n = float(speed_mps) * math.cos(course_rad)
        v_u = float(vertical_mps)
        return np.array([v_e, v_n, v_u], dtype=float)

    @classmethod
    def from_speed_course(
        cls,
        speed_mps: float,
        course_rad: float,
        vertical_mps: float = 0.0,
        cov_vel_enu: Optional[np.ndarray] = None,
        horizontal_std_mps: Optional[float] = None,
        vertical_std_mps: Optional[float] = None,
        gating_threshold: float = 16.27,
    ) -> "GNSSVelocityMeasurement":
        """Construct GNSSVelocityMeasurement from ground speed and clockwise-from-North course."""
        z_enu = cls.speed_course_to_enu(speed_mps, course_rad, vertical_mps)
        return cls(
            z_vel_enu=z_enu,
            cov_vel_enu=cov_vel_enu,
            horizontal_std_mps=horizontal_std_mps,
            vertical_std_mps=vertical_std_mps,
            gating_threshold=gating_threshold,
        )

    def evaluate(self, state: INSState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = state.velocity_enu
        innovation = self.z - h
        h_matrix = np.zeros((3, STATE_DIM), dtype=float)
        h_matrix[:, VEL_IDX] = np.eye(3)
        return innovation, h_matrix, self.r


class AIVelocityMeasurement(BaseMeasurement):
    """
    AI Forward Vehicle Velocity measurement from Velocity TCN.
    The measurement corresponds strictly to the vehicle forward axis (X_veh).
    
    Measurement Model:
        z = v_veh_x + v_noise = e_x^T * R_v_n * v_nav + v_noise
        h(x) = e_x^T * R_vehicle_nav * v_nav = v_veh[0]
        y = z - h(x)  (scalar innovation)
        
    Error-State Jacobian (1x15):
        H_velocity = e_x^T * R_vehicle_nav  (columns 3:6)
        H_attitude = [0, -v_veh[2], v_veh[1]]  (columns 6:9, right-multiplicative body-frame)
        H_pos = 0, H_ba = 0, H_bg = 0
    """

    def __init__(
        self,
        v_forward_mps: float,
        variance_mps2: Optional[float] = None,
        sigma_mps: Optional[float] = None,
        fallback_variance_mps2: float = 0.5,
        min_variance_mps2: float = 1e-4,
        max_variance_mps2: float = 100.0,
        gating_threshold: float = 10.83,  # Chi-square 1-DOF @ 99.9%
    ) -> None:
        self.z = float(v_forward_mps)
        self.measurement_type = "AI_VELOCITY"
        self.gating_threshold = gating_threshold
        self.was_clamped = False
        self.used_fallback = False

        # Determine raw variance
        var = variance_mps2
        if var is None and sigma_mps is not None:
            var = float(sigma_mps) ** 2

        # Validity check: finite and strictly positive
        if var is None or not math.isfinite(var) or var <= 0.0:
            var = fallback_variance_mps2
            self.used_fallback = True

        if not math.isfinite(var) or var <= 0.0:
            var = 0.5

        # Clamping check to safe bounds
        if var < min_variance_mps2:
            var = min_variance_mps2
            self.was_clamped = True
        elif var > max_variance_mps2:
            var = max_variance_mps2
            self.was_clamped = True

        self.variance_mps2 = float(var)
        self.sigma_mps = math.sqrt(self.variance_mps2)
        self.r = np.array([[self.variance_mps2]], dtype=float)

    def evaluate(self, state: INSState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # R_nav_vehicle from nominal quaternion
        r_n_v = state.rotation_matrix
        # R_vehicle_nav = R_nav_vehicle.T
        r_v_n = r_n_v.T
        v_nav = state.velocity_enu
        v_veh = r_v_n @ v_nav

        # Forward velocity is index 0
        h = v_veh[0]
        innovation = np.array([self.z - h], dtype=float)

        # Jacobian w.r.t delta_v (cols 3:6) and delta_theta (cols 6:9)
        h_matrix = np.zeros((1, STATE_DIM), dtype=float)
        h_matrix[0, VEL_IDX] = r_v_n[0, :]
        # Right-multiplicative body-frame attitude error:
        # e_x^T [v_veh]_x = [0, -v_veh[2], v_veh[1]]
        h_matrix[0, ATT_IDX] = np.array([0.0, -v_veh[2], v_veh[1]], dtype=float)

        return innovation, h_matrix, self.r


class NHCMeasurement(BaseMeasurement):
    """
    Non-Holonomic Constraints (NHC) for ground vehicle:
    Assumes zero lateral velocity (v_veh_y ≈ 0) and zero vertical velocity (v_veh_z ≈ 0)
    in the vehicle chassis frame.
    
    Measurement Model:
        z_lateral  = 0 = e_y^T * R_v_n * v_nav + v_noise_y
        z_vertical = 0 = e_z^T * R_v_n * v_nav + v_noise_z
        h(x) = [v_veh[1], v_veh[2]]^T
        y = [0.0 - v_veh[1], 0.0 - v_veh[2]]^T = - [v_veh[1], v_veh[2]]^T
        
    Error-State Jacobian (2x15):
        H_vel = [e_y^T * R_v_n; e_z^T * R_v_n]  (cols 3:6)
        H_att = [v_veh]_x[1:3, :]               (cols 6:9, right-multiplicative body-frame)
              = [[v_veh[2], 0.0, -v_veh[0]],
                 [-v_veh[1], v_veh[0], 0.0]]
    """

    def __init__(
        self,
        cov_2x2: Optional[np.ndarray] = None,
        lateral_std_mps: Optional[float] = None,
        vertical_std_mps: Optional[float] = None,
        gating_threshold: float = 13.82,  # Chi-square 2-DOF @ 99.9%
    ) -> None:
        if cov_2x2 is not None:
            self.r = np.asarray(cov_2x2, dtype=float).reshape((2, 2))
        else:
            lat_std = float(lateral_std_mps if lateral_std_mps is not None else 0.1)
            vert_std = float(vertical_std_mps if vertical_std_mps is not None else 0.15)
            self.r = np.diag([lat_std ** 2, vert_std ** 2])
        self.gating_threshold = gating_threshold
        self.measurement_type = "NHC"

    def evaluate(self, state: INSState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_n_v = state.rotation_matrix
        r_v_n = r_n_v.T
        v_nav = state.velocity_enu
        v_veh = r_v_n @ v_nav

        # Innovation: target is zero lateral and vertical speed
        innovation = np.array([-v_veh[1], -v_veh[2]], dtype=float)

        h_matrix = np.zeros((2, STATE_DIM), dtype=float)
        # Velocity error Jacobian
        h_matrix[0, VEL_IDX] = r_v_n[1, :]
        h_matrix[1, VEL_IDX] = r_v_n[2, :]

        # Right-multiplicative body-frame attitude error Jacobian:
        # delta(v_veh) = R_v_n * delta(v_nav) + [v_veh]_x * delta_theta_body
        h_matrix[0, ATT_IDX] = np.array([v_veh[2], 0.0, -v_veh[0]], dtype=float)
        h_matrix[1, ATT_IDX] = np.array([-v_veh[1], v_veh[0], 0.0], dtype=float)

        return innovation, h_matrix, self.r


class ZUPTMeasurement(BaseMeasurement):
    """
    Zero-Velocity Update (ZUPT) when confirmed stationary.
    Constrains 3D velocity in local ENU navigation frame to zero: [0, 0, 0]^T.
    
    Measurement Model:
        z = [0, 0, 0]^T
        h(x) = v_nav = [v_E, v_N, v_U]^T
        y = z - h(x) = -v_nav
        H_vel = I_3 (cols 3:6), all other columns 0
    """

    def __init__(
        self,
        cov_3x3: Optional[np.ndarray] = None,
        noise_std_mps: Optional[float] = None,
        gating_threshold: float = 16.27,  # Chi-square 3-DOF @ 99.9%
    ) -> None:
        if cov_3x3 is not None:
            self.r = np.asarray(cov_3x3, dtype=float).reshape((3, 3))
        else:
            std = float(noise_std_mps if noise_std_mps is not None else 0.05)
            self.r = np.eye(3, dtype=float) * (std ** 2)
        self.gating_threshold = gating_threshold
        self.measurement_type = "ZUPT"

    def evaluate(self, state: INSState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Target velocity is [0, 0, 0] in navigation frame
        innovation = -np.asarray(state.velocity_enu, dtype=float).reshape(3)
        h_matrix = np.zeros((3, STATE_DIM), dtype=float)
        h_matrix[:, VEL_IDX] = np.eye(3)
        return innovation, h_matrix, self.r


class HeadingMeasurement(BaseMeasurement):
    """Yaw/Heading measurement (radians) in ENU navigation frame."""

    def __init__(
        self,
        heading_rad: float,
        variance_rad2: float = 0.01,
        gating_threshold: float = 10.83, # Chi-square 1-DOF @ 99.9%
    ) -> None:
        self.z = float(heading_rad)
        self.r = np.array([[max(1e-4, float(variance_rad2))]], dtype=float)
        self.gating_threshold = gating_threshold

    def evaluate(self, state: INSState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import math
        _, _, current_yaw = state.euler_rad
        # Wrap angle difference to [-pi, pi]
        diff = math.atan2(math.sin(self.z - current_yaw), math.cos(self.z - current_yaw))
        innovation = np.array([diff], dtype=float)
        h_matrix = np.zeros((1, STATE_DIM), dtype=float)
        # Yaw error in ENU corresponds to delta_theta_z (index 8)
        h_matrix[0, ATT_IDX.start + 2] = 1.0
        return innovation, h_matrix, self.r
