from pathlib import Path
from typing import Optional
import numpy as np
import yaml

from edge.fusion.error_state import (
    STATE_DIM, POSITION, VELOCITY, ATTITUDE, ACCEL_BIAS, GYRO_BIAS,
    POS_IDX, VEL_IDX, ATT_IDX, ACC_BIAS_IDX, GYRO_BIAS_IDX,
    skew,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "ekf.yaml"


class CovariancePropagator:
    """
    Computes error-state transition matrix and discrete process noise for 15-state ES-EKF.
    
    Units:
    - accelerometer_noise_density: m/s^2 / sqrt(Hz)
    - gyro_noise_density: rad/s / sqrt(Hz)
    - accelerometer_bias_random_walk: m/s^3 / sqrt(Hz)
    - gyro_bias_random_walk: rad/s^2 / sqrt(Hz)
    """

    def __init__(
        self,
        accel_noise_density: Optional[float] = None,
        gyro_noise_density: Optional[float] = None,
        accel_bias_random_walk: Optional[float] = None,
        gyro_bias_random_walk: Optional[float] = None,
        config_path: Optional[Path | str] = None,
    ) -> None:
        # Load from config file if available and arguments not explicitly given
        cfg_path = Path(config_path) if config_path else CONFIG_PATH
        cfg = {}
        if cfg_path.is_file():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        imu_noise_cfg = cfg.get("imu_noise", {})

        self.accel_noise_density = float(
            accel_noise_density
            if accel_noise_density is not None
            else imu_noise_cfg.get("accelerometer_noise_density", 0.08)
        )
        self.gyro_noise_density = float(
            gyro_noise_density
            if gyro_noise_density is not None
            else imu_noise_cfg.get("gyro_noise_density", 0.005)
        )
        self.accel_bias_random_walk = float(
            accel_bias_random_walk
            if accel_bias_random_walk is not None
            else imu_noise_cfg.get("accelerometer_bias_random_walk", 0.0002)
        )
        self.gyro_bias_random_walk = float(
            gyro_bias_random_walk
            if gyro_bias_random_walk is not None
            else imu_noise_cfg.get("gyro_bias_random_walk", 0.00005)
        )

        # Continuous power spectral densities (PSD)
        # S_a = sigma_a^2 [ (m/s^2)^2 / Hz ]
        # S_g = sigma_g^2 [ (rad/s)^2 / Hz ]
        # S_ba = sigma_ba^2 [ (m/s^3)^2 / Hz ]
        # S_bg = sigma_bg^2 [ (rad/s^2)^2 / Hz ]
        self.accel_psd = self.accel_noise_density ** 2
        self.gyro_psd = self.gyro_noise_density ** 2
        self.accel_bias_psd = self.accel_bias_random_walk ** 2
        self.gyro_bias_psd = self.gyro_bias_random_walk ** 2

    @property
    def continuous_qc(self) -> np.ndarray:
        """
        Continuous 12x12 process noise covariance matrix Qc:
        Qc = diag([sigma_a^2 * I_3, sigma_g^2 * I_3, sigma_ba^2 * I_3, sigma_bg^2 * I_3])
        """
        qc = np.zeros((12, 12), dtype=float)
        qc[0:3, 0:3] = np.eye(3) * self.accel_psd
        qc[3:6, 3:6] = np.eye(3) * self.gyro_psd
        qc[6:9, 6:9] = np.eye(3) * self.accel_bias_psd
        qc[9:12, 9:12] = np.eye(3) * self.gyro_bias_psd
        return qc

    def compute_continuous_f(
        self,
        r_nav_vehicle: np.ndarray,
        f_vehicle: np.ndarray,
        omega_vehicle: np.ndarray,
    ) -> np.ndarray:
        """
        Construct the 15x15 continuous-time error dynamics matrix F.
        
        Structure:
        F =
        [ 0     I       0                  0              0
          0     0   -R[f]x             -R                0
          0     0   -[ω]x                0              -I
          0     0     0                  0              0
          0     0     0                  0              0 ]
        
        where:
          R = r_nav_vehicle (3x3 rotation matrix from vehicle to navigation frame)
          [f]x = skew(f_vehicle) (3x3 skew-symmetric matrix of specific force)
          [ω]x = skew(omega_vehicle) (3x3 skew-symmetric matrix of angular rate)
        """
        f_matrix = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
        r = np.asarray(r_nav_vehicle, dtype=float).reshape((3, 3))
        f_veh = np.asarray(f_vehicle, dtype=float).reshape(3)
        omega_veh = np.asarray(omega_vehicle, dtype=float).reshape(3)

        # Block (0, 1): Position derivative by velocity error -> d(delta_p)/dt = delta_v
        f_matrix[POSITION, VELOCITY] = np.eye(3)

        # Block (1, 2): Velocity derivative by attitude error -> -R @ [f_vehicle]_x
        f_matrix[VELOCITY, ATTITUDE] = -r @ skew(f_veh)

        # Block (1, 3): Velocity derivative by accelerometer bias error -> -R
        f_matrix[VELOCITY, ACCEL_BIAS] = -r

        # Block (2, 2): Attitude derivative by attitude error -> -[omega_vehicle]_x
        f_matrix[ATTITUDE, ATTITUDE] = -skew(omega_veh)

        # Block (2, 4): Attitude derivative by gyro bias error -> -I (both in body frame)
        f_matrix[ATTITUDE, GYRO_BIAS] = -np.eye(3)

        # Rows 3 (accel bias) and 4 (gyro bias) are all zeros: bias random walk has zero drift
        return f_matrix

    def compute_noise_matrix_g(self, r_nav_vehicle: np.ndarray) -> np.ndarray:
        """
        Construct the 15x12 noise coupling matrix G:
        
        G =
        [ 0     0     0     0
          R     0     0     0
          0     I     0     0
          0     0     I     0
          0     0     0     I ]
        """
        g = np.zeros((15, 12), dtype=float)
        r = np.asarray(r_nav_vehicle, dtype=float).reshape((3, 3))

        # Velocity noise injection (from accelerometer): R @ wa
        g[VELOCITY, 0:3] = r

        # Attitude noise injection (from gyroscope): directly wg in body frame
        g[ATTITUDE, 3:6] = np.eye(3)

        # Accelerometer bias random walk: directly w_ba
        g[ACCEL_BIAS, 6:9] = np.eye(3)

        # Gyroscope bias random walk: directly w_bg
        g[GYRO_BIAS, 9:12] = np.eye(3)

        return g

    def compute_discrete_transition(
        self,
        f_continuous: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Discretize continuous F matrix using second-order Taylor expansion:
            Phi = I + F * dt + 0.5 * (F @ F) * dt^2
        """
        f = np.asarray(f_continuous, dtype=float)
        f_dt = f * dt
        return np.eye(STATE_DIM) + f_dt + 0.5 * (f_dt @ f_dt)

    def compute_discrete_q(
        self,
        r_nav_vehicle: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Compute discrete process noise covariance matrix Qd (15x15):
            Qd ≈ G @ Qc @ G^T * dt
        where G is the 15x12 noise injection matrix and Qc is the 12x12 spectral density.
        """
        g = self.compute_noise_matrix_g(r_nav_vehicle)
        qc = self.continuous_qc
        qd = g @ qc @ g.T * dt
        # Enforce numerical symmetry
        return 0.5 * (qd + qd.T)
