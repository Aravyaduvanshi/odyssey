import math
from pathlib import Path
from typing import Optional
import numpy as np
import yaml

from edge.coordinate.quaternion import Quaternion, from_rotation_vector
from edge.fusion.covariance import CovariancePropagator
from edge.fusion.error_state import (
    STATE_DIM, POSITION, VELOCITY, ATTITUDE, ACCEL_BIAS, GYRO_BIAS,
    POS_IDX, VEL_IDX, ATT_IDX, ACC_BIAS_IDX, GYRO_BIAS_IDX,
    skew,
)
from edge.fusion.gating import InnovationGating
from edge.fusion.measurements import (
    BaseMeasurement, MeasurementUpdateResult,
    GNSSPositionMeasurement, GNSSVelocityMeasurement, AIVelocityMeasurement,
    NHCMeasurement, ZUPTMeasurement,
)
from edge.ins.mechanization import StrapdownMechanization
from edge.ins.state import INSState

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "ekf.yaml"


class ErrorStateEKF:
    """
    15-State Error-State Extended Kalman Filter (ES-EKF).
    
    Nominal State:
      p_enu (3), v_enu (3), q_n_v (4), b_a (3), b_g (3), timestamp_ns
    Error State (15 elements):
      [delta_p (3), delta_v (3), delta_theta (3), delta_b_a (3), delta_b_g (3)]
    """

    def __init__(
        self,
        config: Optional[dict | Path | str] = None,
        initial_state: Optional[INSState] = None,
        initial_covariance: Optional[np.ndarray] = None,
        mechanization: Optional[StrapdownMechanization] = None,
        propagator: Optional[CovariancePropagator] = None,
    ) -> None:
        # Load configuration
        self.config = {}
        if isinstance(config, dict):
            self.config = config
        elif isinstance(config, (str, Path)):
            cfg_path = Path(config)
            if cfg_path.is_file():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
        elif CONFIG_PATH.is_file():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

        filter_cfg = self.config.get("filter", {})
        self.max_dt_sec = float(filter_cfg.get("max_valid_dt_sec", 1.0))
        self.min_dt_sec = float(filter_cfg.get("min_valid_dt_sec", 0.00001))

        self.state = initial_state or INSState.zero()
        self.mechanization = mechanization or StrapdownMechanization(max_dt_sec=self.max_dt_sec, min_dt_sec=self.min_dt_sec)
        self.propagator = propagator or CovariancePropagator(config_path=CONFIG_PATH)

        # Initialize covariance matrix P (15x15)
        if initial_covariance is not None:
            p_arr = np.asarray(initial_covariance, dtype=float)
            if p_arr.shape != (STATE_DIM, STATE_DIM):
                raise ValueError(f"initial_covariance must have shape ({STATE_DIM}, {STATE_DIM}), got {p_arr.shape}")
            self.P = p_arr.copy()
        else:
            self.P = self._load_initial_covariance()

        # Validate initial covariance
        self._validate_covariance()

    def _load_initial_covariance(self) -> np.ndarray:
        """Load initial diagonal variances from config file."""
        init_vars = self.config.get("initial_variances", {})
        p = np.zeros((STATE_DIM, STATE_DIM), dtype=float)

        pos_var = np.asarray(init_vars.get("position", [1.0, 1.0, 4.0]), dtype=float)
        vel_var = np.asarray(init_vars.get("velocity", [0.25, 0.25, 0.25]), dtype=float)
        att_var = np.asarray(init_vars.get("attitude", [0.01, 0.01, 0.04]), dtype=float)
        acc_var = np.asarray(init_vars.get("accel_bias", [0.0025, 0.0025, 0.0025]), dtype=float)
        gyro_var = np.asarray(init_vars.get("gyro_bias", [0.000025, 0.000025, 0.000025]), dtype=float)

        p[POSITION, POSITION] = np.diag(pos_var)
        p[VELOCITY, VELOCITY] = np.diag(vel_var)
        p[ATTITUDE, ATTITUDE] = np.diag(att_var)
        p[ACCEL_BIAS, ACCEL_BIAS] = np.diag(acc_var)
        p[GYRO_BIAS, GYRO_BIAS] = np.diag(gyro_var)
        return p

    def predict(
        self,
        accel_veh_or_state: np.ndarray | INSState,
        gyro_veh_or_accel: np.ndarray,
        timestamp_ns_or_gyro: Optional[int | np.ndarray] = None,
        dt: Optional[float] = None,
        timestamp_ns: Optional[int] = None,
        ins_state: Optional[INSState] = None,
    ) -> np.ndarray:
        """
        High-rate ES-EKF prediction step.
        Supports both signatures:
          predict(ins_state, accel_meas, gyro_meas, dt)
        and:
          predict(accel_veh, gyro_veh, timestamp_ns, dt=None)
        
        Performs:
          1. Propagate nominal INS state (if ins_state not already propagated).
          2. Compute continuous F and discrete Phi = I + F*dt + 0.5*(F@F)*dt^2.
          3. Compute discrete Qd ≈ G @ Qc @ G^T * dt.
          4. Predict covariance: P_new = Phi @ P @ Phi^T + Qd.
          5. Enforce numerical symmetry: P = 0.5 * (P_new + P_new^T).
        """
        # Determine calling pattern
        if isinstance(accel_veh_or_state, INSState):
            # Signature: predict(ins_state, accel_meas, gyro_meas, dt)
            self.state = accel_veh_or_state
            accel_veh = np.asarray(gyro_veh_or_accel, dtype=float).reshape(3)
            gyro_veh = np.asarray(timestamp_ns_or_gyro, dtype=float).reshape(3)
            step_dt = float(dt) if dt is not None else None
            step_ts = timestamp_ns
        else:
            accel_veh = np.asarray(accel_veh_or_state, dtype=float).reshape(3)
            gyro_veh = np.asarray(gyro_veh_or_accel, dtype=float).reshape(3)
            if isinstance(timestamp_ns_or_gyro, (int, float)):
                step_ts = int(timestamp_ns_or_gyro)
            else:
                step_ts = timestamp_ns
            step_dt = float(dt) if dt is not None else None

        # Validate step dt
        if step_dt is None:
            if step_ts is None:
                raise ValueError("Either dt or timestamp_ns must be provided for EKF prediction.")
            if not math.isfinite(step_ts):
                raise ValueError(f"Invalid timestamp_ns: {step_ts}")
            if self.state.timestamp_ns == 0:
                # Initialization sample
                self.state.timestamp_ns = step_ts
                return self.P
            step_dt = (step_ts - self.state.timestamp_ns) * 1e-9

        if not math.isfinite(step_dt):
            raise ValueError(f"Invalid dt (NaN or Inf): {step_dt}")
        if step_dt <= 0.0:
            raise ValueError(f"dt must be strictly positive, got {step_dt}")
        if step_dt > self.max_dt_sec:
            raise ValueError(f"dt {step_dt}s exceeds maximum allowed {self.max_dt_sec}s")
        if step_dt < self.min_dt_sec:
            raise ValueError(f"dt {step_dt}s is below minimum allowed {self.min_dt_sec}s")

        # 1. Propagate nominal state using INS mechanization
        # Note: If ins_state was already propagated externally, self.state is already updated
        prev_state = self.state
        self.state = self.mechanization.propagate(
            prev_state,
            accel_veh,
            gyro_veh,
            timestamp_ns=step_ts,
            dt=step_dt,
        )

        # 2. Compute linearized dynamics
        # f_vehicle = accel - accel_bias
        # omega_vehicle = gyro - gyro_bias
        f_veh = accel_veh - prev_state.accel_bias
        omega_veh = gyro_veh - prev_state.gyro_bias

        # R_nav_vehicle from nominal quaternion
        r_nav_veh = self.state.rotation_matrix

        # Continuous F matrix (15x15)
        f_cont = self.propagator.compute_continuous_f(r_nav_veh, f_veh, omega_veh)

        # Discretized state transition matrix Phi = I + F*dt + 0.5*(F@F)*dt^2
        phi = self.propagator.compute_discrete_transition(f_cont, step_dt)

        # Discrete process noise Qd ≈ G @ Qc @ G^T * dt
        q_d = self.propagator.compute_discrete_q(r_nav_veh, step_dt)

        # 3. Predict covariance: P = Phi @ P @ Phi^T + Qd
        p_pred = phi @ self.P @ phi.T + q_d

        # 4. Enforce numerical symmetry: P = 0.5 * (P + P^T)
        self.P = 0.5 * (p_pred + p_pred.T)

        # Numerical safety validation
        self._validate_covariance()

        return self.P

    def _validate_covariance(self) -> None:
        """Validate covariance matrix dimensions, finiteness, symmetry, and positive semi-definiteness."""
        if self.P.shape != (STATE_DIM, STATE_DIM):
            raise ValueError(f"P matrix shape mismatch: {self.P.shape} != ({STATE_DIM}, {STATE_DIM})")
        if not np.all(np.isfinite(self.P)):
            raise FloatingPointError("Covariance matrix P contains NaN or Inf.")

        # Check symmetry
        sym_diff = np.max(np.abs(self.P - self.P.T))
        if sym_diff > 1e-6:
            raise FloatingPointError(f"Covariance matrix P is asymmetric (max diff: {sym_diff})")

        # Check positive semi-definiteness
        eigvals = np.linalg.eigvalsh(self.P)
        min_eig = float(np.min(eigvals))
        if min_eig < -1e-6:
            raise ValueError(f"Covariance matrix P has significant negative eigenvalue: {min_eig}")

        # Ensure diagonal values remain positive
        for i in range(STATE_DIM):
            if self.P[i, i] < 0.0:
                raise ValueError(f"Negative variance encountered at diagonal index {i}: {self.P[i, i]}")

    def update(self, measurement: BaseMeasurement) -> MeasurementUpdateResult:
        """
        Measurement update step with innovation gating, Joseph-form covariance update,
        and error-state injection + reset.
        
        Mathematical sequence:
        1. Innovation: y = z - h(x_nominal)
        2. Innovation Covariance: S = H @ P @ H^T + R  (symmetric)
        3. Mahalanobis Gating: d^2 = y^T @ S^-1 @ y <= threshold
           If rejected: leave nominal state and P untouched.
        4. Kalman Gain: K = P @ H^T @ S^-1  (computed via np.linalg.solve)
        5. Error state correction: delta_x = K @ y
        6. Covariance update (Joseph Form):
           P_updated = (I - K @ H) @ P @ (I - K @ H)^T + K @ R @ K^T
        7. Error state injection:
           p <- p + delta_p
           v <- v + delta_v
           q <- (q_nominal * from_rotation_vector(delta_theta)).normalized()
           ba <- ba + delta_ba
           bg <- bg + delta_bg
        8. Error state reset:
           G_reset = diag(I, I, I - 0.5 * skew(delta_theta), I, I)
           P <- G_reset @ P_updated @ G_reset^T
           P <- 0.5 * (P + P^T)
        """
        y, H, R = measurement.evaluate(self.state)
        meas_type = getattr(measurement, "measurement_type", "measurement")
        threshold = getattr(measurement, "gating_threshold", 16.27)

        # 2. Innovation covariance: S = H @ P @ H^T + R
        S = H @ self.P @ H.T + R
        S = 0.5 * (S + S.T)  # Ensure strict numerical symmetry

        # 3. Innovation gating test (Mahalanobis distance)
        accepted, dist_sq = InnovationGating.test(y, S, threshold)
        if not accepted:
            reason = f"Outlier rejected: Mahalanobis d^2 ({dist_sq:.2f}) > threshold ({threshold:.2f})"
            return MeasurementUpdateResult(
                accepted=False,
                innovation=y,
                innovation_cov=S,
                mahalanobis_dist_sq=dist_sq,
                measurement_type=meas_type,
                rejection_reason=reason,
                kalman_gain=None,
            )

        # 4. Kalman Gain K = P @ H^T @ S^-1 using numerically stable np.linalg.solve
        # S @ K^T = H @ P => K = (np.linalg.solve(S, H @ self.P))^T
        try:
            if S.size == 1:
                s_scalar = float(S[0, 0])
                if s_scalar <= 0.0 or not math.isfinite(s_scalar):
                    return MeasurementUpdateResult(
                        accepted=False,
                        innovation=y,
                        innovation_cov=S,
                        mahalanobis_dist_sq=dist_sq,
                        measurement_type=meas_type,
                        rejection_reason=f"Invalid scalar innovation variance: {s_scalar}",
                        kalman_gain=None,
                    )
                K = (self.P @ H.T) / s_scalar
            else:
                k_t = np.linalg.solve(S, H @ self.P)
                K = k_t.T
        except np.linalg.LinAlgError as e:
            return MeasurementUpdateResult(
                accepted=False,
                innovation=y,
                innovation_cov=S,
                mahalanobis_dist_sq=dist_sq,
                measurement_type=meas_type,
                rejection_reason=f"Numerical singularity solving for Kalman gain: {e}",
                kalman_gain=None,
            )

        # 5. Compute error state correction
        delta_x = K @ y

        # 6. Joseph-form covariance update for numerical stability:
        # P_updated = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        I_KH = np.eye(STATE_DIM) - K @ H
        P_updated = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        P_updated = 0.5 * (P_updated + P_updated.T)

        # 7. Inject error state into nominal state
        self._inject_error_state(delta_x)

        # 8. Error-state reset:
        # Under right-multiplicative body-frame attitude error convention:
        # G_reset = diag(I_3, I_3, I_3 - 0.5 * skew(delta_theta), I_3, I_3)
        delta_theta = delta_x[ATTITUDE]
        g_reset = np.eye(STATE_DIM)
        g_reset[ATTITUDE, ATTITUDE] -= 0.5 * skew(delta_theta)
        P_reset = g_reset @ P_updated @ g_reset.T
        self.P = 0.5 * (P_reset + P_reset.T)

        self._validate_covariance()

        return MeasurementUpdateResult(
            accepted=True,
            innovation=y,
            innovation_cov=S,
            mahalanobis_dist_sq=dist_sq,
            measurement_type=meas_type,
            rejection_reason=None,
            kalman_gain=K,
        )

    def update_gnss_position(
        self,
        z_enu: np.ndarray,
        cov_enu: Optional[np.ndarray] = None,
        gating_threshold: Optional[float] = None,
    ) -> MeasurementUpdateResult:
        """
        Perform GNSS position measurement update in local ENU frame.
        
        Parameters:
            z_enu: 3D array [East, North, Up] in meters.
            cov_enu: Optional 3x3 covariance matrix. If None, loaded from configs/ekf.yaml.
            gating_threshold: Optional Chi-square threshold. If None, loaded from configs/ekf.yaml.
        """
        gnss_cfg = self.config.get("measurements", {}).get("gnss_position", {})
        if cov_enu is None:
            h_std = float(gnss_cfg.get("noise_std_horizontal", 2.5))
            v_std = float(gnss_cfg.get("noise_std_vertical", 4.5))
            cov_enu = np.diag([h_std ** 2, h_std ** 2, v_std ** 2])
        if gating_threshold is None:
            gating_threshold = float(gnss_cfg.get("gating_mahalanobis_threshold", 16.27))

        meas = GNSSPositionMeasurement(
            z_enu=z_enu,
            cov_enu=cov_enu,
            gating_threshold=gating_threshold,
        )
        return self.update(meas)

    def update_gnss_velocity(
        self,
        z_vel_enu: np.ndarray,
        cov_vel_enu: Optional[np.ndarray] = None,
        gating_threshold: Optional[float] = None,
    ) -> MeasurementUpdateResult:
        """
        Perform GNSS velocity measurement update in local ENU frame.
        
        Parameters:
            z_vel_enu: 3D array [v_East, v_North, v_Up] in m/s.
            cov_vel_enu: Optional 3x3 covariance matrix. If None, loaded from configs/ekf.yaml.
            gating_threshold: Optional Chi-square threshold. If None, loaded from configs/ekf.yaml.
        """
        gnss_vel_cfg = self.config.get("measurements", {}).get("gnss_velocity", {})
        if cov_vel_enu is None:
            h_std = float(gnss_vel_cfg.get("noise_std_horizontal", 0.15))
            v_std = float(gnss_vel_cfg.get("noise_std_vertical", 0.3))
            cov_vel_enu = np.diag([h_std ** 2, h_std ** 2, v_std ** 2])
        if gating_threshold is None:
            gating_threshold = float(gnss_vel_cfg.get("gating_mahalanobis_threshold", 16.27))

        meas = GNSSVelocityMeasurement(
            z_vel_enu=z_vel_enu,
            cov_vel_enu=cov_vel_enu,
            gating_threshold=gating_threshold,
        )
        return self.update(meas)

    def update_ai_velocity(
        self,
        v_forward_mps: float,
        variance_mps2: Optional[float] = None,
        sigma_mps: Optional[float] = None,
        gating_threshold: Optional[float] = None,
    ) -> MeasurementUpdateResult:
        """
        Perform AI forward velocity measurement update.
        
        Parameters:
            v_forward_mps: Predicted forward speed in m/s (from Velocity TCN).
            variance_mps2: Optional variance in (m/s)^2. If invalid/None, loaded from config.
            sigma_mps: Optional 1-sigma standard deviation in m/s.
            gating_threshold: Optional Chi-square threshold (1-DOF). If None, loaded from config.
        """
        ai_cfg = self.config.get("measurements", {}).get("ai_velocity", {})
        fallback_var = float(ai_cfg.get("fallback_variance_mps2", 0.5))
        min_var = float(ai_cfg.get("min_variance_mps2", 1e-4))
        max_var = float(ai_cfg.get("max_variance_mps2", 100.0))
        if gating_threshold is None:
            gating_threshold = float(ai_cfg.get("gating_mahalanobis_threshold", 10.83))

        meas = AIVelocityMeasurement(
            v_forward_mps=v_forward_mps,
            variance_mps2=variance_mps2,
            sigma_mps=sigma_mps,
            fallback_variance_mps2=fallback_var,
            min_variance_mps2=min_var,
            max_variance_mps2=max_var,
            gating_threshold=gating_threshold,
        )
        return self.update(meas)

    def update_nhc(
        self,
        cov_2x2: Optional[np.ndarray] = None,
        lateral_std_mps: Optional[float] = None,
        vertical_std_mps: Optional[float] = None,
        gating_threshold: Optional[float] = None,
    ) -> MeasurementUpdateResult:
        """
        Perform Non-Holonomic Constraints (NHC) measurement update in vehicle chassis frame.
        Assumes zero lateral and vertical velocity.
        """
        nhc_cfg = self.config.get("measurements", {}).get("nhc", {})
        if cov_2x2 is None and lateral_std_mps is None and vertical_std_mps is None:
            lat_std = float(nhc_cfg.get("noise_std_lateral", 0.1))
            vert_std = float(nhc_cfg.get("noise_std_vertical", 0.15))
            cov_2x2 = np.diag([lat_std ** 2, vert_std ** 2])
        if gating_threshold is None:
            gating_threshold = float(nhc_cfg.get("gating_mahalanobis_threshold", 13.82))

        meas = NHCMeasurement(
            cov_2x2=cov_2x2,
            lateral_std_mps=lateral_std_mps,
            vertical_std_mps=vertical_std_mps,
            gating_threshold=gating_threshold,
        )
        return self.update(meas)

    def update_zupt(
        self,
        cov_3x3: Optional[np.ndarray] = None,
        noise_std_mps: Optional[float] = None,
        gating_threshold: Optional[float] = None,
    ) -> MeasurementUpdateResult:
        """
        Perform Zero-Velocity Update (ZUPT) when confirmed stationary.
        Assumes zero 3D velocity in local ENU navigation frame.
        """
        zupt_cfg = self.config.get("measurements", {}).get("zupt", {})
        if cov_3x3 is None and noise_std_mps is None:
            std = float(zupt_cfg.get("noise_std", 0.05))
            cov_3x3 = np.eye(3, dtype=float) * (std ** 2)
        if gating_threshold is None:
            gating_threshold = float(zupt_cfg.get("gating_mahalanobis_threshold", 16.27))

        meas = ZUPTMeasurement(
            cov_3x3=cov_3x3,
            noise_std_mps=noise_std_mps,
            gating_threshold=gating_threshold,
        )
        return self.update(meas)

    def _inject_error_state(self, delta_x: np.ndarray) -> None:
        """Correct nominal state using estimated error state vector delta_x."""
        self.state.position_enu += delta_x[POSITION]
        self.state.velocity_enu += delta_x[VELOCITY]

        # Right-multiplicative body-frame attitude error injection:
        # q_true = q_nominal ⊗ δq, where δθ is in the vehicle/body frame.
        delta_theta = delta_x[ATTITUDE]
        delta_q = from_rotation_vector(delta_theta)
        self.state.quaternion = (self.state.quaternion * delta_q).normalized()

        self.state.accel_bias += delta_x[ACCEL_BIAS]
        self.state.gyro_bias += delta_x[GYRO_BIAS]

    @property
    def position_covariance(self) -> np.ndarray:
        return self.P[POSITION, POSITION].copy()

    @property
    def velocity_covariance(self) -> np.ndarray:
        return self.P[VELOCITY, VELOCITY].copy()

    @property
    def attitude_covariance(self) -> np.ndarray:
        return self.P[ATTITUDE, ATTITUDE].copy()

    @property
    def position_uncertainty_1sigma_m(self) -> float:
        """Return 3D 1-sigma position standard deviation (meters)."""
        pos_vars = np.diag(self.P)[POSITION]
        return float(math.sqrt(max(0.0, float(np.sum(pos_vars)))))

    @property
    def velocity_uncertainty_1sigma_mps(self) -> float:
        """Return 3D 1-sigma velocity standard deviation (m/s)."""
        vel_vars = np.diag(self.P)[VELOCITY]
        return float(math.sqrt(max(0.0, float(np.sum(vel_vars)))))

    @property
    def attitude_uncertainty_1sigma_deg(self) -> float:
        """Return 3D 1-sigma attitude error standard deviation (degrees)."""
        att_vars = np.diag(self.P)[ATTITUDE]
        return float(math.degrees(math.sqrt(max(0.0, float(np.sum(att_vars))))))
