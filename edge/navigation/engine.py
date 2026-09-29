import asyncio
import math
from pathlib import Path
import time
from collections.abc import AsyncIterator
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml

from edge.calibration.engine import CalibrationEngine
from edge.coordinate.frames import ImuVector, VehicleAlignedIMU, VehicleFrameTransform
from edge.events.bus import EventBus, EventType
from edge.fusion import (
    ErrorStateEKF,
    CovariancePropagator,
    GNSSPositionMeasurement,
    GNSSVelocityMeasurement,
    AIVelocityMeasurement,
    NHCMeasurement,
    ZUPTMeasurement,
    GNSSOutageManager,
    GNSSOutageState,
)
from edge.ins import EarthModel, GravityModel, StrapdownMechanization, INSState
from edge.models.manager import ModelManager
from edge.motion import (
    MotionDetector,
    MotionThresholds,
    NHCValidator,
    ZUPTValidator,
    VehicleMotionState,
)
from edge.navigation.models import MotionState, NavigationMode, NavigationState
from edge.sensors.buffer import RollingIMUBuffer
from edge.sensors.models import SensorSample, SensorSource
from edge.sensors.providers import MockSensorProvider, SensorProvider
from edge.sensors.time_sync import TimeSynchronizer


class NavigationEngine:
    """
    Central real-time multi-rate navigation orchestrator for OdysseyV1.
    Decoupled Pipelines:
    - High-Rate Pipeline (100 Hz): IMU -> Time Sync -> Alignment -> Strapdown INS -> ES-EKF Prediction & Constraints
    - AI Pipeline (10 Hz): Resampled IMU -> Rolling Buffer -> MountNet & Velocity TCN -> EKF Measurements
    """

    def __init__(self, provider: SensorProvider | None = None, event_bus: EventBus | None = None) -> None:
        self.provider = provider or MockSensorProvider()
        self.event_bus = event_bus or EventBus()
        self.time_sync = TimeSynchronizer(target_rate_hz=getattr(self.provider, "rate_hz", 100.0))
        self.rolling_buffer = RollingIMUBuffer(window_size=100, resample_rate_hz=10.0, mountnet_stride=50, velocity_stride=10)
        self.calibration = CalibrationEngine()
        self.models = ModelManager(auto_load=True)

        # Core Navigation Mathematics
        self.gravity = GravityModel(9.80665)
        self.ins_mechanization = StrapdownMechanization(gravity_model=self.gravity)
        self.propagator = CovariancePropagator()
        self.ekf = ErrorStateEKF(mechanization=self.ins_mechanization, propagator=self.propagator)
        self.standalone_ins_state = INSState.zero()

        # Motion & Constraint Detectors
        nav_config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "navigation.yaml"
        nav_cfg = {}
        if nav_config_path.is_file():
            with open(nav_config_path, "r", encoding="utf-8") as f:
                nav_cfg = yaml.safe_load(f) or {}
        self.motion_detector = MotionDetector(MotionThresholds.from_config(nav_cfg))
        mot_cfg = nav_cfg.get("motion", {})
        nhc_cfg = mot_cfg.get("nhc", {})
        stat_cfg = mot_cfg.get("stationary", {})
        self.nhc_validator = NHCValidator(
            max_yaw_rate_rads=float(nhc_cfg.get("max_yaw_rate_rads", 0.12)),
            max_vibration_score=float(nhc_cfg.get("max_vibration_score", 3.0)),
        )
        self.zupt_validator = ZUPTValidator(
            required_consecutive_samples=int(stat_cfg.get("consecutive_samples", 5)),
            gyro_max_rads=float(stat_cfg.get("gyro_max_rads", 0.05)),
            accel_diff_max_mps2=float(stat_cfg.get("accel_diff_max_mps2", 0.4)),
        )

        # GNSS Outage & Drift Manager
        outage_cfg = nav_cfg.get("gnss_outage", {})
        self.outage_mgr = GNSSOutageManager(
            event_bus=self.event_bus,
            loss_timeout_sec=float(outage_cfg.get("loss_timeout_sec", 1.0)),
            loss_persistence_samples=int(outage_cfg.get("loss_persistence_samples", 3)),
            recovery_persistence_samples=int(outage_cfg.get("recovery_persistence_samples", 5)),
            accuracy_threshold_degraded_m=float(outage_cfg.get("accuracy_threshold_degraded_m", 5.0)),
            max_acceptable_outage_gap_sec=float(outage_cfg.get("max_acceptable_outage_gap_sec", 300.0)),
        )

        # State tracking
        self.running = False
        self.gnss_blackout = False
        self._should_process_gnss = True
        self._last_state: NavigationState | None = None
        self._last_sample: SensorSample | None = None
        self._origin_lat: float | None = None
        self._origin_lon: float | None = None
        self._origin_alt: float = 0.0
        self._session_started_ns: int | None = None
        self._last_emit_monotonic_ns: int | None = None
        self._navigation_rate_hz: float | None = None
        self._calibration_complete_event_sent = False

        # Constraint & Innovation caches
        self.enable_measurement_updates = True
        self.enable_gnss_position_update = True   # Step 4: GNSS position measurement update active
        self.enable_gnss_velocity_update = True   # Step 5: GNSS velocity measurement update active
        self.enable_ai_velocity_update = True     # Step 6: AI velocity measurement update active
        self.enable_nhc_update = True             # Step 7: NHC measurement update active
        self.enable_zupt_update = True            # Step 7: ZUPT measurement update active
        self._nhc_active = False
        self._zupt_active = False
        self._prev_zupt_active = False
        self._last_motion_state: Optional[VehicleMotionState] = None
        self._last_vibration_score: float = 0.0
        self._last_gnss_innovation: Optional[list[float]] = None
        self._last_gnss_accepted = False
        self._last_gnss_mahalanobis_dist: Optional[float] = None
        self._last_gnss_rejection_reason: Optional[str] = None
        self._last_gnss_vel_innovation: Optional[list[float]] = None
        self._last_gnss_vel_accepted = False
        self._last_gnss_vel_mahalanobis_dist: Optional[float] = None
        self._last_gnss_vel_rejection_reason: Optional[str] = None
        self._last_ai_innovation: Optional[float] = None
        self._last_ai_accepted = False
        self._last_ai_mahalanobis_dist: Optional[float] = None
        self._last_ai_rejection_reason: Optional[str] = None

        # GNSS fix de-duplication tracking
        self._last_applied_gnss_fix_id: Optional[str] = None
        self._last_applied_gnss_coords: Optional[Tuple[float, float, float]] = None
        self._gnss_updates_attempted: int = 0
        self._gnss_updates_accepted: int = 0
        self._gnss_repeated_fixes_ignored: int = 0

        # AI Velocity prediction statistics tracking
        self._ai_predictions_attempted: int = 0
        self._ai_predictions_accepted: int = 0
        self._ai_predictions_rejected: int = 0
        self._ai_prediction_values: List[float] = []

        # NHC & ZUPT diagnostic counters
        self._nhc_opportunities: int = 0
        self._nhc_accepted: int = 0
        self._nhc_rejected: int = 0
        self._nhc_suppressed: int = 0
        self._zupt_opportunities: int = 0
        self._zupt_accepted: int = 0
        self._zupt_rejected: int = 0

        # EKF & Outage Diagnostics
        self._innovation_rejections: int = 0
        self._covariance_numerical_failures: int = 0
        self._state_resets: int = 0
        self._outage_transitions: int = 0
        self._recovery_transitions: int = 0
        self._last_outage_state: Optional[str] = None

        # Motion duration statistics (in seconds)
        self._motion_durations: Dict[str, float] = {
            "stationary": 0.0,
            "moving": 0.0,
            "high_vibration": 0.0,
            "turning": 0.0,
        }

    def start(self) -> None:
        if not self.running:
            self.running = True
            self._session_started_ns = time.monotonic_ns()
            self.event_bus.publish(EventType.SYSTEM_STARTED, "OdysseyV1 edge engine online")
            self.event_bus.publish(EventType.SENSOR_CONNECTED, "Mock deterministic sensor provider connected")
            self.event_bus.publish(EventType.CALIBRATION_STARTED, "Initial calibration started")
            if self.models.mountnet.status.value == "READY" and self.models.velocity.status.value == "READY":
                self.event_bus.publish(EventType.MODEL_LOADED, "MountNet V2 and Velocity TCN models active")

    def reset_calibration(self) -> None:
        self.calibration.reset()
        self._calibration_complete_event_sent = False
        self.event_bus.publish(EventType.CALIBRATION_STARTED, "Calibration restarted")

    def stop(self) -> None:
        if self.running:
            self.running = False
            self.event_bus.publish(EventType.SESSION_STOPPED, "Navigation session stopped")

    def set_gnss_blackout(self, enabled: bool) -> None:
        self.gnss_blackout = enabled

    def last_state(self) -> NavigationState | None:
        return self._last_state

    def last_sample(self) -> SensorSample | None:
        return self._last_sample

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return comprehensive diagnostics for evaluation, telemetry, and reporting."""
        ai_preds = self._ai_prediction_values
        ai_arr = np.array(ai_preds, dtype=float) if ai_preds else np.array([], dtype=float)
        return {
            "gnss": {
                "updates_attempted": self._gnss_updates_attempted,
                "updates_accepted": self._gnss_updates_accepted,
                "updates_rejected": max(0, self._gnss_updates_attempted - self._gnss_updates_accepted),
                "repeated_fixes_ignored": self._gnss_repeated_fixes_ignored,
                "outage_transitions": self._outage_transitions,
                "recovery_transitions": self._recovery_transitions,
            },
            "ai_velocity": {
                "predictions_attempted": self._ai_predictions_attempted,
                "predictions_accepted": self._ai_predictions_accepted,
                "predictions_rejected": self._ai_predictions_rejected,
                "acceptance_percent": (100.0 * self._ai_predictions_accepted / self._ai_predictions_attempted) if self._ai_predictions_attempted > 0 else 0.0,
                "min": float(np.min(ai_arr)) if len(ai_arr) > 0 else 0.0,
                "max": float(np.max(ai_arr)) if len(ai_arr) > 0 else 0.0,
                "mean": float(np.mean(ai_arr)) if len(ai_arr) > 0 else 0.0,
                "median": float(np.median(ai_arr)) if len(ai_arr) > 0 else 0.0,
                "p5": float(np.percentile(ai_arr, 5)) if len(ai_arr) > 0 else 0.0,
                "p95": float(np.percentile(ai_arr, 95)) if len(ai_arr) > 0 else 0.0,
                "prediction_values": ai_preds,
            },
            "nhc": {
                "opportunities": self._nhc_opportunities,
                "accepted": self._nhc_accepted,
                "rejected": self._nhc_rejected,
                "suppressed": self._nhc_suppressed,
            },
            "zupt": {
                "opportunities": self._zupt_opportunities,
                "accepted": self._zupt_accepted,
                "rejected": self._zupt_rejected,
            },
            "ekf": {
                "innovation_rejections": self._innovation_rejections,
                "covariance_numerical_failures": self._covariance_numerical_failures,
                "state_resets": self._state_resets,
            },
            "motion": self._motion_durations,
        }

    def step(self, sample: SensorSample) -> Optional[NavigationState]:
        """
        Execute one complete step of the OdysseyV1 navigation pipeline:
        time sync -> buffer -> mountnet -> alignment -> high-rate INS/EKF/NHC/ZUPT -> velocity TCN -> state output.
        """
        if not self.time_sync.accept(sample):
            return None
        self._last_sample = sample
        self.rolling_buffer.add_sample(sample)

        # 1. MountNet inference on 100-sample phone window (stride 50 = 5 seconds)
        if self.rolling_buffer.should_run_mountnet():
            phone_win = self.rolling_buffer.get_phone_window()
            self.models.predict_yaw(phone_win, sample.timestamp_ns)
            self.rolling_buffer.mark_mountnet_run()

        # Active Phone -> Vehicle Alignment
        calibration_state = self.calibration.update(sample)
        alignment = calibration_state.alignment

        effective_yaw = self.models.get_effective_yaw_prediction(sample.timestamp_ns)
        mounting_yaw_rad = effective_yaw.mounting_yaw_rad if effective_yaw and not effective_yaw.is_stale else 0.0

        transform = VehicleFrameTransform.from_alignment(
            alignment.roll_rad,
            alignment.pitch_rad,
            mounting_yaw_rad,
        )

        # High-rate INS Propagation & EKF Prediction (100 Hz step)
        self._step_high_rate(sample, transform, alignment)

        # 2. Velocity TCN inference on 100-sample VEHICLE-ALIGNED window (stride 10 = 1 second)
        if self.rolling_buffer.should_run_velocity():
            gyro_bias = (alignment.gyro_bias_x, alignment.gyro_bias_y, alignment.gyro_bias_z)
            veh_win = self.rolling_buffer.get_vehicle_window(transform, gyro_bias)
            vel_pred = self.models.predict_velocity(veh_win, sample.timestamp_ns)
            self.rolling_buffer.mark_velocity_run()
            if vel_pred and not vel_pred.is_stale and self.enable_ai_velocity_update and self.enable_measurement_updates:
                self._ai_predictions_attempted += 1
                self._ai_prediction_values.append(vel_pred.velocity_mps)
                ai_cfg = self.ekf.config.get("measurements", {}).get("ai_velocity", {})
                min_speed = float(ai_cfg.get("min_speed_threshold_mps", 0.0))
                max_speed = float(ai_cfg.get("max_speed_threshold_mps", 80.0))

                if not math.isfinite(vel_pred.velocity_mps) or vel_pred.velocity_mps < min_speed or vel_pred.velocity_mps > max_speed:
                    self._last_ai_accepted = False
                    self._ai_predictions_rejected += 1
                    self._last_ai_rejection_reason = f"Velocity {vel_pred.velocity_mps:.2f} m/s outside physical sanity bounds [{min_speed}, {max_speed}]"
                    self.event_bus.publish(EventType.ANOMALY_DETECTED, self._last_ai_rejection_reason)
                else:
                    ai_res = self.ekf.update_ai_velocity(
                        v_forward_mps=vel_pred.velocity_mps,
                        variance_mps2=vel_pred.variance_mps2,
                        sigma_mps=vel_pred.sigma_mps,
                    )
                    self._last_ai_innovation = float(ai_res.innovation[0])
                    self._last_ai_accepted = ai_res.accepted
                    self._last_ai_mahalanobis_dist = ai_res.mahalanobis_distance
                    self._last_ai_rejection_reason = ai_res.rejection_reason
                    if ai_res.accepted:
                        self._ai_predictions_accepted += 1
                        self.event_bus.publish(EventType.ODOMETRY_FUSED, f"AI Velocity fused: {vel_pred.velocity_mps:.2f} m/s")
                    else:
                        self._ai_predictions_rejected += 1
                        self.event_bus.publish(EventType.ANOMALY_DETECTED, f"AI Velocity measurement rejected: {ai_res.rejection_reason}")

        state = self._sample_to_state(sample, transform, alignment, calibration_state)
        self._last_state = state
        return state

    async def states(self, publish_rate_hz: float = 10.0) -> AsyncIterator[NavigationState]:
        self.start()
        emit_period_ns = int(1_000_000_000 / publish_rate_hz)
        next_emit_ns = 0
        async for sample in self.provider.samples():
            if not self.running:
                await asyncio.sleep(0.05)
                continue
            state = self.step(sample)
            if state is None:
                continue
            if sample.timestamp_ns < next_emit_ns:
                continue
            next_emit_ns = sample.timestamp_ns + emit_period_ns
            now_ns = time.monotonic_ns()
            if self._last_emit_monotonic_ns is not None:
                dt_ns = now_ns - self._last_emit_monotonic_ns
                if dt_ns > 0:
                    self._navigation_rate_hz = 1_000_000_000 / dt_ns
            self._last_emit_monotonic_ns = now_ns
            yield state

    def _step_high_rate(
        self,
        sample: SensorSample,
        transform: VehicleFrameTransform,
        alignment: object,
    ) -> None:
        """High-rate 100 Hz step: Bias compensation -> Alignment -> Standalone INS -> ES-EKF Prediction & Constraints."""
        # Initialize anchor if needed
        if self._origin_lat is None and sample.latitude is not None and sample.longitude is not None:
            self._origin_lat = sample.latitude
            self._origin_lon = sample.longitude
            self._origin_alt = sample.altitude or 0.0

        calibrated_imu = ImuVector(
            ax=sample.ax,
            ay=sample.ay,
            az=sample.az,
            gx=sample.gx - alignment.gyro_bias_x,
            gy=sample.gy - alignment.gyro_bias_y,
            gz=sample.gz - alignment.gyro_bias_z,
        )
        vehicle_aligned_imu = transform.to_vehicle_aligned_imu(calibrated_imu)
        accel_veh = np.array([
            vehicle_aligned_imu.acc_veh_x,
            vehicle_aligned_imu.acc_veh_y,
            vehicle_aligned_imu.acc_veh_z,
        ], dtype=float)
        gyro_veh = np.array([
            vehicle_aligned_imu.gyro_roll_corrected,
            vehicle_aligned_imu.gyro_pitch_corrected,
            vehicle_aligned_imu.gyro_yaw_corrected,
        ], dtype=float)

        # 1. Standalone Free-running INS Propagation
        self.standalone_ins_state = self.ins_mechanization.propagate(
            self.standalone_ins_state,
            accel_veh,
            gyro_veh,
            sample.timestamp_ns,
        )

        # 2. High-rate ES-EKF Prediction
        self.ekf.predict(accel_veh, gyro_veh, sample.timestamp_ns)

        # 3. Motion Detection
        cur_speed = float(np.linalg.norm(self.ekf.state.velocity_enu))
        motion_state, vibration_score = self.motion_detector.detect(
            accel_veh,
            gyro_veh,
            estimated_speed_mps=cur_speed,
            timestamp_ns=sample.timestamp_ns,
        )
        self._last_motion_state = motion_state
        self._last_vibration_score = vibration_score

        # Accumulate motion durations
        dt_s = (sample.timestamp_ns - self._last_sample.timestamp_ns) * 1e-9 if self._last_sample else 0.1
        if 0.0 < dt_s < 1.0:
            if motion_state == VehicleMotionState.STATIONARY:
                self._motion_durations["stationary"] += dt_s
            else:
                self._motion_durations["moving"] += dt_s
            if vibration_score > 2.0 or motion_state == VehicleMotionState.HIGH_VIBRATION:
                self._motion_durations["high_vibration"] += dt_s
            if abs(float(gyro_veh[2])) > 0.1:
                self._motion_durations["turning"] += dt_s

        # 4. GNSS Outage Management & Availability Evaluation (Step 8)
        effective_gnss = sample.gnss_available and not self.gnss_blackout
        gt_enu = np.array([sample.truth_local_x, sample.truth_local_y, 0.0]) if sample.truth_local_x is not None else None
        gnss_acc = float(sample.gnss_accuracy) if sample.gnss_accuracy is not None else 2.5

        outage_state, should_process_gnss = self.outage_mgr.on_sample(
            gnss_available=effective_gnss,
            gnss_accuracy_m=gnss_acc,
            timestamp_ns=sample.timestamp_ns,
            current_pos_enu=self.ekf.state.position_enu,
            ground_truth_enu=gt_enu,
        )
        cur_outage_str = outage_state.value if hasattr(outage_state, "value") else str(outage_state)
        if self._last_outage_state is not None and cur_outage_str != self._last_outage_state:
            if "LOST" in cur_outage_str or "DEGRADED" in cur_outage_str:
                self._outage_transitions += 1
            elif "RECOVERY" in cur_outage_str or "INS" in cur_outage_str:
                self._recovery_transitions += 1
        self._last_outage_state = cur_outage_str
        self._should_process_gnss = should_process_gnss

        # 5. GNSS Position & Velocity Measurement Updates (Step 9C: Gated by genuine GPS fix de-duplication)
        has_gnss_coords = sample.latitude is not None and sample.longitude is not None
        is_new_gnss_fix = True

        if effective_gnss and has_gnss_coords:
            self._gnss_updates_attempted += 1
            if sample.gnss_fix_id is not None:
                if sample.gnss_fix_id == self._last_applied_gnss_fix_id:
                    # Repeated/stale GPS fix: EKF must receive a measurement ONLY when genuinely new
                    is_new_gnss_fix = False
                    self._gnss_repeated_fixes_ignored += 1
                    self._last_gnss_accepted = False
                    self._last_gnss_vel_accepted = False
                    self._last_gnss_rejection_reason = "Repeated GNSS fix ignored (not a new measurement)"
                else:
                    self._last_applied_gnss_fix_id = sample.gnss_fix_id
                    self._last_applied_gnss_coords = (sample.latitude, sample.longitude, sample.altitude or 0.0)
            else:
                self._last_applied_gnss_coords = (sample.latitude, sample.longitude, sample.altitude or 0.0)

        if should_process_gnss and self.enable_measurement_updates and is_new_gnss_fix:
            # 5a. GNSS Position Update (Step 4 & Step 8 Recovery)
            if self.enable_gnss_position_update and sample.latitude is not None and sample.longitude is not None:
                if self._origin_lat is None:
                    self._origin_lat = sample.latitude
                    self._origin_lon = sample.longitude
                    self._origin_alt = sample.altitude or 0.0

                pos_enu = EarthModel.geodetic_to_enu(
                    sample.latitude,
                    sample.longitude,
                    sample.altitude or 0.0,
                    self._origin_lat,
                    self._origin_lon,
                    self._origin_alt,
                )

                gnss_cfg = self.ekf.config.get("measurements", {}).get("gnss_position", {})
                default_h_std = float(gnss_cfg.get("noise_std_horizontal", 2.5))
                default_v_std = float(gnss_cfg.get("noise_std_vertical", 4.5))
                h_acc = float(sample.gnss_accuracy) if sample.gnss_accuracy is not None else default_h_std
                v_acc = float(h_acc * (default_v_std / default_h_std)) if sample.gnss_accuracy is not None else default_v_std
                pos_cov = np.diag([h_acc ** 2, h_acc ** 2, v_acc ** 2])
                gating_thresh = float(gnss_cfg.get("gating_mahalanobis_threshold", 16.27))

                pos_meas = GNSSPositionMeasurement(
                    z_enu=pos_enu,
                    cov_enu=pos_cov,
                    gating_threshold=gating_thresh,
                )
                pos_res = self.ekf.update(pos_meas)
                self.outage_mgr.on_measurement_result(pos_res.accepted, "position", sample.timestamp_ns)
                self._last_gnss_innovation = pos_res.innovation.tolist()
                self._last_gnss_accepted = pos_res.accepted
                self._last_gnss_mahalanobis_dist = pos_res.mahalanobis_distance
                self._last_gnss_rejection_reason = pos_res.rejection_reason
                if pos_res.accepted:
                    self._gnss_updates_accepted += 1
                else:
                    self._innovation_rejections += 1
            else:
                self._last_gnss_accepted = False

            # 5b. GNSS Velocity Update (Step 5 & Step 8 Recovery)
            if self.enable_gnss_velocity_update and sample.gnss_velocity is not None and sample.gnss_heading is not None:
                v_enu = GNSSVelocityMeasurement.speed_course_to_enu(
                    speed_mps=sample.gnss_velocity,
                    course_rad=sample.gnss_heading,
                    vertical_mps=0.0,
                )
                vel_res = self.ekf.update_gnss_velocity(z_vel_enu=v_enu)
                self.outage_mgr.on_measurement_result(vel_res.accepted, "velocity", sample.timestamp_ns)
                self._last_gnss_vel_innovation = vel_res.innovation.tolist()
                self._last_gnss_vel_accepted = vel_res.accepted
                self._last_gnss_vel_mahalanobis_dist = vel_res.mahalanobis_distance
                self._last_gnss_vel_rejection_reason = vel_res.rejection_reason
                if not vel_res.accepted:
                    self._innovation_rejections += 1
            else:
                self._last_gnss_vel_accepted = False
        else:
            if not is_new_gnss_fix and has_gnss_coords and effective_gnss:
                # Repeated fix - keep rejection reason set above
                pass
            else:
                # During outage (DEAD_RECKONING) or blackout: GNSS updates are completely disabled
                self._last_gnss_accepted = False
                self._last_gnss_vel_accepted = False

        # 6. Motion Constraints & Measurement Updates (Step 7: ZUPT & NHC)
        if self.enable_measurement_updates:
            gyro_norm = float(np.linalg.norm(gyro_veh))
            accel_norm = float(np.linalg.norm(accel_veh))
            last_vel_pred = self.models._last_velocity_prediction
            ai_vel_mps = last_vel_pred.velocity_mps if (last_vel_pred and not last_vel_pred.is_stale) else None
            gnss_vel_mps = sample.gnss_velocity if (sample.gnss_available and not self.gnss_blackout) else None

            # Evaluate ZUPT eligibility
            is_zupt = self.zupt_validator.evaluate(
                motion_state,
                gyro_norm,
                accel_norm,
                ai_velocity_mps=ai_vel_mps,
                gnss_velocity_mps=gnss_vel_mps,
            )
            if is_zupt:
                self._zupt_opportunities += 1
            if self.enable_zupt_update and is_zupt:
                zupt_res = self.ekf.update_zupt()
                self._zupt_active = zupt_res.accepted
                if zupt_res.accepted:
                    self._zupt_accepted += 1
                    if not self._prev_zupt_active:
                        self.event_bus.publish(EventType.ODOMETRY_FUSED, "ZUPT applied: vehicle stationary")
                else:
                    self._zupt_rejected += 1
            else:
                self._zupt_active = False
            self._prev_zupt_active = self._zupt_active

            # Evaluate NHC eligibility (only if ZUPT is not active to prevent redundant constraint)
            nhc_valid = self.nhc_validator.is_valid(
                motion_state, vibration_score, float(gyro_veh[2])
            )
            if nhc_valid:
                self._nhc_opportunities += 1
            if self._zupt_active:
                if nhc_valid:
                    self._nhc_suppressed += 1
                self._nhc_active = False
            elif self.enable_nhc_update and nhc_valid:
                nhc_res = self.ekf.update_nhc()
                self._nhc_active = nhc_res.accepted
                if nhc_res.accepted:
                    self._nhc_accepted += 1
                else:
                    self._nhc_rejected += 1
            else:
                self._nhc_active = False

        if not np.all(np.isfinite(self.ekf.P)):
            self._covariance_numerical_failures += 1

    def _sample_to_state(
        self,
        sample: SensorSample,
        transform: VehicleFrameTransform | None = None,
        alignment: object | None = None,
        calibration_state: object | None = None,
    ) -> NavigationState:
        if calibration_state is None:
            calibration_state = self.calibration.update(sample)
        if alignment is None:
            alignment = calibration_state.alignment

        effective_yaw_pred = self.models.get_effective_yaw_prediction(sample.timestamp_ns)
        mounting_yaw_rad = effective_yaw_pred.mounting_yaw_rad if effective_yaw_pred and not effective_yaw_pred.is_stale else 0.0

        if transform is None:
            transform = VehicleFrameTransform.from_alignment(alignment.roll_rad, alignment.pitch_rad, mounting_yaw_rad)

        # Step high rate if called directly without states() iterator
        if self._last_sample is None or self._last_sample.timestamp_ns != sample.timestamp_ns:
            self._step_high_rate(sample, transform, alignment)

        calibrated_imu = ImuVector(
            ax=sample.ax,
            ay=sample.ay,
            az=sample.az,
            gx=sample.gx - alignment.gyro_bias_x,
            gy=sample.gy - alignment.gyro_bias_y,
            gz=sample.gz - alignment.gyro_bias_z,
        )
        vehicle_aligned_imu = transform.to_vehicle_aligned_imu(calibrated_imu)

        if alignment.status.value == "COMPLETE" and not self._calibration_complete_event_sent:
            self._calibration_complete_event_sent = True
            self.event_bus.publish(EventType.CALIBRATION_COMPLETE, "Initial alignment complete", alignment.model_dump())

        effective_vel_pred = self.models.get_effective_velocity_prediction(sample.timestamp_ns)

        # Vehicle frame IMU dictionary with both generic and contract feature keys
        vehicle_frame_dict: dict[str, float | None] = {
            "ax": vehicle_aligned_imu.acc_veh_x,
            "ay": vehicle_aligned_imu.acc_veh_y,
            "az": vehicle_aligned_imu.acc_veh_z,
            "gx": vehicle_aligned_imu.gyro_roll_corrected,
            "gy": vehicle_aligned_imu.gyro_pitch_corrected,
            "gz": vehicle_aligned_imu.gyro_yaw_corrected,
            "ACC_VEH_X": vehicle_aligned_imu.acc_veh_x,
            "ACC_VEH_Y": vehicle_aligned_imu.acc_veh_y,
            "ACC_VEH_Z": vehicle_aligned_imu.acc_veh_z,
            "GYRO_Yaw_corrected": vehicle_aligned_imu.gyro_yaw_corrected,
            "GYRO_Pitch_corrected": vehicle_aligned_imu.gyro_pitch_corrected,
            "GYRO_Roll_corrected": vehicle_aligned_imu.gyro_roll_corrected,
        }

        # Determine navigation mode from outage manager
        gnss_available = sample.gnss_available and not self.gnss_blackout
        gnss_health = 1.0 if (gnss_available and self.outage_mgr.state != GNSSOutageState.GNSS_LOST) else 0.0

        if self.gnss_blackout or self.outage_mgr.state == GNSSOutageState.GNSS_LOST:
            nav_mode = NavigationMode.DEAD_RECKONING
        elif self.outage_mgr.state == GNSSOutageState.GNSS_DEGRADED:
            nav_mode = NavigationMode.DEGRADED_GNSS_INS
        elif self.outage_mgr.state == GNSSOutageState.GNSS_RECOVERY:
            nav_mode = NavigationMode.GNSS_RECOVERY
        else:
            nav_mode = NavigationMode.GNSS_INS

        # Fused states from ErrorStateEKF (NEVER DIRECT SNAPPING)
        local_x = float(self.ekf.state.position_enu[0])
        local_y = float(self.ekf.state.position_enu[1])
        local_z = float(self.ekf.state.position_enu[2])
        fused_velocity = float(np.linalg.norm(self.ekf.state.velocity_enu))
        roll, pitch, yaw = self.ekf.state.euler_rad
        heading_deg = float(math.degrees(yaw) % 360.0)

        ins_roll, ins_pitch, ins_yaw = self.standalone_ins_state.euler_rad
        position_unc = self.ekf.position_uncertainty_1sigma_m
        nav_confidence = max(0.1, min(0.99, 1.0 - (position_unc / 40.0)))
        vibration_score = abs(sample.ax) + abs(sample.ay) + max(0.0, abs(sample.az - 9.80665))

        return NavigationState(
            timestamp=sample.timestamp_ns,
            latitude=sample.latitude if gnss_available else None,
            longitude=sample.longitude if gnss_available else None,
            altitude=sample.altitude if gnss_available else None,
            local_x=local_x,
            local_y=local_y,
            local_z=local_z,
            velocity_mps=fused_velocity,
            velocity_kmph=fused_velocity * 3.6,
            heading_rad=yaw,
            heading_deg=heading_deg,
            roll_rad=roll,
            pitch_rad=pitch,
            yaw_rad=yaw,
            velocity=fused_velocity,
            heading=yaw,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            gnss_health=gnss_health,

            # Legacy fields
            ai_velocity=effective_vel_pred.velocity_mps if effective_vel_pred else None,
            ai_velocity_confidence=0.85 if effective_vel_pred else 0.0,
            ai_yaw=effective_yaw_pred.mounting_yaw_rad if effective_yaw_pred else None,
            ai_yaw_confidence=0.85 if effective_yaw_pred else 0.0,

            # Actual AI Model Outputs
            mounting_yaw_rad=effective_yaw_pred.mounting_yaw_rad if effective_yaw_pred else None,
            mounting_yaw_deg=effective_yaw_pred.mounting_yaw_deg if effective_yaw_pred else None,
            ai_velocity_mps=effective_vel_pred.velocity_mps if effective_vel_pred else None,
            ai_velocity_kmh=effective_vel_pred.velocity_kmh if effective_vel_pred else None,
            ai_velocity_normalized=effective_vel_pred.velocity_normalized if effective_vel_pred else None,
            ai_velocity_log_variance=effective_vel_pred.log_variance if effective_vel_pred else None,
            ai_velocity_sigma_mps=effective_vel_pred.sigma_mps if effective_vel_pred else None,
            ai_velocity_variance_mps2=effective_vel_pred.variance_mps2 if effective_vel_pred else None,
            ai_velocity_quality=effective_vel_pred.quality if effective_vel_pred else None,
            mountnet_status=self.models.mountnet.status.value,
            velocity_model_status=self.models.velocity.status.value,
            mountnet_latency_ms=round(self.models.mountnet.last_inference_latency_ms, 2) if self.models.mountnet.last_inference_latency_ms > 0 else None,
            velocity_model_latency_ms=round(self.models.velocity.last_inference_latency_ms, 2) if self.models.velocity.last_inference_latency_ms > 0 else None,

            motion_state=self._motion_state(sample, fused_velocity),
            vibration_score=self._last_vibration_score if self._last_vibration_score is not None else vibration_score,
            map_match_confidence=0.0,
            position_uncertainty=position_unc,
            navigation_confidence=nav_confidence,
            navigation_mode=nav_mode,
            sensor_source=sample.sensor_source.value,
            scenario_phase=sample.scenario_phase,

            # Real INS & ES-EKF States
            ins_position={"E": float(self.standalone_ins_state.position_enu[0]), "N": float(self.standalone_ins_state.position_enu[1]), "U": float(self.standalone_ins_state.position_enu[2])},
            ins_velocity={"vE": float(self.standalone_ins_state.velocity_enu[0]), "vN": float(self.standalone_ins_state.velocity_enu[1]), "vU": float(self.standalone_ins_state.velocity_enu[2])},
            ins_orientation={"roll_rad": ins_roll, "pitch_rad": ins_pitch, "yaw_rad": ins_yaw},
            ekf_position={"E": local_x, "N": local_y, "U": local_z},
            ekf_velocity={"vE": float(self.ekf.state.velocity_enu[0]), "vN": float(self.ekf.state.velocity_enu[1]), "vU": float(self.ekf.state.velocity_enu[2])},
            ekf_attitude={"roll_rad": roll, "pitch_rad": pitch, "yaw_rad": yaw},
            position_covariance=np.diag(self.ekf.P)[0:3].tolist(),
            velocity_covariance=np.diag(self.ekf.P)[3:6].tolist(),
            attitude_covariance=np.diag(self.ekf.P)[6:9].tolist(),
            gyro_bias={"bx": float(self.ekf.state.gyro_bias[0]), "by": float(self.ekf.state.gyro_bias[1]), "bz": float(self.ekf.state.gyro_bias[2])},
            accelerometer_bias={"bx": float(self.ekf.state.accel_bias[0]), "by": float(self.ekf.state.accel_bias[1]), "bz": float(self.ekf.state.accel_bias[2])},
            gnss_innovation=self._last_gnss_innovation,
            gnss_measurement_accepted=self._last_gnss_accepted,
            gnss_position_innovation=self._last_gnss_innovation,
            gnss_position_mahalanobis_distance=self._last_gnss_mahalanobis_dist,
            gnss_position_accepted=self._last_gnss_accepted,
            gnss_position_rejection_reason=self._last_gnss_rejection_reason,
            gnss_velocity_innovation=self._last_gnss_vel_innovation,
            gnss_velocity_mahalanobis_distance=self._last_gnss_vel_mahalanobis_dist,
            gnss_velocity_accepted=self._last_gnss_vel_accepted,
            ai_velocity_innovation=self._last_ai_innovation,
            ai_velocity_mahalanobis_distance=self._last_ai_mahalanobis_dist,
            ai_velocity_accepted=self._last_ai_accepted,
            ai_velocity_rejection_reason=self._last_ai_rejection_reason,
            nhc_active=self._nhc_active,
            zupt_active=self._zupt_active,
            dead_reckoning_distance=self.outage_mgr.drift_tracker.outage_distance_m,
            gnss_outage_duration=self.outage_mgr.outage_duration,
            gnss_available=gnss_available and self.outage_mgr.state != GNSSOutageState.GNSS_LOST,
            gnss_last_seen_timestamp=self.outage_mgr.last_valid_gnss_timestamp,
            gnss_recovery_active=self.outage_mgr.state == GNSSOutageState.GNSS_RECOVERY,
            gnss_recovery_measurements=self.outage_mgr.consecutive_valid_measurements,
            gnss_position_updates_enabled=self.enable_gnss_position_update and getattr(self, "_should_process_gnss", True),
            gnss_velocity_updates_enabled=self.enable_gnss_velocity_update and getattr(self, "_should_process_gnss", True),
            position_error_m=self.outage_mgr.drift_tracker.position_error_m,
            drift_percent=self.outage_mgr.drift_tracker.drift_percent,

            raw_imu=self._imu_dict(sample),
            calibrated_imu=self._imu_vector_dict(calibrated_imu),
            vehicle_frame_imu=vehicle_frame_dict,
            calibration=calibration_state.model_dump(),
            model_status=self.models.statuses(),
            rates={
                "imu_hz": self.time_sync.status.estimated_rate_hz,
                "navigation_hz": self._navigation_rate_hz,
                "gnss_hz": None,
                "websocket_latency_ms": None,
                "edge_latency_ms": None,
            },
            ground_truth=self._truth_dict(sample),
            gnss_position=self._gnss_position_dict(sample) if gnss_available else None,
            map_label="LOCAL TEST MAP",
        )

    def calibration_state(self) -> dict[str, object]:
        return self.calibration.alignment.model_dump()

    def _motion_state(self, sample: SensorSample, velocity: float) -> MotionState:
        if self._last_motion_state is not None:
            val = self._last_motion_state.value
            try:
                return MotionState(val)
            except ValueError:
                if val == "HIGH_VIBRATION":
                    return MotionState.IDLE_VIBRATION
        if sample.scenario_phase == "acceleration":
            return MotionState.ACCELERATION
        if sample.scenario_phase == "braking":
            return MotionState.BRAKING
        if sample.scenario_phase in ("gentle_left_turn", "gentle_right_turn"):
            return MotionState.TURNING
        return MotionState.NORMAL_DRIVING if velocity > 0.5 else MotionState.STATIONARY

    def _imu_dict(self, sample: SensorSample) -> dict[str, float | None]:
        return {
            "ax": sample.ax,
            "ay": sample.ay,
            "az": sample.az,
            "gx": sample.gx,
            "gy": sample.gy,
            "gz": sample.gz,
            "mx": sample.mx,
            "my": sample.my,
            "mz": sample.mz,
        }

    def _imu_vector_dict(self, imu: ImuVector) -> dict[str, float | None]:
        return {"ax": imu.ax, "ay": imu.ay, "az": imu.az, "gx": imu.gx, "gy": imu.gy, "gz": imu.gz, "mx": None, "my": None, "mz": None}

    def _truth_dict(self, sample: SensorSample) -> dict[str, float] | None:
        if sample.truth_local_x is None or sample.truth_local_y is None or sample.truth_yaw_rad is None:
            return None
        return {"local_x": sample.truth_local_x, "local_y": sample.truth_local_y, "yaw_rad": sample.truth_yaw_rad}

    def _gnss_position_dict(self, sample: SensorSample) -> dict[str, float] | None:
        if sample.latitude is None or sample.longitude is None:
            return None
        return {"latitude": sample.latitude, "longitude": sample.longitude, "accuracy_m": sample.gnss_accuracy or 0.0}

    def _local_enu(self, sample: SensorSample) -> tuple[float, float, float]:
        if self._origin_lat is None or self._origin_lon is None or sample.latitude is None or sample.longitude is None:
            return 0.0, 0.0, 0.0
        earth_radius_m = 6_378_137.0
        lat0 = math.radians(self._origin_lat)
        d_lat = math.radians(sample.latitude - self._origin_lat)
        d_lon = math.radians(sample.longitude - self._origin_lon)
        east = earth_radius_m * d_lon * math.cos(lat0)
        north = earth_radius_m * d_lat
        up = (sample.altitude or self._origin_alt) - self._origin_alt
        return east, north, up

    def _blackout_seconds(self) -> float:
        if self._session_started_ns is None:
            return 0.0
        return (time.monotonic_ns() - self._session_started_ns) / 1_000_000_000

