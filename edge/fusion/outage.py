from enum import Enum
import math
from typing import Optional, Tuple
import numpy as np

from edge.events.bus import EventBus, EventType
from edge.navigation.models import NavigationMode


class GNSSOutageState(str, Enum):
    GNSS_AVAILABLE = "GNSS_AVAILABLE"
    GNSS_DEGRADED = "GNSS_DEGRADED"
    GNSS_LOST = "GNSS_LOST"
    GNSS_RECOVERY = "GNSS_RECOVERY"


class DriftTracker:
    """Tracks distance traveled, position error, and drift percentage during GNSS outages."""

    def __init__(self) -> None:
        self.is_outage_active: bool = False
        self.outage_start_pos: Optional[np.ndarray] = None
        self.outage_start_time_ns: int = 0
        self.last_pos: Optional[np.ndarray] = None
        self.outage_distance_m: float = 0.0
        self.position_error_m: Optional[float] = None
        self.drift_percent: Optional[float] = None

    def start_outage(self, current_pos_enu: np.ndarray, timestamp_ns: int) -> None:
        self.is_outage_active = True
        self.outage_start_pos = np.asarray(current_pos_enu, dtype=float).copy()
        self.last_pos = np.asarray(current_pos_enu, dtype=float).copy()
        self.outage_start_time_ns = timestamp_ns
        self.outage_distance_m = 0.0
        self.position_error_m = None
        self.drift_percent = None

    def update(
        self,
        current_pos_enu: np.ndarray,
        ground_truth_enu: Optional[np.ndarray] = None,
    ) -> None:
        if not self.is_outage_active:
            return

        pos = np.asarray(current_pos_enu, dtype=float)
        if self.last_pos is not None:
            step_dist = float(np.linalg.norm(pos - self.last_pos))
            if math.isfinite(step_dist):
                self.outage_distance_m += step_dist
        self.last_pos = pos.copy()

        # Calculate drift only if ground truth is explicitly available for evaluation
        if ground_truth_enu is not None:
            gt = np.asarray(ground_truth_enu, dtype=float)
            error = float(np.linalg.norm(pos - gt))
            self.position_error_m = error
            if self.outage_distance_m > 1.0:
                self.drift_percent = float(100.0 * error / self.outage_distance_m)
        else:
            self.position_error_m = None
            self.drift_percent = None

    def duration_seconds(self, current_time_ns: int) -> float:
        if not self.is_outage_active or self.outage_start_time_ns == 0:
            return 0.0
        return max(0.0, (current_time_ns - self.outage_start_time_ns) * 1e-9)

    def end_outage(self) -> None:
        self.is_outage_active = False


class GNSSOutageManager:
    """
    Robust GNSS availability state machine managing:
    GNSS_INS <-> DEGRADED_GNSS_INS <-> DEAD_RECKONING <-> GNSS_RECOVERY
    with persistence and flapping hysteresis.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        loss_timeout_sec: float = 1.0,
        loss_persistence_samples: int = 3,
        recovery_persistence_samples: int = 5,
        accuracy_threshold_degraded_m: float = 5.0,
        max_acceptable_outage_gap_sec: float = 300.0,
        recovery_required_samples: Optional[int] = None,
    ) -> None:
        self.event_bus = event_bus
        self.loss_timeout_sec = loss_timeout_sec
        self.loss_persistence_samples = loss_persistence_samples
        self.recovery_persistence_samples = (
            recovery_required_samples if recovery_required_samples is not None else recovery_persistence_samples
        )
        self.accuracy_threshold_degraded_m = accuracy_threshold_degraded_m
        self.max_acceptable_outage_gap_sec = max_acceptable_outage_gap_sec

        self.state: GNSSOutageState = GNSSOutageState.GNSS_AVAILABLE
        self.last_valid_gnss_timestamp: Optional[int] = None
        self.last_accepted_position_timestamp: Optional[int] = None
        self.last_accepted_velocity_timestamp: Optional[int] = None
        self.outage_start_timestamp: Optional[int] = None
        self.outage_duration: float = 0.0

        self.consecutive_valid_measurements: int = 0
        self.consecutive_invalid_measurements: int = 0
        self._last_recovery_epoch_ts: Optional[int] = None
        self.drift_tracker = DriftTracker()

    @property
    def navigation_mode(self) -> NavigationMode:
        if self.state == GNSSOutageState.GNSS_LOST:
            return NavigationMode.DEAD_RECKONING
        elif self.state == GNSSOutageState.GNSS_DEGRADED:
            return NavigationMode.DEGRADED_GNSS_INS
        elif self.state == GNSSOutageState.GNSS_RECOVERY:
            return NavigationMode.GNSS_RECOVERY
        return NavigationMode.GNSS_INS

    def on_sample(
        self,
        gnss_available: bool,
        gnss_accuracy_m: float,
        timestamp_ns: int,
        current_pos_enu: np.ndarray,
        ground_truth_enu: Optional[np.ndarray] = None,
    ) -> Tuple[GNSSOutageState, bool]:
        """
        Evaluate GNSS availability at sample time with persistence and flapping hysteresis.
        Returns (current_outage_state, should_process_gnss).
        """
        if not gnss_available:
            self.consecutive_invalid_measurements += 1
            self.consecutive_valid_measurements = 0

            # Check if outage timeout or sample persistence reached
            dt_since_last_gnss = (
                (timestamp_ns - self.last_valid_gnss_timestamp) * 1e-9
                if self.last_valid_gnss_timestamp is not None
                else float("inf")
            )
            should_declare_outage = (
                self.consecutive_invalid_measurements >= self.loss_persistence_samples
                or dt_since_last_gnss >= self.loss_timeout_sec
            )

            if should_declare_outage and self.state != GNSSOutageState.GNSS_LOST:
                self._transition_to_lost(current_pos_enu, timestamp_ns)

            if self.state == GNSSOutageState.GNSS_LOST:
                self.outage_duration = self.drift_tracker.duration_seconds(timestamp_ns)
                self.drift_tracker.update(current_pos_enu, ground_truth_enu)
                return self.state, False

            # In transient state prior to confirming outage, allow dead reckoning drift tracking
            self.drift_tracker.update(current_pos_enu, ground_truth_enu)
            return self.state, False

        # GNSS sample is present
        self.last_valid_gnss_timestamp = timestamp_ns

        # 1. If currently in DEAD_RECKONING, transition to GNSS_RECOVERY
        if self.state == GNSSOutageState.GNSS_LOST:
            self.state = GNSSOutageState.GNSS_RECOVERY
            self.consecutive_valid_measurements = 0
            self.consecutive_invalid_measurements = 0
            if self.event_bus:
                self.event_bus.publish(
                    EventType.GNSS_RECOVERED,
                    "GNSS signal reacquired: Recovery convergence started",
                    {"timestamp_ns": timestamp_ns, "accuracy_m": gnss_accuracy_m},
                )
                self.event_bus.publish(
                    EventType.NAVIGATION_MODE_CHANGED,
                    "Mode changed to GNSS_RECOVERY",
                    {"mode": NavigationMode.GNSS_RECOVERY},
                )

        if self.state == GNSSOutageState.GNSS_RECOVERY:
            self.outage_duration = self.drift_tracker.duration_seconds(timestamp_ns)
            self.drift_tracker.update(current_pos_enu, ground_truth_enu)
            return self.state, True

        # Check degraded accuracy (> threshold)
        prev_state = self.state
        if gnss_accuracy_m > self.accuracy_threshold_degraded_m:
            self.state = GNSSOutageState.GNSS_DEGRADED
            if prev_state != GNSSOutageState.GNSS_DEGRADED and self.event_bus:
                self.event_bus.publish(
                    EventType.GNSS_DEGRADED,
                    f"GNSS degraded: accuracy {gnss_accuracy_m:.1f}m > {self.accuracy_threshold_degraded_m:.1f}m",
                )
                self.event_bus.publish(
                    EventType.NAVIGATION_MODE_CHANGED,
                    "Mode changed to DEGRADED_GNSS_INS",
                    {"mode": NavigationMode.DEGRADED_GNSS_INS},
                )
        else:
            self.state = GNSSOutageState.GNSS_AVAILABLE
            if prev_state == GNSSOutageState.GNSS_DEGRADED and self.event_bus:
                self.event_bus.publish(
                    EventType.NAVIGATION_MODE_CHANGED,
                    "Mode changed to GNSS_INS",
                    {"mode": NavigationMode.GNSS_INS},
                )

        self.consecutive_invalid_measurements = 0
        self.drift_tracker.update(current_pos_enu, ground_truth_enu)
        return self.state, True

    def on_measurement_result(
        self,
        accepted: bool,
        measurement_type: str = "position",
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """
        Called after EKF measurement update to update persistence counters, timestamps,
        and recovery convergence without ever snapping position.
        """
        if accepted:
            if timestamp_ns is None or self._last_recovery_epoch_ts != timestamp_ns:
                self.consecutive_valid_measurements += 1
                self._last_recovery_epoch_ts = timestamp_ns
            self.consecutive_invalid_measurements = 0
            if timestamp_ns is not None:
                if "pos" in measurement_type.lower():
                    self.last_accepted_position_timestamp = timestamp_ns
                elif "vel" in measurement_type.lower():
                    self.last_accepted_velocity_timestamp = timestamp_ns

            if self.state == GNSSOutageState.GNSS_RECOVERY:
                if self.consecutive_valid_measurements >= self.recovery_persistence_samples:
                    self.state = GNSSOutageState.GNSS_AVAILABLE
                    self.drift_tracker.end_outage()
                    self.outage_duration = 0.0
                    if self.event_bus:
                        self.event_bus.publish(
                            EventType.NAVIGATION_MODE_CHANGED,
                            "Mode changed to GNSS_INS: GNSS recovery complete",
                            {"mode": NavigationMode.GNSS_INS},
                        )
        else:
            self.consecutive_invalid_measurements += 1
            if self.state == GNSSOutageState.GNSS_RECOVERY:
                self.consecutive_valid_measurements = max(0, self.consecutive_valid_measurements - 1)

    def on_gnss_measurement_result(
        self,
        accepted: bool,
        measurement_type: str = "position",
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """Compatibility alias for on_measurement_result."""
        self.on_measurement_result(accepted, measurement_type, timestamp_ns)

    def _transition_to_lost(self, current_pos_enu: np.ndarray, timestamp_ns: int) -> None:
        """Execute atomic transition from available/degraded to DEAD_RECKONING."""
        self.state = GNSSOutageState.GNSS_LOST
        self.outage_start_timestamp = timestamp_ns
        self.outage_duration = 0.0
        self.drift_tracker.start_outage(current_pos_enu, timestamp_ns)
        if self.event_bus:
            self.event_bus.publish(
                EventType.GNSS_LOST,
                "GNSS signal lost: Dead reckoning active",
                {"timestamp_ns": timestamp_ns},
            )
            self.event_bus.publish(
                EventType.NAVIGATION_MODE_CHANGED,
                "Mode changed to DEAD_RECKONING",
                {"mode": NavigationMode.DEAD_RECKONING},
            )
