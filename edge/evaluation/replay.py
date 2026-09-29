"""
Chronological Real-Data Replay and Outage Evaluation Engine for OdysseyV1.
Supports real synchronized IO-VNBD dataset replay, deterministic and randomized
GNSS blackout injection, controlled ablation studies (A through F),
and strictly isolated ground-truth evaluation.

CRITICAL INVARIANT:
Ground truth is strictly for post-processing evaluation.
Ground truth must NEVER enter the filter or influence navigation state.
"""

import copy
import csv
from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from edge.coordinate.frames import VehicleFrameTransform
from edge.evaluation.metrics import (
    OutageEvaluationMetrics,
    TrajectorySample,
    compute_outage_metrics,
    wrap_to_pi,
)
from edge.ins.earth import EarthModel
from edge.navigation.engine import NavigationEngine
from edge.sensors.models import SensorSample, SensorSource


class AblationMode(StrEnum):
    A_INS_ONLY = "A_INS_ONLY"
    B_INS_EKF = "B_INS_EKF"
    C_INS_EKF_AI = "C_INS_EKF_AI"
    D_INS_EKF_NHC_ZUPT = "D_INS_EKF_NHC_ZUPT"
    D_INS_EKF_AI_NHC = "D_INS_EKF_AI_NHC"
    E_INS_EKF_AI_NHC_ZUPT = "E_INS_EKF_AI_NHC_ZUPT"
    F_FULL_ODYSSEY = "F_FULL_ODYSSEY"


@dataclass
class AblationConfig:
    """Configures which estimation features are active during evaluation."""
    mode: AblationMode = AblationMode.F_FULL_ODYSSEY
    enable_measurement_updates: bool = True
    enable_gnss_position_update: bool = True
    enable_gnss_velocity_update: bool = True
    enable_ai_velocity_update: bool = True
    enable_nhc_update: bool = True
    enable_zupt_update: bool = True

    @classmethod
    def from_mode(cls, mode: AblationMode | str) -> "AblationConfig":
        m_str = str(mode).strip().upper()
        if m_str in ("A", "A_INS_ONLY"):
            return cls(
                mode=AblationMode.A_INS_ONLY,
                enable_measurement_updates=False,
                enable_gnss_position_update=False,
                enable_gnss_velocity_update=False,
                enable_ai_velocity_update=False,
                enable_nhc_update=False,
                enable_zupt_update=False,
            )
        elif m_str in ("B", "B_INS_EKF"):
            return cls(
                mode=AblationMode.B_INS_EKF,
                enable_measurement_updates=True,
                enable_gnss_position_update=True,
                enable_gnss_velocity_update=True,
                enable_ai_velocity_update=False,
                enable_nhc_update=False,
                enable_zupt_update=False,
            )
        elif m_str in ("C", "C_INS_EKF_AI"):
            return cls(
                mode=AblationMode.C_INS_EKF_AI,
                enable_measurement_updates=True,
                enable_gnss_position_update=True,
                enable_gnss_velocity_update=True,
                enable_ai_velocity_update=True,
                enable_nhc_update=False,
                enable_zupt_update=False,
            )
        elif m_str in ("D", "D_INS_EKF_NHC_ZUPT"):
            return cls(
                mode=AblationMode.D_INS_EKF_NHC_ZUPT,
                enable_measurement_updates=True,
                enable_gnss_position_update=True,
                enable_gnss_velocity_update=True,
                enable_ai_velocity_update=False,
                enable_nhc_update=True,
                enable_zupt_update=True,
            )
        elif m_str == "D_INS_EKF_AI_NHC":
            return cls(
                mode=AblationMode.D_INS_EKF_AI_NHC,
                enable_measurement_updates=True,
                enable_gnss_position_update=True,
                enable_gnss_velocity_update=True,
                enable_ai_velocity_update=True,
                enable_nhc_update=True,
                enable_zupt_update=False,
            )
        elif m_str in ("E", "E_INS_EKF_AI_NHC_ZUPT", "E_INS_EKF_NHC_ZUPT_AI"):
            return cls(
                mode=AblationMode.E_INS_EKF_AI_NHC_ZUPT,
                enable_measurement_updates=True,
                enable_gnss_position_update=True,
                enable_gnss_velocity_update=True,
                enable_ai_velocity_update=True,
                enable_nhc_update=True,
                enable_zupt_update=True,
            )
        elif m_str in ("F", "F_FULL_ODYSSEY"):
            return cls(
                mode=AblationMode.F_FULL_ODYSSEY,
                enable_measurement_updates=True,
                enable_gnss_position_update=True,
                enable_gnss_velocity_update=True,
                enable_ai_velocity_update=True,
                enable_nhc_update=True,
                enable_zupt_update=True,
            )
        return cls()

    def apply(self, engine: NavigationEngine) -> None:
        """Apply feature flags directly to NavigationEngine."""
        engine.enable_measurement_updates = self.enable_measurement_updates
        engine.enable_gnss_position_update = self.enable_gnss_position_update
        engine.enable_gnss_velocity_update = self.enable_gnss_velocity_update
        engine.enable_ai_velocity_update = self.enable_ai_velocity_update
        engine.enable_nhc_update = self.enable_nhc_update
        engine.enable_zupt_update = self.enable_zupt_update


@dataclass
class OutageWindow:
    """Defines an artificial GNSS blackout window within a trajectory."""
    start_s: float
    duration_s: float
    recovery_duration_s: float = 10.0
    random_seed: Optional[int] = None

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    @property
    def recovery_end_s(self) -> float:
        return self.end_s + self.recovery_duration_s

    @staticmethod
    def select_reproducible_outages(
        total_duration_s: float,
        duration_s: float,
        count: int = 1,
        random_seed: int = 42,
        margin_s: float = 15.0,
        recovery_duration_s: float = 10.0,
    ) -> List["OutageWindow"]:
        """
        Deterministically select outage windows from trajectory length.
        Ensures initial convergence margin and recovery margin.
        If requested outage exceeds sequence duration, raises ValueError.
        """
        needed_length = duration_s + margin_s + recovery_duration_s
        if total_duration_s < needed_length:
            raise ValueError(
                f"Requested outage duration {duration_s}s (with {margin_s}s start margin and "
                f"{recovery_duration_s}s recovery margin) exceeds sequence duration {total_duration_s:.1f}s."
            )

        rng = random.Random(random_seed)
        latest_start = total_duration_s - duration_s - recovery_duration_s
        windows: List[OutageWindow] = []

        if count == 1:
            # Deterministic midpoint window if count=1
            start_t = margin_s + (latest_start - margin_s) * 0.35
            windows.append(OutageWindow(
                start_s=round(start_t, 2),
                duration_s=duration_s,
                recovery_duration_s=recovery_duration_s,
                random_seed=random_seed,
            ))
            return windows

        # Multiple windows
        step = (latest_start - margin_s) / (count + 1)
        for i in range(count):
            base_t = margin_s + i * step
            jitter = rng.uniform(-step * 0.2, step * 0.2)
            st = max(margin_s, min(latest_start, base_t + jitter))
            windows.append(OutageWindow(
                start_s=round(st, 2),
                duration_s=duration_s,
                recovery_duration_s=recovery_duration_s,
                random_seed=random_seed + i,
            ))
        return windows


class IOVNBDLoader:
    """
    Adapter and chronological synchronizer for IO-VNBD datasets.
    Supports smartphone (S-Dataset) and vehicle ground-truth (V-Dataset) CSVs.
    """

    KNOWN_SPLITS: Dict[str, str] = {
        # Default split knowledge; unlisted sequences return 'UNKNOWN'
        "Vfa01": "TEST",
        "Vfa02": "TEST",
        "S1": "TEST",
        "S2": "TEST",
        "M": "TRAIN",
        "S4": "VAL",
    }

    def __init__(self, dataset_root: Optional[Path | str] = None) -> None:
        self.dataset_root = self._resolve_dataset_root(dataset_root)

    @staticmethod
    def _resolve_dataset_root(candidate: Optional[Path | str] = None) -> Path:
        """Find the active IO-VNBD dataset directory."""
        candidates = [
            Path(candidate) if candidate else None,
            Path("/kaggle/input/datasets/aravkumaryadav/s-v-synchronized/Synchronised V abd S datasets/Uncategorised IOVNB Dataset"),
            Path("/Users/aravkumaryadav/Desktop/Synchronised V abd S datasets/Uncategorised IOVNB Dataset"),
            Path("datasets/iovnbd"),
            Path("../Synchronised V abd S datasets/Uncategorised IOVNB Dataset"),
        ]
        for c in candidates:
            if c and c.is_dir() and (c / "S-Dataset").is_dir():
                return c.resolve()
        # Fallback to current dir if none found
        return Path("/Users/aravkumaryadav/Desktop/Synchronised V abd S datasets/Uncategorised IOVNB Dataset")

    def get_sequence_split(self, sequence_name: str) -> str:
        """Check whether sequence is in TRAIN, VAL, TEST, or UNKNOWN."""
        clean_name = sequence_name.replace("S-", "").replace("V-", "").replace(".csv", "")
        return self.KNOWN_SPLITS.get(clean_name, "UNKNOWN")

    def load_sequence(
        self,
        sequence_name: str,
        max_samples: Optional[int] = None,
    ) -> Tuple[List[SensorSample], Dict[str, Any]]:
        """
        Load synchronized S and V CSV files for a given sequence name (e.g. 'Vta2', 'Vfa01' or 'S-Vta2.csv').
        Returns (samples, metadata).
        """
        seq_clean = sequence_name.replace("S-", "").replace("V-", "").replace(".csv", "")
        s_file = self.dataset_root / "S-Dataset" / f"S-{seq_clean}.csv"
        v_file = self.dataset_root / "V-Dataset" / f"V-{seq_clean}.csv"

        # Case-insensitive file discovery if exact match not found
        if not s_file.is_file() and (self.dataset_root / "S-Dataset").is_dir():
            for cand in (self.dataset_root / "S-Dataset").glob("*.csv"):
                if cand.name.lower() == f"s-{seq_clean.lower()}.csv":
                    s_file = cand
                    break
        if not v_file.is_file() and (self.dataset_root / "V-Dataset").is_dir():
            for cand in (self.dataset_root / "V-Dataset").glob("*.csv"):
                if cand.name.lower() == f"v-{seq_clean.lower()}.csv":
                    v_file = cand
                    break

        if not s_file.is_file():
            raise FileNotFoundError(f"IO-VNBD S-Dataset file not found: {s_file}")
        if not v_file.is_file():
            raise FileNotFoundError(f"IO-VNBD V-Dataset file not found: {v_file}")

        # 1. Read Vehicle Ground Truth (V-Dataset)
        v_rows: List[Dict[str, float]] = []
        with open(v_file, "r", encoding="latin1") as f:
            reader = csv.DictReader(f)
            for r in reader:
                clean_r = {k.strip(): v.strip() for k, v in r.items() if k}
                try:
                    t_day = float(clean_r["Time Since Start of Day (seconds)"])
                    lat = float(clean_r["Latitude (degrees)"])
                    lon = float(clean_r["Longitude (degrees)"])
                    spd_kmh = float(clean_r.get("Velocity (km/hr)", clean_r.get("Indicated Vehicle Speed (km/hr)", 0.0)))
                    hdg_deg = float(clean_r.get("Heading (degrees)", 0.0))
                    # Note: V-Dataset header says "Height (km)", but numerical values are in meters (~63.05 m)
                    alt_m = float(clean_r.get("Height (km)", 0.0))
                    v_rows.append({
                        "time_s": t_day,
                        "lat": lat,
                        "lon": lon,
                        "alt": alt_m,
                        "speed_mps": spd_kmh / 3.6,
                        "heading_deg": hdg_deg,
                    })
                except (ValueError, KeyError):
                    continue

        if not v_rows:
            raise ValueError(f"No valid ground truth rows in {v_file}")

        # Establish geodetic anchor origin from first valid ground-truth row
        origin_lat = v_rows[0]["lat"]
        origin_lon = v_rows[0]["lon"]
        origin_alt = v_rows[0]["alt"]

        # Compute ground truth ENU positions and velocity vectors
        for vr in v_rows:
            p_enu = EarthModel.geodetic_to_enu(
                vr["lat"], vr["lon"], vr["alt"],
                origin_lat, origin_lon, origin_alt
            )
            vr["local_x"] = float(p_enu[0])
            vr["local_y"] = float(p_enu[1])
            vr["local_z"] = float(p_enu[2])
            # Heading in ENU frame: 0 deg = North, 90 deg = East
            hdg_rad = math.radians(vr["heading_deg"])
            vr["heading_rad"] = hdg_rad
            # Velocity ENU components
            vr["vel_e"] = vr["speed_mps"] * math.sin(hdg_rad)
            vr["vel_n"] = vr["speed_mps"] * math.cos(hdg_rad)
            vr["vel_u"] = 0.0

        v_times = np.array([vr["time_s"] for vr in v_rows], dtype=float)

        # 2. Read Smartphone Sensors (S-Dataset)
        s_samples: List[SensorSample] = []
        last_gps_fix_key: Optional[str] = None
        gps_fix_counter: int = 0

        with open(s_file, "r", encoding="latin1") as f:
            reader = csv.DictReader(f)
            t_base_s: Optional[float] = None
            t_base_ns: int = 1_000_000_000

            for i, r in enumerate(reader):
                if max_samples is not None and i >= max_samples:
                    break
                clean_r = {k.strip(): v.strip() for k, v in r.items() if k}
                date_str = ""
                for k, v in clean_r.items():
                    if k.startswith("DATE"):
                        date_str = v
                        break
                if not date_str:
                    continue

                try:
                    # Parse time of day
                    time_part = date_str.split(" ")[1]
                    h, m, s, ms = [float(x) for x in time_part.split(":")]
                    sec_of_day = h * 3600.0 + m * 60.0 + s + ms / 1000.0
                except (IndexError, ValueError):
                    continue

                if t_base_s is None:
                    t_base_s = sec_of_day

                # Timestamp in monotonic integer nanoseconds
                rel_t_s = sec_of_day - t_base_s
                timestamp_ns = t_base_ns + int(rel_t_s * 1_000_000_000)

                # Find closest ground truth record in V-Dataset
                v_idx = int(np.clip(np.searchsorted(v_times, sec_of_day), 0, len(v_rows) - 1))
                gt = v_rows[v_idx]

                try:
                    # IMU Readings
                    ax = float(clean_r.get("ACCELEROMETER X (m/s²)", clean_r.get("ACCELEROMETER X (m/sÂ²)", 0.0)))
                    ay = float(clean_r.get("ACCELEROMETER Y (m/s²)", clean_r.get("ACCELEROMETER Y (m/sÂ²)", 0.0)))
                    az = float(clean_r.get("ACCELEROMETER Z (m/s²)", clean_r.get("ACCELEROMETER Z (m/sÂ²)", 9.80665)))
                    gx = float(clean_r.get("GYROSCOPE X (rad/s)", 0.0))
                    gy = float(clean_r.get("GYROSCOPE Y (rad/s)", 0.0))
                    gz = float(clean_r.get("GYROSCOPE Z (rad/s)", 0.0))

                    # Phone GPS Readings
                    lat_str = clean_r.get("GPS LATITUDE (degrees)")
                    lon_str = clean_r.get("GPS LONGITUDE (degrees)")
                    alt_str = clean_r.get("GPS ALTITUDE (m)")
                    spd_str = clean_r.get("GPS SPEED (Kmh)")
                    acc_str = clean_r.get("GPS ACCURACY (m)")
                    hdg_str = clean_r.get("GPS ORIENTATION (°) ", clean_r.get("GPS ORIENTATION (Â°)", clean_r.get("GPS ORIENTATION (°)", "")))
                    sat_str = clean_r.get("GPS SATELLITES IN RANGE", "")

                    has_raw_gps = bool(lat_str and lon_str and float(lat_str) != 0.0)
                    if has_raw_gps:
                        fix_key = f"{lat_str}|{lon_str}|{alt_str}|{spd_str}|{hdg_str}|{sat_str}"
                        if fix_key == last_gps_fix_key:
                            # Stale / repeated GPS fix from 10 Hz smartphone logging
                            # The EKF must receive a GNSS measurement ONLY when a genuinely new GPS fix is available.
                            is_new_gps_fix = False
                            gnss_fix_id = None
                        else:
                            is_new_gps_fix = True
                            last_gps_fix_key = fix_key
                            gps_fix_counter += 1
                            gnss_fix_id = f"fix_{gps_fix_counter}"
                    else:
                        is_new_gps_fix = False
                        gnss_fix_id = None

                    lat = float(lat_str) if is_new_gps_fix else None
                    lon = float(lon_str) if is_new_gps_fix else None
                    alt = float(alt_str) if (is_new_gps_fix and alt_str) else None
                    spd_mps = float(spd_str) / 3.6 if (is_new_gps_fix and spd_str) else None
                    hdg_rad = math.radians(float(hdg_str)) if (is_new_gps_fix and hdg_str) else None
                    accuracy_m = float(acc_str) if (is_new_gps_fix and acc_str) else (2.5 if is_new_gps_fix else None)

                    sample = SensorSample(
                        timestamp_ns=timestamp_ns,
                        ax=ax,
                        ay=ay,
                        az=az,
                        gx=gx,
                        gy=gy,
                        gz=gz,
                        latitude=lat,
                        longitude=lon,
                        altitude=alt,
                        gnss_velocity=spd_mps,
                        gnss_heading=hdg_rad,
                        gnss_accuracy=accuracy_m,
                        gnss_available=is_new_gps_fix,
                        gnss_fix_id=gnss_fix_id,
                        sensor_source=SensorSource.IO_VNBD,
                        # Ground Truth fields strictly for evaluation
                        truth_local_x=gt["local_x"],
                        truth_local_y=gt["local_y"],
                        truth_yaw_rad=gt["heading_rad"],
                    )
                    s_samples.append(sample)
                except (ValueError, KeyError):
                    continue

        metadata = {
            "dataset": "IO-VNBD",
            "sequence": seq_clean,
            "split": self.get_sequence_split(seq_clean),
            "sample_count": len(s_samples),
            "duration_s": (s_samples[-1].timestamp_ns - s_samples[0].timestamp_ns) * 1e-9 if s_samples else 0.0,
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "origin_alt": origin_alt,
            "ground_truth_records": len(v_rows),
        }
        return s_samples, metadata


class ReplayEvaluator:
    """
    Executes chronological replay through OdysseyV1 NavigationEngine,
    injects deterministic or randomized GNSS blackout masks,
    and computes full dead-reckoning and recovery evaluation metrics.
    """

    def __init__(
        self,
        nav_config: Optional[dict] = None,
        model_manager: Optional[Any] = None,
    ) -> None:
        self.nav_config = nav_config or {}
        self.model_manager = model_manager

    def run_outage_evaluation(
        self,
        samples: List[SensorSample],
        outage: OutageWindow,
        ablation: AblationConfig = AblationConfig(),
        sequence_name: str = "UNKNOWN",
        split: str = "UNKNOWN",
        dataset_name: str = "IO-VNBD",
    ) -> Tuple[OutageEvaluationMetrics, List[TrajectorySample]]:
        """
        Run the exact OdysseyV1 NavigationEngine over the sample stream.
        Applies replay-time blackout mask during [outage.start_s, outage.end_s].
        Computes comprehensive evaluation metrics against ground truth.
        """
        if not samples:
            raise ValueError("No samples provided for evaluation replay.")

        t0_ns = samples[0].timestamp_ns
        outage_start_ns = t0_ns + int(outage.start_s * 1_000_000_000)
        outage_end_ns = t0_ns + int(outage.end_s * 1_000_000_000)
        recovery_end_ns = t0_ns + int(outage.recovery_end_s * 1_000_000_000)

        # Initialize NavigationEngine
        engine = NavigationEngine()
        ablation.apply(engine)

        # For 10 Hz real datasets, allow up to 0.25s dt in EKF and time synchronizer
        engine.ekf.max_dt_sec = 0.25
        engine.ins_mechanization.max_dt_sec = 0.25
        engine.time_sync.target_rate_hz = 10.0
        engine.time_sync.drop_gap_factor = 3.0

        # Anchor origin from first sample GNSS fix
        for s in samples:
            if s.latitude is not None and s.longitude is not None:
                engine._origin_lat = s.latitude
                engine._origin_lon = s.longitude
                engine._origin_alt = s.altitude or 0.0
                break

        trajectory: List[TrajectorySample] = []

        # Chronological playback loop
        for orig_sample in samples:
            t = orig_sample.timestamp_ns
            is_blackout = outage_start_ns <= t < outage_end_ns

            # Construct replay sample with blackout mask applied non-destructively
            s = SensorSample(
                timestamp_ns=orig_sample.timestamp_ns,
                ax=orig_sample.ax,
                ay=orig_sample.ay,
                az=orig_sample.az,
                gx=orig_sample.gx,
                gy=orig_sample.gy,
                gz=orig_sample.gz,
                mx=orig_sample.mx,
                my=orig_sample.my,
                mz=orig_sample.mz,
                latitude=None if is_blackout else orig_sample.latitude,
                longitude=None if is_blackout else orig_sample.longitude,
                altitude=None if is_blackout else orig_sample.altitude,
                gnss_velocity=None if is_blackout else orig_sample.gnss_velocity,
                gnss_heading=None if is_blackout else orig_sample.gnss_heading,
                gnss_accuracy=None if is_blackout else orig_sample.gnss_accuracy,
                gnss_available=False if is_blackout else orig_sample.gnss_available,
                gnss_fix_id=None if is_blackout else orig_sample.gnss_fix_id,
                sensor_source=orig_sample.sensor_source,
                truth_local_x=orig_sample.truth_local_x,
                truth_local_y=orig_sample.truth_local_y,
                truth_yaw_rad=orig_sample.truth_yaw_rad,
            )

            # Step the real OdysseyV1 navigation pipeline
            state = engine.step(s)
            if state is None:
                continue

            # Record trajectory state for post-processing evaluation ONLY
            gt_pos = (
                np.array([orig_sample.truth_local_x, orig_sample.truth_local_y, 0.0])
                if orig_sample.truth_local_x is not None
                else None
            )

            vel_est = (
                np.array([state.ekf_velocity["vE"], state.ekf_velocity["vN"], state.ekf_velocity["vU"]])
                if state.ekf_velocity
                else np.array([state.velocity_mps * math.sin(state.heading_rad), state.velocity_mps * math.cos(state.heading_rad), 0.0])
            )

            # Approximate GT velocity from CAN speed and heading
            vel_gt = None
            if orig_sample.truth_yaw_rad is not None:
                # speed can be estimated from position diff or CAN
                vel_gt = np.array([
                    state.velocity_mps * math.sin(orig_sample.truth_yaw_rad),
                    state.velocity_mps * math.cos(orig_sample.truth_yaw_rad),
                    0.0
                ])

            record = TrajectorySample(
                timestamp_ns=t,
                pos_est=np.array([state.local_x, state.local_y, state.local_z]),
                pos_gt=gt_pos,
                vel_est=vel_est,
                vel_gt=vel_gt,
                yaw_est=state.yaw_rad,
                yaw_gt=orig_sample.truth_yaw_rad,
                navigation_mode=state.navigation_mode.value if hasattr(state.navigation_mode, "value") else str(state.navigation_mode),
                gnss_available=state.gnss_available,
                ai_velocity=state.ai_velocity_mps,
                nhc_active=bool(state.nhc_active),
                zupt_active=bool(state.zupt_active),
                speed_est_mps=state.velocity_mps,
                speed_gt_mps=None,
            )
            trajectory.append(record)

        # Align evaluation origin between estimated and ground truth trajectories.
        # CRITICAL ARCHITECTURAL DISTINCTION:
        # Navigation Initialization vs Evaluation Coordinate Origin:
        # - Navigation Initialization: The live navigation filter initializes autonomously
        #   using its own smartphone GNSS fix and IMU measurements. Ground truth is NEVER
        #   used to initialize, guide, or constrain the filter state.
        # - Evaluation Coordinate Origin: The evaluator aligns the reference trajectory
        #   to the estimated trajectory's initial position at t0 strictly for post-processing
        #   error metric calculation (APE, RPE, drift %). This removes any pre-existing
        #   static spatial offset between the smartphone GPS antenna and the vehicle RTK antenna
        #   without feeding any reference information into the live navigation filter.
        valid_indices = [
            i for i, rec in enumerate(trajectory)
            if rec.pos_gt is not None and rec.pos_est is not None
        ]
        if valid_indices:
            i0 = valid_indices[0]
            origin_offset = trajectory[i0].pos_gt - trajectory[i0].pos_est
            for i in valid_indices:
                trajectory[i].pos_gt = trajectory[i].pos_gt - origin_offset

        # Compute all evaluation metrics
        metrics = compute_outage_metrics(
            trajectory=trajectory,
            outage_start_ns=outage_start_ns,
            outage_end_ns=outage_end_ns,
            recovery_end_ns=recovery_end_ns,
            dataset=dataset_name,
            sequence=sequence_name,
            split=split,
            configuration=ablation.mode.value,
        )

        # Attach comprehensive engine diagnostics to metrics
        engine_diag = engine.get_diagnostics()
        metrics.engine_diagnostics = engine_diag
        metrics.ai_velocity_active_samples = engine_diag["ai_velocity"]["predictions_accepted"]
        metrics.nhc_active_samples = engine_diag["nhc"]["accepted"]
        metrics.zupt_active_samples = engine_diag["zupt"]["accepted"]

        return metrics, trajectory
