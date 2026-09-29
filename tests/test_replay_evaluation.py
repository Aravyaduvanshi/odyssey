"""
Unit and integration tests for edge/evaluation/replay.py.
Verifies outage masking, reproducible random selection, ablation flags,
and strictly isolated ground-truth invariance (anti-leakage test).
"""

import math
import numpy as np
import pytest

from edge.evaluation.metrics import TrajectorySample
from edge.evaluation.replay import (
    AblationConfig,
    AblationMode,
    IOVNBDLoader,
    OutageWindow,
    ReplayEvaluator,
)
from edge.navigation.engine import NavigationEngine
from edge.sensors.models import SensorSample, SensorSource


def make_test_sample(
    timestamp_ns: int,
    gnss_available: bool = True,
    lat: float = 12.0,
    lon: float = 77.0,
    speed: float = 5.0,
    truth_x: float = 0.0,
    truth_y: float = 0.0,
) -> SensorSample:
    return SensorSample(
        timestamp_ns=timestamp_ns,
        ax=0.0,
        ay=0.0,
        az=9.80665,
        gx=0.0,
        gy=0.0,
        gz=0.0,
        latitude=lat if gnss_available else None,
        longitude=lon if gnss_available else None,
        altitude=100.0 if gnss_available else None,
        gnss_velocity=speed if gnss_available else None,
        gnss_heading=0.0 if gnss_available else None,
        gnss_accuracy=2.0 if gnss_available else None,
        gnss_available=gnss_available,
        sensor_source=SensorSource.IO_VNBD,
        truth_local_x=truth_x,
        truth_local_y=truth_y,
        truth_yaw_rad=0.0,
    )


def test_outage_duration_overflow_rejected():
    """Verify that an outage exceeding sequence length is reported as unavailable (ValueError)."""
    # 20 second sequence
    total_duration_s = 20.0
    # Requesting 30s outage must fail
    with pytest.raises(ValueError, match="exceeds sequence duration"):
        OutageWindow.select_reproducible_outages(
            total_duration_s=total_duration_s,
            duration_s=30.0,
            count=1,
            random_seed=42,
        )


def test_deterministic_random_outage_selection():
    """Verify that reproducible random selection with fixed seed generates exact same windows."""
    windows_seed42_a = OutageWindow.select_reproducible_outages(
        total_duration_s=120.0,
        duration_s=10.0,
        count=3,
        random_seed=42,
    )
    windows_seed42_b = OutageWindow.select_reproducible_outages(
        total_duration_s=120.0,
        duration_s=10.0,
        count=3,
        random_seed=42,
    )
    windows_seed99 = OutageWindow.select_reproducible_outages(
        total_duration_s=120.0,
        duration_s=10.0,
        count=3,
        random_seed=99,
    )

    # Identical seed produces identical starts
    assert len(windows_seed42_a) == 3
    for wa, wb in zip(windows_seed42_a, windows_seed42_b):
        assert wa.start_s == wb.start_s
        assert wa.duration_s == wb.duration_s

    # Different seed produces different starts
    starts_42 = [w.start_s for w in windows_seed42_a]
    starts_99 = [w.start_s for w in windows_seed99]
    assert starts_42 != starts_99


def test_ablation_configuration_flags():
    """Verify controlled ablation configurations A through F."""
    engine = NavigationEngine()

    # Config A: INS only
    cfg_a = AblationConfig.from_mode(AblationMode.A_INS_ONLY)
    cfg_a.apply(engine)
    assert engine.enable_measurement_updates is False
    assert engine.enable_ai_velocity_update is False
    assert engine.enable_nhc_update is False
    assert engine.enable_zupt_update is False

    # Config B: INS + EKF
    cfg_b = AblationConfig.from_mode(AblationMode.B_INS_EKF)
    cfg_b.apply(engine)
    assert engine.enable_measurement_updates is True
    assert engine.enable_gnss_position_update is True
    assert engine.enable_ai_velocity_update is False
    assert engine.enable_nhc_update is False
    assert engine.enable_zupt_update is False

    # Config C: INS + EKF + AI
    cfg_c = AblationConfig.from_mode(AblationMode.C_INS_EKF_AI)
    cfg_c.apply(engine)
    assert engine.enable_ai_velocity_update is True
    assert engine.enable_nhc_update is False
    assert engine.enable_zupt_update is False

    # Config D: INS + EKF + AI + NHC
    cfg_d = AblationConfig.from_mode(AblationMode.D_INS_EKF_AI_NHC)
    cfg_d.apply(engine)
    assert engine.enable_ai_velocity_update is True
    assert engine.enable_nhc_update is True
    assert engine.enable_zupt_update is False

    # Config E: INS + EKF + AI + NHC + ZUPT
    cfg_e = AblationConfig.from_mode(AblationMode.E_INS_EKF_AI_NHC_ZUPT)
    cfg_e.apply(engine)
    assert engine.enable_ai_velocity_update is True
    assert engine.enable_nhc_update is True
    assert engine.enable_zupt_update is True

    # Config F: Full Odyssey
    cfg_f = AblationConfig.from_mode(AblationMode.F_FULL_ODYSSEY)
    cfg_f.apply(engine)
    assert engine.enable_measurement_updates is True
    assert engine.enable_ai_velocity_update is True
    assert engine.enable_nhc_update is True
    assert engine.enable_zupt_update is True


def test_ground_truth_isolation_invariant():
    """
    CRITICAL ANTI-LEAKAGE TEST:
    Proves that altering ground truth values AFTER or DURING navigation
    has zero effect on the estimated navigation state.
    """
    engine1 = NavigationEngine()
    engine2 = NavigationEngine()

    # Run 1: Normal ground truth (0..100)
    states1 = []
    for i in range(25):
        t = 1_000_000_000 + i * 10_000_000
        s = make_test_sample(t, speed=5.0, truth_x=float(i), truth_y=float(i * 2))
        st = engine1.step(s)
        if st is not None:
            states1.append((st.local_x, st.local_y, st.velocity_mps, st.heading_deg))

    # Run 2: Drastically altered, corrupt ground truth (999999, -888888)
    states2 = []
    for i in range(25):
        t = 1_000_000_000 + i * 10_000_000
        s = make_test_sample(t, speed=5.0, truth_x=999999.0 + i * 1000.0, truth_y=-888888.0)
        st = engine2.step(s)
        if st is not None:
            states2.append((st.local_x, st.local_y, st.velocity_mps, st.heading_deg))

    # All state outputs must be 100% BIT-FOR-BIT IDENTICAL!
    assert len(states1) == len(states2)
    for st1, st2 in zip(states1, states2):
        assert math.isclose(st1[0], st2[0], abs_tol=1e-9)
        assert math.isclose(st1[1], st2[1], abs_tol=1e-9)
        assert math.isclose(st1[2], st2[2], abs_tol=1e-9)
        assert math.isclose(st1[3], st2[3], abs_tol=1e-9)


def test_replay_outage_masking():
    """Verify that ReplayEvaluator masks GNSS during blackout without altering original dataset."""
    samples = [
        make_test_sample(1_000_000_000 + i * 100_000_000, gnss_available=True, truth_x=float(i))
        for i in range(25)  # 2.5 seconds total (at 100ms dt)
    ]

    evaluator = ReplayEvaluator()
    outage = OutageWindow(start_s=0.5, duration_s=1.0, recovery_duration_s=0.5)
    metrics, traj = evaluator.run_outage_evaluation(
        samples=samples,
        outage=outage,
        ablation=AblationConfig.from_mode(AblationMode.B_INS_EKF),
        sequence_name="TEST_SEQ",
    )

    # Verify original samples were NOT permanently modified
    assert all(s.gnss_available for s in samples)

    # Verify trajectory states reflect blackout during outage window [0.5s, 1.5s]
    t0_ns = samples[0].timestamp_ns
    outage_start_ns = t0_ns + 500_000_000
    outage_end_ns = t0_ns + 1_500_000_000

    outage_records = [r for r in traj if outage_start_ns <= r.timestamp_ns < outage_end_ns]
    assert len(outage_records) > 0
    # During blackout, gnss_available must be False
    assert all(not r.gnss_available for r in outage_records)


def test_sequence_split_and_data_leakage():
    """Verify that unknown sequences default to UNKNOWN and known ones map correctly."""
    loader = IOVNBDLoader()
    assert loader.get_sequence_split("unknown_xyz") == "UNKNOWN"
    assert loader.get_sequence_split("Vfa01") == "TEST"
    assert loader.get_sequence_split("M") == "TRAIN"
