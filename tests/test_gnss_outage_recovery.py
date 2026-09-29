import math
import numpy as np
import pytest

from edge.coordinate.frames import VehicleFrameTransform
from edge.coordinate.quaternion import Quaternion
from edge.events.bus import EventBus, EventType
from edge.fusion.outage import GNSSOutageManager, GNSSOutageState
from edge.fusion.measurements import GNSSPositionMeasurement, GNSSVelocityMeasurement
from edge.ins.earth import EarthModel
from edge.navigation.engine import NavigationEngine
from edge.navigation.models import NavigationMode
from edge.sensors.models import SensorSample, SensorSource


def make_sample(
    timestamp_ns: int,
    gnss_available: bool = True,
    lat: float = 12.0,
    lon: float = 77.0,
    alt: float = 100.0,
    accel: tuple[float, float, float] = (0.0, 0.0, 9.80665),
    gyro: tuple[float, float, float] = (0.0, 0.0, 0.0),
    speed: float = 0.0,
    heading: float = 0.0,
    accuracy: float = 2.0,
    truth_x: float | None = None,
    truth_y: float | None = None,
) -> SensorSample:
    return SensorSample(
        timestamp_ns=timestamp_ns,
        ax=accel[0],
        ay=accel[1],
        az=accel[2],
        gx=gyro[0],
        gy=gyro[1],
        gz=gyro[2],
        latitude=lat if gnss_available else None,
        longitude=lon if gnss_available else None,
        altitude=alt if gnss_available else None,
        gnss_velocity=speed if gnss_available else None,
        gnss_heading=heading if gnss_available else None,
        gnss_accuracy=accuracy if gnss_available else None,
        gnss_available=gnss_available,
        truth_local_x=truth_x,
        truth_local_y=truth_y,
        truth_yaw_rad=heading if truth_x is not None else None,
        sensor_source=SensorSource.SYNTHETIC,
    )


# 1. test_initial_gnss_ins_mode
def test_initial_gnss_ins_mode():
    engine = NavigationEngine()
    s = make_sample(1_000_000_000, gnss_available=True)
    state = engine._sample_to_state(s)
    assert state.navigation_mode == NavigationMode.GNSS_INS
    assert state.gnss_available is True
    assert state.gnss_position_updates_enabled is True


# 2. test_gnss_loss_transition
def test_gnss_loss_transition():
    bus = EventBus()
    engine = NavigationEngine(event_bus=bus)
    # Establish origin with initial valid sample
    s0 = make_sample(1_000_000_000, gnss_available=True)
    engine._sample_to_state(s0)

    # 3 consecutive invalid samples (meets loss_persistence_samples=3)
    states = []
    for i in range(1, 5):
        s = make_sample(1_000_000_000 + i * 10_000_000, gnss_available=False)
        states.append(engine._sample_to_state(s))

    assert states[-1].navigation_mode == NavigationMode.DEAD_RECKONING
    assert states[-1].gnss_available is False
    assert states[-1].gnss_position_updates_enabled is False

    # Check GNSS_LOST event
    events = [e for e in bus.history() if e.type == EventType.GNSS_LOST]
    assert len(events) == 1


# 3. test_dead_reckoning_continues
def test_dead_reckoning_continues():
    engine = NavigationEngine()
    # Initialize at t=0
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))

    # Trigger blackout and drive forward
    engine.set_gnss_blackout(True)
    engine.ekf.state.velocity_enu = np.array([5.0, 0.0, 0.0])

    pos_history = []
    for i in range(1, 20):
        t_ns = 1_000_000_000 + i * 10_000_000
        s = make_sample(t_ns, gnss_available=False, accel=(0.0, 0.0, 9.80665))
        st = engine._sample_to_state(s)
        pos_history.append((st.local_x, st.velocity_mps))

    assert st.navigation_mode == NavigationMode.DEAD_RECKONING
    # Position should actively advance
    assert pos_history[-1][0] > pos_history[0][0]
    assert pos_history[-1][1] > 0.0


# 4. test_gnss_updates_disabled_during_outage
def test_gnss_updates_disabled_during_outage():
    engine = NavigationEngine()
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))
    engine.set_gnss_blackout(True)

    # Supply sample with coordinates, but blackout active
    s = make_sample(1_010_000_000, gnss_available=False, lat=12.0001, lon=77.0001)
    st = engine._sample_to_state(s)
    assert not st.gnss_position_accepted
    assert not st.gnss_velocity_accepted
    assert st.latitude is None


# 5. test_ai_velocity_continues_during_outage
def test_ai_velocity_continues_during_outage():
    engine = NavigationEngine()
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))
    engine.set_gnss_blackout(True)

    # In dead reckoning, AI forward velocity update must work
    engine.ekf.state.velocity_enu = np.array([8.0, 0.0, 0.0])
    res = engine.ekf.update_ai_velocity(v_forward_mps=8.5, variance_mps2=0.25)
    assert res.accepted
    assert res.measurement_type == "AI_VELOCITY"


# 6. test_nhc_continues_during_outage
def test_nhc_continues_during_outage():
    engine = NavigationEngine()
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))
    engine.set_gnss_blackout(True)

    # Give the vehicle forward velocity so it's in NORMAL_DRIVING
    engine.ekf.state.velocity_enu = np.array([10.0, 0.05, 0.0])

    s = make_sample(1_010_000_000, gnss_available=False, accel=(0.02, 0.0, 9.80665))
    st = engine._sample_to_state(s)
    assert st.nhc_active is True


# 7. test_zupt_during_outage
def test_zupt_during_outage():
    engine = NavigationEngine()
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))
    engine.set_gnss_blackout(True)

    # Vehicle stops (stationary for 5+ samples)
    engine.ekf.state.velocity_enu = np.array([0.01, 0.01, 0.0])
    for i in range(1, 8):
        t_ns = 1_000_000_000 + i * 10_000_000
        s = make_sample(t_ns, gnss_available=False, accel=(0.0, 0.0, 9.80665), speed=0.0)
        st = engine._sample_to_state(s)

    assert st.zupt_active is True
    assert st.nhc_active is False  # NHC suppressed when ZUPT is active


# 8. test_outage_duration
def test_outage_duration():
    engine = NavigationEngine()
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))
    engine.set_gnss_blackout(True)

    for i in range(1, 101):
        t_ns = 1_000_000_000 + i * 10_000_000  # 10ms steps -> 1 second total
        s = make_sample(t_ns, gnss_available=False)
        st = engine._sample_to_state(s)

    # Outage starts after 3 samples (at step 3), so duration = (100 - 3) * 0.01 = 0.97s
    assert math.isclose(st.gnss_outage_duration, 0.97, abs_tol=0.05)


# 9. test_dead_reckoning_distance
def test_dead_reckoning_distance():
    engine = NavigationEngine()
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))
    engine.set_gnss_blackout(True)
    engine.ekf.state.velocity_enu = np.array([10.0, 0.0, 0.0])

    for i in range(1, 21):
        t_ns = 1_000_000_000 + i * 10_000_000
        s = make_sample(t_ns, gnss_available=False, accel=(0.0, 0.0, 9.80665))
        st = engine._sample_to_state(s)

    # In dead reckoning at 10 m/s for 20 steps of 10ms = 0.2s * 10 m/s = 2.0m accumulated
    assert st.dead_reckoning_distance > 1.0


# 10. test_gnss_recovery_transition
def test_gnss_recovery_transition():
    bus = EventBus()
    engine = NavigationEngine(event_bus=bus)
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))
    
    # Enter outage
    for i in range(1, 5):
        engine._sample_to_state(make_sample(1_000_000_000 + i * 10_000_000, gnss_available=False))
    assert engine.outage_mgr.state == GNSSOutageState.GNSS_LOST

    # GNSS returns
    s_rec = make_sample(1_050_000_000, gnss_available=True)
    st_rec = engine._sample_to_state(s_rec)

    assert st_rec.navigation_mode == NavigationMode.GNSS_RECOVERY
    assert st_rec.gnss_recovery_active is True

    recovered_events = [e for e in bus.history() if e.type == EventType.GNSS_RECOVERED]
    assert len(recovered_events) == 1


# 11. test_recovery_uses_ekf_not_snap
def test_recovery_uses_ekf_not_snap():
    engine = NavigationEngine()
    # Initial fix at [0, 0, 0]
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))
    
    # Force state to [5.0, 0.0, 0.0]
    engine.ekf.state.position_enu = np.array([5.0, 0.0, 0.0])

    # Outage
    for i in range(1, 5):
        engine._sample_to_state(make_sample(1_000_000_000 + i * 10_000_000, gnss_available=False))
    
    pos_before = engine.ekf.state.position_enu.copy()

    # GNSS returns with position at origin (gap = 5.0m)
    s_rec = make_sample(1_050_000_000, gnss_available=True)
    engine._sample_to_state(s_rec)
    pos_after = engine.ekf.state.position_enu.copy()

    # CRITICAL: Position must NOT snap completely to 0.0!
    # It must be a Kalman-weighted correction: 0.0 < pos_after[0] < pos_before[0]
    shift = float(np.linalg.norm(pos_after - pos_before))
    assert shift > 0.0
    assert shift < 4.99  # NOT a 100% snap!
    assert pos_after[0] > 0.5  # Gradual correction


# 12. test_recovery_outlier_rejection
def test_recovery_outlier_rejection():
    engine = NavigationEngine()
    # Initialize at lat=12.0, lon=77.0
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))

    # Outage
    for i in range(1, 5):
        engine._sample_to_state(make_sample(1_000_000_000 + i * 10_000_000, gnss_available=False))

    pos_before = engine.ekf.state.position_enu.copy()

    # Return with massive outlier (10 km away)
    s_outlier = make_sample(1_050_000_000, gnss_available=True, lat=12.1, lon=77.1)
    st = engine._sample_to_state(s_outlier)

    # Must be rejected by innovation gating
    assert not st.gnss_position_accepted
    # Position must NOT teleport to outlier
    pos_after = engine.ekf.state.position_enu.copy()
    assert np.linalg.norm(pos_after - pos_before) < 1.0


# 13. test_recovery_persistence
def test_recovery_persistence():
    engine = NavigationEngine()
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))

    # Outage
    for i in range(1, 5):
        engine._sample_to_state(make_sample(1_000_000_000 + i * 10_000_000, gnss_available=False))
    assert engine.outage_mgr.state == GNSSOutageState.GNSS_LOST

    # Feed valid returning samples one by one (needs 5 to transition to GNSS_INS)
    for i in range(1, 5):
        st = engine._sample_to_state(make_sample(1_050_000_000 + i * 10_000_000, gnss_available=True))
        assert st.navigation_mode == NavigationMode.GNSS_RECOVERY
        assert st.gnss_recovery_measurements == i

    # 5th valid sample completes recovery
    st5 = engine._sample_to_state(make_sample(1_050_000_000 + 5 * 10_000_000, gnss_available=True))
    assert st5.navigation_mode == NavigationMode.GNSS_INS
    assert st5.gnss_recovery_active is False


# 14. test_multiple_outages
def test_multiple_outages():
    engine = NavigationEngine()
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))

    # Outage 1
    for i in range(1, 5):
        engine._sample_to_state(make_sample(1_000_000_000 + i * 10_000_000, gnss_available=False))
    assert engine.outage_mgr.state == GNSSOutageState.GNSS_LOST

    # Recovery 1 (5 valid samples)
    for i in range(1, 6):
        st = engine._sample_to_state(make_sample(1_050_000_000 + i * 10_000_000, gnss_available=True))
    assert st.navigation_mode == NavigationMode.GNSS_INS
    assert st.gnss_outage_duration == 0.0

    # Outage 2 (continue seamlessly without timestamp jump > 50ms)
    for i in range(1, 5):
        st = engine._sample_to_state(make_sample(1_100_000_000 + i * 10_000_000, gnss_available=False))
    assert st.navigation_mode == NavigationMode.DEAD_RECKONING
    assert engine.outage_mgr.outage_start_timestamp == 1_130_000_000


# 15. test_gnss_flapping_hysteresis
def test_gnss_flapping_hysteresis():
    engine = NavigationEngine()
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))

    # Flapping: valid, invalid, valid, invalid, valid
    flapping = [True, False, True, False, True]
    modes = []
    for i, avail in enumerate(flapping):
        st = engine._sample_to_state(make_sample(1_000_000_000 + (i + 1) * 10_000_000, gnss_available=avail))
        modes.append(st.navigation_mode)

    # Must NOT switch to DEAD_RECKONING for single dropped packets!
    assert all(m != NavigationMode.DEAD_RECKONING for m in modes)


# 16. test_position_continuity
def test_position_continuity():
    engine = NavigationEngine()
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))

    positions = []
    # Full lifecycle: 5 normal, 10 blackout, 5 recovery
    for i in range(1, 21):
        t = 1_000_000_000 + i * 10_000_000
        avail = not (5 <= i <= 14)
        st = engine._sample_to_state(make_sample(t, gnss_available=avail))
        positions.append(np.array([st.local_x, st.local_y, st.local_z]))

    # Position deltas across all consecutive 10ms steps must remain continuous (< 2.0m per step)
    for i in range(len(positions) - 1):
        step_delta = float(np.linalg.norm(positions[i + 1] - positions[i]))
        assert step_delta < 2.0, f"Discontinuity detected at step {i}: delta={step_delta:.2f}m"


# 17. test_events_emitted_once
def test_events_emitted_once():
    bus = EventBus()
    engine = NavigationEngine(event_bus=bus)
    engine._sample_to_state(make_sample(1_000_000_000, gnss_available=True))

    # Outage lasting 15 steps
    for i in range(1, 16):
        engine._sample_to_state(make_sample(1_000_000_000 + i * 10_000_000, gnss_available=False))

    lost_events = [e for e in bus.history() if e.type == EventType.GNSS_LOST]
    assert len(lost_events) == 1, f"GNSS_LOST emitted {len(lost_events)} times instead of 1"


# 18. test_ground_truth_not_used_by_filter
def test_ground_truth_not_used_by_filter():
    engine1 = NavigationEngine()
    engine2 = NavigationEngine()

    s1 = make_sample(1_000_000_000, gnss_available=True, truth_x=0.0, truth_y=0.0)
    s2 = make_sample(1_000_000_000, gnss_available=True, truth_x=99999.0, truth_y=-88888.0)

    st1 = engine1._sample_to_state(s1)
    st2 = engine2._sample_to_state(s2)

    # State positions must be exactly identical
    assert np.allclose(engine1.ekf.state.position_enu, engine2.ekf.state.position_enu)
    assert math.isclose(st1.local_x, st2.local_x)
    assert math.isclose(st1.local_y, st2.local_y)
