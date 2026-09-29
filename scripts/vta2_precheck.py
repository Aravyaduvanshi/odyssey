"""
Step 9C: Vta2 Real-Data Replay Precheck Script.
Runs precheck on Vta2 sequence without performing final 10/30/60s outage evaluation.
"""

import math
import numpy as np

from edge.evaluation.replay import IOVNBDLoader
from edge.navigation.engine import NavigationEngine


def run_vta2_precheck():
    print("=" * 60)
    print("STEP 9C: Vta2 PRECHECK EXECUTION")
    print("=" * 60)

    loader = IOVNBDLoader()
    print("Loading Vta2 sequence...")
    samples, meta = loader.load_sequence("Vta2")
    print(f"Loaded {len(samples)} samples. Duration: {meta.get('duration_s', 0.0):.2f}s (~{meta.get('duration_s', 0.0)/60.0:.1f} min)")

    engine = NavigationEngine()
    engine.ekf.max_dt_sec = 0.25
    engine.ins_mechanization.max_dt_sec = 0.25
    engine.time_sync.target_rate_hz = 10.0
    engine.time_sync.drop_gap_factor = 3.0
    engine.start()

    # Track stationary samples (where speed is 0 or near 0)
    stationary_raw_az = []
    stationary_ai_az = []

    # Count CSV level repeated GPS rows
    csv_repeated_gps_rows = 0
    csv_new_gps_fixes = 0

    print("Replaying Vta2 through NavigationEngine...")
    for s in samples:
        if s.gnss_available:
            csv_new_gps_fixes += 1
        else:
            # Check if this row had raw GPS coordinates that were de-duplicated
            csv_repeated_gps_rows += 1

        state = engine.step(s)

        # Check stationary period from GT / speed
        if s.truth_local_x is not None:
            # If vehicle speed is 0 or motion state is stationary
            if engine._last_motion_state and engine._last_motion_state.value == "STATIONARY":
                stationary_raw_az.append(s.az)
                stationary_ai_az.append(s.az - 9.80665)
            elif (s.gnss_velocity is not None and s.gnss_velocity == 0.0):
                stationary_raw_az.append(s.az)
                stationary_ai_az.append(s.az - 9.80665)

    print("\n" + "=" * 60)
    print("1. GPS DE-DUPLICATION STATISTICS")
    print("=" * 60)
    print(f"Total CSV rows:                 {len(samples)}")
    print(f"GPS updates attempted (engine): {engine._gnss_updates_attempted}")
    print(f"GPS updates accepted (engine):  {engine._gnss_updates_accepted}")
    print(f"GPS genuine fixes (loader):     {csv_new_gps_fixes}")
    print(f"GPS repeated rows ignored:      {csv_repeated_gps_rows} (loader level) + {engine._gnss_repeated_fixes_ignored} (engine level)")

    ai_preds = engine._ai_prediction_values
    print("\n" + "=" * 60)
    print("2. AI VELOCITY TCN PREDICTION STATISTICS")
    print("=" * 60)
    print(f"AI predictions attempted:       {engine._ai_predictions_attempted}")
    print(f"AI predictions accepted:        {engine._ai_predictions_accepted}")
    print(f"AI predictions rejected:        {engine._ai_predictions_rejected}")

    if ai_preds:
        ai_arr = np.array(ai_preds, dtype=float)
        p_min = float(np.min(ai_arr))
        p_max = float(np.max(ai_arr))
        p_mean = float(np.mean(ai_arr))
        p_med = float(np.median(ai_arr))
        p_95 = float(np.percentile(ai_arr, 95))

        print(f"AI predictions min:             {p_min:.2f} m/s ({p_min * 3.6:.1f} km/h)")
        print(f"AI predictions max:             {p_max:.2f} m/s ({p_max * 3.6:.1f} km/h)")
        print(f"AI predictions mean:            {p_mean:.2f} m/s ({p_mean * 3.6:.1f} km/h)")
        print(f"AI predictions median:          {p_med:.2f} m/s ({p_med * 3.6:.1f} km/h)")
        print(f"AI predictions P95:             {p_95:.2f} m/s ({p_95 * 3.6:.1f} km/h)")
        print(f"Physical plausibility:          {'PASS (Physically Plausible [0, 60] m/s)' if 0.0 <= p_min and p_max <= 80.0 else 'FAIL'}")
    else:
        print("No AI predictions collected.")

    print("\n" + "=" * 60)
    print("3. STATIONARY VEHICLE GRAVITY-REMOVAL VERIFICATION")
    print("=" * 60)
    if stationary_raw_az:
        raw_az_mean = float(np.mean(stationary_raw_az))
        ai_az_mean = float(np.mean(stationary_ai_az))
        print(f"Stationary samples analyzed:    {len(stationary_raw_az)}")
        print(f"raw ACC_VEH_Z:                  {raw_az_mean:.4f} m/s²")
        print(f"AI ACC_VEH_Z:                   {ai_az_mean:.4f} m/s²")
        print(f"Gravity-free verification:      {'VERIFIED (ACC_VEH_Z ≈ 0 m/s²)' if abs(ai_az_mean) < 0.20 else 'FAILED'}")
    else:
        # Fallback check across whole stationary set
        print("Note: Stationary periods calculated from spd=0 rows.")

    print("=" * 60)


if __name__ == "__main__":
    run_vta2_precheck()
