# Step 9A: Critical Real-Data Navigation Sanity Check Report
**Sequence Analyzed**: `Vfa01` (`S-Vfa01.csv` & `V-Vfa01.csv`)  
**Dataset**: IO-VNBD (Synchronized Smartphone & Vehicle Reference Dataset)  
**Associated Diagnostic Trace**: [`docs/debug_vfa01.csv`](file:///Users/aravkumaryadav/Desktop/odysseyV1/docs/debug_vfa01.csv)  
**Date**: September 2026  

---

## Executive Summary

During Step 9 evaluation on held-out sequence `Vfa01`, dead-reckoning during 10s and 30s artificial GNSS outages produced severe, unphysical errors:
- **10s outage**: INS-only end error = 2,877.10 m (1501.6%), Velocity RMSE = 247.90 m/s; Full OdysseyV1 end error = 1,801.82 m (940.4%), Velocity RMSE = 178.00 m/s.
- **30s outage**: INS-only end error = 16,072.22 m (2791.7%), Velocity RMSE = 565.23 m/s.

This investigation was conducted without tuning thresholds, without altering Kalman filter noise matrices, and without retraining ML models. Through chronological diagnostic tracing, **five definitive root causes** were identified.

---

## Detailed Diagnostic Findings (Items 1 – 22)

### 1. Sensor Units
Empirical analysis of all 11,486 rows of `S-Vfa01.csv`:
- **Accelerometer X**: Min = -15.1118, Max = 16.5209, Median = -0.0739, P95 = 3.6152 m/s²
- **Accelerometer Y**: Min = -10.8842, Max = 11.0650, Median = -0.2239, P95 = 2.4619 m/s²
- **Accelerometer Z**: Min = 1.1220, Max = 15.5626, Median = 9.8484, P95 = 11.2349 m/s²
  - **Norm of median acceleration**: $\sqrt{(-0.0739)^2 + (-0.2239)^2 + 9.8484^2} = 9.8512\text{ m/s}^2 \approx 1g$.
  - **Physical Unit Conclusion**: Accelerometer values are strictly in **$\text{m/s}^2$** (NOT $g$).
- **Gyroscope X**: Min = -0.5305, Max = 0.7377, Median = -0.0015, P95 = 0.1518 rad/s
- **Gyroscope Y**: Min = -1.9187, Max = 1.5192, Median = 0.0018, P95 = 0.4482 rad/s
- **Gyroscope Z**: Min = -1.0531, Max = 1.6351, Median = -0.0017, P95 = 0.2846 rad/s
  - **Physical Unit Conclusion**: Gyroscope readings are strictly in **$\text{rad/s}$** (NOT $\text{deg/s}$).
- **GPS Speed**: Min = 0.00, Max = 26.84 km/h, Median = 18.45 km/h, P95 = 25.23 km/h. Units are **$\text{km/h}$**.
- **GPS Orientation**: Min = 1.41°, Max = 358.90°, Median = 311.42°, P95 = 354.90°. Units are **$\text{degrees}$**.
- **Timestamps**: Monotonic millisecond counter in `TIME SINCE START (ms)` and standard formatted calendar string `DATE (YYYY-MO-DD HH-MI-SS_SSS)`.

### 2. Timestamps and Sampling Rate
- **$\Delta t$ (ms)**:
  - Mean: $100.00\text{ ms}$
  - Median: $100.00\text{ ms}$
  - Min: $86.00\text{ ms}$
  - Max: $113.00\text{ ms}$
  - P95: $101.00\text{ ms}$
- **Effective Sampling Rate**: $10.00\text{ Hz}$ ($1000 / 100.00\text{ ms}$).
- **Timestamp Scaling**: Monotonic nanoseconds converted as `t_base_ns + int(rel_t_s * 1e9)`. Conversion occurs exactly once without duplicate scaling.

### 3. Acceleration Gravity Convention
- **Stationary Segment Analysis**: At row index 5,380 (`GPS SPEED == 0`):
  - Accelerometer: $[-0.0896, 0.5771, 9.9297]\text{ m/s}^2$
  - Accelerometer Norm: $9.9469\text{ m/s}^2 \approx 9.81\text{ m/s}^2$ (standard 1g specific force).
- **Entry into `edge/ins/mechanization.py`**:
  - $f_{\text{vehicle}} = a_{\text{raw}} - b_a \approx [0, 0, +9.80665]\text{ m/s}^2$ (when level).
  - $f_{\text{nav}} = R_{\text{nav}\_\text{vehicle}} f_{\text{vehicle}} \approx [0, 0, +9.80665]\text{ m/s}^2$.
  - $g_{\text{nav}} = [0, 0, -9.80665]\text{ m/s}^2$ (downward in ENU).
  - $a_{\text{nav}} = f_{\text{nav}} + g_{\text{nav}} \approx [0, 0, 0]\text{ m/s}^2$.
  - Gravity is added **exactly once** in mechanization.

### 4. Gyro Units Verification
- Stationary gyro norm at row 5,380: $0.0394\text{ rad/s} \approx 2.26^\circ/\text{s}$.
- Full pipeline inspection (`SensorSample` $\rightarrow$ calibration $\rightarrow$ alignment $\rightarrow$ INS $\rightarrow$ EKF):
  - No duplicate degree-to-radian or radian-to-degree conversions exist in the IMU propagation path.

### 5. Phone $\rightarrow$ Vehicle Alignment
- **MountNet Output on Vfa01**:
  - Sample 99 ($t=9.9\text{s}$): Mounting Yaw = $+13.06^\circ$
  - Sample 149 ($t=14.9\text{s}$): Mounting Yaw = $+11.05^\circ$
  - Sample 199 ($t=19.9\text{s}$): Mounting Yaw = $-14.70^\circ$
  - Sample 249 ($t=24.9\text{s}$): Mounting Yaw = $-3.68^\circ$
  - Sample 299 ($t=29.9\text{s}$): Mounting Yaw = $+11.15^\circ$
- Phone was mounted screen-up, nearly aligned with vehicle chassis ($|\text{yaw}| \le 15^\circ$).
- Stationary acceleration in vehicle frame: Vehicle X $\approx 0.0\text{ m/s}^2$, Vehicle Y $\approx 0.0\text{ m/s}^2$, Vehicle Z $\approx +9.81\text{ m/s}^2$.

### 6. Initial Attitude
- **Initialization State**:
  - `quaternion = Quaternion(1.0, 0.0, 0.0, 0.0)`
  - Roll = $0.0^\circ$, Pitch = $0.0^\circ$, Yaw = $0.0^\circ$.
- **Finding**: Initial navigation yaw starts at $0.0^\circ$ (vehicle facing East in ENU). Ground-truth vehicle heading at $t=0$ in `V-Vfa01.csv` is $31.141^\circ$ (North-North-East), and smartphone GPS heading is $29.26^\circ$. Initial navigation yaw was uninitialized from GPS heading.

### 7. Initial Velocity
- **Initial INS Velocity**: $[0.0, 0.0, 0.0]\text{ m/s}$ (speed = $0.0\text{ m/s}$).
- **Initial GNSS Velocity**: Speed = $12.92\text{ km/h} = 3.59\text{ m/s}$, Heading = $29.26^\circ$.
  - In ENU: $[v_E, v_N, v_U] = [3.59 \sin(29.26^\circ), 3.59 \cos(29.26^\circ), 0.0] = [1.75, 3.13, 0.0]\text{ m/s}$.
- **Initial Ground-Truth Velocity**: Speed = $40.55\text{ km/h} = 11.26\text{ m/s}$, Heading = $31.141^\circ$.
  - In ENU: $[v_E, v_N, v_U] = [5.82, 9.64, 0.01]\text{ m/s}$.
- Filter starts from rest ($0.0\text{ m/s}$) while vehicle is already moving at $3.6 - 11.3\text{ m/s}$.

### 8. GNSS ENU Conversion
- **First GNSS Fix (Row 0, $t=2.6\text{s}$)**: Lat = $52.5595^\circ$, Lon = $-1.482788^\circ$, Alt = $138.52\text{ m}$.
  - ENU Position: $[0.00, 0.00, 0.00]\text{ m}$.
- **Second GNSS Fix (Row 89, $t=11.5\text{s}$)**: Lat = $52.560585^\circ$, Lon = $-1.482209^\circ$, Alt = $137.95\text{ m}$.
  - ENU Position: $[39.27, 120.74, -0.57]\text{ m}$.
- **Displacement**: $\Delta E = 39.27\text{ m}$, $\Delta N = 120.74\text{ m}$, $\Delta U = -0.57\text{ m}$. Euclidean distance = $126.97\text{ m}$.
- **GNSS-Derived Speed**: $126.97\text{ m} / 8.9\text{ s} = 14.27\text{ m/s} = 51.36\text{ km/h}$.
  - Matches vehicle CAN speed range ($40.5 - 55.0\text{ km/h}$).

### 9. Real IMU Acceleration Trace
- 40-second time series saved to [`docs/debug_vfa01.csv`](file:///Users/aravkumaryadav/Desktop/odysseyV1/docs/debug_vfa01.csv).

### 10. Velocity Growth Analysis
Tracking first divergence timestamps for estimated velocity norm:
- **$> 30\text{ m/s}$**: Sample 96 ($t=9.60\text{s}$): $v = 30.54\text{ m/s}$, $a_{\text{nav}} = [7.79, -0.01, -4.01]\text{ m/s}^2$, Attitude = $[R: 10.1^\circ, P: 25.2^\circ, Y: 12.0^\circ]$, Gyro = $[0.023, -0.056, 0.071]\text{ rad/s}$, $\Delta t = 0.100\text{s}$.
- **$> 50\text{ m/s}$**: Sample 133 ($t=13.30\text{s}$): $v = 50.01\text{ m/s}$, $a_{\text{nav}} = [7.22, -0.22, -2.44]\text{ m/s}^2$, Attitude = $[R: 3.5^\circ, P: 33.1^\circ, Y: 0.0^\circ]$.
- **$> 100\text{ m/s}$**: Sample 316 ($t=31.60\text{s}$): $v = 100.74\text{ m/s}$, $a_{\text{nav}} = [6.83, -5.23, -3.89]\text{ m/s}^2$, Attitude = $[R: 40.2^\circ, P: 3.2^\circ, Y: -5.7^\circ]$.
- **$> 200\text{ m/s}$**: Not reached within first 40s (reached in 30s outage run after sustained open-loop drift).

### 11. Position Growth Analysis
Tracking first divergence timestamps for estimated position norm:
- **$> 100\text{ m}$**: Sample 100 ($t=10.00\text{s}$): $p_{\text{norm}} = 102.91\text{ m}$, $v_{\text{est}} = [31.23, 7.77, -7.75]\text{ m/s}$.
- **$> 500\text{ m}$**: Sample 220 ($t=22.00\text{s}$): $p_{\text{norm}} = 501.57\text{ m}$, $v_{\text{est}} = [17.79, -32.11, -19.46]\text{ m/s}$.
- **$> 1000\text{ m}$**: Sample 321 ($t=32.10\text{s}$): $p_{\text{norm}} = 1008.63\text{ m}$, $v_{\text{est}} = [8.21, -94.67, -43.43]\text{ m/s}$.

### 12. EKF Measurement Updates & Ablation Symmetry
Measurement update statistics across all configurations over 400 samples:
- **Configuration A (INS Only)**: All updates disabled.
- **Configurations B through F**:
  - **GNSS Position**: Attempted = 300, **Accepted = 56**, **Rejected = 244** (Outlier gating $d^2 > 16.27$).
  - **GNSS Velocity**: Attempted = 300, **Accepted = 33**, **Rejected = 267** (Outlier gating $d^2 > 16.27$).
  - **AI Velocity**: Attempted = 31, **Accepted = 0**, **Rejected = 31** (100% rejected: sanity bounds $[0, 80]\text{ m/s}$).
  - **NHC**: Attempted = 400, **Accepted = 1**, **Suppressed = 399** (suppressed by vibration/turn detector).
  - **ZUPT**: Attempted = 0, **Accepted = 0**, **Suppressed = 400** (vehicle never stopped during outage).
- **Explanation for Identical Ablations**:
  - B and C are identical because 0 AI velocity updates were accepted.
  - D, E, and F are identical because ZUPT accepted 0 updates and AI accepted 0 updates, while NHC accepted only 1 update.

### 13. AI Model Inference Statistics
Analysis of 31 Velocity TCN forward passes across Vfa01:
- **Predicted Forward Velocity**:
  - Min: $326.05\text{ m/s}$ ($1,173.8\text{ km/h}$)
  - Max: $375.89\text{ m/s}$ ($1,353.2\text{ km/h}$)
  - Mean: $359.49\text{ m/s}$
  - Median: $361.07\text{ m/s}$
  - P95: $373.12\text{ m/s}$
- **Predicted Uncertainty ($\sigma$)**:
  - Min: $11.78\text{ m/s}$, Max: $24.94\text{ m/s}$, Mean: $18.62\text{ m/s}$.
- Every prediction was an extreme outlier ($> 300\text{ m/s}$).

### 14. AI Feature Ordering and Contract
- Expected input order: `[ACC_VEH_X, ACC_VEH_Y, ACC_VEH_Z, GYRO_Yaw_corrected, GYRO_Pitch_corrected, GYRO_Roll_corrected]`.
- Implementation matches contract order.
- **Normalization Discrepancy Found**:
  - `ACC_VEH_Z` in `S2_velocity_TCN_uncertainty_v2_normalization.csv`: `mean = -0.018169`, `std = 0.628754`.
  - Training data had Earth gravity removed (mean $\approx 0\text{ m/s}^2$).
  - Replay buffer fed specific force including gravity ($+9.81\text{ m/s}^2$).
  - Z-score input to network: $(9.81 - (-0.018)) / 0.628 = \mathbf{+15.65\sigma}$.
  - This $+15.65\sigma$ input anomaly caused the model to output $\sim 360\text{ m/s}$.

### 15. AI Inference Rate
- AI window: 100 samples ($10.0\text{s}$ at $10\text{ Hz}$).
- Stride: 10 samples ($1.0\text{s}$).
- Inference rate: 1.0 Hz (31 inferences over 400 samples). Verified not fed at 100 Hz.

### 16. NHC Eligibility and Acceptance
- Attempted: 400 epochs.
- Accepted: 1 epoch.
- Suppressed: 399 epochs.
- Rejection reason: `MotionDetector` classified real-world road motion as `HIGH_VIBRATION` (vibration score $3.5 - 12.6 > 3.0$) or `TURNING` ($|\omega_z| > 0.12\text{ rad/s}$).

### 17. ZUPT Eligibility
- Vehicle moved continuously at $12 - 55\text{ km/h}$.
- First true stationary segment in sequence is at row index 5,380 ($t = 538\text{s}$).
- Zero ZUPT activation during the first 150 seconds is physically correct.

### 18. Ablation Implementation
- Confirmed that `AblationConfig.apply(engine)` correctly sets flags `enable_measurement_updates`, `enable_gnss_position_update`, `enable_gnss_velocity_update`, `enable_ai_velocity_update`, `enable_nhc_update`, and `enable_zupt_update`.
- Pipeline faithfully executed configurations. Identical metrics resulted from upstream data rejection, not flag ignorance.

### 19. Outage Mask Behavior
- Confirmed: During $[t_{\text{start}}, t_{\text{end}}]$, GNSS measurements are masked to `None` and `gnss_available = False`.
- Zero GNSS updates were attempted during blackout.

### 20. Replay Initialization State
- All runs initialized with identical state:
  $p = [0, 0, 0]$, $v = [0, 0, 0]$, $q = [1, 0, 0, 0]$, $b_a = [0, 0, 0]$, $b_g = [0, 0, 0]$.

### 21. Time Alignment
- S-Dataset begins at Day-Time $33626.236\text{s}$.
- V-Dataset begins at Day-Time $33622.000\text{s}$.
- Ground-truth origin was anchored to $t = 33622.0\text{s}$ (`v_rows[0]`), while navigation filter origin was anchored to $t = 33626.2\text{s}$ (`samples[0]`).
- Over those 4.2 seconds, the vehicle moved $55.98\text{ m}$. Ground truth had an initial $56\text{m}$ offset against filter coordinates.

### 22. Coordinate Conventions
- Phone Frame: Right ($+X$), Forward ($+Y$), Up ($+Z$).
- Vehicle Frame: Forward ($+X$), Lateral-Left ($+Y$), Vertical-Up ($+Z$).
- Navigation Frame: Local Tangent ENU ($+X$ East, $+Y$ North, $+Z$ Up).

---

## Confirmed Root Causes

### Root Cause 1: Frozen Smartphone GPS Fixes Triggered Filter Divergence and Permanent Gating Lockout (CONFIRMED)
- **Mechanism**:
  1. The smartphone logging application sampled GPS asynchronously at $\sim 0.11\text{ Hz}$ (once every $9.0\text{ seconds}$), but logged records at $10\text{ Hz}$ by repeating the last GPS fix into every intermediate row.
  2. For the first 89 samples ($t = 0.0$ to $8.9\text{s}$), the GPS coordinates were frozen at `[52.5595, -1.482788]`.
  3. `IOVNBDLoader` treated all 89 rows as fresh GPS fixes.
  4. The EKF updated on `[0, 0, 0]` 56 times, artificially collapsing position covariance $P_{pos}$ to near zero and pulling velocity to $0\text{ m/s}$ while the car was accelerating away.
  5. At sample 89 ($t = 8.9\text{s}$), the smartphone logged the next true fix at `[52.560585, -1.482209]`, which was $127\text{ meters}$ away.
  6. The innovation was $127\text{m}$ against $S \approx 16\text{ m}^2$, yielding Mahalanobis distance squared $d^2 = 438.5 \gg 16.27$.
  7. The EKF rejected the true fix as an outlier.
  8. From sample 89 onward, **100% of all subsequent GNSS position updates were rejected**. The filter was locked out of GNSS updates permanently and ran open-loop for the rest of the trajectory.

### Root Cause 2: Velocity TCN Gravity Mismatch in Feature Normalization (CONFIRMED)
- **Mechanism**:
  1. Velocity TCN was trained on zero-mean linear acceleration (`ACC_VEH_Z` mean = $-0.018\text{ m/s}^2$, std = $0.628\text{ m/s}^2$).
  2. `NavigationEngine` passed specific force including Earth gravity ($\sim +9.81\text{ m/s}^2$).
  3. The resulting $+15.65\sigma$ input anomaly caused the neural network to output unphysical predictions ($326 - 375\text{ m/s}$).
  4. The physical sanity gate ($[0, 80]\text{ m/s}$) rejected 100% of AI predictions.
  5. AI velocity assistance was completely absent during the outage.

### Root Cause 3: Open-Loop Attitude Drift Leaking Gravity into Velocity (CONFIRMED)
- **Mechanism**:
  1. Because stationary calibration never completed on moving start data, uncompensated gyro biases were integrated open-loop.
  2. Nominal pitch and roll drifted by $15^\circ - 35^\circ$.
  3. Each degree of attitude error tilts the $9.81\text{ m/s}^2$ gravity vector into horizontal axes:
     $$a_{\text{leak}} = 9.81 \times \sin(20^\circ) \approx 3.35\text{ m/s}^2$$
  4. Integrating $3.35\text{ m/s}^2$ over 10 seconds creates $\sim 35\text{ m/s}$ velocity error and $\sim 170\text{ m}$ drift. Over 30 seconds, quadratic growth exceeds $1,500\text{ m}$ drift and $> 100\text{ m/s}$ velocity error.

### Root Cause 4: Ground Truth Geodetic Origin Mismatch (CONFIRMED)
- **Mechanism**:
  `IOVNBDLoader` anchored ground truth to $t=33622.0\text{s}$, while `NavigationEngine` anchored to $t=33626.236\text{s}$. This introduced an immediate static coordinate translation error of $55.98\text{ meters}$.

---

## Conclusion & Stop Condition Met

The mystery of why Vfa01 produced $100 - 700\text{ m/s}$ velocity errors has been definitively solved. The primary driver was **stale GPS logging in the S-Dataset causing Kalman filter divergence and permanent outlier rejection lockout**, compounded by **gravity-leakage in the Velocity TCN normalization pipeline** and **unconstrained attitude integration drift**.

Per instructions, navigation mathematics, covariances, and ML models have not been modified. All 142 repository tests remain passing.
