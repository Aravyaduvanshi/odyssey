# IO-VNBD Sequence Quality and Clean Validation Selection Report

**Document Reference**: Step 9B Dataset Quality Analysis & Sequence Selection  
**Associated Quality Trace**: [`docs/iovnb_sequence_quality.csv`](file:///Users/aravkumaryadav/Desktop/odysseyV1/docs/iovnb_sequence_quality.csv)  
**Dataset Analyzed**: Synchronized IO-VNBD Dataset (72 paired smartphone/vehicle sequences)  
**Selected Clean Sequence**: **`Vta2`** (`S-Vta2.csv` / `V-vta2.csv`)  
**Date**: September 2026  

---

## 1. Executive Summary

Following the Step 9A sanity check where sequence `Vfa01` was diagnosed with asynchronous, stale smartphone GPS updates (9-second repeat logging cycles), this analysis scanned all **72 matching S/V sequence pairs** in the IO-VNBD dataset to identify a clean, physically sound evaluation sequence for OdysseyV1.

Crucially, sequences were **NOT** ranked or selected based on navigation filter performance. In strict adherence to Step 9B guidelines, selection was based **exclusively on empirical data quality metrics**:
1. True GPS freshness (low repeat rate and $\le 1.0\text{s}$ update interval).
2. Temporal synchronization and initial S/V time offset.
3. Sampling rate stability and absence of missing/corrupt IMU records.
4. Rich, dynamic vehicle driving profiles with valid 10s, 30s, and 60s continuous outage windows.

**Outcome**: Sequence **`Vta2`** is selected as the primary clean validation sequence (`SELECTED_EVALUATION_SEQUENCE`), with `Vta1a` serving as the long-duration secondary benchmark.

---

## 2. Dataset Scan Summary & GPS Freshness Findings

Across all 72 synchronized pairs:
- **Total Paired Sequences**: 72
- **True 1 Hz GPS Sequences ($\le 1.5\text{s}$ update interval)**: **3** (`Vta2`, `Vta1a`, `Vta1b`)
- **Slow GPS Sequences ($\sim 9.0\text{s}$ update interval)**: **67**
- **Sequences with No Usable GPS Fixes**: **2** (`Vw1`, `Vw15` - vehicle stationary throughout with static placeholder GPS)

### The 9-Second GPS Logging Phenomenon
In 67 out of 72 sequences (including `Vfa01`, `Vfa02`, `M`, `S1`, `S2`, `S4`), the smartphone logging application received Android location updates asynchronously every $\sim 9.0\text{ seconds}$, but wrote records to the CSV at $10\text{ Hz}$ by repeating the last GPS fix into 89 to 90 consecutive rows ($98.9\% - 99.1\%$ repeated coordinates). When fed directly to an EKF, this causes severe filter divergence and permanent outlier rejection.

Only **Driver D (sequences `Vta1a`, `Vta1b`, `Vta2`)** had the smartphone GPS configured for high-rate, true **1.0 Hz** logging.

---

## 3. High-Quality Candidate Comparison

The table below contrasts the top candidates against the diagnosed sequence `Vfa01`:

| Sequence | Duration | IMU Rate | Unique GPS | GPS Interval | GPS Repeat % | S/V Overlap | S/V Offset | Speed Max | Stationary % | 10s Windows | 30s Windows | 60s Windows | Overall Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Vta2`** | **1,099.0s** | **10.0 Hz** | **1,023** | **1.0s** | **90.7%** | **1,098.0s** | **+0.96s** | **22.68 m/s** | **6.6%** | **6** | **5** | **5** | **OPTIMAL (Selected)** |
| `Vta1a` | 2,567.5s | 10.0 Hz | 2,520 | 1.0s | 90.2% | 2,553.9s | -13.64s | 28.73 m/s | 0.5% | 5 | 5 | 5 | Excellent (Long secondary) |
| `Vta1b` | 95.3s | 10.0 Hz | 95 | 1.0s | 90.0% | 94.7s | +0.46s | 21.58 m/s | 1.0% | 2 | 1 | 1 | Too short for multi-run |
| `Vtb1` | 3,245.8s | 10.0 Hz | 280 | 9.0s | 99.1% | 3,245.1s | -0.65s | 28.10 m/s | 7.5% | 21 | 15 | 11 | Slow GPS (9s artifact) |
| `Vfa02` | 6,752.2s | 10.0 Hz | 747 | 9.0s | 98.9% | 6,752.2s | +7.75s | 32.75 m/s | 1.1% | 15 | 9 | 7 | Slow GPS (9s artifact) |
| `Vw4` | 12,652.5s | 10.0 Hz | 1,340 | 9.0s | 98.9% | 12,652.3s | +0.30s | 30.55 m/s | 6.8% | 56 | 46 | 38 | Slow GPS (9s artifact) |
| `Vfa01` *(debug)* | 1,148.5s | 10.0 Hz | 123 | 9.0s | 98.9% | 1,148.5s | +4.24s | 27.34 m/s | 6.3% | 4 | 2 | 2 | Sanity baseline (Step 9A) |

*Note on GPS Repeat %*: At 10 Hz IMU rate and 1 Hz GPS rate, 1 GPS fix is logged alongside 9 repeated intermediate readings, yielding an expected theoretical minimum repeat percentage of $\frac{9}{10} = 90.0\%$.

---

## 4. Deep-Dive Quality Analysis of `Vta2`

### A. Temporal Alignment & Synchronization
- **S-Dataset Time**: Day-seconds $[36,000.96\text{s}, 37,099.96\text{s}]$ ($10,991$ rows).
- **V-Dataset Time**: Day-seconds $[36,000.00\text{s}, 37,099.00\text{s}]$ ($10,991$ rows).
- **Row-to-Row Match**: **Identical row count** ($10,991$ rows in S, $10,991$ rows in V).
- **Initial Time Offset**: **$+0.96\text{ seconds}$** (minimal latency, preventing geodetic origin translation drift).
- **Overlap Duration**: $1,098.0\text{ seconds}$ ($99.91\%$ temporal overlap).

### B. IMU Data Integrity & Physical Plausibility
- **Accelerometer ($m/s^2$)**:
  - $a_x$: Min $-11.902$, Max $+12.014$, Median $+0.167\text{ m/s}^2$
  - $a_y$: Min $-21.149$, Max $+16.332$, Median $-0.475\text{ m/s}^2$
  - $a_z$: Min $+5.433$, Max $+13.436$, Median $+9.827\text{ m/s}^2$
  - Accelerometer Norm: Min $6.025$, Max $25.394$, Mean $10.267$, Median $10.071$, P95 $11.986\text{ m/s}^2$.
  - **Empirical Check**: Median $a_z \approx 9.83\text{ m/s}^2$ confirms screen-up phone orientation, zero sensor saturation, and proper $\text{m/s}^2$ physical units.
- **Gyroscope ($rad/s$)**:
  - $\omega_x$: Min $-0.580$, Max $+0.457$, Median $-0.003\text{ rad/s}$
  - $\omega_y$: Min $-1.838$, Max $+1.954$, Median $+0.005\text{ rad/s}$
  - $\omega_z$: Min $-1.165$, Max $+1.105$, Median $-0.004\text{ rad/s}$
  - Gyroscope Norm: Min $0.003$, Max $2.287$, Mean $0.275$, Median $0.209$, P95 $0.738\text{ rad/s}$.
  - **Empirical Check**: Small median biases ($< 0.005\text{ rad/s}$) and typical automotive turn rates confirm $\text{rad/s}$ units.
- **Sampling Interval Stability**:
  - $\Delta t$: Min $90.0\text{ ms}$, Max $108.0\text{ ms}$, Mean $100.00\text{ ms}$, Median $100.00\text{ ms}$, P95 $101.00\text{ ms}$.
  - Effective Sampling Rate: **$10.00\text{ Hz}$** ($0$ dropped frames).
- **Missing / Corrupt Values**: **$0.00\%$** (0 NaN, 0 Inf, 0 null fields).

### C. Motion Profile & Driving Dynamics
- **Speed Statistics**:
  - Min: $0.00\text{ m/s}$ ($0.0\text{ km/h}$)
  - Max: $22.68\text{ m/s}$ ($81.65\text{ km/h}$)
  - Mean: $10.05\text{ m/s}$ ($36.18\text{ km/h}$)
  - Median: $10.23\text{ m/s}$ ($36.83\text{ km/h}$)
- **Motion Distribution**:
  - Stationary: $6.6\%$ of trajectory
  - Active Driving: $93.4\%$ of trajectory
  - Contains multiple natural stops (traffic lights/intersections) ideal for ZUPT verification, alongside sustained high-speed cruising for dead-reckoning stress tests.
- **Heading Diversity**:
  - Full $360^\circ$ coverage (Min $0.01^\circ$, Max $359.99^\circ$, Median $77.71^\circ$).
  - Multiple left/right turns, roundabouts, and straight highway sections.

### D. Outage Window Capacity
- Contains **6 valid continuous driving windows of $\ge 10\text{s}$**.
- Contains **5 valid continuous driving windows of $\ge 30\text{s}$**.
- Contains **5 valid continuous driving windows of $\ge 60\text{s}$**.
- Supports reproducible 10s, 30s, and 60s outage evaluations with zero data gaps.

---

## 5. Selection Decision & Justification

### `SELECTED_EVALUATION_SEQUENCE`: **`Vta2`**

#### Primary Selection Justifications:
1. **True 1.0 Hz GPS Updates**: Unlike $93\%$ of the dataset which suffers from the 9-second logging artifact, `Vta2` has genuine 1 Hz GPS updates ($1,023$ unique fixes across $1,099$ seconds), allowing the EKF to properly track trajectory dynamics without artificial gating lockouts.
2. **Near-Perfect Temporal Alignment**: Initial offset between S-Dataset and V-Dataset is only **$+0.96\text{ seconds}$** (compared to $4.24\text{s}$ in `Vfa01` and $> 3,600\text{s}$ in `M`/`S2`/`S4`), minimizing initial coordinate anchor drift.
3. **Exact Row Count Symmetry**: Both S and V files contain exactly $10,991$ rows at $10.0\text{ Hz}$, guaranteeing 1-to-1 temporal correspondence.
4. **Balanced Motion Characteristics**: $93.4\%$ driving with $6.6\%$ stationary intervals allows simultaneous evaluation of high-speed INS integration, NHC constraints, and stationary ZUPT detection.
5. **Practical Duration**: At $18.3\text{ minutes}$ ($1,099\text{s}$), it is sufficiently long to evaluate multiple 10s, 30s, and 60s outages, yet fast enough for rapid automated testing and CI regression.

---

## 6. Remaining Data Quality Considerations

When executing dead-reckoning evaluations on `Vta2`:
1. **GPS Timestamp De-duplication**: Even with 1 Hz GPS, intermediate 10 Hz rows repeat the GPS fix 9 times. Replay loaders must gate GPS updates on `unique_fix_timestamp` or `sample_delta` to update the Kalman filter only upon receiving a genuinely new satellite fix.
2. **Velocity TCN Normalization**: As established in Step 9A, the vertical accelerometer input to Velocity TCN must have Earth gravity accounted for to match the model's training distribution.
3. **Secondary Benchmark (`Vta1a`)**: If ultra-long evaluation ($> 30\text{ minutes}$) is required in the future, `Vta1a` ($2,567.5\text{s}$) serves as a direct companion sequence.
