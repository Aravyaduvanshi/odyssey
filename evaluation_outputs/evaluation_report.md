# OdysseyV1 Dead-Reckoning Evaluation Report

## Executive Summary

This report measures OdysseyV1 dead-reckoning performance during controlled GNSS blackout windows.
Ground truth is strictly isolated and used solely for post-processing evaluation.

| Sequence | Split | Config | Outage (s) | GT Dist (m) | End Error (m) | RMSE (m) | P95 (m) | Drift % | Velocity RMSE | Recovery Time |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Vfa01` | `TEST` | `A_INS_ONLY` | 10s | 191.6 | 2877.10 | 2082.52 | 2779.08 | **1501.63%** | 247.90 m/s | N/A |
| `Vfa01` | `TEST` | `B_INS_EKF` | 10s | 191.6 | 2024.14 | 1451.58 | 1952.33 | **1056.45%** | 186.33 m/s | N/A |
| `Vfa01` | `TEST` | `C_INS_EKF_AI` | 10s | 191.6 | 2024.14 | 1451.58 | 1952.33 | **1056.45%** | 186.33 m/s | N/A |
| `Vfa01` | `TEST` | `D_INS_EKF_AI_NHC` | 10s | 191.6 | 1801.82 | 1271.70 | 1733.89 | **940.41%** | 178.00 m/s | N/A |
| `Vfa01` | `TEST` | `E_INS_EKF_AI_NHC_ZUPT` | 10s | 191.6 | 1801.82 | 1271.70 | 1733.89 | **940.41%** | 178.00 m/s | N/A |
| `Vfa01` | `TEST` | `F_FULL_ODYSSEY` | 10s | 191.6 | 1801.82 | 1271.70 | 1733.89 | **940.41%** | 178.00 m/s | N/A |
| `Vfa01` | `TEST` | `A_INS_ONLY` | 30s | 575.7 | 16072.22 | 10801.32 | 15429.10 | **2791.68%** | 565.23 m/s | N/A |
| `Vfa01` | `TEST` | `B_INS_EKF` | 30s | 575.7 | 12781.24 | 8178.94 | 12142.16 | **2220.05%** | 605.53 m/s | N/A |
| `Vfa01` | `TEST` | `C_INS_EKF_AI` | 30s | 575.7 | 12781.24 | 8178.94 | 12142.16 | **2220.05%** | 605.53 m/s | N/A |
| `Vfa01` | `TEST` | `D_INS_EKF_AI_NHC` | 30s | 575.7 | 14763.41 | 8853.44 | 13908.62 | **2564.34%** | 747.20 m/s | N/A |
| `Vfa01` | `TEST` | `E_INS_EKF_AI_NHC_ZUPT` | 30s | 575.7 | 14763.41 | 8853.44 | 13908.62 | **2564.34%** | 747.20 m/s | N/A |
| `Vfa01` | `TEST` | `F_FULL_ODYSSEY` | 30s | 575.7 | 14763.41 | 8853.44 | 13908.62 | **2564.34%** | 747.20 m/s | N/A |

## Detailed Analysis per Experiment

### Experiment 1: Vfa01 — A_INS_ONLY (10s Outage)

- **Dataset**: `IO-VNBD` | **Sequence**: `Vfa01` | **Split**: `TEST`
- **Configuration**: `A_INS_ONLY`
- **Outage Window**: `26.0s` to `36.0s` (101 samples)

#### Position & Drift Metrics
- **Ground-Truth Traveled Distance**: 191.60 m
- **Estimated Traveled Distance**: 1464.59 m
- **End Position Error**: 2877.10 m
- **RMSE Position Error**: 2082.52 m
- **Mean Position Error**: 2032.90 m
- **Median Position Error**: 1997.27 m
- **P95 Position Error**: 2779.08 m
- **Maximum Position Error**: 2877.10 m
- **End Drift %**: 1501.63%
- **Max Drift %**: 1501.63%

#### Relative Trajectory Error (RPE)
- **RPE End**: 1548.03 m
- **RPE Max**: 1548.03 m
- **RPE RMSE**: 836.00 m

#### Velocity & Dynamics Metrics
- **Velocity RMSE**: 247.90 m/s
- **Velocity MAE**: 244.14 m/s
- **Velocity P95**: 2779.08 m/s
- **Yaw RMSE**: 15.63°
- **AI Velocity Active Samples**: 101
- **NHC Active Samples**: 0
- **ZUPT Active Samples**: 0

#### GNSS Recovery Behavior
- **Recovery Evaluated**: True
- **Recovery Time until GNSS_INS**: N/A
- **Position Error Pre-Recovery**: 2877.10 m
- **Position Error Post 1st GNSS Update**: 2897.02 m
- **Position Jump at 1st GNSS Update**: 19.10 m
- **Accepted Recovery Updates**: 99

### Experiment 2: Vfa01 — B_INS_EKF (10s Outage)

- **Dataset**: `IO-VNBD` | **Sequence**: `Vfa01` | **Split**: `TEST`
- **Configuration**: `B_INS_EKF`
- **Outage Window**: `26.0s` to `36.0s` (101 samples)

#### Position & Drift Metrics
- **Ground-Truth Traveled Distance**: 191.60 m
- **Estimated Traveled Distance**: 975.90 m
- **End Position Error**: 2024.14 m
- **RMSE Position Error**: 1451.58 m
- **Mean Position Error**: 1415.41 m
- **Median Position Error**: 1385.42 m
- **P95 Position Error**: 1952.33 m
- **Maximum Position Error**: 2024.14 m
- **End Drift %**: 1056.45%
- **Max Drift %**: 1056.45%

#### Relative Trajectory Error (RPE)
- **RPE End**: 1123.80 m
- **RPE Max**: 1123.80 m
- **RPE RMSE**: 600.87 m

#### Velocity & Dynamics Metrics
- **Velocity RMSE**: 186.33 m/s
- **Velocity MAE**: 182.19 m/s
- **Velocity P95**: 1952.33 m/s
- **Yaw RMSE**: 6.74°
- **AI Velocity Active Samples**: 101
- **NHC Active Samples**: 0
- **ZUPT Active Samples**: 0

#### GNSS Recovery Behavior
- **Recovery Evaluated**: True
- **Recovery Time until GNSS_INS**: N/A
- **Position Error Pre-Recovery**: 2024.14 m
- **Position Error Post 1st GNSS Update**: 2038.73 m
- **Position Jump at 1st GNSS Update**: 13.27 m
- **Accepted Recovery Updates**: 99

### Experiment 3: Vfa01 — C_INS_EKF_AI (10s Outage)

- **Dataset**: `IO-VNBD` | **Sequence**: `Vfa01` | **Split**: `TEST`
- **Configuration**: `C_INS_EKF_AI`
- **Outage Window**: `26.0s` to `36.0s` (101 samples)

#### Position & Drift Metrics
- **Ground-Truth Traveled Distance**: 191.60 m
- **Estimated Traveled Distance**: 975.90 m
- **End Position Error**: 2024.14 m
- **RMSE Position Error**: 1451.58 m
- **Mean Position Error**: 1415.41 m
- **Median Position Error**: 1385.42 m
- **P95 Position Error**: 1952.33 m
- **Maximum Position Error**: 2024.14 m
- **End Drift %**: 1056.45%
- **Max Drift %**: 1056.45%

#### Relative Trajectory Error (RPE)
- **RPE End**: 1123.80 m
- **RPE Max**: 1123.80 m
- **RPE RMSE**: 600.87 m

#### Velocity & Dynamics Metrics
- **Velocity RMSE**: 186.33 m/s
- **Velocity MAE**: 182.19 m/s
- **Velocity P95**: 1952.33 m/s
- **Yaw RMSE**: 6.74°
- **AI Velocity Active Samples**: 101
- **NHC Active Samples**: 0
- **ZUPT Active Samples**: 0

#### GNSS Recovery Behavior
- **Recovery Evaluated**: True
- **Recovery Time until GNSS_INS**: N/A
- **Position Error Pre-Recovery**: 2024.14 m
- **Position Error Post 1st GNSS Update**: 2038.73 m
- **Position Jump at 1st GNSS Update**: 13.27 m
- **Accepted Recovery Updates**: 99

### Experiment 4: Vfa01 — D_INS_EKF_AI_NHC (10s Outage)

- **Dataset**: `IO-VNBD` | **Sequence**: `Vfa01` | **Split**: `TEST`
- **Configuration**: `D_INS_EKF_AI_NHC`
- **Outage Window**: `26.0s` to `36.0s` (101 samples)

#### Position & Drift Metrics
- **Ground-Truth Traveled Distance**: 191.60 m
- **Estimated Traveled Distance**: 887.40 m
- **End Position Error**: 1801.82 m
- **RMSE Position Error**: 1271.70 m
- **Mean Position Error**: 1237.06 m
- **Median Position Error**: 1204.91 m
- **P95 Position Error**: 1733.89 m
- **Maximum Position Error**: 1801.82 m
- **End Drift %**: 940.41%
- **Max Drift %**: 940.41%

#### Relative Trajectory Error (RPE)
- **RPE End**: 1062.35 m
- **RPE Max**: 1062.35 m
- **RPE RMSE**: 563.86 m

#### Velocity & Dynamics Metrics
- **Velocity RMSE**: 178.00 m/s
- **Velocity MAE**: 173.13 m/s
- **Velocity P95**: 1733.89 m/s
- **Yaw RMSE**: 5.95°
- **AI Velocity Active Samples**: 101
- **NHC Active Samples**: 0
- **ZUPT Active Samples**: 0

#### GNSS Recovery Behavior
- **Recovery Evaluated**: True
- **Recovery Time until GNSS_INS**: N/A
- **Position Error Pre-Recovery**: 1801.82 m
- **Position Error Post 1st GNSS Update**: 1815.64 m
- **Position Jump at 1st GNSS Update**: 12.49 m
- **Accepted Recovery Updates**: 99

### Experiment 5: Vfa01 — E_INS_EKF_AI_NHC_ZUPT (10s Outage)

- **Dataset**: `IO-VNBD` | **Sequence**: `Vfa01` | **Split**: `TEST`
- **Configuration**: `E_INS_EKF_AI_NHC_ZUPT`
- **Outage Window**: `26.0s` to `36.0s` (101 samples)

#### Position & Drift Metrics
- **Ground-Truth Traveled Distance**: 191.60 m
- **Estimated Traveled Distance**: 887.40 m
- **End Position Error**: 1801.82 m
- **RMSE Position Error**: 1271.70 m
- **Mean Position Error**: 1237.06 m
- **Median Position Error**: 1204.91 m
- **P95 Position Error**: 1733.89 m
- **Maximum Position Error**: 1801.82 m
- **End Drift %**: 940.41%
- **Max Drift %**: 940.41%

#### Relative Trajectory Error (RPE)
- **RPE End**: 1062.35 m
- **RPE Max**: 1062.35 m
- **RPE RMSE**: 563.86 m

#### Velocity & Dynamics Metrics
- **Velocity RMSE**: 178.00 m/s
- **Velocity MAE**: 173.13 m/s
- **Velocity P95**: 1733.89 m/s
- **Yaw RMSE**: 5.95°
- **AI Velocity Active Samples**: 101
- **NHC Active Samples**: 0
- **ZUPT Active Samples**: 0

#### GNSS Recovery Behavior
- **Recovery Evaluated**: True
- **Recovery Time until GNSS_INS**: N/A
- **Position Error Pre-Recovery**: 1801.82 m
- **Position Error Post 1st GNSS Update**: 1815.64 m
- **Position Jump at 1st GNSS Update**: 12.49 m
- **Accepted Recovery Updates**: 99

### Experiment 6: Vfa01 — F_FULL_ODYSSEY (10s Outage)

- **Dataset**: `IO-VNBD` | **Sequence**: `Vfa01` | **Split**: `TEST`
- **Configuration**: `F_FULL_ODYSSEY`
- **Outage Window**: `26.0s` to `36.0s` (101 samples)

#### Position & Drift Metrics
- **Ground-Truth Traveled Distance**: 191.60 m
- **Estimated Traveled Distance**: 887.40 m
- **End Position Error**: 1801.82 m
- **RMSE Position Error**: 1271.70 m
- **Mean Position Error**: 1237.06 m
- **Median Position Error**: 1204.91 m
- **P95 Position Error**: 1733.89 m
- **Maximum Position Error**: 1801.82 m
- **End Drift %**: 940.41%
- **Max Drift %**: 940.41%

#### Relative Trajectory Error (RPE)
- **RPE End**: 1062.35 m
- **RPE Max**: 1062.35 m
- **RPE RMSE**: 563.86 m

#### Velocity & Dynamics Metrics
- **Velocity RMSE**: 178.00 m/s
- **Velocity MAE**: 173.13 m/s
- **Velocity P95**: 1733.89 m/s
- **Yaw RMSE**: 5.95°
- **AI Velocity Active Samples**: 101
- **NHC Active Samples**: 0
- **ZUPT Active Samples**: 0

#### GNSS Recovery Behavior
- **Recovery Evaluated**: True
- **Recovery Time until GNSS_INS**: N/A
- **Position Error Pre-Recovery**: 1801.82 m
- **Position Error Post 1st GNSS Update**: 1815.64 m
- **Position Jump at 1st GNSS Update**: 12.49 m
- **Accepted Recovery Updates**: 99

### Experiment 7: Vfa01 — A_INS_ONLY (30s Outage)

- **Dataset**: `IO-VNBD` | **Sequence**: `Vfa01` | **Split**: `TEST`
- **Configuration**: `A_INS_ONLY`
- **Outage Window**: `46.0s` to `76.0s` (300 samples)

#### Position & Drift Metrics
- **Ground-Truth Traveled Distance**: 575.72 m
- **Estimated Traveled Distance**: 10617.15 m
- **End Position Error**: 16072.22 m
- **RMSE Position Error**: 10801.32 m
- **Mean Position Error**: 10342.46 m
- **Median Position Error**: 10173.28 m
- **P95 Position Error**: 15429.10 m
- **Maximum Position Error**: 16072.22 m
- **End Drift %**: 2791.68%
- **Max Drift %**: 2791.68%

#### Relative Trajectory Error (RPE)
- **RPE End**: 10775.16 m
- **RPE Max**: 10775.16 m
- **RPE RMSE**: 5918.58 m

#### Velocity & Dynamics Metrics
- **Velocity RMSE**: 565.23 m/s
- **Velocity MAE**: 562.85 m/s
- **Velocity P95**: 15429.10 m/s
- **Yaw RMSE**: 27.10°
- **AI Velocity Active Samples**: 300
- **NHC Active Samples**: 0
- **ZUPT Active Samples**: 0

#### GNSS Recovery Behavior
- **Recovery Evaluated**: True
- **Recovery Time until GNSS_INS**: N/A
- **Position Error Pre-Recovery**: 16072.22 m
- **Position Error Post 1st GNSS Update**: 16115.67 m
- **Position Jump at 1st GNSS Update**: 43.45 m
- **Accepted Recovery Updates**: 101

### Experiment 8: Vfa01 — B_INS_EKF (30s Outage)

- **Dataset**: `IO-VNBD` | **Sequence**: `Vfa01` | **Split**: `TEST`
- **Configuration**: `B_INS_EKF`
- **Outage Window**: `46.0s` to `76.0s` (300 samples)

#### Position & Drift Metrics
- **Ground-Truth Traveled Distance**: 575.72 m
- **Estimated Traveled Distance**: 9197.55 m
- **End Position Error**: 12781.24 m
- **RMSE Position Error**: 8178.94 m
- **Mean Position Error**: 7766.80 m
- **Median Position Error**: 7507.63 m
- **P95 Position Error**: 12142.16 m
- **Maximum Position Error**: 12781.24 m
- **End Drift %**: 2220.05%
- **Max Drift %**: 2220.05%

#### Relative Trajectory Error (RPE)
- **RPE End**: 9489.79 m
- **RPE Max**: 9489.79 m
- **RPE RMSE**: 4960.67 m

#### Velocity & Dynamics Metrics
- **Velocity RMSE**: 605.53 m/s
- **Velocity MAE**: 588.38 m/s
- **Velocity P95**: 12142.16 m/s
- **Yaw RMSE**: 61.65°
- **AI Velocity Active Samples**: 300
- **NHC Active Samples**: 0
- **ZUPT Active Samples**: 0

#### GNSS Recovery Behavior
- **Recovery Evaluated**: True
- **Recovery Time until GNSS_INS**: N/A
- **Position Error Pre-Recovery**: 12781.24 m
- **Position Error Post 1st GNSS Update**: 12825.07 m
- **Position Jump at 1st GNSS Update**: 46.74 m
- **Accepted Recovery Updates**: 101

### Experiment 9: Vfa01 — C_INS_EKF_AI (30s Outage)

- **Dataset**: `IO-VNBD` | **Sequence**: `Vfa01` | **Split**: `TEST`
- **Configuration**: `C_INS_EKF_AI`
- **Outage Window**: `46.0s` to `76.0s` (300 samples)

#### Position & Drift Metrics
- **Ground-Truth Traveled Distance**: 575.72 m
- **Estimated Traveled Distance**: 9197.55 m
- **End Position Error**: 12781.24 m
- **RMSE Position Error**: 8178.94 m
- **Mean Position Error**: 7766.80 m
- **Median Position Error**: 7507.63 m
- **P95 Position Error**: 12142.16 m
- **Maximum Position Error**: 12781.24 m
- **End Drift %**: 2220.05%
- **Max Drift %**: 2220.05%

#### Relative Trajectory Error (RPE)
- **RPE End**: 9489.79 m
- **RPE Max**: 9489.79 m
- **RPE RMSE**: 4960.67 m

#### Velocity & Dynamics Metrics
- **Velocity RMSE**: 605.53 m/s
- **Velocity MAE**: 588.38 m/s
- **Velocity P95**: 12142.16 m/s
- **Yaw RMSE**: 61.65°
- **AI Velocity Active Samples**: 300
- **NHC Active Samples**: 0
- **ZUPT Active Samples**: 0

#### GNSS Recovery Behavior
- **Recovery Evaluated**: True
- **Recovery Time until GNSS_INS**: N/A
- **Position Error Pre-Recovery**: 12781.24 m
- **Position Error Post 1st GNSS Update**: 12825.07 m
- **Position Jump at 1st GNSS Update**: 46.74 m
- **Accepted Recovery Updates**: 101

### Experiment 10: Vfa01 — D_INS_EKF_AI_NHC (30s Outage)

- **Dataset**: `IO-VNBD` | **Sequence**: `Vfa01` | **Split**: `TEST`
- **Configuration**: `D_INS_EKF_AI_NHC`
- **Outage Window**: `46.0s` to `76.0s` (300 samples)

#### Position & Drift Metrics
- **Ground-Truth Traveled Distance**: 575.72 m
- **Estimated Traveled Distance**: 11549.90 m
- **End Position Error**: 14763.41 m
- **RMSE Position Error**: 8853.44 m
- **Mean Position Error**: 8243.60 m
- **Median Position Error**: 7778.76 m
- **P95 Position Error**: 13908.62 m
- **Maximum Position Error**: 14763.41 m
- **End Drift %**: 2564.34%
- **Max Drift %**: 2564.34%

#### Relative Trajectory Error (RPE)
- **RPE End**: 11805.41 m
- **RPE Max**: 11805.41 m
- **RPE RMSE**: 6011.39 m

#### Velocity & Dynamics Metrics
- **Velocity RMSE**: 747.20 m/s
- **Velocity MAE**: 717.92 m/s
- **Velocity P95**: 13908.62 m/s
- **Yaw RMSE**: 107.81°
- **AI Velocity Active Samples**: 300
- **NHC Active Samples**: 0
- **ZUPT Active Samples**: 0

#### GNSS Recovery Behavior
- **Recovery Evaluated**: True
- **Recovery Time until GNSS_INS**: N/A
- **Position Error Pre-Recovery**: 14763.41 m
- **Position Error Post 1st GNSS Update**: 14821.93 m
- **Position Jump at 1st GNSS Update**: 61.01 m
- **Accepted Recovery Updates**: 101

### Experiment 11: Vfa01 — E_INS_EKF_AI_NHC_ZUPT (30s Outage)

- **Dataset**: `IO-VNBD` | **Sequence**: `Vfa01` | **Split**: `TEST`
- **Configuration**: `E_INS_EKF_AI_NHC_ZUPT`
- **Outage Window**: `46.0s` to `76.0s` (300 samples)

#### Position & Drift Metrics
- **Ground-Truth Traveled Distance**: 575.72 m
- **Estimated Traveled Distance**: 11549.90 m
- **End Position Error**: 14763.41 m
- **RMSE Position Error**: 8853.44 m
- **Mean Position Error**: 8243.60 m
- **Median Position Error**: 7778.76 m
- **P95 Position Error**: 13908.62 m
- **Maximum Position Error**: 14763.41 m
- **End Drift %**: 2564.34%
- **Max Drift %**: 2564.34%

#### Relative Trajectory Error (RPE)
- **RPE End**: 11805.41 m
- **RPE Max**: 11805.41 m
- **RPE RMSE**: 6011.39 m

#### Velocity & Dynamics Metrics
- **Velocity RMSE**: 747.20 m/s
- **Velocity MAE**: 717.92 m/s
- **Velocity P95**: 13908.62 m/s
- **Yaw RMSE**: 107.81°
- **AI Velocity Active Samples**: 300
- **NHC Active Samples**: 0
- **ZUPT Active Samples**: 0

#### GNSS Recovery Behavior
- **Recovery Evaluated**: True
- **Recovery Time until GNSS_INS**: N/A
- **Position Error Pre-Recovery**: 14763.41 m
- **Position Error Post 1st GNSS Update**: 14821.93 m
- **Position Jump at 1st GNSS Update**: 61.01 m
- **Accepted Recovery Updates**: 101

### Experiment 12: Vfa01 — F_FULL_ODYSSEY (30s Outage)

- **Dataset**: `IO-VNBD` | **Sequence**: `Vfa01` | **Split**: `TEST`
- **Configuration**: `F_FULL_ODYSSEY`
- **Outage Window**: `46.0s` to `76.0s` (300 samples)

#### Position & Drift Metrics
- **Ground-Truth Traveled Distance**: 575.72 m
- **Estimated Traveled Distance**: 11549.90 m
- **End Position Error**: 14763.41 m
- **RMSE Position Error**: 8853.44 m
- **Mean Position Error**: 8243.60 m
- **Median Position Error**: 7778.76 m
- **P95 Position Error**: 13908.62 m
- **Maximum Position Error**: 14763.41 m
- **End Drift %**: 2564.34%
- **Max Drift %**: 2564.34%

#### Relative Trajectory Error (RPE)
- **RPE End**: 11805.41 m
- **RPE Max**: 11805.41 m
- **RPE RMSE**: 6011.39 m

#### Velocity & Dynamics Metrics
- **Velocity RMSE**: 747.20 m/s
- **Velocity MAE**: 717.92 m/s
- **Velocity P95**: 13908.62 m/s
- **Yaw RMSE**: 107.81°
- **AI Velocity Active Samples**: 300
- **NHC Active Samples**: 0
- **ZUPT Active Samples**: 0

#### GNSS Recovery Behavior
- **Recovery Evaluated**: True
- **Recovery Time until GNSS_INS**: N/A
- **Position Error Pre-Recovery**: 14763.41 m
- **Position Error Post 1st GNSS Update**: 14821.93 m
- **Position Jump at 1st GNSS Update**: 61.01 m
- **Accepted Recovery Updates**: 101
