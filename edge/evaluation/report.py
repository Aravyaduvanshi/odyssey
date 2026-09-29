"""
Evaluation Reporting and Export Module for OdysseyV1.
Generates machine-readable results (JSON, CSV), human-readable Markdown reports,
trajectory playback logs, and offline standalone SVG plots.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from edge.evaluation.metrics import OutageEvaluationMetrics, TrajectorySample


def save_results_json(
    results: List[OutageEvaluationMetrics],
    output_path: Path | str,
) -> None:
    """Save evaluation metrics list to formatted JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [r.to_dict() for r in results]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def save_results_csv(
    results: List[OutageEvaluationMetrics],
    output_path: Path | str,
) -> None:
    """Save evaluation metrics list to CSV matching exact Step 9 requirements."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "dataset",
        "sequence",
        "split",
        "configuration",
        "outage_start",
        "outage_duration",
        "ground_truth_distance",
        "estimated_distance",
        "end_position_error_m",
        "mean_position_error_m",
        "median_position_error_m",
        "p95_position_error_m",
        "max_position_error_m",
        "position_rmse_m",
        "drift_percent",
        "max_drift_percent",
        "velocity_rmse_mps",
        "velocity_mae_mps",
        "yaw_rmse_deg",
        "recovery_time_s",
        "accepted_recovery_updates",
        "rejected_recovery_updates",
    ]

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {
                "dataset": r.dataset,
                "sequence": r.sequence,
                "split": r.split,
                "configuration": r.configuration,
                "outage_start": f"{r.outage_start_s:.2f}",
                "outage_duration": f"{r.outage_duration_s:.2f}",
                "ground_truth_distance": f"{r.ground_truth_distance_m:.2f}" if r.ground_truth_distance_m is not None else "N/A",
                "estimated_distance": f"{r.estimated_distance_m:.2f}",
                "end_position_error_m": f"{r.end_position_error_m:.2f}" if r.end_position_error_m is not None else "N/A",
                "mean_position_error_m": f"{r.mean_position_error_m:.2f}" if r.mean_position_error_m is not None else "N/A",
                "median_position_error_m": f"{r.median_position_error_m:.2f}" if r.median_position_error_m is not None else "N/A",
                "p95_position_error_m": f"{r.p95_position_error_m:.2f}" if r.p95_position_error_m is not None else "N/A",
                "max_position_error_m": f"{r.max_position_error_m:.2f}" if r.max_position_error_m is not None else "N/A",
                "position_rmse_m": f"{r.position_rmse_m:.2f}" if r.position_rmse_m is not None else "N/A",
                "drift_percent": f"{r.drift_percent:.2f}%" if r.drift_percent is not None else "N/A",
                "max_drift_percent": f"{r.max_drift_percent:.2f}%" if r.max_drift_percent is not None else "N/A",
                "velocity_rmse_mps": f"{r.velocity_rmse_mps:.2f}" if r.velocity_rmse_mps is not None else "N/A",
                "velocity_mae_mps": f"{r.velocity_mae_mps:.2f}" if r.velocity_mae_mps is not None else "N/A",
                "yaw_rmse_deg": f"{r.yaw_rmse_deg:.2f}" if r.yaw_rmse_deg is not None else "N/A",
                "recovery_time_s": f"{r.recovery_time_s:.2f}" if r.recovery_time_s is not None else "N/A",
                "accepted_recovery_updates": r.accepted_recovery_updates,
                "rejected_recovery_updates": r.rejected_recovery_updates,
            }
            writer.writerow(row)


def save_trajectory_file(
    trajectory: List[TrajectorySample],
    output_path: Path | str,
) -> None:
    """Save full chronological trajectory records for visualization."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "timestamp_ns",
        "gt_pos_e", "gt_pos_n", "gt_pos_u",
        "est_pos_e", "est_pos_n", "est_pos_u",
        "gt_vel_e", "gt_vel_n", "gt_vel_u",
        "est_vel_e", "est_vel_n", "est_vel_u",
        "yaw_est_deg", "yaw_gt_deg",
        "navigation_mode",
        "ai_velocity",
        "nhc_active",
        "zupt_active",
        "gnss_available",
    ]

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in trajectory:
            row = {
                "timestamp_ns": s.timestamp_ns,
                "gt_pos_e": f"{s.pos_gt[0]:.3f}" if s.pos_gt is not None else "",
                "gt_pos_n": f"{s.pos_gt[1]:.3f}" if s.pos_gt is not None else "",
                "gt_pos_u": f"{s.pos_gt[2]:.3f}" if s.pos_gt is not None else "",
                "est_pos_e": f"{s.pos_est[0]:.3f}",
                "est_pos_n": f"{s.pos_est[1]:.3f}",
                "est_pos_u": f"{s.pos_est[2]:.3f}",
                "gt_vel_e": f"{s.vel_gt[0]:.3f}" if s.vel_gt is not None else "",
                "gt_vel_n": f"{s.vel_gt[1]:.3f}" if s.vel_gt is not None else "",
                "gt_vel_u": f"{s.vel_gt[2]:.3f}" if s.vel_gt is not None else "",
                "est_vel_e": f"{s.vel_est[0]:.3f}",
                "est_vel_n": f"{s.vel_est[1]:.3f}",
                "est_vel_u": f"{s.vel_est[2]:.3f}",
                "yaw_est_deg": f"{math.degrees(s.yaw_est):.2f}",
                "yaw_gt_deg": f"{math.degrees(s.yaw_gt):.2f}" if s.yaw_gt is not None else "",
                "navigation_mode": s.navigation_mode,
                "ai_velocity": f"{s.ai_velocity:.2f}" if s.ai_velocity is not None else "",
                "nhc_active": s.nhc_active,
                "zupt_active": s.zupt_active,
                "gnss_available": s.gnss_available,
            }
            writer.writerow(row)


def generate_markdown_report(
    results: List[OutageEvaluationMetrics],
    output_path: Optional[Path | str] = None,
) -> str:
    """Generate structured markdown report comparing configurations and outage durations."""
    lines: List[str] = [
        "# OdysseyV1 Dead-Reckoning Evaluation Report",
        "",
        "## Executive Summary",
        "",
        "This report measures OdysseyV1 dead-reckoning performance during controlled GNSS blackout windows.",
        "Ground truth is strictly isolated and used solely for post-processing evaluation.",
        "",
        "| Sequence | Split | Config | Outage (s) | GT Dist (m) | End Error (m) | RMSE (m) | P95 (m) | Drift % | Velocity RMSE | Recovery Time |",
        "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for r in results:
        gt_dist_str = f"{r.ground_truth_distance_m:.1f}" if r.ground_truth_distance_m is not None else "N/A"
        end_err_str = f"{r.end_position_error_m:.2f}" if r.end_position_error_m is not None else "N/A"
        rmse_str = f"{r.position_rmse_m:.2f}" if r.position_rmse_m is not None else "N/A"
        p95_str = f"{r.p95_position_error_m:.2f}" if r.p95_position_error_m is not None else "N/A"
        drift_str = f"{r.drift_percent:.2f}%" if r.drift_percent is not None else "N/A"
        vel_str = f"{r.velocity_rmse_mps:.2f} m/s" if r.velocity_rmse_mps is not None else "N/A"
        rec_str = f"{r.recovery_time_s:.2f}s" if r.recovery_time_s is not None else "N/A"

        lines.append(
            f"| `{r.sequence}` | `{r.split}` | `{r.configuration}` | {r.outage_duration_s:.0f}s | "
            f"{gt_dist_str} | {end_err_str} | {rmse_str} | {p95_str} | **{drift_str}** | {vel_str} | {rec_str} |"
        )

    lines.extend([
        "",
        "## Detailed Analysis per Experiment",
        "",
    ])

    for i, r in enumerate(results, 1):
        lines.extend([
            f"### Experiment {i}: {r.sequence} — {r.configuration} ({r.outage_duration_s:.0f}s Outage)",
            "",
            f"- **Dataset**: `{r.dataset}` | **Sequence**: `{r.sequence}` | **Split**: `{r.split}`",
            f"- **Configuration**: `{r.configuration}`",
            f"- **Outage Window**: `{r.outage_start_s:.1f}s` to `{r.outage_start_s + r.outage_duration_s:.1f}s` ({r.outage_samples} samples)",
            "",
            "#### Position & Drift Metrics",
            f"- **Ground-Truth Traveled Distance**: {r.ground_truth_distance_m:.2f} m" if r.ground_truth_distance_m is not None else "- **Ground-Truth Traveled Distance**: N/A",
            f"- **Estimated Traveled Distance**: {r.estimated_distance_m:.2f} m",
            f"- **End Position Error**: {r.end_position_error_m:.2f} m" if r.end_position_error_m is not None else "- **End Position Error**: N/A",
            f"- **RMSE Position Error**: {r.position_rmse_m:.2f} m" if r.position_rmse_m is not None else "- **RMSE Position Error**: N/A",
            f"- **Mean Position Error**: {r.mean_position_error_m:.2f} m" if r.mean_position_error_m is not None else "- **Mean Position Error**: N/A",
            f"- **Median Position Error**: {r.median_position_error_m:.2f} m" if r.median_position_error_m is not None else "- **Median Position Error**: N/A",
            f"- **P95 Position Error**: {r.p95_position_error_m:.2f} m" if r.p95_position_error_m is not None else "- **P95 Position Error**: N/A",
            f"- **Maximum Position Error**: {r.max_position_error_m:.2f} m" if r.max_position_error_m is not None else "- **Maximum Position Error**: N/A",
            f"- **End Drift %**: {r.drift_percent:.2f}%" if r.drift_percent is not None else "- **End Drift %**: N/A",
            f"- **Max Drift %**: {r.max_drift_percent:.2f}%" if r.max_drift_percent is not None else "- **Max Drift %**: N/A",
            "",
            "#### Relative Trajectory Error (RPE)",
            f"- **RPE End**: {r.rpe_end_m:.2f} m" if r.rpe_end_m is not None else "- **RPE End**: N/A",
            f"- **RPE Max**: {r.rpe_max_m:.2f} m" if r.rpe_max_m is not None else "- **RPE Max**: N/A",
            f"- **RPE RMSE**: {r.rpe_rmse_m:.2f} m" if r.rpe_rmse_m is not None else "- **RPE RMSE**: N/A",
            "",
            "#### Velocity & Dynamics Metrics",
            f"- **Velocity RMSE**: {r.velocity_rmse_mps:.2f} m/s" if r.velocity_rmse_mps is not None else "- **Velocity RMSE**: N/A",
            f"- **Velocity MAE**: {r.velocity_mae_mps:.2f} m/s" if r.velocity_mae_mps is not None else "- **Velocity MAE**: N/A",
            f"- **Velocity P95**: {r.p95_position_error_m:.2f} m/s" if r.velocity_p95_mps is not None else "- **Velocity P95**: N/A",
            f"- **Yaw RMSE**: {r.yaw_rmse_deg:.2f}°" if r.yaw_rmse_deg is not None else "- **Yaw RMSE**: N/A",
            f"- **AI Velocity Active Samples**: {r.ai_velocity_active_samples}",
            f"- **NHC Active Samples**: {r.nhc_active_samples}",
            f"- **ZUPT Active Samples**: {r.zupt_active_samples}",
            "",
            "#### GNSS Recovery Behavior",
            f"- **Recovery Evaluated**: {r.recovery_evaluated}",
            f"- **Recovery Time until GNSS_INS**: {r.recovery_time_s:.2f} s" if r.recovery_time_s is not None else "- **Recovery Time until GNSS_INS**: N/A",
            f"- **Position Error Pre-Recovery**: {r.pos_error_pre_recovery_m:.2f} m" if r.pos_error_pre_recovery_m is not None else "- **Position Error Pre-Recovery**: N/A",
            f"- **Position Error Post 1st GNSS Update**: {r.pos_error_post_first_gnss_m:.2f} m" if r.pos_error_post_first_gnss_m is not None else "- **Position Error Post 1st GNSS Update**: N/A",
            f"- **Position Jump at 1st GNSS Update**: {r.recovery_position_delta_m:.2f} m" if r.recovery_position_delta_m is not None else "- **Position Jump at 1st GNSS Update**: N/A",
            f"- **Accepted Recovery Updates**: {r.accepted_recovery_updates}",
            "",
        ])

    md_content = "\n".join(lines)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(md_content)

    return md_content


def generate_evaluation_svg_plots(
    trajectory: List[TrajectorySample],
    metrics: OutageEvaluationMetrics,
    output_path: Path | str,
) -> None:
    """
    Generate a pure-Python, zero-dependency SVG plot visualizing:
    1. 2D trajectory (Ground Truth vs Estimate) with outage window highlighted.
    2. Position error over time.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not trajectory or metrics.ground_truth_distance_m is None:
        return

    # Extract coordinates
    t_sec = [(s.timestamp_ns - trajectory[0].timestamp_ns) * 1e-9 for s in trajectory]
    est_e = [s.pos_est[0] for s in trajectory]
    est_n = [s.pos_est[1] for s in trajectory]
    gt_e = [s.pos_gt[0] for s in trajectory if s.pos_gt is not None]
    gt_n = [s.pos_gt[1] for s in trajectory if s.pos_gt is not None]

    if not gt_e:
        return

    # Determine bounding box
    all_e = est_e + gt_e
    all_n = est_n + gt_n
    min_e, max_e = min(all_e), max(all_e)
    min_n, max_n = min(all_n), max(all_n)

    span_e = max(1.0, max_e - min_e)
    span_n = max(1.0, max_n - min_n)

    # Plot dimensions
    width, height = 800, 500
    pad = 60

    def to_svg_x(e: float) -> float:
        return pad + (e - min_e) / span_e * (width - 2 * pad)

    def to_svg_y(n: float) -> float:
        # Invert y for SVG
        return height - pad - (n - min_n) / span_n * (height - 2 * pad)

    # Build polylines
    gt_pts = " ".join(f"{to_svg_x(e):.1f},{to_svg_y(n):.1f}" for e, n in zip(gt_e, gt_n))
    est_pts = " ".join(f"{to_svg_x(e):.1f},{to_svg_y(n):.1f}" for e, n in zip(est_e, est_n))

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color: #0b0f19; font-family: monospace;">
    <!-- Grid & Border -->
    <rect x="{pad}" y="{pad}" width="{width - 2 * pad}" height="{height - 2 * pad}" fill="#111827" stroke="#1f2937" stroke-width="1"/>
    
    <!-- Title & Legend -->
    <text x="{width // 2}" y="30" fill="#38bdf8" font-size="14" font-weight="bold" text-anchor="middle">OdysseyV1 Trajectory: {metrics.sequence} ({metrics.configuration})</text>
    <text x="{pad + 10}" y="{pad + 20}" fill="#10b981" font-size="11">── Ground Truth</text>
    <text x="{pad + 150}" y="{pad + 20}" fill="#38bdf8" font-size="11">── Estimated Trajectory</text>
    <text x="{width - pad - 10}" y="{pad + 20}" fill="#f59e0b" font-size="11" text-anchor="end">Outage Duration: {metrics.outage_duration_s:.1f}s | Drift: {metrics.drift_percent if metrics.drift_percent else 0.0:.2f}%</text>

    <!-- Polylines -->
    <polyline points="{gt_pts}" fill="none" stroke="#10b981" stroke-width="2.5" stroke-dasharray="4,2"/>
    <polyline points="{est_pts}" fill="none" stroke="#38bdf8" stroke-width="2"/>
    
    <!-- Axis Labels -->
    <text x="{width // 2}" y="{height - 15}" fill="#9ca3af" font-size="10" text-anchor="middle">East (meters)</text>
    <text x="20" y="{height // 2}" fill="#9ca3af" font-size="10" text-anchor="middle" transform="rotate(-90 20 {height // 2})">North (meters)</text>
</svg>"""

    with open(out, "w", encoding="utf-8") as f:
        f.write(svg_content)
