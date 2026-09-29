#!/usr/bin/env python3
"""
Step 9 Real Dead-Reckoning Evaluation Runner for OdysseyV1.
Loads held-out IO-VNBD test sequence, executes controlled 10s and 30s outages,
runs controlled ablation configurations (A through F),
and exports machine-readable and human-readable reports.
"""

from pathlib import Path
import sys
import time

from edge.evaluation.metrics import OutageEvaluationMetrics
from edge.evaluation.replay import (
    AblationConfig,
    AblationMode,
    IOVNBDLoader,
    OutageWindow,
    ReplayEvaluator,
)
from edge.evaluation.report import (
    generate_evaluation_svg_plots,
    generate_markdown_report,
    save_results_csv,
    save_results_json,
    save_trajectory_file,
)


def main():
    print("=" * 60)
    print("OdysseyV1 Step 9: Real IO-VNBD Dead-Reckoning Evaluation")
    print("=" * 60)

    # 1. Load IO-VNBD Sequence
    loader = IOVNBDLoader()
    seq_name = "Vfa01"
    split = loader.get_sequence_split(seq_name)
    print(f"Loading sequence '{seq_name}' from {loader.dataset_root}...")

    # Load first 1,500 samples (~150 seconds = 2.5 minutes) for focused, high-precision evaluation
    samples, meta = loader.load_sequence(seq_name, max_samples=1500)
    total_dur_s = meta["duration_s"]
    print(f"Loaded {len(samples)} samples spanning {total_dur_s:.1f}s | Split: {split}")

    evaluator = ReplayEvaluator()
    results: list[OutageEvaluationMetrics] = []
    trajectories_map = {}

    # 2. Outage Scenarios: 10s Outage and 30s Outage
    outage_configs = [
        {"duration_s": 10.0, "start_s": 25.0},
        {"duration_s": 30.0, "start_s": 45.0},
    ]

    ablation_modes = [
        AblationMode.A_INS_ONLY,
        AblationMode.B_INS_EKF,
        AblationMode.C_INS_EKF_AI,
        AblationMode.D_INS_EKF_AI_NHC,
        AblationMode.E_INS_EKF_AI_NHC_ZUPT,
        AblationMode.F_FULL_ODYSSEY,
    ]

    for out_cfg in outage_configs:
        dur = out_cfg["duration_s"]
        st = out_cfg["start_s"]
        outage = OutageWindow(start_s=st, duration_s=dur, recovery_duration_s=10.0)

        print(f"\n--- Running Evaluations for Outage Duration: {dur:.0f}s (Start: {st:.0f}s) ---")

        for mode in ablation_modes:
            ablation = AblationConfig.from_mode(mode)
            print(f"  Evaluating Config: {mode.value:25s} ...", end=" ", flush=True)
            t_start = time.time()
            metrics, traj = evaluator.run_outage_evaluation(
                samples=samples,
                outage=outage,
                ablation=ablation,
                sequence_name=seq_name,
                split=split,
                dataset_name="IO-VNBD",
            )
            elapsed = time.time() - t_start
            results.append(metrics)
            trajectories_map[f"{mode.value}_{dur:.0f}s"] = traj

            drift_str = f"{metrics.drift_percent:.2f}%" if metrics.drift_percent is not None else "N/A"
            end_err_str = f"{metrics.end_position_error_m:.2f}m" if metrics.end_position_error_m is not None else "N/A"
            print(f"Done ({elapsed:.2f}s) | End Err: {end_err_str} | Drift: {drift_str}")

    # 3. Export Reports and Outputs
    out_dir = Path(__file__).resolve().parent.parent / "evaluation_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "evaluation_results.json"
    csv_path = out_dir / "evaluation_results.csv"
    md_path = out_dir / "evaluation_report.md"
    traj_path = out_dir / "trajectory_full_odyssey_10s.csv"
    svg_path = out_dir / "trajectory_full_odyssey_10s.svg"

    save_results_json(results, json_path)
    save_results_csv(results, csv_path)
    report_md = generate_markdown_report(results, md_path)

    # Save representative full trajectory & SVG
    if "F_FULL_ODYSSEY_10s" in trajectories_map:
        save_trajectory_file(trajectories_map["F_FULL_ODYSSEY_10s"], traj_path)
        # Find corresponding metric
        f_metric = next(r for r in results if r.configuration == "F_FULL_ODYSSEY" and r.outage_duration_s == 10.0)
        generate_evaluation_svg_plots(trajectories_map["F_FULL_ODYSSEY_10s"], f_metric, svg_path)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"JSON Output:       {json_path}")
    print(f"CSV Output:        {csv_path}")
    print(f"Markdown Report:   {md_path}")
    print(f"Trajectory CSV:    {traj_path}")
    print(f"Trajectory SVG:    {svg_path}")
    print("=" * 60)

    # Also display summary table in console
    print("\nSummary Results:")
    print(f"{'Config':<25} | {'Outage':<6} | {'End Err (m)':<12} | {'RMSE (m)':<10} | {'Drift %':<10} | {'Vel RMSE':<10}")
    print("-" * 80)
    for r in results:
        drift_str = f"{r.drift_percent:.2f}%" if r.drift_percent is not None else "N/A"
        end_err = f"{r.end_position_error_m:.2f}" if r.end_position_error_m is not None else "N/A"
        rmse = f"{r.position_rmse_m:.2f}" if r.position_rmse_m is not None else "N/A"
        vel_rmse = f"{r.velocity_rmse_mps:.2f}" if r.velocity_rmse_mps is not None else "N/A"
        print(f"{r.configuration:<25} | {r.outage_duration_s:.0f}s     | {end_err:<12} | {rmse:<10} | {drift_str:<10} | {vel_rmse:<10}")


if __name__ == "__main__":
    main()
