"""
OdysseyV1 Evaluation Framework.
Modules for evaluating dead-reckoning drift, IO-VNBD dataset replay, ablation studies,
and post-processing performance metrics.
"""

from edge.evaluation.metrics import (
    OutageEvaluationMetrics,
    TrajectorySample,
    calculate_position_errors,
    calculate_relative_position_error,
    calculate_traveled_distance,
    calculate_velocity_errors,
    calculate_yaw_errors,
    compute_outage_metrics,
    wrap_to_pi,
)
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

__all__ = [
    "wrap_to_pi",
    "TrajectorySample",
    "OutageEvaluationMetrics",
    "calculate_position_errors",
    "calculate_relative_position_error",
    "calculate_traveled_distance",
    "calculate_velocity_errors",
    "calculate_yaw_errors",
    "compute_outage_metrics",
    "AblationConfig",
    "AblationMode",
    "OutageWindow",
    "IOVNBDLoader",
    "ReplayEvaluator",
    "save_results_json",
    "save_results_csv",
    "generate_markdown_report",
    "save_trajectory_file",
    "generate_evaluation_svg_plots",
]
