"""
Safety and Recovery Analysis

Extracts safety metrics from evaluation results:
1. Time-to-safe-band (steps until temp ≤ target)
2. First passage to warning (steps until temp ≤ warning threshold)
3. Maximum temperature (overshoot)
4. Settling time (steps until temp stabilizes)

Usage:
    python scripts/analysis/safety_recovery_analysis.py \
        --eval-results results/policy_eval_v2/combined_per_scenario.csv \
        --scenarios thermal_high_start combined_extreme \
        --output-dir results/safety_analysis
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def compute_time_to_safe_band(temp_trajectory: np.ndarray, target: float = 65.0) -> int:
    """
    Compute time (steps) until temperature reaches safe band (≤ target).
    
    Args:
        temp_trajectory: Temperature trajectory
        target: Target temperature
    
    Returns:
        Steps until safe band reached (or episode length if never reached)
    """
    safe_indices = np.where(temp_trajectory <= target)[0]
    if len(safe_indices) == 0:
        return len(temp_trajectory)
    return int(safe_indices[0])


def compute_first_passage_to_warning(temp_trajectory: np.ndarray, warning: float = 80.0) -> int:
    """
    Compute time (steps) until temperature drops below warning threshold.
    
    Args:
        temp_trajectory: Temperature trajectory
        warning: Warning threshold
    
    Returns:
        Steps until below warning (or episode length if never reached)
    """
    safe_indices = np.where(temp_trajectory <= warning)[0]
    if len(safe_indices) == 0:
        return len(temp_trajectory)
    return int(safe_indices[0])


def compute_overshoot(temp_trajectory: np.ndarray) -> float:
    """
    Compute maximum temperature reached.
    
    Args:
        temp_trajectory: Temperature trajectory
    
    Returns:
        Maximum temperature
    """
    return float(np.max(temp_trajectory))


def compute_settling_time(
    temp_trajectory: np.ndarray,
    target: float = 65.0,
    tolerance: float = 2.0,
    settle_duration: int = 10
) -> int:
    """
    Compute time until temperature settles within ±tolerance of target.
    
    Args:
        temp_trajectory: Temperature trajectory
        target: Target temperature
        tolerance: Tolerance band (±)
        settle_duration: Number of steps to stay settled
    
    Returns:
        Steps until settled (or episode length if never settled)
    """
    settled = np.abs(temp_trajectory - target) <= tolerance
    
    # Find first index where it stays settled for at least settle_duration steps
    for i in range(len(settled) - settle_duration):
        if np.all(settled[i:i+settle_duration]):
            return i
    
    return len(temp_trajectory)


def analyze_episode_trajectory(
    episode_data: pd.DataFrame,
    target: float = 65.0,
    warning: float = 80.0
) -> Dict[str, float]:
    """
    Analyze a single episode trajectory.
    
    Args:
        episode_data: DataFrame with 'temp', 'fan', 'reward' columns
        target: Target temperature
        warning: Warning threshold
    
    Returns:
        Dictionary of safety metrics
    """
    temps = episode_data["temp"].values
    fans = episode_data["fan"].values if "fan" in episode_data.columns else None
    rewards = episode_data["reward"].values if "reward" in episode_data.columns else None
    
    metrics = {
        "time_to_safe_band": compute_time_to_safe_band(temps, target),
        "first_passage_warning": compute_first_passage_to_warning(temps, warning),
        "max_temp": compute_overshoot(temps),
        "settling_time": compute_settling_time(temps, target),
        "initial_temp": float(temps[0]),
        "final_temp": float(temps[-1]),
    }
    
    if fans is not None:
        metrics["mean_fan"] = float(np.mean(fans))
        metrics["max_fan"] = float(np.max(fans))
    
    if rewards is not None:
        metrics["cumulative_reward"] = float(np.sum(rewards))
    
    return metrics


def load_and_analyze_results(
    results_csv: str,
    scenarios: List[str],
    controllers: List[str]
) -> pd.DataFrame:
    """
    Load evaluation results and compute safety metrics.
    
    Note: This assumes the CSV has aggregated results, not per-episode trajectories.
    For detailed trajectory analysis, you'd need the raw episode CSVs.
    
    Args:
        results_csv: Path to combined results CSV
        scenarios: Scenarios to analyze
        controllers: Controllers to analyze
    
    Returns:
        DataFrame with safety metrics
    """
    df = pd.read_csv(results_csv)
    
    # Filter to hot-start scenarios
    df_filtered = df[df["scenario"].isin(scenarios)]
    
    if len(controllers) > 0:
        df_filtered = df_filtered[df_filtered["controller"].isin(controllers)]
    
    # The CSV already has time_to_safe_band if it was computed during evaluation
    # If not, we'd need to load raw episode data
    
    return df_filtered


def create_recovery_comparison_table(
    safety_df: pd.DataFrame,
    output_dir: Path
):
    """Create comparison table of recovery metrics."""
    
    # Group by controller and scenario
    # Only aggregate columns that exist in the dataframe
    agg_dict = {
        "cumulative_reward": ["mean", "std"],
        "mean_fan": ["mean", "std"],
    }
    
    # Add optional columns if they exist
    if "time_to_safe_band" in safety_df.columns:
        agg_dict["time_to_safe_band"] = ["mean", "std"]
    if "max_temp" in safety_df.columns:
        agg_dict["max_temp"] = ["mean", "std"]
    if "violations_warning_entries" in safety_df.columns:
        agg_dict["violations_warning_entries"] = ["mean", "sum"]
    if "violations_critical_entries" in safety_df.columns:
        agg_dict["violations_critical_entries"] = ["mean", "sum"]
    
    summary = safety_df.groupby(["controller", "scenario"]).agg(agg_dict).round(2)
    
    # Save
    table_csv = output_dir / "recovery_comparison_table.csv"
    summary.to_csv(table_csv)
    print(f"✅ Recovery comparison table saved to: {table_csv}")
    
    # Print
    print("\n" + "="*80)
    print("RECOVERY COMPARISON")
    print("="*80 + "\n")
    print(summary)
    print("\n")
    
    return summary


def plot_recovery_trajectories(
    safety_df: pd.DataFrame,
    output_dir: Path,
    scenario: str = "thermal_high_start"
):
    """
    Plot temperature trajectories showing recovery strategies.
    
    Note: This is a simplified version. For actual trajectories,
    you'd need to load the raw episode CSV files.
    """
    # Filter to scenario
    scenario_data = safety_df[safety_df["scenario"] == scenario]
    
    if len(scenario_data) == 0:
        print(f"⚠️  No data for scenario {scenario}, skipping trajectory plot")
        return
    
    # Create bar chart of time-to-safe-band
    fig, ax = plt.subplots(figsize=(10, 6))
    
    controllers = scenario_data["controller"].unique()
    time_to_safe = []
    time_to_safe_std = []
    
    for controller in controllers:
        ctrl_data = scenario_data[scenario_data["controller"] == controller]
        time_to_safe.append(ctrl_data["time_to_safe_band"].mean())
        time_to_safe_std.append(ctrl_data["time_to_safe_band"].std())
    
    x = np.arange(len(controllers))
    bars = ax.bar(x, time_to_safe, yerr=time_to_safe_std, capsize=5, alpha=0.8, edgecolor="black")
    
    # Color bars
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    for bar, color in zip(bars, colors[:len(bars)]):
        bar.set_facecolor(color)
    
    # Labels
    ax.set_xlabel("Controller", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time to Safe Band (steps)", fontsize=12, fontweight="bold")
    ax.set_title(f"Recovery Time Comparison ({scenario})", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper().replace("_", "-") for c in controllers], rotation=0)
    ax.grid(axis="y", alpha=0.3)
    
    # Add horizontal line at target
    ax.axhline(y=65, color="green", linestyle="--", linewidth=2, label="Target (65°C)")
    ax.axhline(y=80, color="orange", linestyle="--", linewidth=2, label="Warning (80°C)")
    
    # Add value labels
    for i, (bar, val, std) in enumerate(zip(bars, time_to_safe, time_to_safe_std)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + std + 5,
            f'{val:.0f}±{std:.0f}',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )
    
    ax.legend()
    plt.tight_layout()
    
    # Save
    recovery_png = output_dir / f"recovery_time_{scenario}.png"
    plt.savefig(recovery_png, dpi=300, bbox_inches="tight")
    print(f"✅ Recovery time plot saved to: {recovery_png}")
    plt.close()


def plot_max_temp_comparison(
    safety_df: pd.DataFrame,
    output_dir: Path
):
    """Plot maximum temperature comparison across controllers."""
    
    # Check if max_temp column exists
    if "max_temp" not in safety_df.columns:
        print("⚠️  Skipping max_temp plot (column not in data)")
        return
    
    # Group by controller
    summary = safety_df.groupby("controller").agg({
        "max_temp": ["mean", "std"]
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    controllers = summary.index.tolist()
    max_temps = summary[("max_temp", "mean")].values
    max_temps_std = summary[("max_temp", "std")].values
    
    x = np.arange(len(controllers))
    bars = ax.bar(x, max_temps, yerr=max_temps_std, capsize=5, alpha=0.8, edgecolor="black")
    
    # Color bars
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    for bar, color in zip(bars, colors[:len(bars)]):
        bar.set_facecolor(color)
    
    # Labels
    ax.set_xlabel("Controller", fontsize=12, fontweight="bold")
    ax.set_ylabel("Maximum Temperature (°C)", fontsize=12, fontweight="bold")
    ax.set_title("Maximum Temperature Comparison (Hot-Start Scenarios)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper().replace("_", "-") for c in controllers], rotation=0)
    ax.grid(axis="y", alpha=0.3)
    
    # Add threshold lines
    ax.axhline(y=80, color="orange", linestyle="--", linewidth=2, label="Warning (80°C)")
    ax.axhline(y=90, color="red", linestyle="--", linewidth=2, label="Critical (90°C)")
    
    # Add value labels
    for i, (bar, val, std) in enumerate(zip(bars, max_temps, max_temps_std)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + std + 1,
            f'{val:.1f}±{std:.1f}',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )
    
    ax.legend()
    plt.tight_layout()
    
    # Save
    max_temp_png = output_dir / "max_temp_comparison.png"
    plt.savefig(max_temp_png, dpi=300, bbox_inches="tight")
    print(f"✅ Max temperature plot saved to: {max_temp_png}")
    plt.close()


def create_thesis_summary(
    safety_df: pd.DataFrame,
    output_dir: Path
):
    """Create thesis-ready summary table."""
    
    # Build aggregation dict with only available columns
    agg_dict = {}
    column_names = []
    
    if "time_to_safe_band" in safety_df.columns:
        agg_dict["time_to_safe_band"] = ["mean", "std"]
        column_names.extend(["Time to Safe (Mean)", "Time to Safe (Std)"])
    
    if "max_temp" in safety_df.columns:
        agg_dict["max_temp"] = ["mean", "std"]
        column_names.extend(["Max Temp (Mean)", "Max Temp (Std)"])
    
    if "settling_time" in safety_df.columns:
        agg_dict["settling_time"] = ["mean", "std"]
        column_names.extend(["Settling Time (Mean)", "Settling Time (Std)"])
    
    if "mean_fan" in safety_df.columns:
        agg_dict["mean_fan"] = ["mean", "std"]
        column_names.extend(["Fan % (Mean)", "Fan % (Std)"])
    
    if "cumulative_reward" in safety_df.columns:
        agg_dict["cumulative_reward"] = ["mean", "std"]
        column_names.extend(["Reward (Mean)", "Reward (Std)"])
    
    # Group by controller
    summary = safety_df.groupby("controller").agg(agg_dict).round(1)
    
    # Flatten columns
    summary.columns = column_names
    
    # Save
    thesis_csv = output_dir / "thesis_safety_summary.csv"
    summary.to_csv(thesis_csv)
    print(f"✅ Thesis summary saved to: {thesis_csv}")
    
    # Print
    print("\n" + "="*80)
    print("THESIS SAFETY SUMMARY")
    print("="*80 + "\n")
    print(summary)
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Safety and recovery analysis")
    
    parser.add_argument("--eval-results", type=str, required=True,
                        help="Path to combined evaluation results CSV")
    parser.add_argument("--scenarios", nargs="+",
                        default=["thermal_high_start", "combined_extreme"],
                        help="Hot-start scenarios to analyze")
    parser.add_argument("--controllers", nargs="+",
                        default=["mpc_rc", "mpc_rcnn", "rl_rc", "rl_rcnn"],
                        help="Controllers to analyze")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Safety and Recovery Analysis")
    print(f"{'='*80}\n")
    
    # Load results
    print("Loading evaluation results...")
    safety_df = load_and_analyze_results(
        args.eval_results,
        args.scenarios,
        args.controllers
    )
    print(f"  Loaded {len(safety_df)} results")
    
    # Create comparison table
    print("\nCreating recovery comparison table...")
    create_recovery_comparison_table(safety_df, output_dir)
    
    # Create plots
    print("\nGenerating plots...")
    for scenario in args.scenarios:
        plot_recovery_trajectories(safety_df, output_dir, scenario)
    
    plot_max_temp_comparison(safety_df, output_dir)
    
    # Create thesis summary
    print("\nCreating thesis summary...")
    create_thesis_summary(safety_df, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ Safety and recovery analysis complete!")
    print(f"📁 Results directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
