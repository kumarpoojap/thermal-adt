"""
Plot cross-evaluation robustness results.

Creates publication-quality plots for thesis:
1. Comparison table (CSV)
2. Bar chart (grouped by train-test combination)
3. Heatmap (train vs test surrogate matrix)

Usage:
    python scripts/analysis/plot_cross_eval.py \
        --cross-eval-dirs results/cross_eval/rl_rc_on_rcnn \
                         results/cross_eval/rl_rcnn_on_rc \
        --baseline-results results/policy_eval_v2/combined_per_scenario.csv \
        --output-dir results/cross_eval/analysis
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def load_cross_eval_results(cross_eval_dirs: List[str]) -> pd.DataFrame:
    """Load all cross-evaluation results."""
    all_results = []
    
    for dir_path in cross_eval_dirs:
        results_csv = Path(dir_path) / "cross_eval_results.csv"
        if not results_csv.exists():
            print(f"⚠️  Warning: {results_csv} not found, skipping")
            continue
        
        df = pd.read_csv(results_csv)
        all_results.append(df)
    
    if not all_results:
        raise ValueError("No cross-evaluation results found!")
    
    return pd.concat(all_results, ignore_index=True)


def create_comparison_table(cross_eval_df: pd.DataFrame, output_dir: Path):
    """
    Create comparison table showing performance for each train-test combination.
    """
    # Aggregate cross-eval results
    cross_summary = cross_eval_df.groupby(["train_surrogate", "test_surrogate", "scenario"]).agg({
        "cumulative_reward": ["mean", "std"],
        "mean_fan": ["mean", "std"],
        "warning_entries": "sum",
        "critical_entries": "sum",
    }).reset_index()
    
    # Flatten column names
    cross_summary.columns = [
        "train_surrogate", "test_surrogate", "scenario",
        "reward_mean", "reward_std", "fan_mean", "fan_std",
        "warning_total", "critical_total"
    ]
    
    all_results = cross_summary
    
    # Compute % change from baseline
    all_results["combination"] = all_results["train_surrogate"] + "→" + all_results["test_surrogate"]
    
    # Initialize % change column
    all_results["reward_pct_change"] = 0.0
    
    # For each scenario, compute % change from same-surrogate baseline
    for scenario in all_results["scenario"].unique():
        scenario_data = all_results[all_results["scenario"] == scenario]
        
        # Get baseline rewards (train == test) if they exist
        rc_baseline_rows = scenario_data[
            (scenario_data["train_surrogate"] == "rc") & 
            (scenario_data["test_surrogate"] == "rc")
        ]
        
        rcnn_baseline_rows = scenario_data[
            (scenario_data["train_surrogate"] == "rcnn") & 
            (scenario_data["test_surrogate"] == "rcnn")
        ]
        
        rc_baseline = rc_baseline_rows["reward_mean"].values[0] if len(rc_baseline_rows) > 0 else None
        rcnn_baseline = rcnn_baseline_rows["reward_mean"].values[0] if len(rcnn_baseline_rows) > 0 else None
        
        # Compute % change for each row
        for idx in scenario_data.index:
            row = all_results.loc[idx]
            
            # Determine which baseline to use
            if row["train_surrogate"] == "rc" and rc_baseline is not None:
                baseline_val = rc_baseline
            elif row["train_surrogate"] == "rcnn" and rcnn_baseline is not None:
                baseline_val = rcnn_baseline
            else:
                baseline_val = None
            
            # Calculate % change
            if baseline_val is not None and baseline_val != 0:
                pct_change = ((row["reward_mean"] - baseline_val) / baseline_val) * 100
                all_results.loc[idx, "reward_pct_change"] = pct_change
    
    # Save table
    table_csv = output_dir / "cross_eval_comparison_table.csv"
    all_results.to_csv(table_csv, index=False)
    print(f"✅ Comparison table saved to: {table_csv}")
    
    return all_results


def plot_bar_chart(comparison_df: pd.DataFrame, output_dir: Path):
    """
    Create grouped bar chart showing reward for each train-test combination.
    """
    # Filter to baseline scenario for simplicity
    baseline_data = comparison_df[comparison_df["scenario"] == "baseline"].copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    combinations = baseline_data["combination"].values
    rewards = baseline_data["reward_mean"].values
    stds = baseline_data["reward_std"].values
    
    # Color by whether train == test
    colors = []
    for _, row in baseline_data.iterrows():
        if row["train_surrogate"] == row["test_surrogate"]:
            colors.append("#2ecc71")  # Green for baseline
        else:
            colors.append("#e74c3c")  # Red for cross-eval
    
    # Plot
    x = np.arange(len(combinations))
    bars = ax.bar(x, rewards, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor="black")
    
    # Labels
    ax.set_xlabel("Train → Test Surrogate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontsize=12, fontweight="bold")
    ax.set_title("Cross-Evaluation Performance (Baseline Scenario)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(combinations, rotation=0, ha="center")
    ax.grid(axis="y", alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Same Surrogate (Baseline)"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Different Surrogate (Cross-Eval)")
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    
    # Add value labels on bars
    for i, (bar, reward, std) in enumerate(zip(bars, rewards, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 50,
                f'{reward:.0f}±{std:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    bar_chart_png = output_dir / "cross_eval_bar_chart.png"
    plt.savefig(bar_chart_png, dpi=300, bbox_inches="tight")
    print(f"✅ Bar chart saved to: {bar_chart_png}")
    plt.close()


def plot_heatmap(comparison_df: pd.DataFrame, output_dir: Path):
    """
    Create heatmap showing performance matrix (train vs test surrogate).
    """
    # Filter to baseline scenario
    baseline_data = comparison_df[comparison_df["scenario"] == "baseline"].copy()
    
    # Pivot to matrix format
    pivot = baseline_data.pivot(index="train_surrogate", columns="test_surrogate", values="reward_mean")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        cbar_kws={"label": "Cumulative Reward"},
        linewidths=2,
        linecolor="black",
        ax=ax,
        vmin=pivot.min().min() * 0.9,
        vmax=pivot.max().max() * 1.1
    )
    
    # Labels
    ax.set_xlabel("Test Surrogate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Train Surrogate", fontsize=12, fontweight="bold")
    ax.set_title("Cross-Evaluation Performance Matrix (Baseline Scenario)", fontsize=14, fontweight="bold")
    
    # Uppercase labels
    ax.set_xticklabels([label.get_text().upper() for label in ax.get_xticklabels()], rotation=0)
    ax.set_yticklabels([label.get_text().upper() for label in ax.get_yticklabels()], rotation=0)
    
    plt.tight_layout()
    
    # Save
    heatmap_png = output_dir / "cross_eval_heatmap.png"
    plt.savefig(heatmap_png, dpi=300, bbox_inches="tight")
    print(f"✅ Heatmap saved to: {heatmap_png}")
    plt.close()


def create_thesis_summary(comparison_df: pd.DataFrame, output_dir: Path):
    """
    Create a concise summary table for thesis.
    """
    # Filter to baseline scenario
    baseline_data = comparison_df[comparison_df["scenario"] == "baseline"].copy()
    
    # Create summary
    summary = baseline_data[[
        "combination", "reward_mean", "reward_std", "reward_pct_change",
        "fan_mean", "warning_total", "critical_total"
    ]].copy()
    
    summary.columns = [
        "Train→Test", "Reward (Mean)", "Reward (Std)", "% Change",
        "Fan Usage (%)", "Warning Violations", "Critical Violations"
    ]
    
    # Round
    summary["Reward (Mean)"] = summary["Reward (Mean)"].round(0)
    summary["Reward (Std)"] = summary["Reward (Std)"].round(0)
    summary["% Change"] = summary["% Change"].round(1)
    summary["Fan Usage (%)"] = summary["Fan Usage (%)"].round(1)
    
    # Save
    thesis_csv = output_dir / "thesis_summary_table.csv"
    summary.to_csv(thesis_csv, index=False)
    print(f"✅ Thesis summary saved to: {thesis_csv}")
    
    # Print to console
    print("\n" + "="*80)
    print("THESIS SUMMARY TABLE (Baseline Scenario)")
    print("="*80 + "\n")
    print(summary.to_string(index=False))
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Plot cross-evaluation results")
    
    parser.add_argument("--cross-eval-dirs", nargs="+", required=True,
                        help="Directories containing cross-evaluation results")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Cross-Evaluation Analysis")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading cross-evaluation results...")
    cross_eval_df = load_cross_eval_results(args.cross_eval_dirs)
    print(f"  Loaded {len(cross_eval_df)} cross-evaluation episodes")
    
    # Create comparison table
    print("\nCreating comparison table...")
    comparison_df = create_comparison_table(cross_eval_df, output_dir)
    
    # Create plots
    print("\nGenerating plots...")
    plot_bar_chart(comparison_df, output_dir)
    plot_heatmap(comparison_df, output_dir)
    
    # Create thesis summary
    print("\nCreating thesis summary...")
    create_thesis_summary(comparison_df, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ Cross-evaluation analysis complete!")
    print(f"📁 Results directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
