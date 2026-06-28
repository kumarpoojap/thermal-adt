"""
Simple cross-evaluation plotting script.

Works with actual cross-eval results format.

Usage:
    python scripts/analysis/plot_cross_eval_simple.py \
        --cross-eval-dirs results/cross_eval/rl_rc_on_rcnn results/cross_eval/rl_rcnn_on_rc \
        --output-dir results/cross_eval/analysis
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def load_all_cross_eval_results(cross_eval_dirs):
    """Load all cross-evaluation results."""
    all_results = []
    
    for dir_path in cross_eval_dirs:
        results_csv = Path(dir_path) / "cross_eval_results.csv"
        if not results_csv.exists():
            print(f"⚠️  Warning: {results_csv} not found, skipping")
            continue
        
        df = pd.read_csv(results_csv)
        all_results.append(df)
        print(f"  Loaded {len(df)} episodes from {dir_path}")
    
    if not all_results:
        raise ValueError("No cross-evaluation results found!")
    
    combined = pd.concat(all_results, ignore_index=True)
    return combined


def create_summary_table(df, output_dir):
    """Create summary table aggregating by train/test surrogate and scenario."""
    
    # Aggregate by train_surrogate, test_surrogate, scenario
    summary = df.groupby(["train_surrogate", "test_surrogate", "scenario"]).agg({
        "cumulative_reward": ["mean", "std"],
        "mean_fan": ["mean", "std"],
        "warning_entries": "sum",
        "critical_entries": "sum",
    }).reset_index()
    
    # Flatten column names
    summary.columns = [
        "train_surrogate", "test_surrogate", "scenario",
        "reward_mean", "reward_std",
        "fan_mean", "fan_std",
        "warning_total", "critical_total"
    ]
    
    # Add combination label
    summary["combination"] = summary["train_surrogate"].str.upper() + "→" + summary["test_surrogate"].str.upper()
    
    # Save
    summary_csv = output_dir / "cross_eval_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"✅ Summary table saved to: {summary_csv}")
    
    return summary


def plot_bar_chart(summary_df, output_dir):
    """Create bar chart comparing rewards across train-test combinations."""
    
    # Filter to baseline scenario for main plot
    baseline = summary_df[summary_df["scenario"] == "baseline"].copy()
    
    if len(baseline) == 0:
        print("⚠️  No baseline scenario found, using all scenarios")
        baseline = summary_df.copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    x = np.arange(len(baseline))
    bars = ax.bar(
        x, 
        baseline["reward_mean"], 
        yerr=baseline["reward_std"],
        capsize=5,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5
    )
    
    # Color bars: green if train==test, red if cross-eval
    colors = []
    for _, row in baseline.iterrows():
        if row["train_surrogate"] == row["test_surrogate"]:
            colors.append("#2ecc71")  # Green
        else:
            colors.append("#e74c3c")  # Red
    
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    
    # Labels
    ax.set_xlabel("Train → Test Surrogate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontsize=12, fontweight="bold")
    ax.set_title("Cross-Evaluation Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(baseline["combination"], rotation=0)
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, baseline["reward_mean"], baseline["reward_std"])):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + std + 50,
            f'{mean:.0f}±{std:.0f}',
            ha='center', va='bottom', 
            fontsize=10, fontweight='bold'
        )
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Same Surrogate"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Cross-Evaluation")
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    
    plt.tight_layout()
    
    # Save
    bar_png = output_dir / "cross_eval_bar_chart.png"
    plt.savefig(bar_png, dpi=300, bbox_inches="tight")
    print(f"✅ Bar chart saved to: {bar_png}")
    plt.close()


def plot_heatmap(summary_df, output_dir):
    """Create heatmap showing train vs test surrogate performance matrix."""
    
    # Filter to baseline scenario
    baseline = summary_df[summary_df["scenario"] == "baseline"].copy()
    
    if len(baseline) == 0:
        print("⚠️  No baseline scenario found, skipping heatmap")
        return
    
    # Pivot to matrix
    pivot = baseline.pivot(
        index="train_surrogate",
        columns="test_surrogate",
        values="reward_mean"
    )
    
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
    ax.set_title("Cross-Evaluation Performance Matrix", fontsize=14, fontweight="bold")
    
    # Uppercase labels
    ax.set_xticklabels([label.get_text().upper() for label in ax.get_xticklabels()], rotation=0)
    ax.set_yticklabels([label.get_text().upper() for label in ax.get_yticklabels()], rotation=0)
    
    plt.tight_layout()
    
    # Save
    heatmap_png = output_dir / "cross_eval_heatmap.png"
    plt.savefig(heatmap_png, dpi=300, bbox_inches="tight")
    print(f"✅ Heatmap saved to: {heatmap_png}")
    plt.close()


def create_thesis_table(summary_df, output_dir):
    """Create publication-ready table for thesis."""
    
    # Filter to baseline scenario
    baseline = summary_df[summary_df["scenario"] == "baseline"].copy()
    
    if len(baseline) == 0:
        baseline = summary_df.copy()
    
    # Select columns for thesis
    thesis_table = baseline[[
        "combination",
        "reward_mean", "reward_std",
        "fan_mean", "fan_std",
        "warning_total", "critical_total"
    ]].copy()
    
    # Rename columns
    thesis_table.columns = [
        "Train→Test",
        "Reward (Mean)", "Reward (Std)",
        "Fan % (Mean)", "Fan % (Std)",
        "Warning Violations", "Critical Violations"
    ]
    
    # Round values
    thesis_table["Reward (Mean)"] = thesis_table["Reward (Mean)"].round(0)
    thesis_table["Reward (Std)"] = thesis_table["Reward (Std)"].round(0)
    thesis_table["Fan % (Mean)"] = thesis_table["Fan % (Mean)"].round(1)
    thesis_table["Fan % (Std)"] = thesis_table["Fan % (Std)"].round(1)
    
    # Save
    thesis_csv = output_dir / "thesis_table.csv"
    thesis_table.to_csv(thesis_csv, index=False)
    print(f"✅ Thesis table saved to: {thesis_csv}")
    
    # Print to console
    print("\n" + "="*80)
    print("THESIS TABLE")
    print("="*80 + "\n")
    print(thesis_table.to_string(index=False))
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
    df = load_all_cross_eval_results(args.cross_eval_dirs)
    print(f"  Total episodes loaded: {len(df)}")
    
    # Create summary
    print("\nCreating summary table...")
    summary = create_summary_table(df, output_dir)
    
    # Create plots
    print("\nGenerating plots...")
    plot_bar_chart(summary, output_dir)
    plot_heatmap(summary, output_dir)
    
    # Create thesis table
    print("\nCreating thesis table...")
    create_thesis_table(summary, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ Cross-evaluation analysis complete!")
    print(f"📁 Results directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
