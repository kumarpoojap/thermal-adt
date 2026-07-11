"""
Plot domain shift robustness results.

Creates publication-quality plots for thesis:
1. Robustness table (controller × perturbation matrix)
2. Heatmap showing % performance change
3. Bar chart comparing performance under perturbations

Usage:
    python scripts/analysis/plot_domain_shift.py \
        --baseline-results results/policy_eval_v2/combined_per_scenario.csv \
        --ambient-results results/domain_shift/ambient_plus5/domain_shift_results.csv \
        --power-results results/domain_shift/power_plus20/domain_shift_results.csv \
        --delay-results results/domain_shift/delay_1step/domain_shift_results.csv \
        --output-dir results/domain_shift/analysis
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


def load_baseline_results(baseline_csv: str) -> pd.DataFrame:
    """Load baseline evaluation results."""
    df = pd.read_csv(baseline_csv)
    
    # Filter to baseline scenario
    baseline_data = df[df["scenario"] == "baseline"].copy()
    
    # Aggregate by controller
    baseline_agg = baseline_data.groupby("controller").agg({
        "cumulative_reward": "mean",
        "mean_fan": "mean",
    }).reset_index()
    
    baseline_agg["perturbation"] = "baseline"
    
    return baseline_agg


def load_domain_shift_results(results_csv: str, perturbation_name: str) -> pd.DataFrame:
    """Load domain shift results."""
    df = pd.read_csv(results_csv)
    
    # Filter to baseline scenario (for fair comparison)
    baseline_data = df[df["scenario"] == "baseline"].copy()
    
    # Aggregate by controller
    agg = baseline_data.groupby("controller").agg({
        "cumulative_reward": "mean",
        "mean_fan": "mean",
    }).reset_index()
    
    agg["perturbation"] = perturbation_name
    
    return agg


def create_robustness_table(
    baseline_df: pd.DataFrame,
    ambient_df: pd.DataFrame,
    power_df: pd.DataFrame,
    delay_df: pd.DataFrame,
    output_dir: Path
):
    """Create robustness table showing performance under each perturbation."""
    
    # Combine all results
    all_results = pd.concat([baseline_df, ambient_df, power_df, delay_df], ignore_index=True)
    
    # Pivot to wide format
    pivot = all_results.pivot(
        index="controller",
        columns="perturbation",
        values="cumulative_reward"
    )
    
    # Reorder columns
    column_order = ["baseline", "ambient_shift", "power_spike", "action_delay"]
    pivot = pivot[[col for col in column_order if col in pivot.columns]]
    
    # Compute % change from baseline
    for col in pivot.columns:
        if col != "baseline":
            pct_change_col = f"{col}_pct"
            pivot[pct_change_col] = ((pivot[col] - pivot["baseline"]) / pivot["baseline"] * 100).round(1)
    
    # Save
    table_csv = output_dir / "robustness_table.csv"
    pivot.to_csv(table_csv)
    print(f"✅ Robustness table saved to: {table_csv}")
    
    # Print
    print("\n" + "="*80)
    print("ROBUSTNESS TABLE")
    print("="*80 + "\n")
    print(pivot)
    print("\n")
    
    return pivot


def plot_robustness_heatmap(
    robustness_table: pd.DataFrame,
    output_dir: Path
):
    """Create heatmap showing % performance change."""
    
    # Extract % change columns
    pct_cols = [col for col in robustness_table.columns if "_pct" in col]
    
    if len(pct_cols) == 0:
        print("⚠️  No % change columns found, skipping heatmap")
        return
    
    # Create heatmap data
    heatmap_data = robustness_table[pct_cols].copy()
    
    # Rename columns for display
    heatmap_data.columns = [col.replace("_pct", "").replace("_", " ").title() for col in heatmap_data.columns]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",  # Red = bad (negative %), Green = good (close to 0%)
        center=0,
        cbar_kws={"label": "% Change from Baseline"},
        linewidths=2,
        linecolor="black",
        ax=ax,
        vmin=-30,
        vmax=5
    )
    
    # Labels
    ax.set_xlabel("Perturbation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Controller", fontsize=12, fontweight="bold")
    ax.set_title("Domain Shift Robustness (% Change from Baseline)", fontsize=14, fontweight="bold")
    
    # Uppercase labels
    ax.set_yticklabels([label.get_text().upper().replace("_", "-") for label in ax.get_yticklabels()], rotation=0)
    
    plt.tight_layout()
    
    # Save
    heatmap_png = output_dir / "robustness_heatmap.png"
    plt.savefig(heatmap_png, dpi=300, bbox_inches="tight")
    print(f"✅ Robustness heatmap saved to: {heatmap_png}")
    plt.close()


def plot_robustness_bar_chart(
    baseline_df: pd.DataFrame,
    ambient_df: pd.DataFrame,
    power_df: pd.DataFrame,
    delay_df: pd.DataFrame,
    output_dir: Path
):
    """Create grouped bar chart comparing performance under perturbations."""
    
    # Combine data
    all_results = pd.concat([baseline_df, ambient_df, power_df, delay_df], ignore_index=True)
    
    # Get unique controllers
    controllers = baseline_df["controller"].unique()
    perturbations = ["baseline", "ambient_shift", "power_spike", "action_delay"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar width and positions
    bar_width = 0.2
    x = np.arange(len(controllers))
    
    # Colors for each perturbation
    colors = {
        "baseline": "#2ecc71",
        "ambient_shift": "#e74c3c",
        "power_spike": "#f39c12",
        "action_delay": "#3498db"
    }
    
    # Plot bars for each perturbation
    for i, pert in enumerate(perturbations):
        pert_data = all_results[all_results["perturbation"] == pert]
        
        # Match controller order
        rewards = []
        for controller in controllers:
            ctrl_data = pert_data[pert_data["controller"] == controller]
            if len(ctrl_data) > 0:
                rewards.append(ctrl_data["cumulative_reward"].values[0])
            else:
                rewards.append(0)
        
        offset = (i - len(perturbations)/2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            rewards,
            bar_width,
            label=pert.replace("_", " ").title(),
            color=colors.get(pert, "#95a5a6"),
            alpha=0.8,
            edgecolor="black"
        )
    
    # Labels
    ax.set_xlabel("Controller", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontsize=12, fontweight="bold")
    ax.set_title("Domain Shift Robustness Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper().replace("_", "-") for c in controllers], rotation=0)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    bar_chart_png = output_dir / "robustness_bar_chart.png"
    plt.savefig(bar_chart_png, dpi=300, bbox_inches="tight")
    print(f"✅ Robustness bar chart saved to: {bar_chart_png}")
    plt.close()


def plot_pareto(
    baseline_df: pd.DataFrame,
    ambient_df: pd.DataFrame,
    power_df: pd.DataFrame,
    delay_df: pd.DataFrame,
    output_dir: Path
):
    """Create Pareto plot showing reward vs fan usage trade-off."""
    
    # Combine data
    all_results = pd.concat([baseline_df, ambient_df, power_df, delay_df], ignore_index=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colors and markers for each perturbation
    colors = {
        "baseline": "#2ecc71",
        "ambient_shift": "#e74c3c",
        "power_spike": "#f39c12",
        "action_delay": "#3498db"
    }
    
    markers = {
        "baseline": "o",
        "ambient_shift": "s",
        "power_spike": "^",
        "action_delay": "D"
    }
    
    # Plot each perturbation
    for pert in all_results["perturbation"].unique():
        pert_data = all_results[all_results["perturbation"] == pert]
        
        ax.scatter(
            pert_data["mean_fan"],
            pert_data["cumulative_reward"],
            s=150,
            c=colors.get(pert, "#95a5a6"),
            marker=markers.get(pert, "o"),
            alpha=0.7,
            edgecolors="black",
            linewidths=1.5,
            label=pert.replace("_", " ").title()
        )
        
        # Add controller labels
        for _, row in pert_data.iterrows():
            ax.annotate(
                row["controller"].upper().replace("_", "-"),
                (row["mean_fan"], row["cumulative_reward"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7
            )
    
    # Labels
    ax.set_xlabel("Mean Fan Usage (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontsize=12, fontweight="bold")
    ax.set_title("Reward vs Efficiency Trade-off (Pareto Plot)", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    pareto_png = output_dir / "pareto_plot.png"
    plt.savefig(pareto_png, dpi=300, bbox_inches="tight")
    print(f"✅ Pareto plot saved to: {pareto_png}")
    plt.close()


def create_thesis_summary(
    robustness_table: pd.DataFrame,
    output_dir: Path
):
    """Create thesis-ready summary table."""
    
    # Select baseline and % change columns
    cols_to_keep = ["baseline"]
    pct_cols = [col for col in robustness_table.columns if "_pct" in col]
    cols_to_keep.extend(pct_cols)
    
    thesis_table = robustness_table[cols_to_keep].copy()
    
    # Rename columns
    thesis_table.columns = [
        "Baseline Reward",
        "Ambient +5°C (%)",
        "Power +20% (%)",
        "Delay 1-step (%)"
    ]
    
    # Round
    thesis_table["Baseline Reward"] = thesis_table["Baseline Reward"].round(0)
    
    # Save
    thesis_csv = output_dir / "thesis_robustness_summary.csv"
    thesis_table.to_csv(thesis_csv)
    print(f"✅ Thesis summary saved to: {thesis_csv}")
    
    # Print
    print("\n" + "="*80)
    print("THESIS ROBUSTNESS SUMMARY")
    print("="*80 + "\n")
    print(thesis_table)
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Plot domain shift results")
    
    parser.add_argument("--baseline-results", type=str, required=True,
                        help="CSV with baseline evaluation results")
    parser.add_argument("--ambient-results", type=str, required=True,
                        help="CSV with ambient shift results")
    parser.add_argument("--power-results", type=str, required=True,
                        help="CSV with power spike results")
    parser.add_argument("--delay-results", type=str, required=True,
                        help="CSV with action delay results")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Domain Shift Analysis")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading results...")
    baseline_df = load_baseline_results(args.baseline_results)
    ambient_df = load_domain_shift_results(args.ambient_results, "ambient_shift")
    power_df = load_domain_shift_results(args.power_results, "power_spike")
    delay_df = load_domain_shift_results(args.delay_results, "action_delay")
    print("  All results loaded")
    
    # Create robustness table
    print("\nCreating robustness table...")
    robustness_table = create_robustness_table(
        baseline_df, ambient_df, power_df, delay_df, output_dir
    )
    
    # Create plots
    print("\nGenerating plots...")
    plot_robustness_heatmap(robustness_table, output_dir)
    plot_robustness_bar_chart(baseline_df, ambient_df, power_df, delay_df, output_dir)
    plot_pareto(baseline_df, ambient_df, power_df, delay_df, output_dir)
    
    # Create thesis summary
    print("\nCreating thesis summary...")
    create_thesis_summary(robustness_table, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ Domain shift analysis complete!")
    print(f"📁 Results directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
