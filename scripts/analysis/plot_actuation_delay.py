"""
Plot actuation delay robustness results.

Creates publication-quality plots for thesis:
1. Delay impact table (controller × delay steps)
2. Bar chart comparing performance under delays
3. Degradation curve (performance vs delay)

Usage:
    python scripts/analysis/plot_actuation_delay.py \
        --baseline-results results/policy_eval_v2/combined_per_scenario.csv \
        --delay1-results results/domain_shift/delay_1step/domain_shift_results.csv \
        --delay2-results results/domain_shift/delay_2step/domain_shift_results.csv \
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
    
    print(f"  Baseline CSV columns: {df.columns.tolist()}")
    print(f"  Baseline scenarios: {df['scenario'].unique()}")
    print(f"  Baseline controllers: {df['controller'].unique() if 'controller' in df.columns else 'N/A'}")
    
    # Filter to baseline scenario
    baseline_data = df[df["scenario"] == "baseline"].copy()
    
    print(f"  Baseline data shape: {baseline_data.shape}")
    print(f"  Baseline rewards:\n{baseline_data[['controller', 'cumulative_reward']] if 'controller' in baseline_data.columns else baseline_data}")
    
    # Check if already aggregated (has mean/std columns) or raw (has seed column)
    if "seed" in baseline_data.columns or "episode" in baseline_data.columns:
        # Raw data - aggregate by controller
        baseline_agg = baseline_data.groupby("controller").agg({
            "cumulative_reward": "mean",
            "mean_fan": "mean",
        }).reset_index()
    else:
        # Already aggregated - use as is
        baseline_agg = baseline_data[["controller", "cumulative_reward", "mean_fan"]].copy()
    
    baseline_agg["delay_steps"] = 0
    
    return baseline_agg


def load_delay_results(results_csv: str, delay_steps: int) -> pd.DataFrame:
    """Load actuation delay results."""
    df = pd.read_csv(results_csv)
    
    # Filter to baseline scenario (for fair comparison)
    baseline_data = df[df["scenario"] == "baseline"].copy()
    
    # Aggregate by controller
    agg = baseline_data.groupby("controller").agg({
        "cumulative_reward": "mean",
        "mean_fan": "mean",
    }).reset_index()
    
    agg["delay_steps"] = delay_steps
    
    return agg


def create_delay_impact_table(
    baseline_df: pd.DataFrame,
    delay1_df: pd.DataFrame,
    delay2_df: pd.DataFrame,
    output_dir: Path
):
    """Create table showing performance impact of actuation delay."""
    
    # Combine all results
    all_results = pd.concat([baseline_df, delay1_df, delay2_df], ignore_index=True)
    
    # Pivot to wide format
    pivot = all_results.pivot(
        index="controller",
        columns="delay_steps",
        values="cumulative_reward"
    )
    
    # Rename columns
    pivot.columns = [f"Delay_{int(col)}_steps" for col in pivot.columns]
    
    # Compute % change from baseline
    baseline_col = "Delay_0_steps"
    for col in pivot.columns:
        if col != baseline_col:
            pct_col = f"{col}_pct"
            pivot[pct_col] = ((pivot[col] - pivot[baseline_col]) / pivot[baseline_col] * 100).round(1)
    
    # Save
    table_csv = output_dir / "actuation_delay_impact_table.csv"
    pivot.to_csv(table_csv)
    print(f"✅ Delay impact table saved to: {table_csv}")
    
    # Print
    print("\n" + "="*80)
    print("ACTUATION DELAY IMPACT TABLE")
    print("="*80 + "\n")
    print(pivot)
    print("\n")
    
    return pivot


def plot_delay_bar_chart(
    baseline_df: pd.DataFrame,
    delay1_df: pd.DataFrame,
    delay2_df: pd.DataFrame,
    output_dir: Path
):
    """Create grouped bar chart comparing performance under delays."""
    
    # Combine data
    all_results = pd.concat([baseline_df, delay1_df, delay2_df], ignore_index=True)
    
    # Get unique controllers
    controllers = baseline_df["controller"].unique()
    delays = [0, 1, 2]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar width and positions
    bar_width = 0.25
    x = np.arange(len(controllers))
    
    # Colors for each delay
    colors = {
        0: "#2ecc71",  # Green - no delay
        1: "#f39c12",  # Orange - 1-step delay
        2: "#e74c3c",  # Red - 2-step delay
    }
    
    # Plot bars for each delay
    for i, delay in enumerate(delays):
        delay_data = all_results[all_results["delay_steps"] == delay]
        
        # Match controller order
        rewards = []
        for controller in controllers:
            ctrl_data = delay_data[delay_data["controller"] == controller]
            if len(ctrl_data) > 0:
                rewards.append(ctrl_data["cumulative_reward"].values[0])
            else:
                rewards.append(0)
        
        offset = (i - len(delays)/2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            rewards,
            bar_width,
            label=f"{delay}-step delay" if delay > 0 else "No delay",
            color=colors.get(delay, "#95a5a6"),
            alpha=0.8,
            edgecolor="black"
        )
    
    # Labels
    ax.set_xlabel("Controller", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontsize=12, fontweight="bold")
    ax.set_title("Actuation Delay Impact on Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper().replace("_", "-") for c in controllers], rotation=0)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    bar_chart_png = output_dir / "actuation_delay_bar_chart.png"
    plt.savefig(bar_chart_png, dpi=300, bbox_inches="tight")
    print(f"✅ Bar chart saved to: {bar_chart_png}")
    plt.close()


def plot_degradation_curve(
    baseline_df: pd.DataFrame,
    delay1_df: pd.DataFrame,
    delay2_df: pd.DataFrame,
    output_dir: Path
):
    """Create line plot showing performance degradation vs delay."""
    
    # Combine data
    all_results = pd.concat([baseline_df, delay1_df, delay2_df], ignore_index=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors and markers for each controller
    colors = {
        "mpc_rc": "#3498db",
        "mpc_rcnn": "#2ecc71",
        "rl_rc": "#e74c3c",
        "rl_rcnn": "#f39c12"
    }
    
    markers = {
        "mpc_rc": "o",
        "mpc_rcnn": "s",
        "rl_rc": "^",
        "rl_rcnn": "D"
    }
    
    # Plot each controller
    for controller in all_results["controller"].unique():
        ctrl_data = all_results[all_results["controller"] == controller].sort_values("delay_steps")
        
        ax.plot(
            ctrl_data["delay_steps"],
            ctrl_data["cumulative_reward"],
            marker=markers.get(controller, "o"),
            markersize=10,
            linewidth=2.5,
            color=colors.get(controller, "#95a5a6"),
            label=controller.upper().replace("_", "-"),
            alpha=0.8
        )
    
    # Labels
    ax.set_xlabel("Actuation Delay (steps)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontsize=12, fontweight="bold")
    ax.set_title("Performance Degradation vs Actuation Delay", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    curve_png = output_dir / "degradation_curve.png"
    plt.savefig(curve_png, dpi=300, bbox_inches="tight")
    print(f"✅ Degradation curve saved to: {curve_png}")
    plt.close()


def plot_heatmap(
    delay_impact_table: pd.DataFrame,
    output_dir: Path
):
    """Create heatmap showing % performance change."""
    
    # Extract % change columns
    pct_cols = [col for col in delay_impact_table.columns if "_pct" in col]
    
    if len(pct_cols) == 0:
        print("⚠️  No % change columns found, skipping heatmap")
        return
    
    # Create heatmap data
    heatmap_data = delay_impact_table[pct_cols].copy()
    
    # Rename columns for display
    heatmap_data.columns = [col.replace("Delay_", "").replace("_steps_pct", "-step") for col in heatmap_data.columns]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
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
        vmin=-20,
        vmax=5
    )
    
    # Labels
    ax.set_xlabel("Actuation Delay", fontsize=12, fontweight="bold")
    ax.set_ylabel("Controller", fontsize=12, fontweight="bold")
    ax.set_title("Actuation Delay Impact (% Change)", fontsize=14, fontweight="bold")
    
    # Uppercase labels
    ax.set_yticklabels([label.get_text().upper().replace("_", "-") for label in ax.get_yticklabels()], rotation=0)
    
    plt.tight_layout()
    
    # Save
    heatmap_png = output_dir / "actuation_delay_heatmap.png"
    plt.savefig(heatmap_png, dpi=300, bbox_inches="tight")
    print(f"✅ Heatmap saved to: {heatmap_png}")
    plt.close()


def create_thesis_summary(
    delay_impact_table: pd.DataFrame,
    output_dir: Path
):
    """Create thesis-ready summary table."""
    
    # Select baseline and % change columns
    cols_to_keep = ["Delay_0_steps"]
    pct_cols = [col for col in delay_impact_table.columns if "_pct" in col]
    cols_to_keep.extend(pct_cols)
    
    thesis_table = delay_impact_table[cols_to_keep].copy()
    
    # Rename columns
    col_names = ["Baseline Reward"]
    col_names.extend([col.replace("Delay_", "").replace("_steps_pct", "-step (%)") for col in pct_cols])
    thesis_table.columns = col_names
    
    # Round
    thesis_table["Baseline Reward"] = thesis_table["Baseline Reward"].round(0)
    
    # Save
    thesis_csv = output_dir / "thesis_actuation_delay_summary.csv"
    thesis_table.to_csv(thesis_csv)
    print(f"✅ Thesis summary saved to: {thesis_csv}")
    
    # Print
    print("\n" + "="*80)
    print("THESIS ACTUATION DELAY SUMMARY")
    print("="*80 + "\n")
    print(thesis_table)
    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Plot actuation delay results")
    
    parser.add_argument("--baseline-results", type=str, required=True,
                        help="CSV with baseline evaluation results")
    parser.add_argument("--delay1-results", type=str, required=True,
                        help="CSV with 1-step delay results")
    parser.add_argument("--delay2-results", type=str, default=None,
                        help="CSV with 2-step delay results (optional)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Actuation Delay Analysis")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading results...")
    baseline_df = load_baseline_results(args.baseline_results)
    delay1_df = load_delay_results(args.delay1_results, 1)
    
    if args.delay2_results:
        delay2_df = load_delay_results(args.delay2_results, 2)
    else:
        # Create empty dataframe with same structure
        delay2_df = baseline_df.copy()
        delay2_df["delay_steps"] = 2
        delay2_df["cumulative_reward"] = np.nan
        delay2_df["mean_fan"] = np.nan
    
    print("  All results loaded")
    
    # Create delay impact table
    print("\nCreating delay impact table...")
    delay_impact_table = create_delay_impact_table(
        baseline_df, delay1_df, delay2_df, output_dir
    )
    
    # Create plots
    print("\nGenerating plots...")
    plot_delay_bar_chart(baseline_df, delay1_df, delay2_df, output_dir)
    
    if args.delay2_results:
        plot_degradation_curve(baseline_df, delay1_df, delay2_df, output_dir)
    
    plot_heatmap(delay_impact_table, output_dir)
    
    # Create thesis summary
    print("\nCreating thesis summary...")
    create_thesis_summary(delay_impact_table, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ Actuation delay analysis complete!")
    print(f"📁 Results directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
