#!/usr/bin/env python3
"""
Standalone plotting script for policy evaluation results.

Reads CSV files from a completed evaluation run and generates plots.
Allows re-plotting with different modes without re-running evaluation.

Usage:
    # Generate all plots (96 multi-episode overlays + combined)
    python scripts/evaluation/plot_results.py \
      --eval-dir results/policy-eval-v2 \
      --plot-mode combined

    # Generate minimal plots (23 plots for dissertation)
    python scripts/evaluation/plot_results.py \
      --eval-dir results/policy-eval-v2 \
      --plot-mode minimal

    # Generate everything including individual episodes (1095 plots)
    python scripts/evaluation/plot_results.py \
      --eval-dir results/policy-eval-v2 \
      --plot-mode all
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def plot_episode(df: pd.DataFrame, out_dir: Path, scenario: str, tag: str, temp_warning: float, temp_critical: float, episode: int = 0):
    """Plot individual episode temperature and fan trajectories."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Temp plot
    plt.figure(figsize=(9,4))
    plt.plot(df["t"], df["temp"], label="temp")
    plt.plot(df["t"], df["ambient"], label="ambient", alpha=0.6)
    plt.axhline(temp_warning, color="orange", linestyle="--", label="warning")
    plt.axhline(temp_critical, color="red", linestyle=":", label="critical")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.title(f"{tag} | {scenario} | ep{episode}: Temperature")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_{scenario}_ep{episode}_temp.png", dpi=150)
    plt.close()

    # Fan plot
    plt.figure(figsize=(9,3.5))
    plt.plot(df["t"], df["fan"], color="tab:green")
    plt.xlabel("Time (s)")
    plt.ylabel("Fan Speed (%)")
    plt.title(f"{tag} | {scenario} | ep{episode}: Fan")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_{scenario}_ep{episode}_fan.png", dpi=150)
    plt.close()


def plot_combined_multi_episode(output_dir: Path, episodes_dir: Path, controllers: list, scenarios: list, temp_warning: float, temp_critical: float):
    """Plot all episodes for each controller-scenario combination on one plot (with mean ± std)."""
    combined_dir = output_dir / "combined_episodes"
    combined_dir.mkdir(exist_ok=True)
    
    for tag in controllers:
        for scen in scenarios:
            # Load all episode CSVs for this controller-scenario
            episode_dfs = []
            for ep_file in sorted(episodes_dir.glob(f"{tag}_{scen}_ep*.csv")):
                episode_dfs.append(pd.read_csv(ep_file))
            
            if not episode_dfs:
                continue
            
            # Temperature plot with all episodes + mean
            plt.figure(figsize=(10,5))
            temps = []
            for i, df in enumerate(episode_dfs):
                plt.plot(df["t"], df["temp"], alpha=0.3, color="tab:blue")
                temps.append(df["temp"].values)
            
            # Mean and std
            temps_arr = np.array(temps)
            mean_temp = temps_arr.mean(axis=0)
            std_temp = temps_arr.std(axis=0)
            t = episode_dfs[0]["t"].values
            plt.plot(t, mean_temp, color="tab:blue", linewidth=2, label=f"mean (n={len(episode_dfs)})")
            plt.fill_between(t, mean_temp - std_temp, mean_temp + std_temp, alpha=0.2, color="tab:blue")
            plt.axhline(temp_warning, color="orange", linestyle="--", label="warning", linewidth=1.5)
            plt.axhline(temp_critical, color="red", linestyle=":", label="critical", linewidth=1.5)
            plt.xlabel("Time (s)")
            plt.ylabel("Temperature (°C)")
            plt.title(f"{tag} | {scen}: Temperature (all episodes)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(combined_dir / f"{tag}_{scen}_temp_all_episodes.png", dpi=150)
            plt.close()
            
            # Fan plot with all episodes + mean
            plt.figure(figsize=(10,4))
            fans = []
            for i, df in enumerate(episode_dfs):
                plt.plot(df["t"], df["fan"], alpha=0.3, color="tab:green")
                fans.append(df["fan"].values)
            
            fans_arr = np.array(fans)
            mean_fan = fans_arr.mean(axis=0)
            std_fan = fans_arr.std(axis=0)
            plt.plot(t, mean_fan, color="tab:green", linewidth=2, label=f"mean (n={len(episode_dfs)})")
            plt.fill_between(t, mean_fan - std_fan, mean_fan + std_fan, alpha=0.2, color="tab:green")
            plt.xlabel("Time (s)")
            plt.ylabel("Fan Speed (%)")
            plt.title(f"{tag} | {scen}: Fan Speed (all episodes)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(combined_dir / f"{tag}_{scen}_fan_all_episodes.png", dpi=150)
            plt.close()


def plot_controller_comparison_per_scenario(output_dir: Path, episodes_dir: Path, controllers: list, scenarios: list, temp_warning: float, temp_critical: float):
    """For each scenario, plot all controllers on one plot (mean trajectory)."""
    comp_dir = output_dir / "controller_comparison"
    comp_dir.mkdir(exist_ok=True)
    
    colors = {"mpc_rc": "tab:blue", "mpc_rcnn": "tab:cyan", "rl_rc": "tab:orange", "rl_rcnn": "tab:red"}
    
    for scen in scenarios:
        # Temperature comparison
        plt.figure(figsize=(10,5))
        for tag in controllers:
            episode_dfs = [pd.read_csv(f) for f in sorted(episodes_dir.glob(f"{tag}_{scen}_ep*.csv"))]
            if not episode_dfs:
                continue
            temps = np.array([df["temp"].values for df in episode_dfs])
            mean_temp = temps.mean(axis=0)
            t = episode_dfs[0]["t"].values
            plt.plot(t, mean_temp, label=tag, color=colors.get(tag, None), linewidth=2)
        
        plt.axhline(temp_warning, color="orange", linestyle="--", label="warning", linewidth=1, alpha=0.7)
        plt.axhline(temp_critical, color="red", linestyle=":", label="critical", linewidth=1, alpha=0.7)
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.title(f"{scen}: Controller Comparison (Temperature)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(comp_dir / f"{scen}_temp_comparison.png", dpi=150)
        plt.close()
        
        # Fan comparison
        plt.figure(figsize=(10,4))
        for tag in controllers:
            episode_dfs = [pd.read_csv(f) for f in sorted(episodes_dir.glob(f"{tag}_{scen}_ep*.csv"))]
            if not episode_dfs:
                continue
            fans = np.array([df["fan"].values for df in episode_dfs])
            mean_fan = fans.mean(axis=0)
            t = episode_dfs[0]["t"].values
            plt.plot(t, mean_fan, label=tag, color=colors.get(tag, None), linewidth=2)
        
        plt.xlabel("Time (s)")
        plt.ylabel("Fan Speed (%)")
        plt.title(f"{scen}: Controller Comparison (Fan Speed)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(comp_dir / f"{scen}_fan_comparison.png", dpi=150)
        plt.close()


def plot_scenario_grid_per_controller(output_dir: Path, episodes_dir: Path, controllers: list, scenarios: list, temp_warning: float, temp_critical: float):
    """For each controller, create a grid of all scenarios (3x4 subplots)."""
    grid_dir = output_dir / "scenario_grids"
    grid_dir.mkdir(exist_ok=True)
    
    for tag in controllers:
        # Temperature grid
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        axes = axes.flatten()
        
        for i, scen in enumerate(scenarios):
            ax = axes[i]
            episode_dfs = [pd.read_csv(f) for f in sorted(episodes_dir.glob(f"{tag}_{scen}_ep*.csv"))]
            if episode_dfs:
                temps = np.array([df["temp"].values for df in episode_dfs])
                mean_temp = temps.mean(axis=0)
                t = episode_dfs[0]["t"].values
                ax.plot(t, mean_temp, color="tab:blue", linewidth=1.5)
                ax.axhline(temp_warning, color="orange", linestyle="--", linewidth=1, alpha=0.7)
                ax.axhline(temp_critical, color="red", linestyle=":", linewidth=1, alpha=0.7)
                ax.set_title(scen, fontsize=9)
                ax.grid(True, alpha=0.3)
                if i >= 8:
                    ax.set_xlabel("Time (s)", fontsize=8)
                if i % 4 == 0:
                    ax.set_ylabel("Temp (°C)", fontsize=8)
        
        plt.suptitle(f"{tag}: Temperature Across All Scenarios", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(grid_dir / f"{tag}_temp_grid.png", dpi=150)
        plt.close()
        
        # Fan grid
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        axes = axes.flatten()
        
        for i, scen in enumerate(scenarios):
            ax = axes[i]
            episode_dfs = [pd.read_csv(f) for f in sorted(episodes_dir.glob(f"{tag}_{scen}_ep*.csv"))]
            if episode_dfs:
                fans = np.array([df["fan"].values for df in episode_dfs])
                mean_fan = fans.mean(axis=0)
                t = episode_dfs[0]["t"].values
                ax.plot(t, mean_fan, color="tab:green", linewidth=1.5)
                ax.set_title(scen, fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(15, 105)
                if i >= 8:
                    ax.set_xlabel("Time (s)", fontsize=8)
                if i % 4 == 0:
                    ax.set_ylabel("Fan (%)", fontsize=8)
        
        plt.suptitle(f"{tag}: Fan Speed Across All Scenarios", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(grid_dir / f"{tag}_fan_grid.png", dpi=150)
        plt.close()


def plot_heatmaps(output_dir: Path, combined_df: pd.DataFrame):
    """Create heatmaps for controller × scenario metrics."""
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    
    for metric, title in [("mean_fan", "Mean Fan Speed (%)"), 
                          ("cumulative_reward", "Mean Cumulative Reward"),
                          ("violations_warning_entries", "Warning Violations (entries)")]:
        if metric not in combined_df.columns:
            continue
        
        pivot = combined_df.pivot(index="controller", columns="scenario", values=metric)
        
        plt.figure(figsize=(14, 4))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r" if "violation" in metric else "RdYlGn", 
                    cbar_kws={'label': title}, linewidths=0.5)
        plt.title(f"Heatmap: {title}")
        plt.xlabel("Scenario")
        plt.ylabel("Controller")
        plt.tight_layout()
        plt.savefig(heatmap_dir / f"heatmap_{metric}.png", dpi=150)
        plt.close()


def plot_bar_charts(output_dir: Path, combined_df: pd.DataFrame, controllers: list):
    """Create bar charts for metric comparison across scenarios."""
    for metric, ylabel in [("cumulative_reward", "Mean Cumulative Reward"), 
                           ("mean_fan", "Mean Fan %"), 
                           ("violations_warning_entries", "Warn Entries"), 
                           ("violations_critical_entries", "Critical Entries")]:
        if metric not in combined_df.columns:
            continue
            
        plt.figure(figsize=(12,5))
        scenarios_unique = combined_df["scenario"].unique()
        x = np.arange(len(scenarios_unique))
        width = 0.18
        
        for i, tag in enumerate(controllers):
            sub = combined_df[combined_df["controller"] == tag]
            vals = [float(sub[sub["scenario"]==s][metric].values[0]) if s in set(sub["scenario"]) else np.nan for s in scenarios_unique]
            plt.bar(x + i*width, vals, width=width, label=tag)
        
        plt.xticks(x + width* (len(controllers)-1)/2, scenarios_unique, rotation=30, ha='right')
        plt.ylabel(ylabel)
        plt.title(f"Comparison: {ylabel} by Scenario")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"combined_{metric}.png", dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots from policy evaluation CSV files")
    parser.add_argument("--eval-dir", type=str, required=True, help="Directory containing evaluation results (with episodes/ subdirectory)")
    parser.add_argument("--plot-mode", type=str, default="combined", 
                        choices=["all", "combined", "minimal"],
                        help="Plot generation mode: 'all' (individual + combined), 'combined' (multi-episode + summaries), 'minimal' (summaries only)")
    parser.add_argument("--temp-warning", type=float, default=80.0, help="Warning temperature threshold")
    parser.add_argument("--temp-critical", type=float, default=90.0, help="Critical temperature threshold")
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir)
    episodes_dir = eval_dir / "episodes"
    
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")
    
    # Discover controllers and scenarios from episode CSV files
    episode_files = list(episodes_dir.glob("*_ep*.csv"))
    if not episode_files:
        raise FileNotFoundError(f"No episode CSV files found in {episodes_dir}")
    
    # Parse filenames to extract controllers and scenarios
    controllers = set()
    scenarios = set()
    for f in episode_files:
        # Format: {controller}_{scenario}_ep{N}.csv
        parts = f.stem.split("_ep")[0].split("_")
        # Controller is first part(s), scenario is remaining
        # Heuristic: if starts with mpc_ or rl_, controller is first 2 parts, else first part
        if parts[0] in ["mpc", "rl"]:
            controller = "_".join(parts[:2])
            scenario = "_".join(parts[2:])
        else:
            controller = parts[0]
            scenario = "_".join(parts[1:])
        controllers.add(controller)
        scenarios.add(scenario)
    
    controllers = sorted(list(controllers))
    scenarios = sorted(list(scenarios))
    
    print(f"[INFO] Found {len(controllers)} controllers: {controllers}")
    print(f"[INFO] Found {len(scenarios)} scenarios: {scenarios}")
    
    # Load combined_per_scenario.csv if it exists
    combined_csv = eval_dir / "combined_per_scenario.csv"
    if combined_csv.exists():
        combined_df = pd.read_csv(combined_csv)
    else:
        print(f"[WARN] combined_per_scenario.csv not found at {combined_csv}, skipping heatmaps and bar charts")
        combined_df = None
    
    # Filter scenarios for minimal mode
    if args.plot_mode == "minimal":
        key_scenarios = ["baseline", "thermal_high_start", "combined_extreme", "low_workload"]
        plot_scenarios = [s for s in scenarios if s in key_scenarios]
        print(f"[INFO] Minimal mode: plotting only {len(plot_scenarios)} key scenarios: {plot_scenarios}")
    else:
        plot_scenarios = scenarios
    
    # Count episodes
    sample_files = list(episodes_dir.glob(f"{controllers[0]}_{scenarios[0]}_ep*.csv"))
    num_episodes = len(sample_files)
    print(f"[INFO] Found {num_episodes} episodes per scenario-controller combination")
    
    # Generate plots based on mode
    print(f"\n[INFO] Generating plots (mode={args.plot_mode})...")
    
    # Individual episode plots (only in 'all' mode)
    if args.plot_mode == "all":
        print("  - Individual episode plots...")
        plots_dir = eval_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        for tag in controllers:
            for scen in scenarios:
                for ep_file in sorted(episodes_dir.glob(f"{tag}_{scen}_ep*.csv")):
                    ep_num = int(ep_file.stem.split("_ep")[-1])
                    df = pd.read_csv(ep_file)
                    plot_episode(df, plots_dir, scen, tag, args.temp_warning, args.temp_critical, episode=ep_num)
        print(f"    Generated {len(controllers) * len(scenarios) * num_episodes * 2} plots")
    
    # Multi-episode overlays (in 'all' and 'combined' modes)
    if args.plot_mode in ["all", "combined"]:
        print("  - Multi-episode overlays (mean ± std)...")
        plot_combined_multi_episode(eval_dir, episodes_dir, controllers, plot_scenarios, args.temp_warning, args.temp_critical)
        print(f"    Generated {len(controllers) * len(plot_scenarios) * 2} plots")
    
    # Controller comparisons (always)
    print("  - Controller comparison plots...")
    plot_controller_comparison_per_scenario(eval_dir, episodes_dir, controllers, plot_scenarios, args.temp_warning, args.temp_critical)
    print(f"    Generated {len(plot_scenarios) * 2} plots")
    
    # Scenario grids (always)
    print("  - Scenario grids...")
    plot_scenario_grid_per_controller(eval_dir, episodes_dir, controllers, scenarios, args.temp_warning, args.temp_critical)
    print(f"    Generated {len(controllers) * 2} plots")
    
    # Heatmaps and bar charts (if combined_df available)
    if combined_df is not None:
        print("  - Heatmaps...")
        plot_heatmaps(eval_dir, combined_df)
        print(f"    Generated 3 plots")
        
        print("  - Bar charts...")
        plot_bar_charts(eval_dir, combined_df, controllers)
        print(f"    Generated 4 plots")
    
    print(f"\n[INFO] Plotting complete! Results saved to {eval_dir}")
    
    # Summary
    total_plots = 0
    if args.plot_mode == "all":
        total_plots += len(controllers) * len(scenarios) * num_episodes * 2
        total_plots += len(controllers) * len(plot_scenarios) * 2
    elif args.plot_mode == "combined":
        total_plots += len(controllers) * len(plot_scenarios) * 2
    
    total_plots += len(plot_scenarios) * 2  # controller comparisons
    total_plots += len(controllers) * 2  # scenario grids
    if combined_df is not None:
        total_plots += 3  # heatmaps
        total_plots += 4  # bar charts
    
    print(f"[INFO] Total plots generated: ~{total_plots}")


if __name__ == "__main__":
    main()
