#!/usr/bin/env python3
"""
Extract training metrics from TensorBoard logs and generate publication-quality plots.

Usage:
    # Plot training curves for RC surrogate
    python scripts/analysis/plot_training_curves.py \
      --run-dir runs/rl/sac_rc_baseline \
      --output-dir results/training_analysis/rc

    # Plot training curves for RCNN surrogate
    python scripts/analysis/plot_training_curves.py \
      --run-dir runs/rl/sac_rcnn_hybrid \
      --output-dir results/training_analysis/rcnn

    # Compare RC vs RCNN training
    python scripts/analysis/plot_training_curves.py \
      --run-dirs runs/rl/sac_rc_baseline runs/rl/sac_rcnn_hybrid \
      --labels "RL-RC" "RL-RCNN" \
      --output-dir results/training_analysis/comparison
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_tensorboard_scalars(log_dir: Path, tags: list = None):
    """
    Extract scalar metrics from TensorBoard event files.
    
    Args:
        log_dir: Path to tensorboard log directory
        tags: List of metric tags to extract (None = all)
    
    Returns:
        dict: {tag: [(step, value), ...]}
    """
    # Find event files
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")
    
    # Use the most recent event file
    event_file = max(event_files, key=lambda p: p.stat().st_mtime)
    print(f"[INFO] Reading TensorBoard events from: {event_file.name}")
    
    # Load events
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    
    # Get available tags
    available_tags = ea.Tags()["scalars"]
    print(f"[INFO] Available metrics: {available_tags}")
    
    # Extract requested tags (or all if None)
    if tags is None:
        tags = available_tags
    
    data = {}
    for tag in tags:
        if tag in available_tags:
            events = ea.Scalars(tag)
            data[tag] = [(e.step, e.value) for e in events]
        else:
            print(f"[WARN] Tag '{tag}' not found in logs")
    
    return data


def plot_single_run(run_dir: Path, output_dir: Path, run_name: str = None):
    """Generate training plots for a single run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if run_name is None:
        run_name = run_dir.name
    
    # Try to load from progress.csv first (if available)
    logs_dir = run_dir / "logs"
    progress_csv = logs_dir / "progress.csv" if logs_dir.exists() else None
    
    if progress_csv and progress_csv.exists():
        print(f"[INFO] Loading from progress.csv: {progress_csv}")
        df = pd.read_csv(progress_csv)
        
        # Determine x-axis
        x_key = None
        for candidate in ["time/total_timesteps", "timesteps", "total_timesteps"]:
            if candidate in df.columns:
                x_key = candidate
                break
        
        if x_key is None:
            x = np.arange(len(df))
            x_label = "Training Steps"
        else:
            x = df[x_key].values
            x_label = "Timesteps"
        
        # Plot 1: Episode Reward
        if "rollout/ep_rew_mean" in df.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(x, df["rollout/ep_rew_mean"], linewidth=2, color="tab:blue")
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel("Mean Episode Reward", fontsize=12)
            plt.title(f"{run_name}: Learning Curve (Reward)", fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "reward_curve.png", dpi=150)
            plt.close()
            print(f"  ✓ Saved reward_curve.png")
        
        # Plot 2: Episode Length
        if "rollout/ep_len_mean" in df.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(x, df["rollout/ep_len_mean"], linewidth=2, color="tab:orange")
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel("Mean Episode Length", fontsize=12)
            plt.title(f"{run_name}: Episode Length", fontsize=14, fontweight="bold")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "episode_length_curve.png", dpi=150)
            plt.close()
            print(f"  ✓ Saved episode_length_curve.png")
        
        # Plot 3: Actor/Critic Losses
        loss_keys = [k for k in df.columns if "loss" in k.lower() or "train/" in k]
        if loss_keys:
            plt.figure(figsize=(10, 5))
            for k in loss_keys[:5]:  # Limit to 5 for readability
                plt.plot(x, df[k], label=k.replace("train/", ""), linewidth=1.5, alpha=0.8)
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.title(f"{run_name}: Training Losses", fontsize=14, fontweight="bold")
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "loss_curves.png", dpi=150)
            plt.close()
            print(f"  ✓ Saved loss_curves.png")
        
        # Summary statistics
        summary = {
            "run_name": run_name,
            "total_timesteps": int(x[-1]) if len(x) > 0 else 0,
            "final_reward_mean": float(df["rollout/ep_rew_mean"].iloc[-1]) if "rollout/ep_rew_mean" in df.columns else None,
            "max_reward_mean": float(df["rollout/ep_rew_mean"].max()) if "rollout/ep_rew_mean" in df.columns else None,
            "final_ep_len_mean": float(df["rollout/ep_len_mean"].iloc[-1]) if "rollout/ep_len_mean" in df.columns else None,
        }
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(output_dir / "training_summary.csv", index=False)
        print(f"  ✓ Saved training_summary.csv")
        
    else:
        # Fallback to TensorBoard logs
        print(f"[INFO] progress.csv not found, extracting from TensorBoard logs...")
        tb_dir = run_dir / "tensorboard"
        if not tb_dir.exists():
            print(f"[ERROR] No tensorboard directory found in {run_dir}")
            return
        
        try:
            data = extract_tensorboard_scalars(
                tb_dir,
                tags=["rollout/ep_rew_mean", "rollout/ep_len_mean", "train/actor_loss", "train/critic_loss"]
            )
            
            # Plot reward
            if "rollout/ep_rew_mean" in data:
                steps, values = zip(*data["rollout/ep_rew_mean"])
                plt.figure(figsize=(10, 5))
                plt.plot(steps, values, linewidth=2, color="tab:blue")
                plt.xlabel("Timesteps", fontsize=12)
                plt.ylabel("Mean Episode Reward", fontsize=12)
                plt.title(f"{run_name}: Learning Curve (Reward)", fontsize=14, fontweight="bold")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "reward_curve.png", dpi=150)
                plt.close()
                print(f"  ✓ Saved reward_curve.png")
            
            # Plot episode length
            if "rollout/ep_len_mean" in data:
                steps, values = zip(*data["rollout/ep_len_mean"])
                plt.figure(figsize=(10, 5))
                plt.plot(steps, values, linewidth=2, color="tab:orange")
                plt.xlabel("Timesteps", fontsize=12)
                plt.ylabel("Mean Episode Length", fontsize=12)
                plt.title(f"{run_name}: Episode Length", fontsize=14, fontweight="bold")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "episode_length_curve.png", dpi=150)
                plt.close()
                print(f"  ✓ Saved episode_length_curve.png")
            
        except Exception as e:
            print(f"[ERROR] Failed to extract TensorBoard data: {e}")


def plot_comparison(run_dirs: list, labels: list, output_dir: Path):
    """Compare training curves across multiple runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    # Collect data from all runs
    all_data = []
    for run_dir, label in zip(run_dirs, labels):
        logs_dir = run_dir / "logs"
        progress_csv = logs_dir / "progress.csv" if logs_dir.exists() else None
        
        if progress_csv and progress_csv.exists():
            df = pd.read_csv(progress_csv)
            
            # Determine x-axis
            x_key = None
            for candidate in ["time/total_timesteps", "timesteps", "total_timesteps"]:
                if candidate in df.columns:
                    x_key = candidate
                    break
            
            x = df[x_key].values if x_key else np.arange(len(df))
            all_data.append((label, x, df))
        else:
            print(f"[WARN] No progress.csv found for {run_dir.name}, skipping")
    
    if not all_data:
        print("[ERROR] No data to compare")
        return
    
    # Plot 1: Reward Comparison
    plt.figure(figsize=(12, 6))
    for i, (label, x, df) in enumerate(all_data):
        if "rollout/ep_rew_mean" in df.columns:
            plt.plot(x, df["rollout/ep_rew_mean"], label=label, linewidth=2, color=colors[i % len(colors)])
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Mean Episode Reward", fontsize=12)
    plt.title("Training Comparison: Episode Reward", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_reward.png", dpi=150)
    plt.close()
    print(f"  ✓ Saved comparison_reward.png")
    
    # Plot 2: Episode Length Comparison
    plt.figure(figsize=(12, 6))
    for i, (label, x, df) in enumerate(all_data):
        if "rollout/ep_len_mean" in df.columns:
            plt.plot(x, df["rollout/ep_len_mean"], label=label, linewidth=2, color=colors[i % len(colors)])
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Mean Episode Length", fontsize=12)
    plt.title("Training Comparison: Episode Length", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_episode_length.png", dpi=150)
    plt.close()
    print(f"  ✓ Saved comparison_episode_length.png")
    
    # Summary table
    summary_rows = []
    for label, x, df in all_data:
        row = {
            "run": label,
            "total_timesteps": int(x[-1]) if len(x) > 0 else 0,
            "final_reward": float(df["rollout/ep_rew_mean"].iloc[-1]) if "rollout/ep_rew_mean" in df.columns else None,
            "max_reward": float(df["rollout/ep_rew_mean"].max()) if "rollout/ep_rew_mean" in df.columns else None,
            "final_ep_len": float(df["rollout/ep_len_mean"].iloc[-1]) if "rollout/ep_len_mean" in df.columns else None,
        }
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "comparison_summary.csv", index=False)
    print(f"  ✓ Saved comparison_summary.csv")


def main():
    parser = argparse.ArgumentParser(description="Plot RL training curves from logs")
    parser.add_argument("--run-dir", type=str, help="Single run directory to plot")
    parser.add_argument("--run-dirs", nargs="+", help="Multiple run directories for comparison")
    parser.add_argument("--labels", nargs="+", help="Labels for comparison runs (must match --run-dirs)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--run-name", type=str, help="Custom name for single run (default: directory name)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.run_dir:
        # Single run mode
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        print(f"[INFO] Plotting training curves for: {run_dir.name}")
        plot_single_run(run_dir, output_dir, args.run_name)
        
    elif args.run_dirs:
        # Comparison mode
        run_dirs = [Path(d) for d in args.run_dirs]
        
        # Validate
        for rd in run_dirs:
            if not rd.exists():
                raise FileNotFoundError(f"Run directory not found: {rd}")
        
        # Default labels if not provided
        if args.labels:
            if len(args.labels) != len(run_dirs):
                raise ValueError("Number of labels must match number of run directories")
            labels = args.labels
        else:
            labels = [rd.name for rd in run_dirs]
        
        print(f"[INFO] Comparing {len(run_dirs)} training runs")
        plot_comparison(run_dirs, labels, output_dir)
        
    else:
        parser.error("Must specify either --run-dir or --run-dirs")
    
    print(f"\n[INFO] All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
