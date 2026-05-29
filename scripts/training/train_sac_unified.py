#!/usr/bin/env python3
"""
Unified SAC training script for thermal control with surrogate adapter support.

Supports RC, RF, and PINN surrogates via unified interface.

Usage:
    python scripts/training/train_sac_unified.py --config configs/rl/sac_unified.yaml
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.environments.thermal_unified import ThermalControlEnv
from src.rl.surrogates import create_surrogate
from src.rl.safety.shield import SafetyWrapper


class VecNormalizeSaveCallback(BaseCallback):
    """Periodically save VecNormalize statistics."""

    def __init__(self, vec_env, save_path: Path, save_freq: int):
        super().__init__()
        self.vec_env = vec_env
        self.save_path = save_path
        self.save_freq = int(save_freq)

    def _on_step(self) -> bool:
        if self.save_freq <= 0:
            return True
        if (self.num_timesteps % self.save_freq) == 0:
            if isinstance(self.vec_env, VecNormalize):
                self.vec_env.save(str(self.save_path))
                print(f"[INFO] Saved VecNormalize stats at step {self.num_timesteps}")
        return True


class DebugStatsCallback(BaseCallback):
    """Lightweight callback to log optimizer/param stats to a CSV for debugging."""
    def __init__(self, save_path: Path, log_every: int = 1000):
        super().__init__()
        self.save_path = save_path
        self.log_every = int(log_every)
        self._rows = []

    def _on_step(self) -> bool:
        if self.log_every <= 0:
            return True
        if (self.num_timesteps % self.log_every) != 0:
            return True

        try:
            # Collect parameter norms (actor and critics)
            actor_norm = 0.0
            critic0_norm = 0.0
            critic1_norm = 0.0
            with torch.no_grad():
                for p in self.model.actor.parameters():
                    actor_norm += float((p.data ** 2).sum().sqrt().item())
                # Critic networks names differ by SB3 version; try common ones
                qfs = []
                if hasattr(self.model, "critic"):
                    qfs.append(self.model.critic)
                if hasattr(self.model, "critic_target"):
                    qfs.append(self.model.critic_target)
                if hasattr(self.model, "critic") and hasattr(self.model.critic, "qf1"):
                    qfs = [self.model.critic.qf1, getattr(self.model.critic, "qf2", None)]
                norms = []
                for q in qfs:
                    if q is None:
                        continue
                    n = 0.0
                    for p in q.parameters():
                        n += float((p.data ** 2).sum().sqrt().item())
                    norms.append(n)
                if norms:
                    critic0_norm = norms[0]
                    critic1_norm = norms[-1]

            # Read last logged losses if available
            critic_loss = None
            for k in [
                "train/critic_loss", "train/qf_loss", "train/qf0_loss", "train/qf1_loss",
                "train/value_loss"
            ]:
                if hasattr(self.model.logger, "name_to_value") and k in self.model.logger.name_to_value:
                    critic_loss = float(self.model.logger.name_to_value[k])
                    break

            self._rows.append({
                "timesteps": int(self.num_timesteps),
                "actor_param_norm": actor_norm,
                "critic0_param_norm": critic0_norm,
                "critic1_param_norm": critic1_norm,
                "critic_loss": critic_loss if critic_loss is not None else float("nan"),
            })
        except Exception:
            pass
        return True

    def _on_training_end(self) -> None:
        try:
            import pandas as pd
            if self._rows:
                df = pd.DataFrame(self._rows)
            else:
                df = pd.DataFrame(columns=[
                    "timesteps",
                    "actor_param_norm",
                    "critic0_param_norm",
                    "critic1_param_norm",
                    "critic_loss",
                ])
            df.to_csv(self.save_path, index=False)
            print(f"[DEBUG] Saved debug stats to {self.save_path}")
        except Exception as e:
            print(f"[WARN] Failed to save debug stats: {e}")

def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Find latest checkpoint based on step number."""
    if not ckpt_dir.exists():
        return None
    cands = list(ckpt_dir.glob("sac_*_steps.zip"))
    if not cands:
        return None
    
    def step_num(p: Path) -> int:
        m = re.search(r"_(\d+)_steps\.zip$", p.name)
        return int(m.group(1)) if m else -1
    
    return sorted(cands, key=step_num)[-1]


def checkpoint_step(ckpt_path: Path) -> int:
    """Extract step count from a checkpoint filename like sac_200_steps.zip."""
    m = re.search(r"_(\d+)_steps\.zip$", ckpt_path.name)
    return int(m.group(1)) if m else 0


def find_replay_buffer(ckpt_path: Path) -> Path | None:
    """Find matching replay buffer for checkpoint."""
    m = re.search(r"_(\d+)_steps\.zip$", ckpt_path.name)
    if not m:
        return None
    steps = m.group(1)
    rb = ckpt_path.parent / f"sac_replay_buffer_{steps}_steps.pkl"
    return rb if rb.exists() else None


def make_env(surrogate_config: dict, env_config: dict, safety_config: dict, use_safety: bool = True, monitor_dir: Path | None = None):
    """
    Create a single thermal control environment with unified surrogate.
    
    Args:
        surrogate_config: Config for surrogate adapter (type, model_path, etc.)
        env_config: Environment configuration
        safety_config: Safety wrapper configuration
        use_safety: Whether to use safety wrapper
    
    Returns:
        Configured environment
    """
    surrogate = create_surrogate(
        surrogate_type=surrogate_config["type"],
        config=surrogate_config
    )
    
    env = ThermalControlEnv(
        surrogate=surrogate,
        config=env_config
    )
    
    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, str(monitor_dir))
    else:
        env = Monitor(env)
    
    if use_safety:
        env = SafetyWrapper(env, safety_config)
    
    return env


def main():
    parser = argparse.ArgumentParser(description="Train SAC agent with unified surrogate interface")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device for training (cpu/cuda/auto)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_name = config.get("run_name", f"sac_unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir = Path(config.get("output_dir", "runs/rl")) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    surrogate_config = config["surrogate"]
    env_config = config.get("env", {})
    safety_config = config.get("safety", {})
    use_safety = config.get("use_safety", True)

    print(f"[INFO] Creating environment with {surrogate_config['type'].upper()} surrogate")
    env = DummyVecEnv([lambda: make_env(surrogate_config, env_config, safety_config, use_safety, output_dir / "monitor")])

    if config.get("normalize_obs", True):
        vecnorm_path = output_dir / "vecnormalize.pkl"
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=config.get("normalize_reward", True),
            clip_obs=10.0,
            clip_reward=10.0
        )

    sac_config = config.get("sac", {})
    
    latest_ckpt = None
    if args.resume:
        latest_ckpt = find_latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            print(f"[INFO] Resuming from checkpoint: {latest_ckpt}")
            model = SAC.load(
                str(latest_ckpt),
                env=env,
                device=args.device,
                print_system_info=True
            )
            
            rb_path = find_replay_buffer(latest_ckpt)
            if rb_path and rb_path.exists():
                print(f"[INFO] Loading replay buffer from {rb_path}")
                model.load_replay_buffer(str(rb_path))
            
            if config.get("normalize_obs", True) and vecnorm_path.exists():
                print(f"[INFO] Loading VecNormalize stats from {vecnorm_path}")
                env = VecNormalize.load(str(vecnorm_path), env)
        else:
            print("[INFO] No checkpoint found, starting fresh")

    if latest_ckpt is None:
        print("[INFO] Creating new SAC model")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=float(sac_config.get("learning_rate", 3e-4)),
            buffer_size=int(sac_config.get("buffer_size", 100000)),
            learning_starts=int(sac_config.get("learning_starts", 1000)),
            batch_size=int(sac_config.get("batch_size", 256)),
            tau=float(sac_config.get("tau", 0.005)),
            gamma=float(sac_config.get("gamma", 0.99)),
            train_freq=int(sac_config.get("train_freq", 1)),
            gradient_steps=int(sac_config.get("gradient_steps", 1)),
            verbose=1,
            device=args.device,
            tensorboard_log=str(output_dir / "tensorboard")
        )

    # Configure logger to write CSV (progress.csv) and TensorBoard scalars
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    # SB3 logger API expects 'format_strings' or positional formats list depending on version
    try:
        new_logger = configure(folder=str(log_dir), format_strings=["csv", "tensorboard", "stdout"])
    except TypeError:
        # Fallback to positional signature: configure(folder, format_strings)
        new_logger = configure(str(log_dir), ["csv", "tensorboard", "stdout"])
    model.set_logger(new_logger)

    total_timesteps = int(config.get("total_timesteps", 100000))
    
    if latest_ckpt:
        steps_done = checkpoint_step(latest_ckpt)
        remaining = max(0, total_timesteps - steps_done)
        print(f"[INFO] Already trained {steps_done} steps, {remaining} remaining")
        total_timesteps = remaining

    checkpoint_callback = CheckpointCallback(
        save_freq=int(config.get("checkpoint_freq", 10000)),
        save_path=str(ckpt_dir),
        name_prefix="sac",
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    callbacks = [checkpoint_callback]

    # Attach debug stats callback to verify critic updates during training
    debug_csv = output_dir / "debug_stats.csv"
    callbacks.append(DebugStatsCallback(save_path=debug_csv, log_every=int(config.get("debug_log_every", 1000))))
    
    if config.get("normalize_obs", True):
        vecnorm_callback = VecNormalizeSaveCallback(
            vec_env=env,
            save_path=vecnorm_path,
            save_freq=int(config.get("checkpoint_freq", 10000))
        )
        callbacks.append(vecnorm_callback)

    print(f"[INFO] Starting training for {total_timesteps} timesteps")
    print(f"[INFO] Surrogate type: {surrogate_config['type']}")
    print(f"[INFO] Output directory: {output_dir}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10,
        progress_bar=True
    )

    final_model_path = output_dir / "sac_final.zip"
    model.save(str(final_model_path))
    print(f"[INFO] Saved final model to {final_model_path}")

    if isinstance(env, VecNormalize):
        env.save(str(vecnorm_path))
        print(f"[INFO] Saved final VecNormalize stats to {vecnorm_path}")

    metrics = {
        "total_timesteps": model.num_timesteps,
        "surrogate_type": surrogate_config["type"],
        "run_name": run_name,
        "completed_at": datetime.now().isoformat()
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[INFO] Training complete!")

    # Auto-generate learning curve PNGs from CSV logger if available
    progress_csv = log_dir / "progress.csv"
    try:
        if progress_csv.exists() and progress_csv.stat().st_size > 0:
            df = pd.read_csv(progress_csv)
            # X-axis: total timesteps if present
            x_key_candidates = [
                "time/total_timesteps",
                "time/iterations",
                "timesteps"
            ]
            x_key = next((k for k in x_key_candidates if k in df.columns), None)
            if x_key is None:
                x = range(len(df))
            else:
                x = df[x_key].values

            # 1) Mean episode reward
            if "rollout/ep_rew_mean" in df.columns:
                plt.figure(figsize=(8,4))
                plt.plot(x, df["rollout/ep_rew_mean"], label="ep_rew_mean")
                plt.xlabel("timesteps" if x_key is None else x_key)
                plt.ylabel("mean episode reward")
                plt.title("Learning Curve: Episode Reward")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "learning_curve_reward.png", dpi=150)
                plt.close()

            # 2) Episode length
            if "rollout/ep_len_mean" in df.columns:
                plt.figure(figsize=(8,4))
                plt.plot(x, df["rollout/ep_len_mean"], label="ep_len_mean", color="tab:orange")
                plt.xlabel("timesteps" if x_key is None else x_key)
                plt.ylabel("mean episode length")
                plt.title("Learning Curve: Episode Length")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "learning_curve_ep_len.png", dpi=150)
                plt.close()

            # 3) Loss diagnostics (best-effort)
            loss_keys = [
                k for k in [
                    "train/actor_loss",
                    "train/critic_loss",
                    "train/entropy",
                    "train/ent_coef_loss",
                    "train/qf_loss",
                    "train/qf0_loss",
                    "train/qf1_loss",
                    "train/value_loss",
                ] if k in df.columns
            ]
            if loss_keys:
                plt.figure(figsize=(8,4))
                for k in loss_keys:
                    plt.plot(x, df[k], label=k)
                plt.xlabel("timesteps" if x_key is None else x_key)
                plt.ylabel("loss / metric")
                plt.title("Learning Diagnostics")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "learning_curve_losses.png", dpi=150)
                plt.close()
            print(f"[INFO] Saved learning curves to {output_dir}")
        else:
            # Fallback: derive plots from Monitor CSV (per-episode logs)
            monitor_csv = output_dir / "monitor" / "monitor.csv"
            if monitor_csv.exists() and monitor_csv.stat().st_size > 0:
                mdf = pd.read_csv(monitor_csv, comment="#")
                # Expect columns: r (reward), l (length), t (time)
                if all(col in mdf.columns for col in ["r", "l"]):
                    mdf["cum_steps"] = mdf["l"].cumsum()
                    # Reward curves
                    plt.figure(figsize=(8,4))
                    plt.plot(mdf["cum_steps"], mdf["r"], alpha=0.4, label="episode reward")
                    if len(mdf) >= 5:
                        plt.plot(mdf["cum_steps"], mdf["r"].rolling(window=5, min_periods=1).mean(),
                                 color="tab:blue", label="rolling mean (w=5)")
                    plt.xlabel("timesteps")
                    plt.ylabel("episode reward")
                    plt.title("Learning Curve (from monitor): Episode Reward")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(output_dir / "learning_curve_reward.png", dpi=150)
                    plt.close()

                    # Episode length curve
                    plt.figure(figsize=(8,4))
                    plt.plot(mdf["cum_steps"], mdf["l"], alpha=0.6, color="tab:orange")
                    if len(mdf) >= 5:
                        plt.plot(mdf["cum_steps"], mdf["l"].rolling(window=5, min_periods=1).mean(),
                                 color="tab:red", label="rolling mean (w=5)")
                    plt.xlabel("timesteps")
                    plt.ylabel("episode length")
                    plt.title("Learning Curve (from monitor): Episode Length")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(output_dir / "learning_curve_ep_len.png", dpi=150)
                    plt.close()
                    print(f"[INFO] Saved learning curves from monitor.csv to {output_dir}")
                else:
                    print(f"[WARN] monitor.csv missing required columns at {monitor_csv}")
            else:
                print(f"[WARN] progress.csv not found or empty at {progress_csv} and no usable monitor.csv; skipping PNG export")
    except Exception as e:
        print(f"[WARN] Failed to export learning curves: {e}")


if __name__ == "__main__":
    main()
