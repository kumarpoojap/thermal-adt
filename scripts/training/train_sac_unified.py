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
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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


def make_env(surrogate_config: dict, env_config: dict, safety_config: dict, use_safety: bool = True):
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
    env = DummyVecEnv([lambda: make_env(surrogate_config, env_config, safety_config, use_safety)])

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


if __name__ == "__main__":
    main()
