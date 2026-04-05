#!/usr/bin/env python3
"""
Self-contained SAC training script for thermal control.
Works with the thermal-adt repository structure.

Usage:
    python scripts/training/train_sac.py --config configs/rl/sac_validation.yaml
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.environments.thermal_rf import ThermalControlEnvRF
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


def make_env(rf_model_path: str, env_config: dict, safety_config: dict, use_safety: bool = True):
    """Create a single thermal control environment."""
    # Create base environment (match ThermalControlEnvRF signature)
    env = ThermalControlEnvRF(
        rf_model_path=Path(rf_model_path),
        config=env_config,
        k_ahead=int(env_config.get("k_ahead", 10)),
        cadence_seconds=float(env_config.get("cadence_seconds", 1.0)),
    )
    
    # Wrap with Monitor for episode stats
    env = Monitor(env)
    
    # Add safety wrapper
    if use_safety:
        env = SafetyWrapper(env, safety_config)
    
    return env


def main():
    parser = argparse.ArgumentParser(description="Train SAC agent for thermal control")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("SAC TRAINING - THERMAL CONTROL")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Resume: {args.resume}")
    
    # Paths
    rf_model_path = config.get("rf_model_path", "models/rf_teacher.pkl")
    base_save_dir = Path(config.get("save_dir", "results/rl_training/sac"))
    
    # Create run directory
    if not args.resume:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        save_dir = base_save_dir / run_id
    else:
        # Find most recent run dir
        if base_save_dir.exists():
            runs = sorted([d for d in base_save_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
            save_dir = runs[-1] if runs else base_save_dir / "run_default"
        else:
            save_dir = base_save_dir / "run_default"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = save_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    vec_normalize_path = save_dir / "vec_normalize.pkl"
    
    print(f"Save directory: {save_dir}")
    
    # Save config snapshot
    if not args.resume:
        with open(save_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f)
    
    # Extract configs
    env_config = config.get("environment", {})
    safety_config = config.get("safety", {})
    sac_config = config.get("sac", {})
    training_config = config.get("training", {})
    
    # Training params
    total_timesteps = training_config.get("total_timesteps", 100000)
    checkpoint_freq = training_config.get("checkpoint_freq", 10000)
    n_envs = config.get("n_envs", 1)
    seed = config.get("seed", 42)
    use_safety = config.get("use_safety_shield", True)
    
    print(f"\nTraining Configuration:")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Checkpoint freq: {checkpoint_freq}")
    print(f"  Safety shield: {use_safety}")
    print(f"  Seed: {seed}")
    
    # Create vectorized environment
    print(f"\n[1/4] Creating environment...")
    print(f"  RF model: {rf_model_path}")
    
    def env_fn():
        return make_env(rf_model_path, env_config, safety_config, use_safety)
    
    vec_env = DummyVecEnv([env_fn for _ in range(n_envs)])
    
    # Wrap with VecNormalize
    if args.resume and vec_normalize_path.exists():
        print(f"  Loading VecNormalize stats from {vec_normalize_path}")
        vec_env = VecNormalize.load(str(vec_normalize_path), vec_env)
        vec_env.training = True
    else:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            gamma=sac_config.get("gamma", 0.99),
        )
    
    print(f"  ✓ Environment created")
    
    # Create or load SAC agent
    print(f"\n[2/4] Creating SAC agent...")
    
    latest_ckpt = find_latest_checkpoint(checkpoints_dir) if args.resume else None
    already_trained_steps = checkpoint_step(latest_ckpt) if latest_ckpt is not None else 0
    if args.resume:
        # Interpret total_timesteps as an absolute cap across resumes.
        remaining_timesteps = max(int(total_timesteps) - int(already_trained_steps), 0)
        if remaining_timesteps == 0:
            print(
                f"[INFO] Latest checkpoint already at {already_trained_steps} steps, "
                f"which meets/exceeds total_timesteps={total_timesteps}. Nothing to do."
            )
            return
    else:
        remaining_timesteps = int(total_timesteps)
    
    if latest_ckpt:
        print(f"  Loading checkpoint: {latest_ckpt}")
        agent = SAC.load(str(latest_ckpt), env=vec_env)
        
        # Load replay buffer if available
        rb_path = find_replay_buffer(latest_ckpt)
        if rb_path:
            print(f"  Loading replay buffer: {rb_path}")
            agent.load_replay_buffer(str(rb_path))
    else:
        print(f"  Creating new agent...")
        agent = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=sac_config.get("learning_rate", 3e-4),
            buffer_size=sac_config.get("buffer_size", 100000),
            learning_starts=sac_config.get("learning_starts", 1000),
            batch_size=sac_config.get("batch_size", 256),
            tau=sac_config.get("tau", 0.005),
            gamma=sac_config.get("gamma", 0.99),
            train_freq=sac_config.get("train_freq", 1),
            gradient_steps=sac_config.get("gradient_steps", 1),
            ent_coef=sac_config.get("ent_coef", "auto"),
            tensorboard_log=str(save_dir / "tensorboard"),
            verbose=1,
            seed=seed,
        )
    
    print(f"  ✓ Agent ready")
    
    # Setup callbacks
    print(f"\n[3/4] Setting up callbacks...")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoints_dir),
        name_prefix="sac",
        save_replay_buffer=True,
        save_vecnormalize=False,  # We handle this separately
    )
    
    vecnorm_callback = VecNormalizeSaveCallback(
        vec_env=vec_env,
        save_path=vec_normalize_path,
        save_freq=checkpoint_freq,
    )
    
    callbacks = [checkpoint_callback, vecnorm_callback]
    print(f"  ✓ Callbacks configured")
    
    # Train
    print(f"\n[4/4] Training...")
    print("="*70)
    
    agent.learn(
        total_timesteps=remaining_timesteps,
        callback=callbacks,
        log_interval=training_config.get("log_interval", 10),
        reset_num_timesteps=not args.resume,
    )
    
    # Save final model
    final_model_path = save_dir / "sac_final.zip"
    agent.save(str(final_model_path))
    vec_env.save(str(vec_normalize_path))
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"\nArtifacts saved to: {save_dir}")
    print(f"  Final model: {final_model_path}")
    print(f"  VecNormalize: {vec_normalize_path}")
    print(f"  Checkpoints: {checkpoints_dir}")
    print(f"  TensorBoard: {save_dir / 'tensorboard'}")
    print("\nTo resume training:")
    print(f"  python scripts/training/train_sac.py --config {args.config} --resume")
    print()
    
    vec_env.close()


if __name__ == "__main__":
    main()
