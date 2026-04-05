"""
Training script for SAC agent on thermal control task.

Usage:
    python -m rl.training.train_sac --config configs/rl_training.yaml
"""

import argparse
from pathlib import Path
import yaml
import sys
import os
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rl.env_thermal_rf import make_thermal_env
from rl.safety_shield import SafetyWrapper
from rl.agents.sac_agent import (
    create_sac_agent,
    train_sac_agent,
    make_vec_env
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


def main():
    parser = argparse.ArgumentParser(description="Train SAC agent for thermal control")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in save_dir (if present)")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("SAC Training for Thermal Control")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print()
    
    # Paths
    rf_model_path = Path(config["rf_model_path"])
    base_save_dir = Path(config["save_dir"]).expanduser()
    # For training, create a timestamped run directory under save_dir; for eval-only, use base as-is
    if not args.eval_only:
        from datetime import datetime
        run_id = datetime.now().strftime("run_%Y-%m-%d_%H-%M") + f"_seed{args.seed}"
        save_dir = base_save_dir / run_id
    else:
        save_dir = base_save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = save_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    vec_normalize_path = save_dir / "vec_normalize.pkl"

    # Ensure SAC has a tensorboard_log path (default to run-local folder if not provided)
    sac_cfg = config.get("sac", {})
    sac_cfg.setdefault("tensorboard_log", str((save_dir / "tensorboard").resolve()))
    config["sac"] = sac_cfg

    # Snapshot configs and manifest for reproducibility (training only)
    if not args.eval_only:
        try:
            # Save raw config used to launch
            raw_cfg_out = save_dir / "config.yaml"
            with open(args.config, "r") as f_in, open(raw_cfg_out, "w") as f_out:
                f_out.write(f_in.read())

            # Save resolved config (with runtime fields)
            resolved = dict(config)
            resolved.setdefault("_resolved", {})
            resolved["_resolved"].update({
                "device": args.device,
                "seed": args.seed,
                "save_dir": str(save_dir.resolve()),
                "tensorboard_log": sac_cfg.get("tensorboard_log"),
                "use_safety_shield": config.get("use_safety_shield", True),
                "surrogate_type": "rf",
            })
            with open(save_dir / "resolved_config.yaml", "w") as f:
                yaml.safe_dump(resolved, f, sort_keys=False)

            # Minimal manifest.json
            import json, platform, datetime as _dt
            manifest = {
                "run_id": save_dir.name,
                "timestamp": _dt.datetime.now().isoformat(),
                "surrogate": "rf",
                "policy": "sac",
                "shield": "shielded" if config.get("use_safety_shield", True) else "unshielded",
                "seed": args.seed,
                "paths": {"run_dir": str(save_dir.resolve())},
                "versions": {
                    "python": platform.python_version(),
                },
            }
            with open(save_dir / "manifest.json", "w") as f:
                import json as _json
                _json.dump(manifest, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write config snapshots/manifest: {e}")

    def _find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
        """Return latest checkpoint path based on step number in filename."""
        if not ckpt_dir.exists():
            return None
        # SB3 CheckpointCallback names: <prefix>_<steps>_steps.zip
        cands = list(ckpt_dir.glob("sac_thermal_*_steps.zip"))
        if not cands:
            return None
        def step_num(p: Path) -> int:
            m = re.search(r"_(\d+)_steps\.zip$", p.name)
            return int(m.group(1)) if m else -1
        return sorted(cands, key=step_num)[-1]

    def _find_replay_buffer_for_checkpoint(ckpt_path: Path) -> Path | None:
        """Find matching replay buffer file saved by CheckpointCallback."""
        # Expected: <prefix>_replay_buffer_<steps>_steps.pkl
        m = re.search(r"_(\d+)_steps\.zip$", ckpt_path.name)
        if not m:
            return None
        steps = m.group(1)
        rb = ckpt_path.parent / f"sac_thermal_replay_buffer_{steps}_steps.pkl"
        return rb if rb.exists() else None

    class VecNormalizeSaveCallback(BaseCallback):
        """Periodically save VecNormalize statistics."""

        def __init__(self, env, save_path: Path, save_freq: int):
            super().__init__()
            self._env = env
            self._save_path = save_path
            self._save_freq = int(save_freq)

        def _on_step(self) -> bool:
            if self._save_freq <= 0:
                return True
            if (self.num_timesteps % self._save_freq) == 0:
                if isinstance(self._env, VecNormalize):
                    self._env.save(str(self._save_path))
            return True
    
    # Environment configuration
    env_config = config.get("environment", {})
    safety_config = config.get("safety", {})
    
    # Create environment factory
    def make_env(workload_profile="steady", with_safety=True):
        """Create environment with optional safety wrapper."""
        env = make_thermal_env(
            rf_model_path=rf_model_path,
            config=env_config,
            workload_profile=workload_profile
        )
        
        # Wrap with Monitor for episode statistics
        env = Monitor(env)
        
        # Add safety wrapper if requested
        if with_safety:
            env = SafetyWrapper(env, safety_config)
        
        return env
    
    # Create training environment (vectorized)
    print("[INFO] Creating training environment...")
    normalize_env = config.get("normalize_env", True)
    train_env = make_vec_env(
        env_fn=lambda: make_env(
            workload_profile=config.get("initial_workload_profile", "steady"),
            with_safety=config.get("use_safety_shield", True),
        ),
        n_envs=config.get("n_envs", 1),
        normalize=normalize_env,
        norm_obs=config.get("normalize_obs", True),
        norm_reward=config.get("normalize_reward", True),
        gamma=config.get("gamma", 0.99),
        vec_normalize_load_path=(vec_normalize_path if args.resume else None),
    )
    
    # Create evaluation environment (separate, no normalization)
    print("[INFO] Creating evaluation environment...")
    eval_env_base = DummyVecEnv([
        lambda: make_env(
            workload_profile="stress",  # Evaluate on hardest scenario
            with_safety=config.get("use_safety_shield", True),
        )
    ])
    if normalize_env and isinstance(train_env, VecNormalize):
        eval_env = VecNormalize(eval_env_base, training=False, norm_obs=True, norm_reward=False)
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms
    else:
        eval_env = eval_env_base
    
    if not args.eval_only:
        # Create or resume SAC agent
        print("[INFO] Creating SAC agent...")
        agent = None
        latest_ckpt = _find_latest_checkpoint(checkpoints_dir) if args.resume else None
        if latest_ckpt is not None:
            from stable_baselines3 import SAC
            print(f"[INFO] Resuming from checkpoint: {latest_ckpt}")
            agent = SAC.load(str(latest_ckpt), env=train_env, device=args.device)

            rb_path = _find_replay_buffer_for_checkpoint(latest_ckpt)
            if rb_path is not None:
                print(f"[INFO] Loading replay buffer: {rb_path}")
                agent.load_replay_buffer(str(rb_path))
        else:
            agent = create_sac_agent(
                env=train_env,
                config=config.get("sac", {}),
                device=args.device,
                seed=args.seed,
            )
        
        print(f"[INFO] Agent created with policy: {agent.policy}")
        print(f"[INFO] Total parameters: {sum(p.numel() for p in agent.policy.parameters())}")
        print()
        
        # Training configuration
        total_timesteps = config.get("total_timesteps", 1000000)
        eval_freq = config.get("eval_freq", 10000)
        n_eval_episodes = config.get("n_eval_episodes", 10)
        
        # Curriculum learning configuration
        curriculum_config = None
        if config.get("use_curriculum", True):
            curriculum_config = {
                "phase_thresholds": config.get("curriculum_phases", {
                    "steady": 0,
                    "moderate": 250000,
                    "bursty": 500000,
                    "stress": 750000
                })
            }
        
        print(f"[INFO] Training for {total_timesteps} timesteps")
        print(f"[INFO] Curriculum learning: {config.get('use_curriculum', True)}")
        print(f"[INFO] Safety shield: {config.get('use_safety_shield', True)}")
        print()
        
        # Train agent
        print("[INFO] Starting training...")
        checkpoint_freq = int(config.get("save_freq", 50000))
        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoints_dir),
            name_prefix="sac_thermal",
            save_replay_buffer=True,
            save_vecnormalize=False,
        )
        vecnorm_cb = VecNormalizeSaveCallback(train_env, vec_normalize_path, checkpoint_freq)

        trained_agent = train_sac_agent(
            agent=agent,
            total_timesteps=total_timesteps,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            save_path=save_dir,
            curriculum_config=curriculum_config,
            verbose=1,
            extra_callbacks=[checkpoint_cb, vecnorm_cb],
        )
        
        # Save final model
        final_model_path = save_dir / "final_model"
        trained_agent.save(final_model_path)
        print(f"\n[INFO] Final model saved to: {final_model_path}")
        
        # Save normalized environment statistics
        if normalize_env and isinstance(train_env, VecNormalize):
            train_env.save(str(vec_normalize_path))
            print(f"[INFO] VecNormalize stats saved to: {vec_normalize_path}")
    
    else:
        # Evaluation only mode
        print("[INFO] Evaluation mode - loading trained model...")
        from rl.agents.sac_agent import load_trained_agent
        
        model_path = save_dir / "best_model" / "best_model.zip"
        if not model_path.exists():
            model_path = save_dir / "final_model.zip"
        
        agent = load_trained_agent(model_path, device=args.device)
        print(f"[INFO] Loaded model from: {model_path}")
        
        # Read eval settings from config (define locally for eval-only mode)
        n_eval_episodes = config.get("n_eval_episodes", 10)

        # Run evaluation episodes
        print(f"\n[INFO] Running {n_eval_episodes} evaluation episodes...")
        eval_rewards = []
        eval_throttles = []
        
        for ep in range(n_eval_episodes):
            obs, info = eval_env.reset()
            done = False
            ep_reward = 0.0
            
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated
            
            # Try to retrieve episode metrics robustly through wrapper chain
            metrics = {}
            try:
                getter = getattr(eval_env, "get_episode_metrics", None)
                if callable(getter):
                    metrics = getter()
                else:
                    # Walk possible attributes to reach base envs
                    env_ref = getattr(eval_env, "venv", None) or getattr(eval_env, "envs", None)
                    if env_ref is not None:
                        base_env = env_ref[0] if isinstance(env_ref, list) else env_ref
                        unwrap = getattr(base_env, "env", base_env)
                        getter2 = getattr(unwrap, "get_episode_metrics", None)
                        if callable(getter2):
                            metrics = getter2()
            except Exception:
                metrics = {}
            eval_rewards.append(ep_reward)
            eval_throttles.append(metrics.get("throttle_events", 0))
            
            print(f"  Episode {ep+1}: Reward={ep_reward:.2f}, Throttles={metrics['throttle_events']}")
        
        print(f"\n[EVAL] Mean Reward: {sum(eval_rewards)/len(eval_rewards):.2f}")
        print(f"[EVAL] Mean Throttles: {sum(eval_throttles)/len(eval_throttles):.2f}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
