"""
Soft Actor-Critic (SAC) agent for thermal control.

Uses Stable-Baselines3 implementation with custom callbacks
for monitoring and curriculum learning.
"""

from pathlib import Path
from typing import Dict, Optional, Callable
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym


class ThermalControlCallback(BaseCallback):
    """
    Custom callback for monitoring thermal control training.
    
    Tracks:
    - Throttle events
    - Energy usage
    - Temperature statistics
    - Reward components
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_throttles = []
        self.episode_energies = []
        self.episode_temps = []
    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Check if episode ended
        if self.locals.get("dones", [False])[0]:
            # Get episode info
            info = self.locals.get("infos", [{}])[0]
            
            if "episode" in info:
                ep_info = info["episode"]
                self.episode_rewards.append(ep_info["r"])
            
            # Get environment metrics if available
            if hasattr(self.training_env.envs[0], "get_episode_metrics"):
                metrics = self.training_env.envs[0].get_episode_metrics()
                self.episode_throttles.append(metrics.get("throttle_events", 0))
                self.episode_energies.append(metrics.get("total_energy", 0.0))
                self.episode_temps.append(metrics.get("max_temp", 0.0))
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            mean_throttles = np.mean(self.episode_throttles[-100:])
            mean_energy = np.mean(self.episode_energies[-100:])
            
            self.logger.record("rollout/ep_rew_mean", mean_reward)
            self.logger.record("thermal/throttle_events", mean_throttles)
            self.logger.record("thermal/energy_usage", mean_energy)
            
            if len(self.episode_temps) > 0:
                self.logger.record("thermal/max_temp", np.mean(self.episode_temps[-100:]))


class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning.
    
    Progressively increases workload difficulty:
    - Phase 1 (0-250k steps): Steady workloads
    - Phase 2 (250k-500k steps): Moderate bursts
    - Phase 3 (500k-750k steps): Bursty workloads
    - Phase 4 (750k+ steps): High stress
    """
    
    def __init__(
        self,
        phase_thresholds: Dict[str, int],
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.phase_thresholds = phase_thresholds
        self.current_phase = "steady"
    
    def _on_step(self) -> bool:
        """Check if curriculum phase should advance."""
        total_steps = self.num_timesteps
        
        # Determine current phase
        new_phase = "steady"
        for phase, threshold in sorted(self.phase_thresholds.items(), key=lambda x: x[1]):
            if total_steps >= threshold:
                new_phase = phase
        
        # Update environment if phase changed
        if new_phase != self.current_phase:
            if self.verbose > 0:
                print(f"\n[Curriculum] Advancing to phase: {new_phase} at step {total_steps}")
            
            self.current_phase = new_phase
            
            # Update workload profile in environment
            if hasattr(self.training_env.envs[0], "set_workload_profile"):
                self.training_env.envs[0].set_workload_profile(new_phase)
            
            self.logger.record("curriculum/phase", self.current_phase)
        
        return True


def create_sac_agent(
    env: gym.Env,
    config: Dict,
    device: str = "auto",
    seed: Optional[int] = None
) -> SAC:
    """
    Create SAC agent with specified configuration.
    
    Args:
        env: Training environment
        config: SAC hyperparameters
        device: Device to use ("auto", "cpu", or "cuda")
        seed: Random seed
    
    Returns:
        SAC agent
    """
    # Default SAC hyperparameters
    default_config = {
        "learning_rate": 3e-4,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "target_entropy": "auto",
        "use_sde": False,
        "policy_kwargs": {
            "net_arch": [256, 256],
            "activation_fn": torch.nn.ReLU
        }
    }
    
    # Merge with provided config
    agent_config = {**default_config, **config}
    
    # Create agent
    agent = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=agent_config["learning_rate"],
        buffer_size=agent_config["buffer_size"],
        learning_starts=agent_config["learning_starts"],
        batch_size=agent_config["batch_size"],
        tau=agent_config["tau"],
        gamma=agent_config["gamma"],
        train_freq=agent_config["train_freq"],
        gradient_steps=agent_config["gradient_steps"],
        ent_coef=agent_config["ent_coef"],
        target_update_interval=agent_config["target_update_interval"],
        target_entropy=agent_config["target_entropy"],
        use_sde=agent_config["use_sde"],
        policy_kwargs=agent_config["policy_kwargs"],
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log=agent_config.get("tensorboard_log", "./rl_logs/tensorboard/")
    )
    
    return agent


def make_vec_env(
    env_fn: Callable,
    n_envs: int = 1,
    normalize: bool = True,
    norm_obs: bool = True,
    norm_reward: bool = True,
    clip_obs: float = 10.0,
    clip_reward: float = 10.0,
    gamma: float = 0.99,
    vec_normalize_load_path: Optional[Path] = None
) -> gym.Env:
    """
    Create vectorized environment with optional normalization.
    
    Args:
        env_fn: Function that creates environment
        n_envs: Number of parallel environments
        normalize: Whether to normalize observations and rewards
        norm_obs: Normalize observations
        norm_reward: Normalize rewards
        clip_obs: Observation clipping value
        clip_reward: Reward clipping value
        gamma: Discount factor for reward normalization
    
    Returns:
        Vectorized environment
    """
    # Create vectorized environment
    venv = DummyVecEnv([env_fn for _ in range(n_envs)])

    # Add (or load) normalization wrapper
    if not normalize:
        return venv

    if vec_normalize_load_path is not None and vec_normalize_load_path.exists():
        env = VecNormalize.load(str(vec_normalize_load_path), venv)
        # Ensure loaded env uses current clipping/gamma settings
        env.norm_obs = norm_obs
        env.norm_reward = norm_reward
        env.clip_obs = clip_obs
        env.clip_reward = clip_reward
        env.gamma = gamma
        return env

    env = VecNormalize(
        venv,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=clip_obs,
        clip_reward=clip_reward,
        gamma=gamma,
    )

    return env


def train_sac_agent(
    agent: SAC,
    total_timesteps: int,
    eval_env: Optional[gym.Env] = None,
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    save_path: Optional[Path] = None,
    curriculum_config: Optional[Dict] = None,
    verbose: int = 1,
    extra_callbacks: Optional[list[BaseCallback]] = None
) -> SAC:
    """
    Train SAC agent with callbacks and evaluation.
    
    Args:
        agent: SAC agent to train
        total_timesteps: Total training steps
        eval_env: Evaluation environment
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        save_path: Path to save best model
        curriculum_config: Curriculum learning configuration
        verbose: Verbosity level
    
    Returns:
        Trained agent
    """
    callbacks: list[BaseCallback] = []
    
    # Add thermal control monitoring callback
    callbacks.append(ThermalControlCallback(verbose=verbose))
    
    # Add curriculum learning callback if configured
    if curriculum_config is not None:
        callbacks.append(CurriculumCallback(
            phase_thresholds=curriculum_config.get("phase_thresholds", {
                "steady": 0,
                "moderate": 250000,
                "bursty": 500000,
                "stress": 750000
            }),
            verbose=verbose
        ))
    
    # Add evaluation callback if eval environment provided
    if eval_env is not None and save_path is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(save_path / "best_model"),
            log_path=str(save_path / "eval_logs"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=verbose
        )
        callbacks.append(eval_callback)

    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    
    # Train agent
    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10,
        progress_bar=True
    )
    
    return agent


def load_trained_agent(
    model_path: Path,
    env: Optional[gym.Env] = None,
    device: str = "auto"
) -> SAC:
    """
    Load trained SAC agent from checkpoint.
    
    Args:
        model_path: Path to saved model
        env: Environment (optional, for continuing training)
        device: Device to load model on
    
    Returns:
        Loaded SAC agent
    """
    agent = SAC.load(
        str(model_path),
        env=env,
        device=device
    )
    
    return agent
