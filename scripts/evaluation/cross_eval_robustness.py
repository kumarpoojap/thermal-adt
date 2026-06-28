"""
Cross-Evaluation Robustness Study

Tests RL agents trained on one surrogate by evaluating them on different surrogates.
This demonstrates generalization capability and surrogate quality impact.

Example:
    Train RL on RC surrogate, evaluate on RC+NN surrogate.
    Measure performance improvement/degradation.

Usage:
    python scripts/evaluation/cross_eval_robustness.py \
        --rl-model runs/rl/sac_rc_baseline/sac_final.zip \
        --rl-vecnorm runs/rl/sac_rc_baseline/vecnormalize.pkl \
        --train-surrogate rc \
        --test-surrogate rcnn \
        --rcnn-bundle models/rc_nn_hybrid.pkl \
        --scenarios baseline thermal_high_start combined_extreme \
        --episodes 10 \
        --output-dir results/cross_eval/rl_rc_on_rcnn
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from collections import deque

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl.environments.thermal_unified import ThermalControlEnv
from src.rl.surrogates.factory import create_surrogate
from src.control.mpc_controller import MPCController


def make_env(surrogate_type: str, surrogate_cfg: dict, env_cfg: dict) -> ThermalControlEnv:
    """Create thermal control environment with specified surrogate."""
    surrogate = create_surrogate(surrogate_type, surrogate_cfg)
    env = ThermalControlEnv(surrogate=surrogate, config=env_cfg)
    return env


def load_rl_agent(model_path: str, vecnorm_path: str, test_env):
    """
    Load RL agent and wrap test environment with VecNormalize.
    
    Args:
        model_path: Path to saved SAC model
        vecnorm_path: Path to saved VecNormalize stats
        test_env: Environment to wrap (already created with test surrogate)
    
    Returns:
        model: Loaded SAC model
        vec_env: Wrapped environment with normalization
    """
    # Load model
    model = SAC.load(model_path)
    
    # Wrap environment
    vec_env = DummyVecEnv([lambda: test_env])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False  # Disable training mode
    vec_env.norm_reward = False  # Don't normalize rewards during eval
    
    return model, vec_env


def get_scenario_config(scenario: str, base_cfg: dict) -> dict:
    """Get scenario-specific configuration."""
    scenarios = {
        "baseline": {
            "initial_temp_range": (60.0, 80.0),
            "ambient_range": (20.0, 30.0),
            "power_range": (100.0, 300.0),
        },
        "thermal_high_start": {
            "initial_temp_range": (80.0, 85.0),  # Hot start
            "ambient_range": (25.0, 30.0),
            "power_range": (150.0, 300.0),
        },
        "combined_extreme": {
            "initial_temp_range": (75.0, 85.0),
            "ambient_range": (28.0, 35.0),  # Hot ambient
            "power_range": (200.0, 350.0),  # High power
        },
        "low_workload": {
            "initial_temp_range": (60.0, 70.0),
            "ambient_range": (20.0, 25.0),
            "power_range": (50.0, 150.0),  # Low power
        },
        "high_workload": {
            "initial_temp_range": (70.0, 80.0),
            "ambient_range": (25.0, 30.0),
            "power_range": (250.0, 350.0),  # High power
        },
    }
    
    cfg = base_cfg.copy()
    if scenario in scenarios:
        cfg.update(scenarios[scenario])
    return cfg


def rollout_episode(
    model,
    vec_env,
    scenario: str,
    seed: int,
    T: int = 300
) -> Dict[str, Any]:
    """
    Run single episode with RL agent.
    
    Args:
        model: SAC model
        vec_env: VecNormalize wrapped environment
        scenario: Scenario name
        seed: Random seed
        T: Episode length
    
    Returns:
        Dictionary with episode metrics
    """
    # Reset environment
    obs = vec_env.reset()
    
    # Storage
    temps = []
    fans = []
    rewards = []
    actions_raw = []
    
    for t in range(T):
        # Agent acts
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, info = vec_env.step(action)
        
        # Extract metrics (from info dict)
        if isinstance(info, list):
            info = info[0]
        
        # Extract action (fan speed) - handle VecEnv format
        if isinstance(action, np.ndarray):
            if action.ndim == 2:  # Shape: (1, 1) from VecEnv
                fan_action = float(action[0][0])
            elif action.ndim == 1:  # Shape: (1,)
                fan_action = float(action[0])
            else:
                fan_action = float(action)
        else:
            fan_action = float(action)
        
        # Clip to valid range (same as environment does)
        fan_action = np.clip(fan_action, 20.0, 100.0)
        
        temps.append(info.get("temp", 0.0))
        fans.append(fan_action)
        rewards.append(reward[0] if isinstance(reward, np.ndarray) else reward)
        actions_raw.append(fan_action)
        
        if done:
            break
    
    # Compute metrics
    temps = np.array(temps)
    fans = np.array(fans)
    rewards = np.array(rewards)
    
    # Violation metrics
    warning_entries = np.sum((temps[:-1] <= 80.0) & (temps[1:] > 80.0))
    critical_entries = np.sum((temps[:-1] <= 90.0) & (temps[1:] > 90.0))
    
    metrics = {
        "scenario": scenario,
        "seed": seed,
        "cumulative_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "mean_temp": float(np.mean(temps)),
        "max_temp": float(np.max(temps)),
        "mean_fan": float(np.mean(fans)),
        "warning_entries": int(warning_entries),
        "critical_entries": int(critical_entries),
        "episode_length": len(temps),
    }
    
    return metrics


def cross_evaluate(
    rl_model_path: str,
    rl_vecnorm_path: str,
    train_surrogate: str,
    test_surrogate: str,
    test_surrogate_cfg: dict,
    env_cfg: dict,
    scenarios: List[str],
    episodes: int = 10
) -> pd.DataFrame:
    """
    Cross-evaluate RL agent trained on one surrogate using a different test surrogate.
    
    Args:
        rl_model_path: Path to trained SAC model
        rl_vecnorm_path: Path to VecNormalize stats
        train_surrogate: Surrogate the agent was trained on
        test_surrogate: Surrogate to use for evaluation
        test_surrogate_cfg: Configuration for test surrogate
        env_cfg: Environment configuration
        scenarios: List of scenarios to test
        episodes: Episodes per scenario
    
    Returns:
        DataFrame with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Cross-Evaluation: RL trained on {train_surrogate.upper()}, testing on {test_surrogate.upper()}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario}")
        print(f"{'-'*40}")
        
        # Get scenario-specific config
        scenario_cfg = get_scenario_config(scenario, env_cfg)
        
        # Create test environment with test surrogate
        test_env = make_env(test_surrogate, test_surrogate_cfg, scenario_cfg)
        
        # Load RL agent and wrap environment
        model, vec_env = load_rl_agent(rl_model_path, rl_vecnorm_path, test_env)
        
        # Run episodes
        for ep in range(episodes):
            seed = 1000 + ep
            metrics = rollout_episode(model, vec_env, scenario, seed)
            all_results.append(metrics)
            
            print(f"  Episode {ep+1}/{episodes}: "
                  f"Reward={metrics['cumulative_reward']:.1f}, "
                  f"Fan={metrics['mean_fan']:.1f}%, "
                  f"Violations={metrics['warning_entries']}/{metrics['critical_entries']}")
        
        # Close environment
        vec_env.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Add metadata
    df["train_surrogate"] = train_surrogate
    df["test_surrogate"] = test_surrogate
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Cross-evaluation robustness study")
    
    # Model paths
    parser.add_argument("--rl-model", type=str, required=True,
                        help="Path to trained SAC model")
    parser.add_argument("--rl-vecnorm", type=str, required=True,
                        help="Path to VecNormalize stats")
    
    # Surrogate specification
    parser.add_argument("--train-surrogate", type=str, required=True,
                        choices=["rc", "rcnn", "rf", "xgb", "pinn"],
                        help="Surrogate the agent was trained on")
    parser.add_argument("--test-surrogate", type=str, required=True,
                        choices=["rc", "rcnn", "rf", "xgb", "pinn"],
                        help="Surrogate to use for evaluation")
    
    # Surrogate bundles (if needed)
    parser.add_argument("--rcnn-bundle", type=str, default=None,
                        help="Path to RC+NN bundle (if test-surrogate=rcnn)")
    parser.add_argument("--rf-bundle", type=str, default=None,
                        help="Path to RF bundle (if test-surrogate=rf)")
    parser.add_argument("--xgb-bundle", type=str, default=None,
                        help="Path to XGB bundle (if test-surrogate=xgb)")
    parser.add_argument("--pinn-bundle", type=str, default=None,
                        help="Path to PINN bundle (if test-surrogate=pinn)")
    
    # Evaluation settings
    parser.add_argument("--scenarios", nargs="+", 
                        default=["baseline", "thermal_high_start", "combined_extreme"],
                        help="Scenarios to evaluate")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per scenario")
    
    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test surrogate configuration
    test_surrogate_cfg = {}
    
    if args.test_surrogate == "rc":
        test_surrogate_cfg = {
            "thermal_capacity": 100.0,
            "heat_transfer_coeff": 0.05,
            "cooling_effectiveness": -0.03,
            "power_to_heat": 0.01,
            "dt": 1.0,
            "temp_min": 30.0,
            "temp_max": 95.0,
        }
    elif args.test_surrogate == "rcnn":
        if not args.rcnn_bundle:
            raise ValueError("--rcnn-bundle required when test-surrogate=rcnn")
        test_surrogate_cfg = {"bundle_path": args.rcnn_bundle}
    elif args.test_surrogate == "rf":
        if not args.rf_bundle:
            raise ValueError("--rf-bundle required when test-surrogate=rf")
        test_surrogate_cfg = {"bundle_path": args.rf_bundle}
    elif args.test_surrogate == "xgb":
        if not args.xgb_bundle:
            raise ValueError("--xgb-bundle required when test-surrogate=xgb")
        test_surrogate_cfg = {"bundle_path": args.xgb_bundle}
    elif args.test_surrogate == "pinn":
        if not args.pinn_bundle:
            raise ValueError("--pinn-bundle required when test-surrogate=pinn")
        test_surrogate_cfg = {"bundle_path": args.pinn_bundle}
    
    # Environment configuration
    env_cfg = {
        "max_steps": 300,
        "temp_warning": 80.0,
        "temp_critical": 90.0,
        "temp_target": 65.0,
        "initial_temp_range": (60.0, 80.0),
        "ambient_range": (20.0, 30.0),
        "power_range": (100.0, 300.0),
        "reward_weights": {
            "thermal": 10.0,
            "energy": 0.1,
            "oscillation": 1.0,
            "headroom": 2.0
        },
    }
    
    # Run cross-evaluation
    results_df = cross_evaluate(
        rl_model_path=args.rl_model,
        rl_vecnorm_path=args.rl_vecnorm,
        train_surrogate=args.train_surrogate,
        test_surrogate=args.test_surrogate,
        test_surrogate_cfg=test_surrogate_cfg,
        env_cfg=env_cfg,
        scenarios=args.scenarios,
        episodes=args.episodes
    )
    
    # Save results
    results_csv = output_dir / "cross_eval_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n✅ Results saved to: {results_csv}")
    
    # Compute summary statistics
    summary = results_df.groupby("scenario").agg({
        "cumulative_reward": ["mean", "std"],
        "mean_fan": ["mean", "std"],
        "warning_entries": "sum",
        "critical_entries": "sum",
    }).round(2)
    
    summary_csv = output_dir / "summary_by_scenario.csv"
    summary.to_csv(summary_csv)
    print(f"✅ Summary saved to: {summary_csv}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(summary)
    
    # Save metadata
    metadata = {
        "train_surrogate": args.train_surrogate,
        "test_surrogate": args.test_surrogate,
        "rl_model": args.rl_model,
        "scenarios": args.scenarios,
        "episodes_per_scenario": args.episodes,
        "total_episodes": len(results_df),
    }
    
    metadata_json = output_dir / "metadata.json"
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Cross-evaluation complete!")
    print(f"📁 Results directory: {output_dir}")


if __name__ == "__main__":
    main()
