"""
Safety Shield Evaluation: Shielded vs Unshielded RL.

Compares RL policy performance with and without safety shield to demonstrate:
1. Safety improvements (fewer violations)
2. Performance trade-offs (reward impact)
3. Intervention frequency and types
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl.environments.thermal_unified import ThermalControlEnv
from src.rl.surrogates.factory import create_surrogate
from src.rl.safety.shield import SafetyWrapper


def rollout_unshielded(
    model,
    vec_env,
    scenario: str,
    seed: int,
    T: int = 300
) -> Dict[str, Any]:
    """Rollout WITHOUT safety shield."""
    vec_env.seed(seed)
    obs = vec_env.reset()
    base_env = vec_env.venv.envs[0]
    
    temps = []
    fans = []
    rewards = []
    violations = {
        "warning": 0,  # >80°C
        "critical": 0,  # >85°C (realistic throttling threshold)
        "emergency": 0,  # >88°C (emergency threshold - kept for safety shield)
    }
    
    # Check initial temperature BEFORE any action
    initial_temp = base_env.state[0]
    temps.append(initial_temp)
    if initial_temp > 80.0:
        violations["warning"] += 1
    if initial_temp > 85.0:
        violations["critical"] += 1
    if initial_temp > 88.0:
        violations["emergency"] += 1
    
    for t in range(T):
        # RL decides action (no safety filter!)
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, info = vec_env.step(action)
        
        # Track metrics AFTER step
        current_temp = base_env.state[0]
        temps.append(current_temp)
        fans.append(float(action[0][0]))
        rewards.append(reward[0])
        
        # Track violations
        if current_temp > 80.0:
            violations["warning"] += 1
        if current_temp > 85.0:
            violations["critical"] += 1
        if current_temp > 88.0:
            violations["emergency"] += 1
        
        if done[0]:
            break
    
    temps = np.array(temps)
    fans = np.array(fans)
    rewards = np.array(rewards)
    
    return {
        "scenario": scenario,
        "seed": seed,
        "shield": "none",
        "initial_temp": float(initial_temp),
        "cumulative_reward": float(np.sum(rewards)),
        "mean_temp": float(np.mean(temps)),
        "max_temp": float(np.max(temps)),
        "mean_fan": float(np.mean(fans)),
        "std_fan": float(np.std(fans)),
        "warning_violations": violations["warning"],
        "critical_violations": violations["critical"],
        "emergency_violations": violations["emergency"],
        "safety_interventions": 0,
        "intervention_rate": 0.0,
    }


def rollout_shielded(
    model,
    vec_env,
    surrogate,
    env_cfg: Dict,
    scenario: str,
    seed: int,
    T: int = 300
) -> Dict[str, Any]:
    """Rollout WITH safety shield."""
    # Create shielded environment
    base_env = ThermalControlEnv(surrogate=surrogate, config=env_cfg)
    shielded_env = SafetyWrapper(base_env)
    
    # Wrap in VecEnv with same normalization as training
    vec_shielded = DummyVecEnv([lambda: shielded_env])
    vec_shielded = VecNormalize(vec_shielded, training=False, norm_reward=False)
    vec_shielded.obs_rms = vec_env.obs_rms
    vec_shielded.ret_rms = vec_env.ret_rms
    
    vec_shielded.seed(seed)
    obs = vec_shielded.reset()
    
    temps = []
    fans = []
    rewards = []
    violations = {
        "warning": 0,
        "critical": 0,
        "emergency": 0,
    }
    safety_interventions = 0
    
    # Check initial temperature BEFORE any action
    initial_temp = shielded_env.state[0]
    temps.append(initial_temp)
    if initial_temp > 80.0:
        violations["warning"] += 1
    if initial_temp > 85.0:
        violations["critical"] += 1
    if initial_temp > 88.0:
        violations["emergency"] += 1
    
    for t in range(T):
        # RL decides action
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment (safety shield filters action automatically)
        obs, reward, done, info = vec_shielded.step(action)
        
        # Track metrics AFTER step
        current_temp = shielded_env.state[0]
        temps.append(current_temp)
        fans.append(float(shielded_env.state[3]))  # Actual applied fan
        rewards.append(reward[0])
        
        # Track violations
        if current_temp > 80.0:
            violations["warning"] += 1
        if current_temp > 85.0:
            violations["critical"] += 1
        if current_temp > 88.0:
            violations["emergency"] += 1
        
        # Track safety interventions
        if "safety" in info[0] and not info[0]["safety"]["is_safe"]:
            safety_interventions += 1
        
        if done[0]:
            break
    
    temps = np.array(temps)
    fans = np.array(fans)
    rewards = np.array(rewards)
    
    # Get safety stats
    safety_stats = shielded_env.safety_shield.get_stats()
    
    vec_shielded.close()
    
    return {
        "scenario": scenario,
        "seed": seed,
        "shield": "active",
        "initial_temp": float(initial_temp),
        "cumulative_reward": float(np.sum(rewards)),
        "mean_temp": float(np.mean(temps)),
        "max_temp": float(np.max(temps)),
        "mean_fan": float(np.mean(fans)),
        "std_fan": float(np.std(fans)),
        "warning_violations": violations["warning"],
        "critical_violations": violations["critical"],
        "emergency_violations": violations["emergency"],
        "safety_interventions": safety_interventions,
        "intervention_rate": safety_stats.get("intervention_rate", 0.0),
        "emergency_overrides": safety_stats.get("emergency_overrides", 0),
        "min_cooling_enforced": safety_stats.get("min_cooling_enforced", 0),
        "rate_limited": safety_stats.get("rate_limited_actions", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Safety shield evaluation")
    
    parser.add_argument("--controller", type=str, required=True,
                        choices=["rl_rc", "rl_rcnn"],
                        help="RL controller to evaluate")
    parser.add_argument("--scenarios", nargs="+", default=["baseline", "combined_extreme", "thermal_high_start"],
                        help="Scenarios to test")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per scenario")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Safety Shield Evaluation: {args.controller}")
    print(f"{'='*80}\n")
    
    # Determine surrogate type
    if "rcnn" in args.controller:
        surrogate_type = "rcnn"
        surrogate_cfg = {"bundle_path": "models/rc_nn_hybrid.pkl"}
        model_path = "runs/rl/sac_rcnn_hybrid/sac_final.zip"
        vecnorm_path = "runs/rl/sac_rcnn_hybrid/vecnormalize.pkl"
    else:
        surrogate_type = "rc"
        surrogate_cfg = {
            "thermal_capacity": 100.0,
            "heat_transfer_coeff": 0.05,
            "cooling_effectiveness": -0.03,
            "power_to_heat": 0.01,
            "dt": 1.0,
        }
        model_path = "runs/rl/sac_rc_baseline/sac_final.zip"
        vecnorm_path = "runs/rl/sac_rc_baseline/vecnormalize.pkl"
    
    # Load model
    print(f"Loading model: {model_path}")
    model = SAC.load(model_path)
    
    # Define scenario-specific configurations
    scenario_configs = {
        "baseline": {
            "initial_temp_range": (60.0, 80.0),
            "ambient_range": (20.0, 30.0),
            "power_range": (100.0, 300.0),
        },
        "combined_extreme": {
            "initial_temp_range": (60.0, 80.0),
            "ambient_range": (20.0, 30.0),
            "power_range": (100.0, 300.0),
        },
        "thermal_high_start": {
            "initial_temp_range": (80.0, 85.0),  # High initial temp!
            "ambient_range": (25.0, 30.0),
            "power_range": (150.0, 300.0),
        },
    }
    
    all_results = []
    
    for scenario in args.scenarios:
        print(f"\nScenario: {scenario}")
        print(f"{'-'*40}")
        
        # Get scenario-specific config
        scenario_cfg = scenario_configs.get(scenario, scenario_configs["baseline"])
        
        # Create environment with scenario-specific config
        env_cfg = {
            "max_steps": 300,
            "temp_warning": 80.0,
            "temp_critical": 85.0,  # Realistic GPU throttling threshold
            "temp_target": 65.0,
            "initial_temp_range": scenario_cfg["initial_temp_range"],
            "ambient_range": scenario_cfg["ambient_range"],
            "power_range": scenario_cfg["power_range"],
            "reward_weights": {"thermal": 10.0, "energy": 0.1, "oscillation": 1.0, "headroom": 2.0},
        }
        
        surrogate = create_surrogate(surrogate_type, surrogate_cfg)
        env = ThermalControlEnv(surrogate=surrogate, config=env_cfg)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        
        for ep in range(args.episodes):
            seed = 2000 + ep
            
            # Test unshielded
            print(f"  Episode {ep+1}/{args.episodes} (unshielded)...", end=" ")
            metrics_unshielded = rollout_unshielded(model, vec_env, scenario, seed)
            metrics_unshielded["episode"] = ep + 1
            all_results.append(metrics_unshielded)
            print(f"InitTemp={metrics_unshielded['initial_temp']:.1f}°C, "
                  f"Reward={metrics_unshielded['cumulative_reward']:.1f}, "
                  f"MaxTemp={metrics_unshielded['max_temp']:.1f}°C, "
                  f"Violations={metrics_unshielded['critical_violations']}")
            
            # Test shielded
            print(f"  Episode {ep+1}/{args.episodes} (shielded)...", end=" ")
            metrics_shielded = rollout_shielded(model, vec_env, surrogate, env_cfg, scenario, seed)
            metrics_shielded["episode"] = ep + 1
            all_results.append(metrics_shielded)
            print(f"InitTemp={metrics_shielded['initial_temp']:.1f}°C, "
                  f"Reward={metrics_shielded['cumulative_reward']:.1f}, "
                  f"MaxTemp={metrics_shielded['max_temp']:.1f}°C, "
                  f"Violations={metrics_shielded['critical_violations']}, "
                  f"Interventions={metrics_shielded['safety_interventions']}")
        
        # Close environment after scenario
        vec_env.close()
    
    # Save results
    df = pd.DataFrame(all_results)
    results_path = output_dir / "safety_shield_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path}")
    
    # Create summary
    summary = df.groupby(["scenario", "shield"]).agg({
        "cumulative_reward": ["mean", "std"],
        "max_temp": ["mean", "max"],
        "mean_fan": "mean",
        "warning_violations": "sum",
        "critical_violations": "sum",
        "emergency_violations": "sum",
        "safety_interventions": "sum",
        "intervention_rate": "mean",
    }).round(2)
    
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path)
    print(f"✅ Summary saved to: {summary_path}")
    
    print(f"\n{'='*80}")
    print("✅ Safety shield evaluation complete!")
    print(f"📁 Results directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
