"""
Sensor fault robustness evaluation.

Tests controller robustness to sensor faults:
- Stuck sensors (constant value)
- Biased sensors (systematic offset)
- Noisy sensors (random noise)
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl.environments.thermal_unified import ThermalControlEnv
from src.rl.surrogates.factory import create_surrogate
from src.control.mpc_controller import MPCController


def apply_sensor_fault(state: np.ndarray, fault_type: str, fault_params: dict) -> np.ndarray:
    """
    Apply sensor fault to state observation.
    
    Args:
        state: True state [temp, ambient, power, fan, temp_delta]
        fault_type: Type of fault ('stuck', 'bias', 'noise', 'intermittent')
        fault_params: Fault parameters (sensor, value, noise_std, etc.)
    
    Returns:
        Corrupted state observation
    """
    faulty_state = state.copy()
    sensor_idx = fault_params.get('sensor_idx', 1)  # Default: ambient (index 1)
    
    if fault_type == 'stuck':
        # Sensor stuck at constant value
        stuck_value = fault_params.get('stuck_value', 25.0)
        faulty_state[sensor_idx] = stuck_value
    
    elif fault_type == 'bias':
        # Sensor reports biased value
        bias = fault_params.get('bias', 5.0)
        faulty_state[sensor_idx] += bias
    
    elif fault_type == 'noise':
        # Sensor adds Gaussian noise
        noise_std = fault_params.get('noise_std', 2.0)
        noise = np.random.normal(0, noise_std)
        faulty_state[sensor_idx] += noise
    
    elif fault_type == 'intermittent':
        # Sensor occasionally drops out (reports previous value)
        dropout_prob = fault_params.get('dropout_prob', 0.1)
        if np.random.random() < dropout_prob:
            # Report stuck value during dropout
            faulty_state[sensor_idx] = fault_params.get('last_value', faulty_state[sensor_idx])
    
    return faulty_state


def rollout_mpc_with_fault(
    mpc, 
    env, 
    scenario: str, 
    seed: int, 
    fault_type: str,
    fault_params: dict,
    T: int = 300
) -> Dict[str, Any]:
    """MPC rollout with sensor fault."""
    mpc.reset()
    obs, info = env.reset(seed=seed)
    
    temps = []
    fans = []
    rewards = []
    fault_magnitudes = []  # Track how wrong the sensor is
    
    for t in range(T):
        # Get true state
        true_state = env.state.copy()
        
        # Apply sensor fault to observation
        faulty_obs = apply_sensor_fault(true_state, fault_type, fault_params)
        
        # Track fault magnitude
        sensor_idx = fault_params.get('sensor_idx', 1)
        fault_mag = abs(faulty_obs[sensor_idx] - true_state[sensor_idx])
        fault_magnitudes.append(fault_mag)
        
        # MPC sees faulty observation
        action, _ = mpc.compute_action(faulty_obs)
        
        # Environment steps with true state (fault only affects observation)
        obs, reward, done, truncated, info = env.step(action)
        
        temps.append(info.get("temp", obs[0]))
        fans.append(float(action[0]))
        rewards.append(reward)
        
        # Update last_value for intermittent faults
        if fault_type == 'intermittent':
            fault_params['last_value'] = true_state[sensor_idx]
        
        if done or truncated:
            break
    
    temps = np.array(temps)
    fans = np.array(fans)
    rewards = np.array(rewards)
    fault_magnitudes = np.array(fault_magnitudes)
    
    warning_entries = np.sum((temps[:-1] <= 80.0) & (temps[1:] > 80.0))
    critical_entries = np.sum((temps[:-1] <= 90.0) & (temps[1:] > 90.0))
    
    return {
        "scenario": scenario,
        "seed": seed,
        "fault_type": fault_type,
        "cumulative_reward": float(np.sum(rewards)),
        "mean_temp": float(np.mean(temps)),
        "max_temp": float(np.max(temps)),
        "mean_fan": float(np.mean(fans)),
        "mean_fault_magnitude": float(np.mean(fault_magnitudes)),
        "max_fault_magnitude": float(np.max(fault_magnitudes)),
        "warning_entries": int(warning_entries),
        "critical_entries": int(critical_entries),
    }


def rollout_rl_with_fault(
    model,
    vec_env,
    scenario: str,
    seed: int,
    fault_type: str,
    fault_params: dict,
    T: int = 300
) -> Dict[str, Any]:
    """RL rollout with sensor fault."""
    vec_env.seed(seed)
    obs = vec_env.reset()
    base_env = vec_env.venv.envs[0]
    
    temps = []
    fans = []
    rewards = []
    fault_magnitudes = []
    
    for t in range(T):
        # Get true state
        true_state = base_env.state.copy()
        
        # Apply sensor fault
        faulty_state = apply_sensor_fault(true_state, fault_type, fault_params)
        
        # Track fault magnitude
        sensor_idx = fault_params.get('sensor_idx', 1)
        fault_mag = abs(faulty_state[sensor_idx] - true_state[sensor_idx])
        fault_magnitudes.append(fault_mag)
        
        # Normalize faulty observation
        faulty_obs = vec_env.normalize_obs(faulty_state.reshape(1, -1))
        
        # RL sees faulty observation
        action, _ = model.predict(faulty_obs, deterministic=True)
        
        # Environment steps with true state
        obs, reward, done, info = vec_env.step(action)
        
        temps.append(base_env.state[0])
        fans.append(float(action[0][0]))
        rewards.append(reward[0])
        
        # Update last_value for intermittent faults
        if fault_type == 'intermittent':
            fault_params['last_value'] = true_state[sensor_idx]
        
        if done[0]:
            break
    
    temps = np.array(temps)
    fans = np.array(fans)
    rewards = np.array(rewards)
    fault_magnitudes = np.array(fault_magnitudes)
    
    warning_entries = np.sum((temps[:-1] <= 80.0) & (temps[1:] > 80.0))
    critical_entries = np.sum((temps[:-1] <= 90.0) & (temps[1:] > 90.0))
    
    return {
        "scenario": scenario,
        "seed": seed,
        "fault_type": fault_type,
        "cumulative_reward": float(np.sum(rewards)),
        "mean_temp": float(np.mean(temps)),
        "max_temp": float(np.max(temps)),
        "mean_fan": float(np.mean(fans)),
        "mean_fault_magnitude": float(np.mean(fault_magnitudes)),
        "max_fault_magnitude": float(np.max(fault_magnitudes)),
        "warning_entries": int(warning_entries),
        "critical_entries": int(critical_entries),
    }


def create_controller(controller_type: str, surrogate_type: str, surrogate_cfg: dict, env_cfg: dict):
    """Create controller (MPC or RL)."""
    if controller_type.startswith("mpc"):
        surrogate = create_surrogate(surrogate_type, surrogate_cfg)
        mpc = MPCController(
            surrogate=surrogate,
            horizon=10,
            temp_target=env_cfg["temp_target"],
            weight_temp=100.0,
            weight_effort=0.01,
            weight_rate=0.5
        )
        return mpc, None
    
    elif controller_type.startswith("rl"):
        if surrogate_type == "rc":
            model_path = "runs/rl/sac_rc_baseline/sac_final.zip"
            vecnorm_path = "runs/rl/sac_rc_baseline/vecnormalize.pkl"
        else:
            model_path = "runs/rl/sac_rcnn_hybrid/sac_final.zip"
            vecnorm_path = "runs/rl/sac_rcnn_hybrid/vecnormalize.pkl"
        
        model = SAC.load(model_path)
        surrogate = create_surrogate(surrogate_type, surrogate_cfg)
        env = ThermalControlEnv(surrogate=surrogate, config=env_cfg)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        
        return model, vec_env


def main():
    parser = argparse.ArgumentParser(description="Sensor fault robustness evaluation")
    
    parser.add_argument("--fault-type", type=str, required=True,
                        choices=["stuck", "bias", "noise", "intermittent"],
                        help="Type of sensor fault")
    parser.add_argument("--sensor", type=str, default="ambient",
                        choices=["ambient", "power", "temperature"],
                        help="Which sensor to fault")
    parser.add_argument("--stuck-value", type=float, default=25.0,
                        help="Value for stuck sensor")
    parser.add_argument("--bias", type=float, default=5.0,
                        help="Bias offset for biased sensor")
    parser.add_argument("--noise-std", type=float, default=2.0,
                        help="Noise standard deviation")
    parser.add_argument("--dropout-prob", type=float, default=0.1,
                        help="Dropout probability for intermittent faults")
    
    parser.add_argument("--controllers", nargs="+", required=True,
                        choices=["mpc_rc", "mpc_rcnn", "rl_rc", "rl_rcnn"],
                        help="Controllers to evaluate")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodes per controller")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Map sensor name to index
    sensor_map = {"temperature": 0, "ambient": 1, "power": 2}
    sensor_idx = sensor_map[args.sensor]
    
    # Create fault parameters
    fault_params = {
        "sensor_idx": sensor_idx,
        "stuck_value": args.stuck_value,
        "bias": args.bias,
        "noise_std": args.noise_std,
        "dropout_prob": args.dropout_prob,
        "last_value": None,
    }
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Sensor Fault Evaluation: {args.fault_type} on {args.sensor} sensor")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for controller_name in args.controllers:
        print(f"\nController: {controller_name}")
        print(f"{'-'*40}")
        
        # Determine surrogate type
        if "rcnn" in controller_name:
            surrogate_type = "rcnn"
            surrogate_cfg = {"bundle_path": "models/rc_nn_hybrid.pkl"}
        else:
            surrogate_type = "rc"
            surrogate_cfg = {
                "thermal_capacity": 100.0,
                "heat_transfer_coeff": 0.05,
                "cooling_effectiveness": -0.03,
                "power_to_heat": 0.01,
                "dt": 1.0,
            }
        
        env_cfg = {
            "max_steps": 300,
            "temp_warning": 80.0,
            "temp_critical": 90.0,
            "temp_target": 65.0,
            "initial_temp_range": (60.0, 80.0),
            "ambient_range": (20.0, 30.0),
            "power_range": (100.0, 300.0),
            "reward_weights": {"thermal": 10.0, "energy": 0.1, "oscillation": 1.0, "headroom": 2.0},
        }
        
        # Create controller
        controller, vec_env = create_controller(controller_name, surrogate_type, surrogate_cfg, env_cfg)
        
        # Run episodes
        for ep in range(args.episodes):
            seed = 2000 + ep
            
            if controller_name.startswith("mpc"):
                # Create environment for MPC
                surrogate = create_surrogate(surrogate_type, surrogate_cfg)
                env = ThermalControlEnv(surrogate=surrogate, config=env_cfg)
                metrics = rollout_mpc_with_fault(controller, env, "baseline", seed, args.fault_type, fault_params.copy())
            else:
                metrics = rollout_rl_with_fault(controller, vec_env, "baseline", seed, args.fault_type, fault_params.copy())
            
            metrics["controller"] = controller_name
            metrics["episode"] = ep + 1
            all_results.append(metrics)
            
            print(f"  Episode {ep+1}/{args.episodes}: Reward={metrics['cumulative_reward']:.1f}, "
                  f"Fan={metrics['mean_fan']:.1f}%, Fault={metrics['mean_fault_magnitude']:.2f}")
        
        if vec_env is not None:
            vec_env.close()
    
    # Save results
    df = pd.DataFrame(all_results)
    results_path = output_dir / "sensor_fault_results.csv"
    df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path}")
    
    # Create summary
    summary = df.groupby("controller").agg({
        "cumulative_reward": ["mean", "std"],
        "mean_temp": "mean",
        "max_temp": "max",
        "mean_fan": "mean",
        "mean_fault_magnitude": "mean",
        "warning_entries": "sum",
        "critical_entries": "sum",
    }).round(2)
    
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path)
    print(f"✅ Summary saved to: {summary_path}")
    
    print(f"\n{'='*80}")
    print("✅ Sensor fault evaluation complete!")
    print(f"📁 Results directory: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
