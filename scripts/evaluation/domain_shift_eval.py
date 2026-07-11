"""
Domain Shift Robustness Evaluation

Tests controllers under realistic deployment perturbations:
1. Ambient temperature shift (+5°C)
2. Power consumption spike (+20%)
3. Action delay (1-step lag)

Usage:
    python scripts/evaluation/domain_shift_eval.py \
        --perturbation ambient_shift \
        --shift-deg 5.0 \
        --controllers rl_rc rl_rcnn \
        --scenarios baseline thermal_high_start \
        --episodes 10 \
        --output-dir results/domain_shift/ambient_plus5
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.rl.environments.thermal_unified import ThermalControlEnv
from src.rl.surrogates.factory import create_surrogate
from src.control.mpc_controller import MPCController


def scenario_schedule(name: str, T: int):
    """
    Return ambient and power schedules for a scenario over T steps.
    
    This provides deterministic, realistic workload patterns instead of
    random constant conditions.
    
    Returns:
        amb: Array of ambient temperatures [T]
        poww: Array of power consumption [T]
    """
    amb = np.full(T, 25.0, dtype=float)
    poww = np.full(T, 200.0, dtype=float)

    if name == "baseline":
        # Constant conditions
        pass
    
    elif name == "thermal_high_start":
        # Moderate workload, high initial temp (set in rollout)
        pass
    
    elif name == "combined_extreme":
        # Hot ambient + oscillating power
        amb[:] = 32.0
        for t in range(T):
            poww[t] = 240.0 + 70.0 * math.sin(2 * math.pi * t / 30.0)
    
    elif name == "variable_workload":
        # Moderate ambient + slower power oscillation
        for t in range(T):
            poww[t] = 200.0 + 60.0 * math.sin(2 * math.pi * t / 60.0)
    
    elif name == "workload_spike":
        # Sudden power spike
        poww[:] = 180.0
        spike_idx = np.arange(50, min(70, T))
        poww[spike_idx] = 320.0
    
    elif name == "low_workload":
        poww[:] = 120.0
    
    elif name == "high_workload":
        poww[:] = 300.0
    
    elif name == "warm_ambient":
        amb[:] = 30.0
    
    elif name == "ambient_extreme":
        amb[:] = 33.0
    
    elif name == "workload_oscillation":
        for t in range(T):
            poww[t] = 220.0 + 90.0 * math.sin(2 * math.pi * t / 20.0)
    
    elif name == "recovery":
        # Gradually cooling ambient and workload
        for t in range(T):
            amb[t] = 31.0 - 0.02 * t
            poww[t] = 260.0 - 0.5 * t
    
    elif name == "sustained_limit":
        amb[:] = 31.0
        poww[:] = 280.0
    
    else:
        raise ValueError(f"Unknown scenario: {name}")
    
    return amb, poww


def get_scenario_config(scenario: str) -> dict:
    """Get scenario-specific configuration."""
    scenarios = {
        "baseline": {
            "initial_temp_range": (60.0, 80.0),
            "ambient_range": (20.0, 30.0),
            "power_range": (100.0, 300.0),
        },
        "thermal_high_start": {
            "initial_temp_range": (80.0, 85.0),
            "ambient_range": (25.0, 30.0),
            "power_range": (150.0, 300.0),
        },
        "combined_extreme": {
            "initial_temp_range": (75.0, 85.0),
            "ambient_range": (28.0, 35.0),
            "power_range": (200.0, 350.0),
        },
    }
    return scenarios.get(scenario, scenarios["baseline"])


def create_controller(controller_type: str, surrogate_type: str, surrogate_cfg: dict, env_cfg: dict):
    """Create controller (MPC or RL)."""
    if controller_type.startswith("mpc"):
        # Create MPC controller
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
        # Load RL model
        if surrogate_type == "rc":
            model_path = "runs/rl/sac_rc_baseline/sac_final.zip"
            vecnorm_path = "runs/rl/sac_rc_baseline/vecnormalize.pkl"
        else:  # rcnn
            model_path = "runs/rl/sac_rcnn_hybrid/sac_final.zip"
            vecnorm_path = "runs/rl/sac_rcnn_hybrid/vecnormalize.pkl"
        
        model = SAC.load(model_path)
        
        # Create environment
        surrogate = create_surrogate(surrogate_type, surrogate_cfg)
        env = ThermalControlEnv(surrogate=surrogate, config=env_cfg)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        
        return model, vec_env
    
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


def rollout_ambient_shift(
    controller_type: str,
    controller,
    vec_env: Optional[Any],
    scenario: str,
    seed: int,
    shift_deg: float = 5.0,
    T: int = 300
) -> Dict[str, Any]:
    """
    Rollout with ambient temperature shifted by shift_deg.
    
    Note: This is a simplified version. In practice, you'd need to modify
    the environment's ambient schedule during the episode.
    """
    # Get scenario config
    scenario_cfg = get_scenario_config(scenario)
    
    # Shift ambient range
    amb_low, amb_high = scenario_cfg["ambient_range"]
    shifted_ambient = (amb_low + shift_deg, amb_high + shift_deg)
    
    # Create new environment with shifted ambient
    env_cfg = {
        "max_steps": T,
        "temp_warning": 80.0,
        "temp_critical": 90.0,
        "temp_target": 65.0,
        "initial_temp_range": scenario_cfg["initial_temp_range"],
        "ambient_range": shifted_ambient,
        "power_range": scenario_cfg["power_range"],
        "reward_weights": {"thermal": 10.0, "energy": 0.1, "oscillation": 1.0, "headroom": 2.0},
    }
    
    # Run episode with shifted environment
    if controller_type.startswith("rl"):
        # For RL, we need to recreate the environment
        # Extract surrogate type from vec_env
        if "rc" in str(type(vec_env.venv.envs[0].surrogate)):
            surrogate_type = "rc"
            surrogate_cfg = {
                "thermal_capacity": 100.0,
                "heat_transfer_coeff": 0.05,
                "cooling_effectiveness": -0.03,
                "power_to_heat": 0.01,
                "dt": 1.0,
            }
        else:
            surrogate_type = "rcnn"
            surrogate_cfg = {"bundle_path": "models/rc_nn_hybrid.pkl"}
        
        # Create shifted environment
        surrogate = create_surrogate(surrogate_type, surrogate_cfg)
        shifted_env = ThermalControlEnv(surrogate=surrogate, config=env_cfg)
        shifted_vec_env = DummyVecEnv([lambda: shifted_env])
        
        # Load normalization stats from original
        shifted_vec_env = VecNormalize(shifted_vec_env, training=False, norm_reward=False)
        shifted_vec_env.obs_rms = vec_env.obs_rms
        shifted_vec_env.ret_rms = vec_env.ret_rms
        
        metrics = rollout_rl(controller, shifted_vec_env, scenario, seed, T)
        shifted_vec_env.close()
    else:
        # For MPC, recreate with shifted environment
        if "rc" in str(type(controller.surrogate)):
            surrogate_type = "rc"
            surrogate_cfg = {
                "thermal_capacity": 100.0,
                "heat_transfer_coeff": 0.05,
                "cooling_effectiveness": -0.03,
                "power_to_heat": 0.01,
                "dt": 1.0,
            }
        else:
            surrogate_type = "rcnn"
            surrogate_cfg = {"bundle_path": "models/rc_nn_hybrid.pkl"}
        
        surrogate = create_surrogate(surrogate_type, surrogate_cfg)
        shifted_env = ThermalControlEnv(surrogate=surrogate, config=env_cfg)
        
        metrics = rollout_mpc(controller, shifted_env, scenario, seed, T)
    
    return metrics


def rollout_power_spike(
    controller_type: str,
    controller,
    vec_env: Optional[Any],
    scenario: str,
    seed: int,
    spike_factor: float = 1.2,
    T: int = 300
) -> Dict[str, Any]:
    """Rollout with power consumption increased by spike_factor."""
    scenario_cfg = get_scenario_config(scenario)
    
    # Spike power range
    pow_low, pow_high = scenario_cfg["power_range"]
    spiked_power = (pow_low * spike_factor, pow_high * spike_factor)
    
    env_cfg = {
        "max_steps": T,
        "temp_warning": 80.0,
        "temp_critical": 90.0,
        "temp_target": 65.0,
        "initial_temp_range": scenario_cfg["initial_temp_range"],
        "ambient_range": scenario_cfg["ambient_range"],
        "power_range": spiked_power,
        "reward_weights": {"thermal": 10.0, "energy": 0.1, "oscillation": 1.0, "headroom": 2.0},
    }
    
    # Run with spiked power (similar to ambient shift)
    if controller_type.startswith("rl"):
        if "rc" in str(type(vec_env.venv.envs[0].surrogate)):
            surrogate_type = "rc"
            surrogate_cfg = {
                "thermal_capacity": 100.0,
                "heat_transfer_coeff": 0.05,
                "cooling_effectiveness": -0.03,
                "power_to_heat": 0.01,
                "dt": 1.0,
            }
        else:
            surrogate_type = "rcnn"
            surrogate_cfg = {"bundle_path": "models/rc_nn_hybrid.pkl"}
        
        surrogate = create_surrogate(surrogate_type, surrogate_cfg)
        spiked_env = ThermalControlEnv(surrogate=surrogate, config=env_cfg)
        spiked_vec_env = DummyVecEnv([lambda: spiked_env])
        spiked_vec_env = VecNormalize(spiked_vec_env, training=False, norm_reward=False)
        spiked_vec_env.obs_rms = vec_env.obs_rms
        spiked_vec_env.ret_rms = vec_env.ret_rms
        
        metrics = rollout_rl(controller, spiked_vec_env, scenario, seed, T)
        spiked_vec_env.close()
    else:
        if "rc" in str(type(controller.surrogate)):
            surrogate_type = "rc"
            surrogate_cfg = {
                "thermal_capacity": 100.0,
                "heat_transfer_coeff": 0.05,
                "cooling_effectiveness": -0.03,
                "power_to_heat": 0.01,
                "dt": 1.0,
            }
        else:
            surrogate_type = "rcnn"
            surrogate_cfg = {"bundle_path": "models/rc_nn_hybrid.pkl"}
        
        surrogate = create_surrogate(surrogate_type, surrogate_cfg)
        spiked_env = ThermalControlEnv(surrogate=surrogate, config=env_cfg)
        
        metrics = rollout_mpc(controller, spiked_env, scenario, seed, T)
    
    return metrics


def rollout_action_delay(
    controller_type: str,
    controller,
    vec_env: Optional[Any],
    scenario: str,
    seed: int,
    delay_steps: int = 1,
    T: int = 300
) -> Dict[str, Any]:
    """Rollout with action applied with delay_steps lag."""
    scenario_cfg = get_scenario_config(scenario)
    
    env_cfg = {
        "max_steps": T,
        "temp_warning": 80.0,
        "temp_critical": 90.0,
        "temp_target": 65.0,
        "initial_temp_range": scenario_cfg["initial_temp_range"],
        "ambient_range": scenario_cfg["ambient_range"],
        "power_range": scenario_cfg["power_range"],
        "reward_weights": {"thermal": 10.0, "energy": 0.1, "oscillation": 1.0, "headroom": 2.0},
    }
    
    # Create environment
    if controller_type.startswith("rl"):
        # Check for rcnn FIRST (before rc) since "rc" is substring of "rcnn"
        surrogate_type_str = str(type(vec_env.venv.envs[0].surrogate)).lower()
        if "rcnn" in surrogate_type_str:
            surrogate_type = "rcnn"
            surrogate_cfg = {"bundle_path": "models/rc_nn_hybrid.pkl"}
        elif "rc" in surrogate_type_str:
            surrogate_type = "rc"
            surrogate_cfg = {
                "thermal_capacity": 100.0,
                "heat_transfer_coeff": 0.05,
                "cooling_effectiveness": -0.03,
                "power_to_heat": 0.01,
                "dt": 1.0,
            }
        else:
            raise ValueError(f"Unknown surrogate type: {type(vec_env.venv.envs[0].surrogate)}")
        
        surrogate = create_surrogate(surrogate_type, surrogate_cfg)
        
        # Create environment factory function (avoid lambda closure bug)
        def make_delay_env():
            return ThermalControlEnv(surrogate=surrogate, config=env_cfg)
        
        delay_vec_env = DummyVecEnv([make_delay_env])
        delay_vec_env = VecNormalize(delay_vec_env, training=False, norm_reward=False)
        delay_vec_env.obs_rms = vec_env.obs_rms
        delay_vec_env.ret_rms = vec_env.ret_rms
        
        metrics = rollout_rl_with_delay(controller, delay_vec_env, scenario, seed, delay_steps, T)
        delay_vec_env.close()
    else:
        # Check for rcnn FIRST (before rc) since "rc" is substring of "rcnn"
        surrogate_type_str = str(type(controller.surrogate)).lower()
        if "rcnn" in surrogate_type_str:
            surrogate_type = "rcnn"
            surrogate_cfg = {"bundle_path": "models/rc_nn_hybrid.pkl"}
        elif "rc" in surrogate_type_str:
            surrogate_type = "rc"
            surrogate_cfg = {
                "thermal_capacity": 100.0,
                "heat_transfer_coeff": 0.05,
                "cooling_effectiveness": -0.03,
                "power_to_heat": 0.01,
                "dt": 1.0,
            }
        else:
            raise ValueError(f"Unknown surrogate type: {type(controller.surrogate)}")
        
        surrogate = create_surrogate(surrogate_type, surrogate_cfg)
        delay_env = ThermalControlEnv(surrogate=surrogate, config=env_cfg)
        
        # Create fresh MPC controller with same surrogate as environment
        fresh_mpc = MPCController(
            surrogate=surrogate,
            horizon=controller.horizon,
            temp_target=controller.temp_target,
            temp_max=controller.temp_max,
            fan_min=controller.fan_min,
            fan_max=controller.fan_max,
            max_fan_delta=controller.max_fan_delta,
            weight_temp=controller.weight_temp,
            weight_effort=controller.weight_effort,
            weight_rate=controller.weight_rate
        )
        
        metrics = rollout_mpc_with_delay(fresh_mpc, delay_env, scenario, seed, delay_steps, T)
    
    return metrics


def rollout_rl(model, vec_env, scenario: str, seed: int, T: int = 300) -> Dict[str, Any]:
    """Standard RL rollout."""
    vec_env.seed(seed)
    obs = vec_env.reset()
    
    temps = []
    fans = []
    rewards = []
    
    for t in range(T):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        if isinstance(info, list):
            info = info[0]
        
        # Extract fan from action
        if isinstance(action, np.ndarray):
            if action.ndim == 2:
                fan_action = float(action[0][0])
            elif action.ndim == 1:
                fan_action = float(action[0])
            else:
                fan_action = float(action)
        else:
            fan_action = float(action)
        
        fan_action = np.clip(fan_action, 20.0, 100.0)
        
        temps.append(info.get("temp", 0.0))
        fans.append(fan_action)
        rewards.append(reward[0] if isinstance(reward, np.ndarray) else reward)
        
        if done:
            break
    
    temps = np.array(temps)
    fans = np.array(fans)
    rewards = np.array(rewards)
    
    warning_entries = np.sum((temps[:-1] <= 80.0) & (temps[1:] > 80.0))
    critical_entries = np.sum((temps[:-1] <= 90.0) & (temps[1:] > 90.0))
    
    return {
        "scenario": scenario,
        "seed": seed,
        "cumulative_reward": float(np.sum(rewards)),
        "mean_temp": float(np.mean(temps)),
        "max_temp": float(np.max(temps)),
        "mean_fan": float(np.mean(fans)),
        "warning_entries": int(warning_entries),
        "critical_entries": int(critical_entries),
    }


def rollout_rl_with_delay(model, vec_env, scenario: str, seed: int, delay_steps: int, T: int = 300) -> Dict[str, Any]:
    """RL rollout with action delay."""
    vec_env.seed(seed)
    obs = vec_env.reset()
    
    # Get scenario schedule for dynamic ambient/power
    amb_sched, pow_sched = scenario_schedule(scenario, T)
    
    # Handle thermal_high_start: set high initial temperature
    if scenario == "thermal_high_start":
        base_env = vec_env.venv.envs[0]
        base_env.state[0] = max(base_env.state[0], 82.0)
    
    # Initialize buffer with default action (50% fan)
    action_buffer = deque([np.array([[50.0]]) for _ in range(delay_steps)], maxlen=delay_steps)
    temps = []
    fans = []
    rewards = []
    
    for t in range(T):
        # Inject scenario schedule (dynamic ambient/power)
        base_env = vec_env.venv.envs[0]
        base_env.state[1] = float(amb_sched[t])  # Ambient temperature
        base_env.state[2] = float(pow_sched[t])  # Power consumption
        
        # Controller decides action
        action_decided, _ = model.predict(obs, deterministic=True)
        
        # Get action to apply (oldest in buffer = delay_steps ago)
        action_applied = action_buffer[0]
        
        # Add new action to buffer (will push out oldest)
        action_buffer.append(action_decided)
        
        obs, reward, done, info = vec_env.step(action_applied)
        
        if isinstance(info, list):
            info = info[0]
        
        # Extract fan from applied action
        if isinstance(action_applied, np.ndarray):
            if action_applied.ndim == 2:
                fan_action = float(action_applied[0][0])
            elif action_applied.ndim == 1:
                fan_action = float(action_applied[0])
            else:
                fan_action = float(action_applied)
        else:
            fan_action = float(action_applied)
        
        fan_action = np.clip(fan_action, 20.0, 100.0)
        
        temps.append(info.get("temp", 0.0))
        fans.append(fan_action)
        rewards.append(reward[0] if isinstance(reward, np.ndarray) else reward)
        
        if done:
            break
    
    temps = np.array(temps)
    fans = np.array(fans)
    rewards = np.array(rewards)
    
    warning_entries = np.sum((temps[:-1] <= 80.0) & (temps[1:] > 80.0))
    critical_entries = np.sum((temps[:-1] <= 90.0) & (temps[1:] > 90.0))
    
    return {
        "scenario": scenario,
        "seed": seed,
        "cumulative_reward": float(np.sum(rewards)),
        "mean_temp": float(np.mean(temps)),
        "max_temp": float(np.max(temps)),
        "mean_fan": float(np.mean(fans)),
        "warning_entries": int(warning_entries),
        "critical_entries": int(critical_entries),
    }


def rollout_mpc(mpc, env, scenario: str, seed: int, T: int = 300) -> Dict[str, Any]:
    """Standard MPC rollout."""
    mpc.reset()
    obs, info = env.reset(seed=seed)
    
    temps = []
    fans = []
    rewards = []
    
    for t in range(T):
        action, _ = mpc.compute_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        temps.append(info.get("temp", obs[0]))
        fans.append(float(action[0]))
        rewards.append(reward)
        
        if done or truncated:
            break
    
    temps = np.array(temps)
    fans = np.array(fans)
    rewards = np.array(rewards)
    
    warning_entries = np.sum((temps[:-1] <= 80.0) & (temps[1:] > 80.0))
    critical_entries = np.sum((temps[:-1] <= 90.0) & (temps[1:] > 90.0))
    
    return {
        "scenario": scenario,
        "seed": seed,
        "cumulative_reward": float(np.sum(rewards)),
        "mean_temp": float(np.mean(temps)),
        "max_temp": float(np.max(temps)),
        "mean_fan": float(np.mean(fans)),
        "warning_entries": int(warning_entries),
        "critical_entries": int(critical_entries),
    }


def rollout_mpc_with_delay(mpc, env, scenario: str, seed: int, delay_steps: int, T: int = 300) -> Dict[str, Any]:
    """MPC rollout with action delay."""
    mpc.reset()
    obs, info = env.reset(seed=seed)
    
    # Get scenario schedule for dynamic ambient/power
    amb_sched, pow_sched = scenario_schedule(scenario, T)
    
    # Handle thermal_high_start: set high initial temperature
    if scenario == "thermal_high_start":
        env.state[0] = max(env.state[0], 82.0)
    
    # Initialize buffer with default action (50% fan)
    action_buffer = deque([np.array([50.0]) for _ in range(delay_steps)], maxlen=delay_steps)
    temps = []
    fans = []
    rewards = []
    
    for t in range(T):
        # Inject scenario schedule (dynamic ambient/power)
        env.state[1] = float(amb_sched[t])  # Ambient temperature
        env.state[2] = float(pow_sched[t])  # Power consumption
        
        # Update obs to reflect injected schedule (critical for MPC!)
        obs = env.state.copy()
        
        # MPC decides action
        action_decided, _ = mpc.compute_action(obs)
        
        # Get action to apply (oldest in buffer = delay_steps ago)
        action_applied = action_buffer[0]
        
        # Add new action to buffer (will push out oldest)
        action_buffer.append(action_decided)
        
        obs, reward, done, truncated, info = env.step(action_applied)
        
        temps.append(info.get("temp", obs[0]))
        fans.append(float(action_applied[0]))
        rewards.append(reward)
        
        if done or truncated:
            break
    
    temps = np.array(temps)
    fans = np.array(fans)
    rewards = np.array(rewards)
    
    warning_entries = np.sum((temps[:-1] <= 80.0) & (temps[1:] > 80.0))
    critical_entries = np.sum((temps[:-1] <= 90.0) & (temps[1:] > 90.0))
    
    return {
        "scenario": scenario,
        "seed": seed,
        "cumulative_reward": float(np.sum(rewards)),
        "mean_temp": float(np.mean(temps)),
        "max_temp": float(np.max(temps)),
        "mean_fan": float(np.mean(fans)),
        "warning_entries": int(warning_entries),
        "critical_entries": int(critical_entries),
    }


def main():
    parser = argparse.ArgumentParser(description="Domain shift robustness evaluation")
    
    parser.add_argument("--perturbation", type=str, required=True,
                        choices=["ambient_shift", "power_spike", "action_delay"],
                        help="Type of perturbation")
    parser.add_argument("--shift-deg", type=float, default=5.0,
                        help="Ambient temperature shift (°C)")
    parser.add_argument("--spike-factor", type=float, default=1.2,
                        help="Power spike factor")
    parser.add_argument("--delay-steps", type=int, default=1,
                        help="Action delay (steps)")
    
    parser.add_argument("--controllers", nargs="+", required=True,
                        choices=["mpc_rc", "mpc_rcnn", "rl_rc", "rl_rcnn"],
                        help="Controllers to evaluate")
    parser.add_argument("--scenarios", nargs="+", required=True,
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
    print(f"Domain Shift Evaluation: {args.perturbation}")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for controller_name in args.controllers:
        print(f"\nController: {controller_name}")
        print(f"{'-'*40}")
        
        # Determine surrogate type
        if "rc" in controller_name and "rcnn" not in controller_name:
            surrogate_type = "rc"
            surrogate_cfg = {
                "thermal_capacity": 100.0,
                "heat_transfer_coeff": 0.05,
                "cooling_effectiveness": -0.03,
                "power_to_heat": 0.01,
                "dt": 1.0,
            }
        else:
            surrogate_type = "rcnn"
            surrogate_cfg = {"bundle_path": "models/rc_nn_hybrid.pkl"}
        
        env_cfg = {
            "max_steps": 300,
            "temp_warning": 80.0,
            "temp_critical": 90.0,
            "temp_target": 65.0,
            "reward_weights": {"thermal": 10.0, "energy": 0.1, "oscillation": 1.0, "headroom": 2.0},
        }
        
        # Create controller
        controller, vec_env = create_controller(controller_name, surrogate_type, surrogate_cfg, env_cfg)
        
        for scenario in args.scenarios:
            print(f"  Scenario: {scenario}")
            
            for ep in range(args.episodes):
                seed = 2000 + ep
                
                # Run with perturbation
                if args.perturbation == "ambient_shift":
                    metrics = rollout_ambient_shift(
                        controller_name, controller, vec_env, scenario, seed, args.shift_deg
                    )
                elif args.perturbation == "power_spike":
                    metrics = rollout_power_spike(
                        controller_name, controller, vec_env, scenario, seed, args.spike_factor
                    )
                else:  # action_delay
                    metrics = rollout_action_delay(
                        controller_name, controller, vec_env, scenario, seed, args.delay_steps
                    )
                
                metrics["controller"] = controller_name
                metrics["perturbation"] = args.perturbation
                all_results.append(metrics)
                
                print(f"    Episode {ep+1}/{args.episodes}: "
                      f"Reward={metrics['cumulative_reward']:.1f}, "
                      f"Fan={metrics['mean_fan']:.1f}%")
        
        # Clean up
        if vec_env is not None:
            vec_env.close()
    
    # Save results
    df = pd.DataFrame(all_results)
    results_csv = output_dir / "domain_shift_results.csv"
    df.to_csv(results_csv, index=False)
    print(f"\n✅ Results saved to: {results_csv}")
    
    # Summary
    summary = df.groupby(["controller", "scenario"]).agg({
        "cumulative_reward": ["mean", "std"],
        "mean_fan": ["mean", "std"],
        "warning_entries": "sum",
        "critical_entries": "sum",
    }).round(2)
    
    summary_csv = output_dir / "summary.csv"
    summary.to_csv(summary_csv)
    print(f"✅ Summary saved to: {summary_csv}")
    
    # Metadata
    metadata = {
        "perturbation": args.perturbation,
        "shift_deg": args.shift_deg if args.perturbation == "ambient_shift" else None,
        "spike_factor": args.spike_factor if args.perturbation == "power_spike" else None,
        "delay_steps": args.delay_steps if args.perturbation == "action_delay" else None,
        "controllers": args.controllers,
        "scenarios": args.scenarios,
        "episodes_per_scenario": args.episodes,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Domain shift evaluation complete!")


if __name__ == "__main__":
    main()
