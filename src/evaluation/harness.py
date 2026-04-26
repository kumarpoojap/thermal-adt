"""
Evaluation harness for thermal control policies.

Supports evaluation of both RL agents and classical controllers (MPC)
on various test scenarios with comprehensive metrics collection.
"""

from typing import Dict, List, Optional, Callable, Any
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime


class EvaluationHarness:
    """
    Unified evaluation harness for thermal control policies.
    
    Supports:
    - RL agents (Stable Baselines3)
    - MPC controllers
    - Custom policy functions
    
    Collects comprehensive metrics:
    - Temperature tracking (RMSE, max deviation, violations)
    - Energy efficiency (average fan speed, total energy)
    - Control smoothness (fan speed variance, rate-of-change)
    - Safety (constraint violations, emergency overrides)
    """
    
    def __init__(
        self,
        env,
        policy,
        policy_type: str = "rl",
        output_dir: Optional[str] = None,
        save_trajectory: bool = True
    ):
        """
        Initialize evaluation harness.
        
        Args:
            env: Gymnasium environment
            policy: Policy to evaluate (RL agent, MPC controller, or callable)
            policy_type: Type of policy ("rl", "mpc", or "custom")
            output_dir: Directory to save results
            save_trajectory: Whether to save full trajectory data
        """
        self.env = env
        self.policy = policy
        self.policy_type = policy_type
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_trajectory = save_trajectory
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.episode_metrics = []
        self.trajectory_data = []
    
    def evaluate_episode(
        self,
        scenario: Dict,
        seed: Optional[int] = None,
        render: bool = False
    ) -> Dict:
        """
        Evaluate policy on a single episode.
        
        Args:
            scenario: Scenario configuration (workload profile, initial conditions)
            seed: Random seed for reproducibility
            render: Whether to render environment
        
        Returns:
            metrics: Episode metrics dictionary
        """
        # Reset environment with scenario
        obs, info = self.env.reset(seed=seed)
        
        # Apply scenario-specific initial conditions if provided
        if "initial_temp" in scenario:
            self.env.unwrapped.state[0] = scenario["initial_temp"]
            obs[0] = scenario["initial_temp"]
        
        if "ambient_temp" in scenario:
            self.env.unwrapped.state[1] = scenario["ambient_temp"]
            obs[1] = scenario["ambient_temp"]
        
        # Reset policy if applicable
        if self.policy_type == "mpc" and hasattr(self.policy, "reset"):
            self.policy.reset(seed=seed)
        
        # Episode data
        trajectory = []
        done = False
        step = 0
        total_reward = 0.0
        
        # Metrics accumulators
        temp_errors = []
        fan_speeds = []
        fan_deltas = []
        temp_violations = 0
        safety_interventions = 0
        
        prev_fan = None
        
        while not done:
            # Get action from policy
            if self.policy_type == "rl":
                action, _ = self.policy.predict(obs, deterministic=True)
            elif self.policy_type == "mpc":
                action, policy_info = self.policy.compute_action(self.env.unwrapped.state)
            else:  # custom callable
                action = self.policy(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, step_info = self.env.step(action)
            done = terminated or truncated
            
            # Extract state information
            temp = next_obs[0]
            ambient = next_obs[1]
            power = next_obs[2]
            fan = action[0]
            
            # Compute metrics
            temp_target = scenario.get("temp_target", 75.0)
            temp_max = scenario.get("temp_max", 85.0)
            
            temp_error = abs(temp - temp_target)
            temp_errors.append(temp_error)
            fan_speeds.append(fan)
            
            if prev_fan is not None:
                fan_deltas.append(abs(fan - prev_fan))
            prev_fan = fan
            
            if temp > temp_max:
                temp_violations += 1
            
            # Check for safety interventions
            if "safety" in step_info and not step_info["safety"].get("is_safe", True):
                safety_interventions += 1
            
            # Store trajectory
            if self.save_trajectory:
                trajectory.append({
                    "step": step,
                    "temp": float(temp),
                    "ambient": float(ambient),
                    "power": float(power),
                    "fan": float(fan),
                    "action": float(action[0]),
                    "reward": float(reward),
                    "temp_error": float(temp_error),
                    "temp_violation": temp > temp_max
                })
            
            total_reward += reward
            obs = next_obs
            step += 1
            
            if render:
                self.env.render()
        
        # Compute episode metrics
        metrics = {
            "scenario": scenario.get("name", "unknown"),
            "seed": seed,
            "total_steps": step,
            "total_reward": float(total_reward),
            "avg_reward": float(total_reward / step) if step > 0 else 0.0,
            
            # Temperature tracking
            "temp_rmse": float(np.sqrt(np.mean(np.array(temp_errors) ** 2))),
            "temp_mae": float(np.mean(temp_errors)),
            "temp_max_error": float(np.max(temp_errors)),
            "temp_std": float(np.std(temp_errors)),
            
            # Safety
            "temp_violations": temp_violations,
            "temp_violation_rate": float(temp_violations / step) if step > 0 else 0.0,
            "safety_interventions": safety_interventions,
            "safety_intervention_rate": float(safety_interventions / step) if step > 0 else 0.0,
            
            # Energy efficiency
            "avg_fan_speed": float(np.mean(fan_speeds)),
            "fan_speed_std": float(np.std(fan_speeds)),
            "max_fan_speed": float(np.max(fan_speeds)),
            "min_fan_speed": float(np.min(fan_speeds)),
            
            # Control smoothness
            "avg_fan_delta": float(np.mean(fan_deltas)) if fan_deltas else 0.0,
            "max_fan_delta": float(np.max(fan_deltas)) if fan_deltas else 0.0,
            "fan_delta_std": float(np.std(fan_deltas)) if fan_deltas else 0.0,
        }
        
        # Add policy-specific metrics
        if self.policy_type == "mpc" and hasattr(self.policy, "get_stats"):
            metrics["mpc_stats"] = self.policy.get_stats()
        
        # Store
        self.episode_metrics.append(metrics)
        if self.save_trajectory:
            self.trajectory_data.append({
                "scenario": scenario.get("name", "unknown"),
                "seed": seed,
                "trajectory": trajectory
            })
        
        return metrics
    
    def evaluate_scenarios(
        self,
        scenarios: List[Dict],
        n_episodes_per_scenario: int = 5,
        seeds: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Evaluate policy across multiple scenarios.
        
        Args:
            scenarios: List of scenario configurations
            n_episodes_per_scenario: Number of episodes per scenario
            seeds: Optional list of seeds (one per episode)
        
        Returns:
            results_df: DataFrame with all episode metrics
        """
        if seeds is None:
            seeds = list(range(n_episodes_per_scenario))
        
        print(f"Evaluating {len(scenarios)} scenarios with {n_episodes_per_scenario} episodes each...")
        
        for scenario in scenarios:
            scenario_name = scenario.get("name", "unknown")
            print(f"\nScenario: {scenario_name}")
            
            for i, seed in enumerate(seeds[:n_episodes_per_scenario]):
                print(f"  Episode {i+1}/{n_episodes_per_scenario} (seed={seed})...", end=" ")
                
                metrics = self.evaluate_episode(scenario, seed=seed)
                
                print(f"Reward: {metrics['total_reward']:.2f}, "
                      f"Temp RMSE: {metrics['temp_rmse']:.2f}°C, "
                      f"Violations: {metrics['temp_violations']}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.episode_metrics)
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame, prefix: str = "eval"):
        """
        Save evaluation results to disk.
        
        Args:
            results_df: Results DataFrame
            prefix: Filename prefix
        """
        if self.output_dir is None:
            print("Warning: No output_dir specified, skipping save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics CSV
        metrics_path = self.output_dir / f"{prefix}_metrics_{timestamp}.csv"
        results_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")
        
        # Save summary statistics
        summary = self._compute_summary(results_df)
        summary_path = self.output_dir / f"{prefix}_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to {summary_path}")
        
        # Save trajectories
        if self.save_trajectory and self.trajectory_data:
            traj_path = self.output_dir / f"{prefix}_trajectories_{timestamp}.json"
            with open(traj_path, 'w') as f:
                json.dump(self.trajectory_data, f, indent=2)
            print(f"Saved trajectories to {traj_path}")
    
    def _compute_summary(self, results_df: pd.DataFrame) -> Dict:
        """
        Compute summary statistics across all episodes.
        
        Args:
            results_df: Results DataFrame
        
        Returns:
            summary: Summary statistics dictionary
        """
        # Overall statistics
        summary = {
            "total_episodes": len(results_df),
            "policy_type": self.policy_type,
            "timestamp": datetime.now().isoformat(),
            
            # Aggregate metrics
            "overall": {
                "avg_reward": float(results_df["total_reward"].mean()),
                "std_reward": float(results_df["total_reward"].std()),
                "avg_temp_rmse": float(results_df["temp_rmse"].mean()),
                "avg_temp_mae": float(results_df["temp_mae"].mean()),
                "avg_violations": float(results_df["temp_violations"].mean()),
                "avg_fan_speed": float(results_df["avg_fan_speed"].mean()),
                "avg_fan_delta": float(results_df["avg_fan_delta"].mean()),
            }
        }
        
        # Per-scenario statistics
        if "scenario" in results_df.columns:
            summary["by_scenario"] = {}
            for scenario in results_df["scenario"].unique():
                scenario_df = results_df[results_df["scenario"] == scenario]
                summary["by_scenario"][scenario] = {
                    "n_episodes": len(scenario_df),
                    "avg_reward": float(scenario_df["total_reward"].mean()),
                    "std_reward": float(scenario_df["total_reward"].std()),
                    "avg_temp_rmse": float(scenario_df["temp_rmse"].mean()),
                    "avg_violations": float(scenario_df["temp_violations"].mean()),
                    "avg_fan_speed": float(scenario_df["avg_fan_speed"].mean()),
                }
        
        return summary
    
    def print_summary(self, results_df: pd.DataFrame):
        """
        Print evaluation summary to console.
        
        Args:
            results_df: Results DataFrame
        """
        summary = self._compute_summary(results_df)
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Policy Type: {self.policy_type}")
        print(f"Total Episodes: {summary['total_episodes']}")
        print()
        
        print("Overall Performance:")
        print(f"  Average Reward: {summary['overall']['avg_reward']:.2f} ± {summary['overall']['std_reward']:.2f}")
        print(f"  Temp RMSE: {summary['overall']['avg_temp_rmse']:.2f}°C")
        print(f"  Temp MAE: {summary['overall']['avg_temp_mae']:.2f}°C")
        print(f"  Avg Violations: {summary['overall']['avg_violations']:.2f}")
        print(f"  Avg Fan Speed: {summary['overall']['avg_fan_speed']:.1f}%")
        print(f"  Avg Fan Delta: {summary['overall']['avg_fan_delta']:.1f}%")
        print()
        
        if "by_scenario" in summary:
            print("Per-Scenario Performance:")
            for scenario, stats in summary["by_scenario"].items():
                print(f"\n  {scenario}:")
                print(f"    Reward: {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
                print(f"    Temp RMSE: {stats['avg_temp_rmse']:.2f}°C")
                print(f"    Violations: {stats['avg_violations']:.2f}")
                print(f"    Fan Speed: {stats['avg_fan_speed']:.1f}%")
        
        print("="*80)
