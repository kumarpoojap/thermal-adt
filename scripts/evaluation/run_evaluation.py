"""
Run evaluation of thermal control policies.

Evaluates MPC and RL agents on nominal and stress test scenarios,
collecting comprehensive performance metrics.
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import EvaluationHarness, create_scenarios
from src.control import MPCController
from src.rl.surrogates import create_surrogate
from src.rl.environments.thermal_unified import ThermalControlEnv
from src.rl.safety.shield import SafetyWrapper
from stable_baselines3 import SAC


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_environment(config: dict):
    """
    Create thermal control environment from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        env: Gymnasium environment
    """
    # Create surrogate
    surrogate = create_surrogate(config["surrogate"])
    
    # Create environment
    env = ThermalControlEnv(
        surrogate=surrogate,
        **config.get("env", {})
    )
    
    # Add safety wrapper if configured
    if config.get("use_safety", False):
        env = SafetyWrapper(env, config.get("safety", {}))
    
    return env


def create_mpc_policy(config: dict, env):
    """
    Create MPC controller.
    
    Args:
        config: Configuration dictionary
        env: Environment (for surrogate)
    
    Returns:
        mpc: MPC controller
    """
    # Get surrogate from environment
    if hasattr(env, "unwrapped"):
        surrogate = env.unwrapped.surrogate
    else:
        surrogate = env.surrogate
    
    # Create MPC controller
    mpc = MPCController(
        surrogate=surrogate,
        **config.get("mpc", {})
    )
    
    return mpc


def create_rl_policy(config: dict, model_path: str):
    """
    Load trained RL agent.
    
    Args:
        config: Configuration dictionary
        model_path: Path to trained model
    
    Returns:
        agent: Trained RL agent
    """
    # Load trained model
    agent = SAC.load(model_path)
    return agent


def main():
    parser = argparse.ArgumentParser(description="Evaluate thermal control policies")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/evaluation/eval_config.yaml",
        help="Path to evaluation config"
    )
    parser.add_argument(
        "--policy-config",
        type=str,
        default="configs/evaluation/mpc_baseline.yaml",
        help="Path to policy config (MPC or RL)"
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        choices=["mpc", "rl"],
        default="mpc",
        help="Type of policy to evaluate"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained RL model (required for RL policy)"
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["nominal", "stress"],
        help="Scenario types to evaluate"
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=5,
        help="Number of episodes per scenario"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        default=True,
        help="Save full trajectory data"
    )
    
    args = parser.parse_args()
    
    # Load configurations
    print(f"Loading policy config from {args.policy_config}")
    policy_config = load_config(args.policy_config)
    
    # Create environment
    print("Creating environment...")
    env = create_environment(policy_config)
    
    # Create policy
    print(f"Creating {args.policy_type} policy...")
    if args.policy_type == "mpc":
        policy = create_mpc_policy(policy_config, env)
        policy_type = "mpc"
    elif args.policy_type == "rl":
        if args.model_path is None:
            raise ValueError("--model-path required for RL policy")
        policy = create_rl_policy(policy_config, args.model_path)
        policy_type = "rl"
    else:
        raise ValueError(f"Unknown policy type: {args.policy_type}")
    
    # Create scenarios
    print(f"Creating scenarios: {args.scenarios}")
    scenarios = create_scenarios(args.scenarios)
    print(f"Total scenarios: {len(scenarios)}")
    
    # Create evaluation harness
    print(f"Initializing evaluation harness (output: {args.output_dir})")
    harness = EvaluationHarness(
        env=env,
        policy=policy,
        policy_type=policy_type,
        output_dir=args.output_dir,
        save_trajectory=args.save_trajectory
    )
    
    # Run evaluation
    print("\n" + "="*80)
    print("STARTING EVALUATION")
    print("="*80)
    
    results_df = harness.evaluate_scenarios(
        scenarios=scenarios,
        n_episodes_per_scenario=args.n_episodes
    )
    
    # Print summary
    harness.print_summary(results_df)
    
    # Save results
    print("\nSaving results...")
    run_name = policy_config.get("run_name", "eval")
    harness.save_results(results_df, prefix=run_name)
    
    print("\nEvaluation complete!")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
