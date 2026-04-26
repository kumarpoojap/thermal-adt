"""
Quick smoke test for MPC controller.

Tests MPC controller with RC surrogate on a simple scenario.
"""

import sys
from pathlib import Path
import numpy as np

# Fix encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.control import MPCController
from src.rl.surrogates import RCAdapter
from src.rl.environments.thermal_unified import ThermalControlEnv


def test_mpc_basic():
    """Test basic MPC functionality."""
    print("="*80)
    print("MPC SMOKE TEST")
    print("="*80)
    
    # Create RC surrogate
    print("\n1. Creating RC surrogate...")
    surrogate = RCAdapter(
        thermal_capacity=100.0,
        heat_transfer_coeff=0.05,
        cooling_effectiveness=-0.03,
        power_to_heat=0.01,
        dt=1.0
    )
    print("   ✓ RC surrogate created")
    
    # Create environment
    print("\n2. Creating environment...")
    env_config = {
        "max_steps": 50,
        "temp_target": 75.0,
        "temp_warning": 80.0,
        "temp_critical": 90.0,
        "initial_temp_range": [40.0, 60.0],
        "ambient_range": [20.0, 30.0],
        "power_range": [100.0, 300.0],
        "reward_weights": {
            "thermal": 10.0,
            "energy": 0.1,
            "oscillation": 1.0,
            "headroom": 2.0
        }
    }
    env = ThermalControlEnv(
        surrogate=surrogate,
        config=env_config
    )
    print("   ✓ Environment created")
    
    # Create MPC controller
    print("\n3. Creating MPC controller...")
    mpc = MPCController(
        surrogate=surrogate,
        horizon=10,
        temp_target=75.0,
        temp_max=85.0,
        fan_min=20.0,
        fan_max=100.0,
        max_fan_delta=20.0,
        weight_temp=10.0,
        weight_effort=0.1,
        weight_rate=1.0
    )
    print("   ✓ MPC controller created")
    
    # Run episode
    print("\n4. Running test episode (50 steps)...")
    obs, info = env.reset(seed=42)
    mpc.reset(seed=42)
    
    total_reward = 0.0
    temps = []
    fans = []
    
    for step in range(50):
        # Get MPC action
        action, mpc_info = mpc.compute_action(env.state)
        
        # Step environment
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        
        # Record
        temps.append(obs[0])
        fans.append(action[0])
        
        if step % 10 == 0:
            print(f"   Step {step:3d}: Temp={obs[0]:.2f}°C, Fan={action[0]:.1f}%, "
                  f"Reward={reward:.2f}, OptCost={mpc_info['optimization_cost']:.2f}")
        
        if terminated or truncated:
            break
    
    # Summary
    print("\n5. Episode Summary:")
    print(f"   Total steps: {len(temps)}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Avg reward: {total_reward/len(temps):.2f}")
    print(f"   Temperature: {np.mean(temps):.2f}°C ± {np.std(temps):.2f}°C")
    print(f"   Temp range: [{np.min(temps):.2f}, {np.max(temps):.2f}]°C")
    print(f"   Fan speed: {np.mean(fans):.1f}% ± {np.std(fans):.1f}%")
    print(f"   Fan range: [{np.min(fans):.1f}, {np.max(fans):.1f}]%")
    
    # MPC stats
    mpc_stats = mpc.get_stats()
    print(f"\n6. MPC Statistics:")
    print(f"   Avg temp error: {mpc_stats['avg_temp_error']:.2f}°C")
    print(f"   Avg fan effort: {mpc_stats['avg_fan_effort']:.1f}%")
    print(f"   Avg fan delta: {mpc_stats['avg_fan_delta']:.1f}%")
    print(f"   Violations: {mpc_stats['constraint_violations']}")
    
    print("\n" + "="*80)
    print("✓ MPC SMOKE TEST PASSED")
    print("="*80)
    
    env.close()


if __name__ == "__main__":
    test_mpc_basic()
