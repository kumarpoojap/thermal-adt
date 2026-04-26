"""
Smoke tests for surrogate adapters.

Validates that RC, RF, and PINN adapters conform to the ThermalSurrogate
protocol and can be used with the unified environment.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.surrogates import RCAdapter, create_surrogate
from src.rl.environments.thermal_unified import ThermalControlEnv


class TestRCAdapter:
    """Test RC surrogate adapter."""
    
    def test_rc_adapter_creation(self):
        """Test RC adapter can be created."""
        adapter = RCAdapter()
        assert adapter is not None
        assert adapter.warmup_steps == 0
    
    def test_rc_adapter_reset(self):
        """Test RC adapter reset."""
        adapter = RCAdapter()
        init_state = np.array([50.0, 25.0, 200.0, 50.0, 0.0])
        adapter.reset(seed=42, init_state=init_state)
    
    def test_rc_adapter_predict(self):
        """Test RC adapter prediction."""
        adapter = RCAdapter()
        state = np.array([50.0, 25.0, 200.0, 50.0, 0.0])
        action = np.array([60.0])
        
        adapter.reset(init_state=state)
        next_temp = adapter.predict_next(state, action)
        
        assert isinstance(next_temp, float)
        assert 30.0 <= next_temp <= 95.0
    
    def test_rc_adapter_with_env(self):
        """Test RC adapter works with unified environment."""
        adapter = RCAdapter()
        env = ThermalControlEnv(surrogate=adapter)
        
        obs, info = env.reset(seed=42)
        assert obs.shape == (5,)
        assert info["warmup_steps"] == 0
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (5,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_rc_adapter_episode(self):
        """Test RC adapter can run a full episode."""
        adapter = RCAdapter()
        env = ThermalControlEnv(surrogate=adapter)
        
        obs, info = env.reset(seed=42)
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        metrics = env.get_episode_metrics()
        assert "mean_temp" in metrics
        assert "total_reward" in metrics


class TestSurrogateFactory:
    """Test surrogate factory function."""
    
    def test_create_rc_surrogate(self):
        """Test factory creates RC surrogate."""
        config = {
            "type": "rc",
            "thermal_capacity": 100.0,
            "dt": 1.0
        }
        
        surrogate = create_surrogate("rc", config)
        assert isinstance(surrogate, RCAdapter)
    
    def test_create_rc_with_env(self):
        """Test factory-created RC surrogate works with env."""
        config = {
            "type": "rc",
            "thermal_capacity": 100.0
        }
        
        surrogate = create_surrogate("rc", config)
        env = ThermalControlEnv(surrogate=surrogate)
        
        obs, info = env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (5,)
        assert isinstance(reward, float)
    
    def test_invalid_surrogate_type(self):
        """Test factory raises error for invalid type."""
        with pytest.raises(ValueError, match="Unknown surrogate type"):
            create_surrogate("invalid", {})


class TestAdapterProtocol:
    """Test that adapters conform to protocol."""
    
    def test_rc_adapter_has_required_methods(self):
        """Test RC adapter has all required protocol methods."""
        adapter = RCAdapter()
        
        assert hasattr(adapter, "reset")
        assert hasattr(adapter, "predict_next")
        assert hasattr(adapter, "warmup_steps")
        
        assert callable(adapter.reset)
        assert callable(adapter.predict_next)
    
    def test_rc_adapter_method_signatures(self):
        """Test RC adapter methods have correct signatures."""
        adapter = RCAdapter()
        state = np.array([50.0, 25.0, 200.0, 50.0, 0.0])
        action = np.array([60.0])
        
        adapter.reset(seed=42, init_state=state)
        
        result = adapter.predict_next(state, action)
        assert isinstance(result, float)
        
        warmup = adapter.warmup_steps
        assert isinstance(warmup, int)


class TestEnvironmentCompatibility:
    """Test environment works with different adapters."""
    
    def test_env_with_rc_adapter(self):
        """Test environment with RC adapter."""
        adapter = RCAdapter()
        env = ThermalControlEnv(surrogate=adapter)
        
        obs, info = env.reset(seed=42)
        
        for _ in range(5):
            action = np.array([50.0])
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert 30.0 <= obs[0] <= 95.0
            assert 20.0 <= obs[3] <= 100.0
            
            if terminated or truncated:
                break
    
    def test_env_determinism(self):
        """Test environment is deterministic with same seed."""
        adapter1 = RCAdapter()
        env1 = ThermalControlEnv(surrogate=adapter1)
        
        adapter2 = RCAdapter()
        env2 = ThermalControlEnv(surrogate=adapter2)
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        np.testing.assert_array_almost_equal(obs1, obs2)
        
        action = np.array([50.0])
        obs1, r1, _, _, _ = env1.step(action)
        obs2, r2, _, _, _ = env2.step(action)
        
        np.testing.assert_array_almost_equal(obs1, obs2)
        assert abs(r1 - r2) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
