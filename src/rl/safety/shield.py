"""
Safety Shield for Thermal Control.

Implements hard constraints and safety mechanisms to ensure RL agent
actions don't violate physical or operational constraints.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import gymnasium as gym


class SafetyShield:
    """
    Safety layer that filters RL agent actions to ensure safe operation.
    
    Safety mechanisms:
    1. Action clamping: Ensure fan speed in valid range [20%, 100%]
    2. Rate limiting: Limit maximum fan speed change per step
    3. Emergency override: Force max cooling if temp exceeds threshold
    4. Minimum cooling: Ensure minimum fan speed at high temps
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize safety shield.
        
        Args:
            config: Safety configuration parameters
        """
        self.config = config or self._default_config()
        
        # Action constraints
        self.fan_min = self.config["fan_min"]
        self.fan_max = self.config["fan_max"]
        self.max_fan_delta = self.config["max_fan_delta"]
        
        # Temperature thresholds
        self.temp_emergency = self.config["temp_emergency"]
        self.temp_high = self.config["temp_high"]
        
        # Minimum fan speeds at different temperatures
        self.temp_fan_map = self.config["temp_fan_map"]
        
        # Statistics
        self.stats = {
            "total_actions": 0,
            "clamped_actions": 0,
            "rate_limited_actions": 0,
            "emergency_overrides": 0,
            "min_cooling_enforced": 0
        }
    
    def _default_config(self) -> Dict:
        """Default safety configuration."""
        return {
            "fan_min": 20.0,
            "fan_max": 100.0,
            "max_fan_delta": 20.0,  # Max 20% change per step
            "temp_emergency": 88.0,  # Emergency override threshold
            "temp_high": 80.0,       # High temperature threshold
            "temp_fan_map": {        # Minimum fan speeds at different temps
                85.0: 80.0,  # At 85°C, min 80% fan
                80.0: 60.0,  # At 80°C, min 60% fan
                75.0: 40.0,  # At 75°C, min 40% fan
            }
        }
    
    def filter_action(
        self,
        state: np.ndarray,
        action: np.ndarray,
        prev_action: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Filter and modify action to ensure safety.
        
        Args:
            state: Current environment state [temp, ambient, power, fan, temp_delta]
            action: Proposed action from RL agent
            prev_action: Previous action (for rate limiting)
        
        Returns:
            safe_action: Filtered safe action
            info: Information about safety interventions
        """
        self.stats["total_actions"] += 1
        
        # Extract state
        current_temp = state[0]
        current_fan = state[3] if len(state) > 3 else (prev_action or 50.0)
        
        # Extract proposed action
        proposed_fan = float(action[0])
        
        # Track interventions
        interventions = []
        
        # 1. Emergency override (highest priority)
        if current_temp >= self.temp_emergency:
            safe_fan = self.fan_max
            interventions.append("emergency_override")
            self.stats["emergency_overrides"] += 1
        else:
            safe_fan = proposed_fan
            
            # 2. Enforce minimum cooling at high temperatures
            min_fan = self._get_minimum_fan(current_temp)
            if safe_fan < min_fan:
                safe_fan = min_fan
                interventions.append("min_cooling_enforced")
                self.stats["min_cooling_enforced"] += 1
            
            # 3. Clamp to valid range
            if safe_fan < self.fan_min or safe_fan > self.fan_max:
                safe_fan = np.clip(safe_fan, self.fan_min, self.fan_max)
                interventions.append("action_clamped")
                self.stats["clamped_actions"] += 1
            
            # 4. Rate limiting
            if prev_action is not None:
                max_change = self.max_fan_delta
                fan_delta = safe_fan - prev_action
                
                if abs(fan_delta) > max_change:
                    # Limit the change
                    if fan_delta > 0:
                        safe_fan = prev_action + max_change
                    else:
                        safe_fan = prev_action - max_change
                    
                    interventions.append("rate_limited")
                    self.stats["rate_limited_actions"] += 1
        
        # Create safe action
        safe_action = np.array([safe_fan], dtype=np.float32)
        
        # Info
        info = {
            "proposed_fan": proposed_fan,
            "safe_fan": safe_fan,
            "interventions": interventions,
            "current_temp": current_temp,
            "is_safe": len(interventions) == 0
        }
        
        return safe_action, info
    
    def _get_minimum_fan(self, temp: float) -> float:
        """
        Get minimum required fan speed for given temperature.
        
        Args:
            temp: Current temperature (°C)
        
        Returns:
            min_fan: Minimum fan speed (%)
        """
        # Find applicable minimum from temp_fan_map
        min_fan = self.fan_min
        
        for temp_threshold, required_fan in sorted(self.temp_fan_map.items()):
            if temp >= temp_threshold:
                min_fan = max(min_fan, required_fan)
        
        return min_fan
    
    def check_safety(self, state: np.ndarray) -> Dict:
        """
        Check if current state is safe.
        
        Args:
            state: Current environment state
        
        Returns:
            safety_status: Dictionary with safety information
        """
        current_temp = state[0]
        current_fan = state[3] if len(state) > 3 else 50.0
        
        # Check various safety conditions
        is_emergency = current_temp >= self.temp_emergency
        is_high_temp = current_temp >= self.temp_high
        min_required_fan = self._get_minimum_fan(current_temp)
        fan_adequate = current_fan >= min_required_fan
        
        return {
            "is_safe": not is_emergency and fan_adequate,
            "is_emergency": is_emergency,
            "is_high_temp": is_high_temp,
            "current_temp": current_temp,
            "current_fan": current_fan,
            "min_required_fan": min_required_fan,
            "fan_adequate": fan_adequate
        }
    
    def get_stats(self) -> Dict:
        """Get safety shield statistics."""
        stats = self.stats.copy()
        
        if stats["total_actions"] > 0:
            stats["intervention_rate"] = (
                (stats["clamped_actions"] + 
                 stats["rate_limited_actions"] + 
                 stats["emergency_overrides"] + 
                 stats["min_cooling_enforced"]) / 
                stats["total_actions"]
            )
        else:
            stats["intervention_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            "total_actions": 0,
            "clamped_actions": 0,
            "rate_limited_actions": 0,
            "emergency_overrides": 0,
            "min_cooling_enforced": 0
        }


class SafetyWrapper(gym.Wrapper):
    """
    Gym environment wrapper that adds safety shield.
    
    Wraps any thermal control environment and automatically filters
    actions through the safety shield.
    """
    
    def __init__(self, env: gym.Env, safety_config: Optional[Dict] = None):
        """
        Initialize safety wrapper.
        
        Args:
            env: Base thermal control environment
            safety_config: Safety shield configuration
        """
        super().__init__(env)
        self.safety_shield = SafetyShield(safety_config)
        self.prev_action = None
    
    def reset(self, **kwargs):
        """Reset environment and safety shield."""
        obs, info = self.env.reset(**kwargs)
        self.prev_action = None
        self.safety_shield.reset_stats()
        return obs, info
    
    def step(self, action):
        """Step with safety filtering."""
        # Get current state (handle gym wrappers like Monitor)
        base_env = self.env
        if hasattr(base_env, "state"):
            state = base_env.state
        elif hasattr(base_env, "unwrapped") and hasattr(base_env.unwrapped, "state"):
            state = base_env.unwrapped.state
        else:
            raise AttributeError("Wrapped env has no 'state' attribute for SafetyShield")
        
        # Filter action through safety shield
        safe_action, safety_info = self.safety_shield.filter_action(
            state, action, self.prev_action
        )
        
        # Execute safe action
        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        
        # Update previous action
        self.prev_action = safe_action[0]
        
        # Add safety info to environment info
        info["safety"] = safety_info
        info["safety_stats"] = self.safety_shield.get_stats()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render environment."""
        return self.env.render()
    
    def close(self):
        """Close environment."""
        return self.env.close()
    
    def get_episode_metrics(self):
        """Get episode metrics including safety stats."""
        # Walk wrapper chain to find the base env that implements get_episode_metrics
        base = self.env
        for _ in range(10):
            if hasattr(base, "get_episode_metrics"):
                break
            if hasattr(base, "env"):
                base = base.env
                continue
            if hasattr(base, "unwrapped"):
                base = base.unwrapped
                continue
            break

        if hasattr(base, "get_episode_metrics"):
            metrics = base.get_episode_metrics()
        else:
            metrics = {}
        metrics["safety_stats"] = self.safety_shield.get_stats()
        return metrics
