"""
Reward function components for thermal control RL.

Provides modular reward components that can be combined and weighted
for different training objectives.
"""

import numpy as np
from typing import Dict, Tuple


class RewardFunction:
    """
    Modular reward function for thermal control.
    
    Combines multiple objectives:
    - Throttle avoidance (safety)
    - Energy efficiency (cost)
    - Control smoothness (stability)
    - Thermal headroom (proactive cooling)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize reward function.
        
        Args:
            config: Reward configuration with weights and thresholds
        """
        self.config = config
        self.weights = config.get("weights", {
            "throttle_risk": 10.0,
            "energy": 0.1,
            "oscillation": 1.0,
            "headroom": 2.0
        })
        
        self.temp_throttle = config.get("temp_throttle", 85.0)
        self.temp_target = config.get("temp_target", 75.0)
        self.temp_safe = config.get("temp_safe", 80.0)
        self.temp_danger_zone = config.get("temp_danger_zone", 5.0)
    
    def compute(
        self,
        current_temp: float,
        next_temp: float,
        fan_speed: float,
        prev_fan_speed: float,
        workload: float = 50.0
    ) -> Tuple[float, Dict]:
        """
        Compute total reward and components.
        
        Args:
            current_temp: Current temperature (°C)
            next_temp: Predicted next temperature (°C)
            fan_speed: Current fan speed (%)
            prev_fan_speed: Previous fan speed (%)
            workload: Current workload (%)
        
        Returns:
            total_reward: Weighted sum of components
            components: Dictionary of individual reward components
        """
        # Compute individual components
        throttle_risk = self._throttle_risk(next_temp)
        energy_cost = self._energy_cost(fan_speed)
        oscillation = self._oscillation_penalty(fan_speed, prev_fan_speed)
        headroom_bonus = self._headroom_bonus(next_temp)
        
        # Weighted sum
        total_reward = (
            self.weights["throttle_risk"] * throttle_risk +
            self.weights["energy"] * energy_cost +
            self.weights["oscillation"] * oscillation +
            self.weights["headroom"] * headroom_bonus
        )
        
        components = {
            "throttle_risk": throttle_risk,
            "energy_cost": energy_cost,
            "oscillation": oscillation,
            "headroom_bonus": headroom_bonus,
            "total": total_reward
        }
        
        return total_reward, components
    
    def _throttle_risk(self, temp: float) -> float:
        """
        Throttle risk penalty (exponential as temp approaches threshold).
        
        Returns large negative reward if throttling occurs or temp is
        in danger zone.
        """
        margin = self.temp_throttle - temp
        
        if margin < 0:
            # Already throttling - catastrophic penalty
            return -100.0
        elif margin < self.temp_danger_zone:
            # Danger zone - exponential penalty
            return -np.exp(self.temp_danger_zone - margin)
        else:
            # Safe - no penalty
            return 0.0
    
    def _energy_cost(self, fan_speed: float) -> float:
        """
        Energy cost penalty (linear in fan speed).
        
        Encourages agent to use minimal cooling necessary.
        """
        # Normalize to [0, 1] and negate
        return -(fan_speed / 100.0)
    
    def _oscillation_penalty(self, fan_speed: float, prev_fan_speed: float) -> float:
        """
        Oscillation penalty (penalize rapid fan speed changes).
        
        Encourages smooth, stable control.
        """
        if prev_fan_speed is None:
            return 0.0
        
        # Penalize large changes
        delta = abs(fan_speed - prev_fan_speed)
        
        # Normalize by max reasonable change (20% per step)
        normalized_delta = delta / 20.0
        
        return -normalized_delta
    
    def _headroom_bonus(self, temp: float) -> float:
        """
        Thermal headroom bonus (reward staying in safe operating zone).
        
        Encourages proactive cooling to maintain margin.
        """
        if temp <= self.temp_target:
            # Ideal operating zone
            return 1.0
        elif temp <= self.temp_safe:
            # Safe zone (acceptable)
            return 0.5
        else:
            # Approaching danger - no bonus
            return 0.0


class SparseRewardFunction:
    """
    Sparse reward variant - only penalizes throttle events.
    
    Useful for comparison and ablation studies.
    """
    
    def __init__(self, config: Dict):
        self.temp_throttle = config.get("temp_throttle", 85.0)
    
    def compute(
        self,
        current_temp: float,
        next_temp: float,
        fan_speed: float,
        prev_fan_speed: float,
        workload: float = 50.0
    ) -> Tuple[float, Dict]:
        """Compute sparse reward (only throttle penalty)."""
        if next_temp >= self.temp_throttle:
            reward = -100.0
        else:
            reward = 0.0
        
        components = {
            "throttle_penalty": reward,
            "total": reward
        }
        
        return reward, components


class DenseShapedRewardFunction(RewardFunction):
    """
    Dense reward with additional shaping for faster learning.
    
    Adds:
    - Temperature delta penalty (penalize temp increases)
    - Predictive risk (penalize trajectory toward throttle)
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.weights["temp_delta"] = config.get("temp_delta_weight", 0.5)
        self.weights["predictive_risk"] = config.get("predictive_risk_weight", 2.0)
    
    def compute(
        self,
        current_temp: float,
        next_temp: float,
        fan_speed: float,
        prev_fan_speed: float,
        workload: float = 50.0
    ) -> Tuple[float, Dict]:
        """Compute dense shaped reward."""
        # Base components
        total_reward, components = super().compute(
            current_temp, next_temp, fan_speed, prev_fan_speed, workload
        )
        
        # Additional shaping: temperature delta
        temp_delta = next_temp - current_temp
        temp_delta_penalty = -abs(temp_delta) if temp_delta > 0 else 0.0
        
        # Additional shaping: predictive risk (trajectory toward throttle)
        if temp_delta > 0 and next_temp > self.temp_safe:
            # Temperature rising and already in warm zone
            time_to_throttle = (self.temp_throttle - next_temp) / max(temp_delta, 0.1)
            if time_to_throttle < 10.0:  # Less than 10 steps to throttle
                predictive_risk = -np.exp(10.0 - time_to_throttle)
            else:
                predictive_risk = 0.0
        else:
            predictive_risk = 0.0
        
        # Add to total
        total_reward += (
            self.weights["temp_delta"] * temp_delta_penalty +
            self.weights["predictive_risk"] * predictive_risk
        )
        
        components["temp_delta_penalty"] = temp_delta_penalty
        components["predictive_risk"] = predictive_risk
        components["total"] = total_reward
        
        return total_reward, components


def create_reward_function(reward_type: str = "standard", config: Dict = None) -> RewardFunction:
    """
    Factory function to create reward function.
    
    Args:
        reward_type: One of ["standard", "sparse", "dense"]
        config: Reward configuration
    
    Returns:
        RewardFunction instance
    """
    config = config or {}
    
    if reward_type == "standard":
        return RewardFunction(config)
    elif reward_type == "sparse":
        return SparseRewardFunction(config)
    elif reward_type == "dense":
        return DenseShapedRewardFunction(config)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
