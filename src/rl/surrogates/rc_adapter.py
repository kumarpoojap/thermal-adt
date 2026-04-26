"""
RC (Resistor-Capacitor) thermal model adapter.

Implements analytical thermal dynamics using RC circuit analogy.
"""

from typing import Optional, Dict
import numpy as np


class RCAdapter:
    """
    Adapter for analytical RC thermal model.
    
    Uses a simple resistor-capacitor circuit analogy for thermal dynamics:
    T(t+1) = T(t) + dt * [gamma*P - beta*Fan - h*(T - T_amb)] / C
    
    No history buffer needed - stateless analytical model.
    """
    
    def __init__(
        self,
        thermal_capacity: float = 100.0,
        heat_transfer_coeff: float = 0.05,
        cooling_effectiveness: float = -0.03,
        power_to_heat: float = 0.01,
        dt: float = 1.0,
        temp_min: float = 30.0,
        temp_max: float = 95.0,
        config: Optional[Dict] = None
    ):
        """
        Initialize RC thermal model.
        
        Args:
            thermal_capacity: Thermal capacitance (C)
            heat_transfer_coeff: Heat transfer coefficient (h)
            cooling_effectiveness: Fan cooling coefficient (beta)
            power_to_heat: Power-to-heat conversion (gamma)
            dt: Time step (seconds)
            temp_min: Minimum physical temperature (°C)
            temp_max: Maximum physical temperature (°C)
            config: Optional config dict to override defaults
        """
        if config is not None:
            self.C = config.get("thermal_capacity", thermal_capacity)
            self.h = config.get("heat_transfer_coeff", heat_transfer_coeff)
            self.beta = config.get("cooling_effectiveness", cooling_effectiveness)
            self.gamma = config.get("power_to_heat", power_to_heat)
            self.dt = config.get("dt", dt)
            self.temp_min = config.get("temp_min", temp_min)
            self.temp_max = config.get("temp_max", temp_max)
        else:
            self.C = thermal_capacity
            self.h = heat_transfer_coeff
            self.beta = cooling_effectiveness
            self.gamma = power_to_heat
            self.dt = dt
            self.temp_min = temp_min
            self.temp_max = temp_max
    
    def reset(
        self,
        seed: Optional[int] = None,
        init_state: Optional[np.ndarray] = None
    ) -> None:
        """
        Reset (no-op for stateless RC model).
        
        Args:
            seed: Random seed (unused)
            init_state: Initial state (unused)
        """
        if seed is not None:
            np.random.seed(seed)
    
    def predict_next(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> float:
        """
        Predict next GPU temperature using RC thermal model.
        
        Args:
            state: Current state [temp, ambient, power, fan_speed, temp_delta]
            action: Action [fan_speed]
        
        Returns:
            Predicted next GPU temperature (°C)
        """
        current_temp = float(state[0])
        ambient_temp = float(state[1])
        gpu_power = float(state[2])
        fan_speed = float(action[0])
        
        heat_gen = self.gamma * gpu_power
        cooling = self.beta * fan_speed
        heat_transfer = -self.h * (current_temp - ambient_temp)
        
        dT_dt = (heat_gen + cooling + heat_transfer) / self.C
        next_temp = current_temp + dT_dt * self.dt
        
        next_temp = np.clip(next_temp, self.temp_min, self.temp_max)
        
        return float(next_temp)
    
    @property
    def warmup_steps(self) -> int:
        """RC model is stateless, no warmup needed."""
        return 0
