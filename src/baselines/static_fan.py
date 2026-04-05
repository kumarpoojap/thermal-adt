"""
Static Fan Curve Controller (Baseline).

Industry-standard piecewise temperature-to-fan mapping.
"""

import numpy as np
from typing import Dict, List, Tuple


class StaticFanController:
    """
    Static fan curve controller.
    
    Maps temperature to fan speed using piecewise linear function.
    This is the most common baseline in real systems.
    """
    
    def __init__(self, fan_curve: List[Tuple[float, float]]):
        """
        Initialize static fan controller.
        
        Args:
            fan_curve: List of (temperature, fan_speed) breakpoints
                       Example: [(60, 30), (70, 50), (80, 75), (85, 100)]
        """
        self.fan_curve = sorted(fan_curve, key=lambda x: x[0])
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict fan speed based on current temperature.
        
        Args:
            state: [temp, workload, power, ambient, fan_speed]
        
        Returns:
            action: Fan speed (%)
        """
        temp = state[0]
        
        # Find appropriate fan speed from curve
        if temp <= self.fan_curve[0][0]:
            fan_speed = self.fan_curve[0][1]
        elif temp >= self.fan_curve[-1][0]:
            fan_speed = self.fan_curve[-1][1]
        else:
            # Linear interpolation between breakpoints
            for i in range(len(self.fan_curve) - 1):
                temp_low, fan_low = self.fan_curve[i]
                temp_high, fan_high = self.fan_curve[i + 1]
                
                if temp_low <= temp <= temp_high:
                    # Linear interpolation
                    alpha = (temp - temp_low) / (temp_high - temp_low)
                    fan_speed = fan_low + alpha * (fan_high - fan_low)
                    break
        
        return np.array([fan_speed], dtype=np.float32)
    
    def reset(self):
        """Reset controller state (no state for static controller)."""
        pass


def create_default_fan_curve() -> StaticFanController:
    """
    Create controller with default industry-standard fan curve.
    
    Returns:
        StaticFanController with default curve
    """
    default_curve = [
        (60.0, 30.0),   # Below 60°C: 30% fan
        (70.0, 50.0),   # At 70°C: 50% fan
        (80.0, 75.0),   # At 80°C: 75% fan
        (85.0, 100.0)   # At 85°C: 100% fan
    ]
    
    return StaticFanController(default_curve)
