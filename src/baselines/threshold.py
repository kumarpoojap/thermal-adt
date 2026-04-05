"""
Threshold Controller (Baseline).

Simple reactive controller: max cooling above threshold, low cooling below.
"""

import numpy as np


class ThresholdController:
    """
    Threshold-based controller.
    
    Simple bang-bang control:
    - If temp > threshold: fan = 100%
    - If temp <= threshold: fan = base_fan%
    
    Optionally adds hysteresis to prevent oscillation.
    """
    
    def __init__(
        self,
        threshold: float = 80.0,
        base_fan: float = 40.0,
        max_fan: float = 100.0,
        hysteresis: float = 2.0
    ):
        """
        Initialize threshold controller.
        
        Args:
            threshold: Temperature threshold (°C)
            base_fan: Fan speed when below threshold (%)
            max_fan: Fan speed when above threshold (%)
            hysteresis: Temperature hysteresis to prevent oscillation (°C)
        """
        self.threshold = threshold
        self.base_fan = base_fan
        self.max_fan = max_fan
        self.hysteresis = hysteresis
        
        # State
        self.high_mode = False
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict fan speed based on temperature threshold.
        
        Args:
            state: [temp, workload, power, ambient, fan_speed]
        
        Returns:
            action: Fan speed (%)
        """
        temp = state[0]
        
        # Threshold with hysteresis
        if not self.high_mode:
            # Currently in low mode
            if temp > self.threshold:
                self.high_mode = True
                fan_speed = self.max_fan
            else:
                fan_speed = self.base_fan
        else:
            # Currently in high mode
            if temp < (self.threshold - self.hysteresis):
                self.high_mode = False
                fan_speed = self.base_fan
            else:
                fan_speed = self.max_fan
        
        return np.array([fan_speed], dtype=np.float32)
    
    def reset(self):
        """Reset controller state."""
        self.high_mode = False


class AdaptiveThresholdController(ThresholdController):
    """
    Adaptive threshold controller.
    
    Adjusts threshold based on workload to be more proactive.
    """
    
    def __init__(
        self,
        base_threshold: float = 80.0,
        base_fan: float = 40.0,
        max_fan: float = 100.0,
        hysteresis: float = 2.0,
        workload_sensitivity: float = 0.1
    ):
        super().__init__(base_threshold, base_fan, max_fan, hysteresis)
        self.base_threshold = base_threshold
        self.workload_sensitivity = workload_sensitivity
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict with adaptive threshold based on workload."""
        temp, workload = state[0], state[1]
        
        # Adjust threshold based on workload
        # Higher workload -> lower threshold (more aggressive cooling)
        self.threshold = self.base_threshold - (workload * self.workload_sensitivity)
        
        return super().predict(state)
