"""
Unified interface for thermal surrogate models.

Defines the protocol that all surrogate adapters must implement.
"""

from typing import Protocol, Optional
import numpy as np


class ThermalSurrogate(Protocol):
    """
    Protocol for thermal dynamics surrogate models.
    
    All surrogate adapters (RC, RF, PINN) must implement this interface
    to be used interchangeably in the RL environment.
    """
    
    def reset(
        self,
        seed: Optional[int] = None,
        init_state: Optional[np.ndarray] = None
    ) -> None:
        """
        Reset the surrogate's internal state (e.g., history buffers).
        
        Args:
            seed: Random seed for reproducibility
            init_state: Initial state vector [temp, ambient, power, fan_speed, temp_delta]
        """
        ...
    
    def predict_next(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> float:
        """
        Predict the next GPU temperature given current state and action.
        
        Args:
            state: Current state vector [temp, ambient, power, fan_speed, temp_delta]
            action: Action vector [fan_speed]
        
        Returns:
            Predicted next GPU temperature (°C)
        """
        ...
    
    @property
    def warmup_steps(self) -> int:
        """
        Number of warmup steps required before predictions are valid.
        
        Returns:
            Number of steps needed for warmup (0 for stateless models)
        """
        ...
