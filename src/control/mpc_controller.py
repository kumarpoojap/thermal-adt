"""
Model Predictive Control (MPC) for Thermal Management.

Lightweight MPC controller that optimizes fan speed to track temperature target
while minimizing fan effort and rate-of-change, subject to safety constraints.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.optimize import minimize


class MPCController:
    """
    Model Predictive Control for thermal management.
    
    Objective:
    - Track temperature target (minimize deviation)
    - Minimize fan effort (energy consumption)
    - Minimize fan speed rate-of-change (smooth control)
    
    Constraints:
    - Fan speed bounds [fan_min, fan_max]
    - Maximum fan speed change per step
    - Temperature safety limits
    """
    
    def __init__(
        self,
        surrogate,
        horizon: int = 10,
        temp_target: float = 75.0,
        temp_max: float = 85.0,
        fan_min: float = 20.0,
        fan_max: float = 100.0,
        max_fan_delta: float = 20.0,
        weight_temp: float = 10.0,
        weight_effort: float = 0.1,
        weight_rate: float = 1.0,
        dt: float = 1.0,
        config: Optional[Dict] = None
    ):
        """
        Initialize MPC controller.
        
        Args:
            surrogate: Thermal surrogate model (RC, RF, or PINN adapter)
            horizon: Prediction horizon (number of steps)
            temp_target: Target temperature (°C)
            temp_max: Maximum allowed temperature (°C)
            fan_min: Minimum fan speed (%)
            fan_max: Maximum fan speed (%)
            max_fan_delta: Maximum fan speed change per step (%)
            weight_temp: Weight for temperature tracking error
            weight_effort: Weight for fan effort penalty
            weight_rate: Weight for fan rate-of-change penalty
            dt: Time step (seconds)
            config: Optional config dict to override defaults
        """
        self.surrogate = surrogate
        
        if config is not None:
            self.horizon = config.get("horizon", horizon)
            self.temp_target = config.get("temp_target", temp_target)
            self.temp_max = config.get("temp_max", temp_max)
            self.fan_min = config.get("fan_min", fan_min)
            self.fan_max = config.get("fan_max", fan_max)
            self.max_fan_delta = config.get("max_fan_delta", max_fan_delta)
            self.weight_temp = config.get("weight_temp", weight_temp)
            self.weight_effort = config.get("weight_effort", weight_effort)
            self.weight_rate = config.get("weight_rate", weight_rate)
            self.dt = config.get("dt", dt)
        else:
            self.horizon = horizon
            self.temp_target = temp_target
            self.temp_max = temp_max
            self.fan_min = fan_min
            self.fan_max = fan_max
            self.max_fan_delta = max_fan_delta
            self.weight_temp = weight_temp
            self.weight_effort = weight_effort
            self.weight_rate = weight_rate
            self.dt = dt
        
        # Previous action for rate limiting
        self.prev_action = None
        
        # Statistics
        self.stats = {
            "total_steps": 0,
            "avg_temp_error": 0.0,
            "avg_fan_effort": 0.0,
            "avg_fan_delta": 0.0,
            "constraint_violations": 0
        }
        
        # Running sums for statistics
        self._sum_temp_error = 0.0
        self._sum_fan_effort = 0.0
        self._sum_fan_delta = 0.0
    
    def reset(self, seed: Optional[int] = None):
        """
        Reset controller state.
        
        Args:
            seed: Random seed (unused)
        """
        self.prev_action = None
        self.stats = {
            "total_steps": 0,
            "avg_temp_error": 0.0,
            "avg_fan_effort": 0.0,
            "avg_fan_delta": 0.0,
            "constraint_violations": 0
        }
        self._sum_temp_error = 0.0
        self._sum_fan_effort = 0.0
        self._sum_fan_delta = 0.0
    
    def compute_action(self, state: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Compute optimal fan speed using MPC.
        
        Args:
            state: Current state [temp, ambient, power, fan_speed, temp_delta]
        
        Returns:
            action: Optimal fan speed [fan_speed]
            info: Information about optimization
        """
        # Initial guess: maintain current fan speed or use middle of range
        if self.prev_action is not None:
            u0 = np.full(self.horizon, self.prev_action)
        else:
            u0 = np.full(self.horizon, (self.fan_min + self.fan_max) / 2)
        
        # Bounds for each control input
        bounds = [(self.fan_min, self.fan_max)] * self.horizon
        
        # Constraints
        constraints = []
        
        # Rate limiting constraint
        if self.prev_action is not None:
            # First action must respect rate limit from previous action
            constraints.append({
                'type': 'ineq',
                'fun': lambda u: self.max_fan_delta - abs(u[0] - self.prev_action)
            })
        
        # Consecutive action rate limits
        for i in range(self.horizon - 1):
            constraints.append({
                'type': 'ineq',
                'fun': lambda u, idx=i: self.max_fan_delta - abs(u[idx+1] - u[idx])
            })
        
        # Temperature safety constraint (soft - handled in objective)
        # Hard constraint would make problem infeasible in some cases
        
        # Optimize
        result = minimize(
            fun=self._objective,
            x0=u0,
            args=(state,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        # Extract first action (receding horizon)
        optimal_action = result.x[0]
        
        # Clip to ensure bounds (numerical safety)
        optimal_action = np.clip(optimal_action, self.fan_min, self.fan_max)
        
        # Update statistics
        self.stats["total_steps"] += 1
        temp_error = abs(state[0] - self.temp_target)
        self._sum_temp_error += temp_error
        self._sum_fan_effort += optimal_action
        
        if self.prev_action is not None:
            fan_delta = abs(optimal_action - self.prev_action)
            self._sum_fan_delta += fan_delta
        
        self.stats["avg_temp_error"] = self._sum_temp_error / self.stats["total_steps"]
        self.stats["avg_fan_effort"] = self._sum_fan_effort / self.stats["total_steps"]
        if self.stats["total_steps"] > 1:
            self.stats["avg_fan_delta"] = self._sum_fan_delta / (self.stats["total_steps"] - 1)
        
        # Check for constraint violations
        if state[0] > self.temp_max:
            self.stats["constraint_violations"] += 1
        
        # Store for next step
        self.prev_action = optimal_action
        
        # Info
        info = {
            "optimal_action": optimal_action,
            "optimization_success": result.success,
            "optimization_cost": result.fun,
            "n_iterations": result.nit,
            "temp_error": temp_error,
            "fan_delta": abs(optimal_action - self.prev_action) if self.prev_action is not None else 0.0
        }
        
        return np.array([optimal_action], dtype=np.float32), info
    
    def _objective(self, u: np.ndarray, state: np.ndarray) -> float:
        """
        MPC objective function.
        
        Minimizes:
        - Temperature tracking error
        - Fan effort (energy)
        - Fan rate-of-change (smoothness)
        
        Args:
            u: Control sequence [u_0, u_1, ..., u_{H-1}]
            state: Current state
        
        Returns:
            cost: Total cost
        """
        cost = 0.0
        current_state = state.copy()
        
        for k in range(self.horizon):
            # Predict next temperature
            action = np.array([u[k]], dtype=np.float32)
            next_temp = self.surrogate.predict_next(current_state, action)
            
            # Temperature tracking error
            temp_error = (next_temp - self.temp_target) ** 2
            cost += self.weight_temp * temp_error
            
            # Fan effort penalty
            fan_effort = (u[k] / 100.0) ** 2  # Normalize to [0, 1]
            cost += self.weight_effort * fan_effort
            
            # Fan rate-of-change penalty
            if k == 0 and self.prev_action is not None:
                fan_delta = ((u[k] - self.prev_action) / 100.0) ** 2
            elif k > 0:
                fan_delta = ((u[k] - u[k-1]) / 100.0) ** 2
            else:
                fan_delta = 0.0
            cost += self.weight_rate * fan_delta
            
            # Soft temperature constraint (penalty for exceeding max)
            if next_temp > self.temp_max:
                temp_violation = (next_temp - self.temp_max) ** 2
                cost += 100.0 * temp_violation  # Large penalty
            
            # Update state for next prediction
            # State: [temp, ambient, power, fan_speed, temp_delta]
            temp_delta = next_temp - current_state[0]
            current_state = np.array([
                next_temp,
                current_state[1],  # ambient (assumed constant)
                current_state[2],  # power (assumed constant)
                u[k],              # fan speed
                temp_delta
            ], dtype=np.float32)
        
        return cost
    
    def get_stats(self) -> Dict:
        """Get controller statistics."""
        return self.stats.copy()
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """
        Convenience method for calling compute_action.
        
        Args:
            state: Current state
        
        Returns:
            action: Optimal fan speed
        """
        action, _ = self.compute_action(state)
        return action
