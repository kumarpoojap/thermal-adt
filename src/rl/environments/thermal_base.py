"""
Thermal Control Gym Environment for RL-based GPU cooling.

This environment wraps the thermal surrogate model (PINN/RF/RC) and provides
a standard Gym interface for training RL agents to control fan speed.
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np

# Simple Space classes (minimal gym-like interface)
class Box:
    """Simple Box space for continuous values."""
    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.shape = shape or self.low.shape
        self.dtype = dtype
    
    def sample(self):
        """Sample random value from space."""
        if self.shape == (1,):
            return np.array([np.random.uniform(self.low, self.high)], dtype=self.dtype)
        else:
            return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)
    
    def contains(self, x):
        """Check if x is in space."""
        x = np.array(x)
        return np.all(x >= self.low) and np.all(x <= self.high)


class ThermalControlEnv:
    """
    GPU Thermal Control Environment.
    
    State Space:
        - gpu_temp: Current GPU temperature (°C)
        - ambient_temp: Ambient temperature (°C)
        - gpu_power: GPU power consumption (W)
        - fan_speed: Current fan speed (%)
        - temp_delta: Recent temperature change (°C/s)
    
    Action Space:
        - fan_speed: Target fan speed (%) in [20, 100]
    
    Reward:
        - Penalize thermal violations (temp > threshold)
        - Penalize energy consumption (high fan speed)
        - Penalize oscillations (rapid fan changes)
        - Reward thermal headroom (staying cool)
    
    Episode:
        - Length: 300 steps (5 minutes at 1s cadence)
        - Termination: Temperature exceeds critical threshold or episode ends
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(
        self,
        surrogate_model,
        config: Optional[Dict] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize thermal control environment.
        
        Args:
            surrogate_model: Trained surrogate (PINN, RF, or RC)
            config: Environment configuration
            render_mode: Rendering mode (None or "human")
        """
        self.surrogate = surrogate_model
        self.config = config or self._default_config()
        self.render_mode = render_mode
        
        # State space: [temp, ambient, power, fan_speed, temp_delta]
        self.observation_space = Box(
            low=np.array([30.0, 15.0, 50.0, 20.0, -10.0]),
            high=np.array([95.0, 35.0, 350.0, 100.0, 10.0]),
            dtype=np.float32
        )
        
        # Action space: fan_speed in [20, 100]%
        self.action_space = Box(
            low=20.0,
            high=100.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.max_steps = self.config["max_steps"]
        self.state = None
        self.prev_action = None
        self.episode_history = []
        
        # Thermal thresholds
        self.temp_warning = self.config["temp_warning"]  # 80°C
        self.temp_critical = self.config["temp_critical"]  # 90°C
        self.temp_target = self.config["temp_target"]  # 70°C
        
        # Reward weights
        self.w_thermal = self.config["w_thermal"]
        self.w_energy = self.config["w_energy"]
        self.w_oscillation = self.config["w_oscillation"]
        self.w_headroom = self.config["w_headroom"]
    
    def _default_config(self) -> Dict:
        """Default environment configuration."""
        return {
            "max_steps": 300,  # 5 minutes
            "temp_warning": 80.0,
            "temp_critical": 90.0,
            "temp_target": 70.0,
            "w_thermal": 10.0,
            "w_energy": 0.1,
            "w_oscillation": 1.0,
            "w_headroom": 0.5,
            "ambient_range": (20.0, 30.0),
            "power_range": (100.0, 300.0),
            "initial_temp_range": (50.0, 70.0),
            "dt": 1.0  # 1 second time step
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Returns:
            observation: Initial state
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()
        
        # Sample initial conditions
        initial_temp = self.np_random.uniform(*self.config["initial_temp_range"])
        ambient_temp = self.np_random.uniform(*self.config["ambient_range"])
        gpu_power = self.np_random.uniform(*self.config["power_range"])
        fan_speed = 50.0  # Start at 50%
        temp_delta = 0.0
        
        self.state = np.array([
            initial_temp,
            ambient_temp,
            gpu_power,
            fan_speed,
            temp_delta
        ], dtype=np.float32)
        
        self.current_step = 0
        self.prev_action = fan_speed
        self.episode_history = []
        
        info = {
            "initial_temp": initial_temp,
            "ambient_temp": ambient_temp,
            "gpu_power": gpu_power
        }
        
        return self.state, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step.
        
        Args:
            action: Fan speed (%) in [20, 100]
        
        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode ended due to critical condition
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        # Extract action
        fan_speed = float(np.clip(action[0], 20.0, 100.0))
        
        # Current state
        current_temp = self.state[0]
        ambient_temp = self.state[1]
        gpu_power = self.state[2]
        
        # Predict next temperature using surrogate
        next_temp = self._predict_next_temp(
            current_temp, ambient_temp, gpu_power, fan_speed
        )
        
        # Compute temperature delta
        temp_delta = next_temp - current_temp
        
        # Update state
        self.state = np.array([
            next_temp,
            ambient_temp,
            gpu_power,
            fan_speed,
            temp_delta
        ], dtype=np.float32)
        
        # Compute reward
        reward = self._compute_reward(
            next_temp, fan_speed, self.prev_action
        )
        
        # Check termination conditions
        terminated = next_temp >= self.temp_critical
        truncated = self.current_step >= self.max_steps - 1
        
        # Update episode state
        self.current_step += 1
        self.prev_action = fan_speed
        
        # Store history
        self.episode_history.append({
            "step": self.current_step,
            "temp": next_temp,
            "fan_speed": fan_speed,
            "reward": reward,
            "power": gpu_power
        })
        
        # Info
        info = {
            "temp": next_temp,
            "fan_speed": fan_speed,
            "temp_delta": temp_delta,
            "thermal_violation": next_temp > self.temp_warning,
            "critical_violation": terminated
        }
        
        return self.state, reward, terminated, truncated, info
    
    def _predict_next_temp(
        self,
        current_temp: float,
        ambient_temp: float,
        gpu_power: float,
        fan_speed: float
    ) -> float:
        """
        Predict next temperature using surrogate model.
        
        This is a simplified prediction - in reality you'd need to:
        1. Build feature vector with lags and rolling windows
        2. Normalize inputs
        3. Get prediction from surrogate
        4. Denormalize output
        
        For now, using a simple RC model approximation.
        """
        # Simple RC model: T(t+1) = T(t) + dt * [gamma*P - beta*Fan - h*(T - T_amb)] / C
        C = 100.0  # Thermal capacity
        h = 0.05   # Heat transfer coefficient
        beta = -0.03  # Cooling effectiveness
        gamma = 0.01  # Power-to-heat conversion
        dt = self.config["dt"]
        
        # Temperature dynamics
        heat_gen = gamma * gpu_power
        cooling = beta * fan_speed
        heat_transfer = -h * (current_temp - ambient_temp)
        
        dT_dt = (heat_gen + cooling + heat_transfer) / C
        next_temp = current_temp + dT_dt * dt
        
        # Clip to physical range
        next_temp = np.clip(next_temp, 30.0, 95.0)
        
        return float(next_temp)
    
    def _compute_reward(
        self,
        temp: float,
        fan_speed: float,
        prev_fan_speed: float
    ) -> float:
        """
        Compute reward for current state and action.
        
        Reward components:
        1. Thermal penalty: Penalize high temperatures
        2. Energy penalty: Penalize high fan speed
        3. Oscillation penalty: Penalize rapid fan changes
        4. Headroom bonus: Reward staying below target temp
        """
        # 1. Thermal penalty (exponential beyond warning threshold)
        if temp > self.temp_warning:
            thermal_penalty = self.w_thermal * ((temp - self.temp_warning) ** 2)
        else:
            thermal_penalty = 0.0
        
        # 2. Energy penalty (quadratic in fan speed)
        energy_penalty = self.w_energy * ((fan_speed / 100.0) ** 2)
        
        # 3. Oscillation penalty (penalize rapid changes)
        fan_delta = abs(fan_speed - prev_fan_speed)
        oscillation_penalty = self.w_oscillation * (fan_delta / 100.0)
        
        # 4. Headroom bonus (reward staying cool)
        if temp < self.temp_target:
            headroom_bonus = self.w_headroom * (self.temp_target - temp) / 10.0
        else:
            headroom_bonus = 0.0
        
        # Total reward
        reward = -thermal_penalty - energy_penalty - oscillation_penalty + headroom_bonus
        
        return float(reward)
    
    def render(self):
        """Render environment state (optional)."""
        if self.render_mode == "human":
            temp = self.state[0]
            fan = self.state[3]
            print(f"Step {self.current_step}: Temp={temp:.1f}°C, Fan={fan:.0f}%")
    
    def close(self):
        """Clean up environment resources."""
        pass
    
    def get_episode_metrics(self) -> Dict:
        """Get metrics for completed episode."""
        if not self.episode_history:
            return {}
        
        temps = [h["temp"] for h in self.episode_history]
        fan_speeds = [h["fan_speed"] for h in self.episode_history]
        rewards = [h["reward"] for h in self.episode_history]
        
        return {
            "mean_temp": np.mean(temps),
            "max_temp": np.max(temps),
            "min_temp": np.min(temps),
            "temp_std": np.std(temps),
            "mean_fan": np.mean(fan_speeds),
            "max_fan": np.max(fan_speeds),
            "total_reward": np.sum(rewards),
            "thermal_violations": sum(1 for t in temps if t > self.temp_warning),
            "critical_violations": sum(1 for t in temps if t > self.temp_critical),
            "episode_length": len(self.episode_history)
        }
