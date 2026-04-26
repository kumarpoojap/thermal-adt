"""
Unified Thermal Control Gym Environment using surrogate interface.

This environment accepts any surrogate adapter (RC, RF, PINN) conforming
to the ThermalSurrogate protocol, enabling seamless switching between
different thermal models for RL training and evaluation.
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.rl.surrogates.interface import ThermalSurrogate


class ThermalControlEnv(gym.Env):
    """
    GPU Thermal Control Environment with unified surrogate interface.
    
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
        - Length: Configurable (default 300 steps = 5 minutes at 1s cadence)
        - Termination: Temperature exceeds critical threshold or episode ends
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(
        self,
        surrogate: ThermalSurrogate,
        config: Optional[Dict] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize thermal control environment.
        
        Args:
            surrogate: Surrogate model conforming to ThermalSurrogate protocol
            config: Environment configuration
            render_mode: Rendering mode (None or "human")
        """
        super().__init__()
        
        self.surrogate = surrogate
        self.config = config or self._default_config()
        self.render_mode = render_mode
        
        self.observation_space = spaces.Box(
            low=np.array([30.0, 15.0, 50.0, 20.0, -10.0], dtype=np.float32),
            high=np.array([95.0, 35.0, 350.0, 100.0, 10.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=20.0,
            high=100.0,
            shape=(1,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = self.config["max_steps"]
        self.state = None
        self.prev_action = None
        self.episode_history = []
        
        self.temp_warning = self.config["temp_warning"]
        self.temp_critical = self.config["temp_critical"]
        self.temp_target = self.config["temp_target"]
        
        self.w_thermal = self.config["reward_weights"]["thermal"]
        self.w_energy = self.config["reward_weights"]["energy"]
        self.w_oscillation = self.config["reward_weights"]["oscillation"]
        self.w_headroom = self.config["reward_weights"]["headroom"]
        
        self.np_random = None
    
    def _default_config(self) -> Dict:
        """Default environment configuration."""
        return {
            "max_steps": 300,
            "temp_warning": 80.0,
            "temp_critical": 90.0,
            "temp_target": 75.0,
            "initial_temp_range": (40.0, 60.0),
            "ambient_range": (20.0, 30.0),
            "power_range": (100.0, 300.0),
            "reward_weights": {
                "thermal": 10.0,
                "energy": 0.1,
                "oscillation": 1.0,
                "headroom": 2.0
            }
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
        
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()
        
        initial_temp = self.np_random.uniform(*self.config["initial_temp_range"])
        ambient_temp = self.np_random.uniform(*self.config["ambient_range"])
        gpu_power = self.np_random.uniform(*self.config["power_range"])
        fan_speed = 50.0
        temp_delta = 0.0
        
        self.state = np.array([
            initial_temp,
            ambient_temp,
            gpu_power,
            fan_speed,
            temp_delta
        ], dtype=np.float32)
        
        self.surrogate.reset(seed=seed, init_state=self.state)
        
        self.current_step = 0
        self.prev_action = fan_speed
        self.episode_history = []
        
        info = {
            "initial_temp": initial_temp,
            "ambient_temp": ambient_temp,
            "gpu_power": gpu_power,
            "warmup_steps": self.surrogate.warmup_steps
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
        fan_speed = float(np.clip(action[0], 20.0, 100.0))
        
        current_temp = self.state[0]
        
        action_vec = np.array([fan_speed], dtype=np.float32)
        next_temp = self.surrogate.predict_next(self.state, action_vec)
        
        temp_delta = next_temp - current_temp
        
        self.state = np.array([
            next_temp,
            self.state[1],
            self.state[2],
            fan_speed,
            temp_delta
        ], dtype=np.float32)
        
        reward = self._compute_reward(next_temp, fan_speed, self.prev_action)
        
        terminated = next_temp >= self.temp_critical
        truncated = self.current_step >= self.max_steps - 1
        
        self.current_step += 1
        self.prev_action = fan_speed
        
        self.episode_history.append({
            "step": self.current_step,
            "temp": next_temp,
            "fan_speed": fan_speed,
            "reward": reward,
            "power": self.state[2]
        })
        
        info = {
            "temp": next_temp,
            "fan_speed": fan_speed,
            "temp_delta": temp_delta,
            "thermal_violation": next_temp > self.temp_warning,
            "critical_violation": terminated
        }
        
        return self.state, reward, terminated, truncated, info
    
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
        if temp > self.temp_warning:
            thermal_penalty = self.w_thermal * ((temp - self.temp_warning) ** 2)
        else:
            thermal_penalty = 0.0
        
        energy_penalty = self.w_energy * ((fan_speed / 100.0) ** 2)
        
        fan_delta = abs(fan_speed - prev_fan_speed)
        oscillation_penalty = self.w_oscillation * (fan_delta / 100.0)
        
        if temp < self.temp_target:
            headroom_bonus = self.w_headroom * (self.temp_target - temp) / 10.0
        else:
            headroom_bonus = 0.0
        
        reward = -thermal_penalty - energy_penalty - oscillation_penalty + headroom_bonus
        
        return float(reward)
    
    def render(self):
        """Render environment state."""
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
