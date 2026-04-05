"""
Thermal Control Gym Environment using Random Forest Surrogate.

This environment uses the trained RF teacher model as the dynamics model
for model-based RL training. It follows the OpenAI Gym API.

State: [gpu_temp_current, workload_pct, power_w, ambient_temp_c, fan_speed_pct]
Action: fan_speed_pct (continuous, 20-100%)
Reward: Multi-objective (throttle risk, energy, smoothness, headroom)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import joblib


class TeacherRF:
    """Minimal wrapper to load and use RF teacher bundles."""
    
    def __init__(
        self,
        bundle_path: str | None = None,
        *,
        model_path: Path | str | None = None,
        cache_dir=None,
        use_cache: bool = False,
    ):
        # NOTE: cache_dir/use_cache are accepted for compatibility with older codepaths.
        _ = cache_dir
        _ = use_cache
        if bundle_path is None and model_path is None:
            raise ValueError("Either bundle_path or model_path must be provided")
        path = bundle_path if bundle_path is not None else model_path
        self.bundle = joblib.load(path)
        self.model = self.bundle["model"]
        self.feature_columns = self.bundle["feature_columns"]
        self.target_columns = self.bundle["target_columns"]
        # Backwards-compatible attribute names used elsewhere in the env
        self.feature_cols = self.feature_columns
        self.target_cols = self.target_columns
    
    def predict(self, X: pd.DataFrame, return_tensor: bool = False, **kwargs) -> np.ndarray:
        """Predict using the RF model.

        Args:
            X: DataFrame of features.
            return_tensor: Accepted for compatibility; this wrapper always returns numpy.
            kwargs: Ignored compatibility parameters.
        """
        _ = return_tensor
        _ = kwargs
        # Ensure columns match
        if not all(col in X.columns for col in self.feature_columns):
            raise ValueError(f"Missing features. Expected: {self.feature_columns}")
        
        # Select and order features correctly
        X_ordered = X[self.feature_columns]
        
        # Predict
        pred = self.model.predict(X_ordered)
        
        # Ensure (n, n_targets)
        if isinstance(pred, (list, tuple)):
            pred = np.asarray(pred)
        if getattr(pred, "ndim", 0) == 1:
            pred = np.asarray(pred).reshape(-1, 1)
        
        return pred


def materialize_features_from_list(
    history,
    feature_cols: list,
    lags: list | None = None,
) -> pd.DataFrame:
    """
    Build feature vector from history buffer with lags.
    
    Args:
        history: List of dicts with keys matching base feature columns
        feature_cols: Base feature column names (without lags)
        lags: List of lag values to include
    
    Returns:
        DataFrame with one row containing all features (base + lags)
    """
    if history is None:
        raise ValueError("History buffer is empty")

    # Allow history as DataFrame (preferred) or list[dict]
    if isinstance(history, pd.DataFrame):
        if history.empty:
            raise ValueError("History buffer is empty")
        rows = history.to_dict("records")
    else:
        rows = list(history)
        if len(rows) == 0:
            raise ValueError("History buffer is empty")

    # Infer required lags from feature_cols if not provided.
    # Expected convention: <base>_lagN
    if lags is None:
        import re as _re
        inferred = set()
        for c in feature_cols:
            m = _re.search(r"_lag(\d+)$", str(c))
            if m:
                inferred.add(int(m.group(1)))
        lags = sorted(inferred)

    # Determine base feature names (no lag suffix)
    base_features = set()
    for c in feature_cols:
        s = str(c)
        if "_lag" in s:
            base_features.add(s.rsplit("_lag", 1)[0])
        else:
            base_features.add(s)

    # Build one-row feature dict
    features: dict = {}

    # Current (most recent)
    current = rows[-1]
    for base in base_features:
        if base in current:
            features[base] = current[base]

    # Lagged
    for lag in sorted(lags):
        if lag >= len(rows):
            lag_idx = 0
        else:
            lag_idx = len(rows) - 1 - lag
        lag_row = rows[lag_idx]
        for base in base_features:
            if base in lag_row:
                features[f"{base}_lag{lag}"] = lag_row[base]

    # Ensure all requested feature_cols exist and are ordered
    out = {c: float(features.get(c, 0.0)) for c in feature_cols}
    return pd.DataFrame([out], columns=feature_cols)


class ThermalControlEnvRF(gym.Env):
    """
    Gym environment for thermal control using RF surrogate.
    
    The environment simulates GPU thermal dynamics using a trained
    Random Forest model that predicts temperature k steps ahead.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(
        self,
        rf_model_path: Path,
        config: Optional[Dict] = None,
        render_mode: Optional[str] = None,
        k_ahead: int = 10,
        cadence_seconds: float = 1.0
    ):
        """
        Initialize thermal control environment.
        
        Args:
            rf_model_path: Path to trained RF teacher model (.pkl)
            config: Environment configuration
            render_mode: Rendering mode
            k_ahead: Prediction horizon (steps)
            cadence_seconds: Time step duration (seconds)
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.k_ahead = k_ahead
        self.cadence_seconds = cadence_seconds
        self.dt = k_ahead * cadence_seconds  # Prediction horizon in seconds
        
        # Load configuration
        self.config = config or self._default_config()
        
        # Load RF surrogate model
        self.rf_model = self._load_rf_model(rf_model_path)

        # History buffer for feature materialization
        self._history: Optional[pd.DataFrame] = None
        self._warmup_len = self._infer_warmup_len(self.rf_model.feature_cols)
        self._step_dt = pd.Timedelta(seconds=float(self.cadence_seconds))
        
        # Define action and observation spaces
        # Action: fan_speed_pct [20, 100]
        self.action_space = spaces.Box(
            low=np.array([20.0], dtype=np.float32),
            high=np.array([100.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation: [temp, workload, power, ambient, fan_speed]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 15.0, 20.0], dtype=np.float32),
            high=np.array([100.0, 100.0, 500.0, 40.0, 100.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # State variables
        self.state = None
        self.prev_action = None
        self.timestep = 0
        self.episode_length = self.config["episode_length"]
        
        # Episode statistics
        self.episode_stats = {
            "throttle_events": 0,
            "total_energy": 0.0,
            "max_temp": 0.0,
            "min_headroom": float('inf'),
            "action_changes": []
        }
        
        # Workload generator (for curriculum learning)
        self.workload_generator = None
        self.set_workload_profile("steady")

    def _infer_warmup_len(self, feature_cols) -> int:
        """Infer warmup length needed to avoid NaNs from lag/rolling features."""
        max_lag = 0
        max_roll = 0
        for c in feature_cols:
            if "_lag" in c:
                try:
                    max_lag = max(max_lag, int(c.rsplit("_lag", 1)[1]))
                except Exception:
                    pass
            if "_roll" in c:
                # format: base_roll{w}_{kind}
                try:
                    after = c.split("_roll", 1)[1]
                    w_str = after.split("_", 1)[0]
                    max_roll = max(max_roll, int(w_str))
                except Exception:
                    pass
        # +2 guard to ensure rolling std has enough points
        return int(max(max_lag, max_roll) + 2)

    def _base_value_from_state(self, base_col: str, state: np.ndarray) -> float:
        """Map required base columns to values from the compact state."""
        temp, workload_pct, power_w, ambient_c, fan_pct = [float(x) for x in state]

        mapping = {
            "gpu_temp_current": temp,
            "gpu_temp_c": temp,
            "workload_pct": workload_pct,
            "workload_intensity": workload_pct / 100.0,
            "gpu_power_w": power_w,
            "power_w": power_w,
            "ambient_temp_c": ambient_c,
            "ambient_c": ambient_c,
            "fan_speed_pct": fan_pct,
            "fan_pct": fan_pct,
        }

        if base_col in mapping:
            return float(mapping[base_col])

        # Unknown base feature: default to 0.0 to satisfy column requirements.
        return 0.0

    def _build_teacher_features_row(self, state: np.ndarray) -> pd.DataFrame:
        """Build a single-row DataFrame with exactly the columns expected by the teacher."""
        assert self._history is not None

        # Ensure the most recent history row reflects current state
        base_cols_needed = set()
        for col in self.rf_model.feature_cols:
            if "_lag" in col:
                base_cols_needed.add(col.rsplit("_lag", 1)[0])
            elif "_roll" in col:
                base_cols_needed.add(col.split("_roll", 1)[0])
            else:
                base_cols_needed.add(col)

        # Update last row with base values
        last_idx = self._history.index[-1]
        for base in base_cols_needed:
            if base in self._history.columns:
                self._history.loc[last_idx, base] = self._base_value_from_state(base, state)
            else:
                self._history[base] = 0.0
                self._history.loc[last_idx, base] = self._base_value_from_state(base, state)

        # Materialize engineered columns from teacher feature list
        X_full = materialize_features_from_list(self._history, self.rf_model.feature_cols)
        x_last = X_full.iloc[[-1]]
        
        # If any NaNs remain (shouldn't after warmup), fill with zeros as last resort
        x_last = x_last.fillna(0.0)
        return x_last
    
    def _default_config(self) -> Dict:
        """Default environment configuration."""
        return {
            "episode_length": 300,  # 300 steps = 5 minutes at 1s cadence
            "temp_throttle": 85.0,  # Throttling threshold (°C)
            "temp_target": 75.0,    # Target operating temperature (°C)
            "temp_safe": 80.0,      # Safe zone upper bound (°C)
            "ambient_mean": 25.0,   # Mean ambient temperature (°C)
            "ambient_std": 2.0,     # Ambient temperature variation (°C)
            "power_base": 200.0,    # Base power consumption (W)
            "power_per_workload": 2.0,  # Power increase per % workload (W)
            "reward_weights": {
                "throttle_risk": 10.0,
                "energy": 0.1,
                "oscillation": 1.0,
                "headroom": 2.0
            }
        }
    
    def _load_rf_model(self, model_path: Path) -> TeacherRF:
        """Load trained RF teacher model."""
        return TeacherRF(model_path=model_path, cache_dir=None, use_cache=False)
    
    def set_workload_profile(self, profile: str):
        """
        Set workload generation profile for curriculum learning.
        
        Args:
            profile: One of ["steady", "moderate", "bursty", "stress"]
        """
        if profile == "steady":
            # Low, steady workload (20-40%)
            self.workload_generator = lambda: np.random.uniform(20, 40)
        elif profile == "moderate":
            # Moderate workload with some variation (30-60%)
            self.workload_generator = lambda: np.random.uniform(30, 60)
        elif profile == "bursty":
            # Bursty workload (20-80% with spikes)
            def bursty():
                if np.random.random() < 0.2:  # 20% chance of spike
                    return np.random.uniform(70, 90)
                else:
                    return np.random.uniform(20, 50)
            self.workload_generator = bursty
        elif profile == "stress":
            # High stress (50-100% with frequent spikes)
            def stress():
                if np.random.random() < 0.4:  # 40% chance of spike
                    return np.random.uniform(80, 100)
                else:
                    return np.random.uniform(50, 80)
            self.workload_generator = stress
        else:
            raise ValueError(f"Unknown workload profile: {profile}")
    
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
        super().reset(seed=seed)
        
        # Initialize state: [temp, workload, power, ambient, fan_speed]
        initial_temp = np.random.uniform(60.0, 70.0)
        initial_workload = self.workload_generator()
        initial_ambient = np.random.normal(
            self.config["ambient_mean"],
            self.config["ambient_std"]
        )
        initial_fan = 50.0  # Start at 50% fan speed
        initial_power = (
            self.config["power_base"] +
            initial_workload * self.config["power_per_workload"]
        )
        
        self.state = np.array([
            initial_temp,
            initial_workload,
            initial_power,
            initial_ambient,
            initial_fan
        ], dtype=np.float32)

        # Initialize history buffer with warmup rows
        start = pd.Timestamp("2026-01-01")
        idx = pd.date_range(start=start, periods=self._warmup_len, freq=self._step_dt)
        # Create columns for all base features that might be needed
        base_cols_needed = set()
        for col in self.rf_model.feature_cols:
            if "_lag" in col:
                base_cols_needed.add(col.rsplit("_lag", 1)[0])
            elif "_roll" in col:
                base_cols_needed.add(col.split("_roll", 1)[0])
            else:
                base_cols_needed.add(col)

        history = pd.DataFrame(index=idx)
        for base in base_cols_needed:
            history[base] = self._base_value_from_state(base, self.state)

        # Add tiny noise to avoid zero-variance issues in rolling std
        for base in history.columns:
            history[base] = history[base].astype(float) + np.random.normal(0.0, 1e-3, size=len(history))

        self._history = history
        
        self.prev_action = initial_fan
        self.timestep = 0
        
        # Reset episode statistics
        self.episode_stats = {
            "throttle_events": 0,
            "total_energy": 0.0,
            "max_temp": initial_temp,
            "min_headroom": self.config["temp_throttle"] - initial_temp,
            "action_changes": []
        }
        
        info = {"initial_state": self.state.copy()}
        
        return self.state.copy(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step.
        
        Args:
            action: Fan speed (%) [20, 100]
        
        Returns:
            observation: Next state
            reward: Reward signal
            terminated: Whether episode ended (throttle event)
            truncated: Whether episode was cut off (max steps)
            info: Additional information
        """
        # Extract current state
        current_temp, workload, power, ambient, _ = self.state
        fan_speed = float(action[0])

        # Update compact state with the chosen action for feature generation
        state_for_features = np.array([
            float(current_temp),
            float(workload),
            float(power),
            float(ambient),
            float(fan_speed),
        ], dtype=np.float32)
        
        # Predict next temperature using RF surrogate with engineered features
        # Advance history index by one step
        assert self._history is not None
        next_idx = self._history.index[-1] + self._step_dt
        self._history = pd.concat(
            [self._history, self._history.iloc[[-1]].copy().set_index(pd.Index([next_idx]))],
            axis=0,
        )
        # Keep history bounded
        if len(self._history) > (self._warmup_len + 50):
            self._history = self._history.iloc[-(self._warmup_len + 50):]

        X_teacher = self._build_teacher_features_row(state_for_features)
        y_pred = self.rf_model.predict(X_teacher, return_tensor=False)
        y_arr = np.asarray(y_pred)
        if y_arr.ndim == 0:
            next_temp_pred = float(y_arr)
        elif y_arr.ndim == 1:
            next_temp_pred = float(y_arr[0])
        else:
            next_temp_pred = float(y_arr[0, 0])
        
        # Generate next workload and ambient (environment dynamics)
        next_workload = self.workload_generator()
        next_ambient = np.random.normal(
            self.config["ambient_mean"],
            self.config["ambient_std"]
        )
        next_power = (
            self.config["power_base"] +
            next_workload * self.config["power_per_workload"]
        )
        
        # Update state
        next_state = np.array([
            next_temp_pred,
            next_workload,
            next_power,
            next_ambient,
            fan_speed
        ], dtype=np.float32)
        
        # Compute reward
        reward, reward_components = self._compute_reward(
            current_temp=current_temp,
            next_temp=next_temp_pred,
            fan_speed=fan_speed,
            prev_fan_speed=self.prev_action
        )
        
        # Update episode statistics
        self.episode_stats["total_energy"] += (fan_speed / 100.0) * self.dt
        self.episode_stats["max_temp"] = max(self.episode_stats["max_temp"], next_temp_pred)
        headroom = self.config["temp_throttle"] - next_temp_pred
        self.episode_stats["min_headroom"] = min(self.episode_stats["min_headroom"], headroom)
        if self.prev_action is not None:
            self.episode_stats["action_changes"].append(abs(fan_speed - self.prev_action))
        
        # Check termination conditions
        terminated = False
        if next_temp_pred >= self.config["temp_throttle"]:
            self.episode_stats["throttle_events"] += 1
            terminated = True  # Episode ends on throttle event
        
        truncated = False
        self.timestep += 1
        if self.timestep >= self.episode_length:
            truncated = True
        
        # Update state and previous action
        self.state = next_state
        self.prev_action = fan_speed
        
        # Info dictionary
        info = {
            "timestep": self.timestep,
            "temperature": float(next_temp_pred),
            "workload": float(next_workload),
            "fan_speed": float(fan_speed),
            "headroom": float(headroom),
            "reward_components": reward_components,
            "throttled": terminated
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _compute_reward(
        self,
        current_temp: float,
        next_temp: float,
        fan_speed: float,
        prev_fan_speed: Optional[float]
    ) -> Tuple[float, Dict]:
        """
        Compute multi-objective reward.
        
        Components:
        1. Throttle risk: Penalize approaching throttle threshold
        2. Energy cost: Penalize high fan usage
        3. Oscillation: Penalize rapid fan speed changes
        4. Headroom bonus: Reward maintaining thermal headroom
        
        Returns:
            reward: Total reward
            components: Dictionary of reward components
        """
        weights = self.config["reward_weights"]
        
        # 1. Throttle risk (exponential penalty as temp approaches threshold)
        temp_margin = self.config["temp_throttle"] - next_temp
        if temp_margin < 0:
            # Already throttling - large penalty
            throttle_risk = -100.0
        elif temp_margin < 5.0:
            # Danger zone (within 5°C of throttle)
            throttle_risk = -np.exp(5.0 - temp_margin)
        else:
            # Safe zone
            throttle_risk = 0.0
        
        # 2. Energy cost (linear penalty for fan usage)
        energy_cost = -(fan_speed / 100.0)
        
        # 3. Oscillation penalty (penalize rapid changes)
        if prev_fan_speed is not None:
            fan_delta = abs(fan_speed - prev_fan_speed)
            oscillation = -fan_delta / 10.0  # Normalize by max reasonable change
        else:
            oscillation = 0.0
        
        # 4. Headroom bonus (reward staying in safe zone)
        target_temp = self.config["temp_target"]
        safe_temp = self.config["temp_safe"]
        
        if next_temp <= target_temp:
            # Ideal zone
            headroom_bonus = 1.0
        elif next_temp <= safe_temp:
            # Safe zone
            headroom_bonus = 0.5
        else:
            # Approaching danger
            headroom_bonus = 0.0
        
        # Weighted sum
        reward = (
            weights["throttle_risk"] * throttle_risk +
            weights["energy"] * energy_cost +
            weights["oscillation"] * oscillation +
            weights["headroom"] * headroom_bonus
        )
        
        components = {
            "throttle_risk": throttle_risk,
            "energy_cost": energy_cost,
            "oscillation": oscillation,
            "headroom_bonus": headroom_bonus,
            "total": reward
        }
        
        return reward, components
    
    def render(self):
        """Render environment state."""
        if self.render_mode == "human":
            temp, workload, power, ambient, fan = self.state
            print(f"\n=== Timestep {self.timestep} ===")
            print(f"Temperature: {temp:.1f}°C")
            print(f"Workload: {workload:.1f}%")
            print(f"Power: {power:.1f}W")
            print(f"Ambient: {ambient:.1f}°C")
            print(f"Fan Speed: {fan:.1f}%")
            print(f"Headroom: {self.config['temp_throttle'] - temp:.1f}°C")
    
    def close(self):
        """Cleanup."""
        pass
    
    def get_episode_metrics(self) -> Dict:
        """Get episode-level metrics."""
        metrics = self.episode_stats.copy()
        
        if len(metrics["action_changes"]) > 0:
            metrics["mean_action_change"] = np.mean(metrics["action_changes"])
            metrics["action_oscillation"] = np.std(metrics["action_changes"])
        else:
            metrics["mean_action_change"] = 0.0
            metrics["action_oscillation"] = 0.0
        
        del metrics["action_changes"]  # Remove raw list
        
        return metrics


def make_thermal_env(
    rf_model_path: Path,
    config: Optional[Dict] = None,
    workload_profile: str = "steady",
    render_mode: Optional[str] = None
) -> ThermalControlEnvRF:
    """
    Factory function to create thermal control environment.
    
    Args:
        rf_model_path: Path to RF teacher model
        config: Environment configuration
        workload_profile: Workload generation profile
        render_mode: Rendering mode
    
    Returns:
        ThermalControlEnvRF instance
    """
    env = ThermalControlEnvRF(
        rf_model_path=rf_model_path,
        config=config,
        render_mode=render_mode
    )
    env.set_workload_profile(workload_profile)
    return env
