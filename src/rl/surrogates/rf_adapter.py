"""
Random Forest surrogate adapter.

Wraps TeacherRF and handles all RF-specific feature materialization,
history buffer management, and lag/rolling window computations.
"""

from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd

from src.pinn.models.teacher_rf import TeacherRF
from src.common.features import materialize_features_from_list


class RFAdapter:
    """
    Adapter for Random Forest surrogate model.
    
    Handles:
    - Feature materialization from state history
    - Lag and rolling window features
    - Warmup period for feature computation
    - State-to-feature mapping
    """
    
    def __init__(
        self,
        model_path: Path,
        cache_dir: Optional[Path] = None,
        use_cache: bool = False,
        cadence_seconds: float = 1.0
    ):
        """
        Initialize RF adapter.
        
        Args:
            model_path: Path to trained RF model bundle
            cache_dir: Optional cache directory for predictions
            use_cache: Whether to use prediction caching
        """
        self.rf_model = TeacherRF(
            model_path=model_path,
            cache_dir=cache_dir,
            use_cache=use_cache
        )

        self._cadence = pd.Timedelta(seconds=float(cadence_seconds))
        
        self._warmup_steps = self._infer_warmup_len(self.rf_model.feature_cols)
        self._history: Optional[pd.DataFrame] = None
        self._step_count = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        init_state: Optional[np.ndarray] = None
    ) -> None:
        """
        Reset history buffer and step counter.
        
        Args:
            seed: Random seed (unused for RF)
            init_state: Initial state [temp, ambient, power, fan_speed, temp_delta]
        """
        if seed is not None:
            np.random.seed(seed)
        
        warmup_len = self._warmup_steps
        base_cols_needed = self._extract_base_columns(self.rf_model.feature_cols)

        # materialize_features_from_list() requires a DatetimeIndex (or timestamp col)
        start = pd.Timestamp("2000-01-01")
        idx = pd.date_range(start=start, periods=warmup_len, freq=self._cadence)
        self._history = pd.DataFrame(0.0, index=idx, columns=list(base_cols_needed))
        
        if init_state is not None:
            for i in range(warmup_len):
                for base_col in base_cols_needed:
                    self._history.iloc[i, self._history.columns.get_loc(base_col)] = self._base_value_from_state(
                        base_col, init_state
                    )
        
        self._step_count = 0
    
    def predict_next(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> float:
        """
        Predict next GPU temperature using RF model.
        
        Args:
            state: Current state [temp, ambient, power, fan_speed, temp_delta]
            action: Action [fan_speed]
        
        Returns:
            Predicted next GPU temperature (°C)
        """
        if self._history is None:
            raise RuntimeError("RFAdapter must be reset before prediction")
        
        state_with_action = self._merge_state_action(state, action)
        
        X_teacher = self._build_teacher_features_row(state_with_action)
        
        y_pred = self.rf_model.predict(X_teacher, return_tensor=False)
        
        next_temp = float(y_pred[0])
        
        self._append_to_history(state_with_action)
        self._step_count += 1
        
        return next_temp
    
    @property
    def warmup_steps(self) -> int:
        """Number of warmup steps needed for lag/rolling features."""
        return self._warmup_steps
    
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
                try:
                    after = c.split("_roll", 1)[1]
                    w_str = after.split("_", 1)[0]
                    max_roll = max(max_roll, int(w_str))
                except Exception:
                    pass
        return int(max(max_lag, max_roll) + 2)
    
    def _extract_base_columns(self, feature_cols) -> set:
        """Extract base column names from engineered feature names."""
        base_cols = set()
        for col in feature_cols:
            if "_lag" in col:
                base_cols.add(col.rsplit("_lag", 1)[0])
            elif "_roll" in col:
                base_cols.add(col.split("_roll", 1)[0])
            else:
                base_cols.add(col)
        return base_cols
    
    def _base_value_from_state(self, base_col: str, state: np.ndarray) -> float:
        """
        Map base column names to values from state vector.
        
        State format: [temp, ambient, power, fan_speed, temp_delta]
        """
        temp, ambient, power, fan_speed, temp_delta = [float(x) for x in state[:5]]
        
        mapping = {
            "gpu_temp_current": temp,
            "gpu_temp_c": temp,
            "temp": temp,
            "temperature": temp,
            "ambient_temp_c": ambient,
            "ambient_c": ambient,
            "ambient": ambient,
            "gpu_power_w": power,
            "power_w": power,
            "power": power,
            "fan_speed_pct": fan_speed,
            "fan_pct": fan_speed,
            "fan_speed": fan_speed,
            "temp_delta": temp_delta,
            "workload_pct": 0.0,
            "workload_intensity": 0.0,
        }
        
        return mapping.get(base_col, 0.0)
    
    def _merge_state_action(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> np.ndarray:
        """
        Merge state and action, updating fan_speed with action value.
        
        Args:
            state: [temp, ambient, power, fan_speed, temp_delta]
            action: [fan_speed]
        
        Returns:
            Updated state with action applied
        """
        state_copy = state.copy()
        state_copy[3] = float(action[0])
        return state_copy
    
    def _build_teacher_features_row(self, state: np.ndarray) -> pd.DataFrame:
        """Build feature row with lag/rolling features for RF model."""
        assert self._history is not None
        
        base_cols_needed = self._extract_base_columns(self.rf_model.feature_cols)
        
        last_idx = self._history.index[-1]
        for base in base_cols_needed:
            if base in self._history.columns:
                self._history.loc[last_idx, base] = self._base_value_from_state(base, state)
            else:
                self._history[base] = 0.0
                self._history.loc[last_idx, base] = self._base_value_from_state(base, state)
        
        X_full = materialize_features_from_list(self._history, self.rf_model.feature_cols)
        x_last = X_full.iloc[[-1]]
        
        x_last = x_last.fillna(0.0)
        return x_last
    
    def _append_to_history(self, state: np.ndarray) -> None:
        """Append current state to history buffer (rolling window)."""
        assert self._history is not None
        
        new_row = {}
        for col in self._history.columns:
            new_row[col] = self._base_value_from_state(col, state)

        next_ts = self._history.index[-1] + self._cadence
        self._history = pd.concat(
            [
                self._history.iloc[1:],
                pd.DataFrame([new_row], index=[next_ts]),
            ],
            ignore_index=False,
        )
