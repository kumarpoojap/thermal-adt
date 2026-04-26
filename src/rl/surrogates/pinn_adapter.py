"""
PINN (Physics-Informed Neural Network) surrogate adapter.

Wraps HybridPINN model and handles tensorization, device management,
normalization, and inference.
"""

from typing import Optional, Dict, List
from pathlib import Path
import json
import numpy as np
import torch
import pandas as pd

from src.pinn.models.hybrid_pinn import HybridPINN
from src.common.features import materialize_features_from_list


class PINNAdapter:
    """
    Adapter for PINN surrogate model.
    
    Handles:
    - Model loading and device management
    - Input tensorization and normalization
    - Inference and output denormalization
    - State tracking for delta prediction
    """
    
    def __init__(
        self,
        model_path: Path,
        device: Optional[str] = None,
        input_mean: Optional[np.ndarray] = None,
        input_std: Optional[np.ndarray] = None,
        output_mean: Optional[float] = None,
        output_std: Optional[float] = None,
        config: Optional[Dict] = None,
        feature_columns_path: Optional[Path] = None,
        scalers_path: Optional[Path] = None,
        cadence_seconds: float = 1.0,
        strict_features: bool = True,
    ):
        """
        Initialize PINN adapter.
        
        Args:
            model_path: Path to trained PINN checkpoint
            device: Device for inference ('cpu', 'cuda', or None for auto)
            input_mean: Mean for input normalization
            input_std: Std for input normalization
            output_mean: Mean for output denormalization
            output_std: Std for output denormalization
            config: Optional config dict with model architecture params
        """
        self.model_path = Path(model_path)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std

        self._cadence = pd.Timedelta(seconds=float(cadence_seconds))
        self._strict_features = bool(strict_features)
        self._feature_cols: Optional[List[str]] = self._load_feature_cols(feature_columns_path)
        self._history: Optional[pd.DataFrame] = None

        self._target_mean: Optional[float] = None
        self._target_std: Optional[float] = None
        self._load_target_scaler(scalers_path)

        self.model = self._load_model(config)
        self.model.eval()
        
        self._current_temp: Optional[float] = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        init_state: Optional[np.ndarray] = None
    ) -> None:
        """
        Reset internal state tracking.
        
        Args:
            seed: Random seed (unused for PINN)
            init_state: Initial state [temp, ambient, power, fan_speed, temp_delta]
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        if init_state is not None:
            self._current_temp = float(init_state[0])
        else:
            self._current_temp = None

        # Initialize feature history if we have a feature schema.
        if self._feature_cols is not None:
            warmup_len = self._infer_warmup_len(self._feature_cols)
            base_cols_needed = self._extract_base_columns(self._feature_cols)
            start = pd.Timestamp("2000-01-01")
            idx = pd.date_range(start=start, periods=warmup_len, freq=self._cadence)
            self._history = pd.DataFrame(0.0, index=idx, columns=list(base_cols_needed))

            if init_state is not None:
                for i in range(warmup_len):
                    for base_col in base_cols_needed:
                        self._history.iloc[i, self._history.columns.get_loc(base_col)] = self._base_value_from_state(
                            base_col, init_state, action=None
                        )
        else:
            self._history = None
    
    def predict_next(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> float:
        """
        Predict next GPU temperature using PINN model.
        
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
        
        features = self._build_feature_vector(
            current_temp, ambient_temp, gpu_power, fan_speed
        )
        
        x_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        if self.input_mean is not None and self.input_std is not None:
            mean_tensor = torch.from_numpy(self.input_mean).float().to(self.device)
            std_tensor = torch.from_numpy(self.input_std).float().to(self.device)
            x_tensor = (x_tensor - mean_tensor) / (std_tensor + 1e-8)
        
        with torch.no_grad():
            if self._target_mean is not None and self._target_std is not None:
                y_current_norm = (current_temp - self._target_mean) / (self._target_std + 1e-8)
                y_current = torch.tensor([[y_current_norm]], dtype=torch.float32, device=self.device)
                y_pred_norm = self.model.predict_absolute(x_tensor, y_current)
                next_temp_tensor = (y_pred_norm[0, 0] * self._target_std) + self._target_mean
            else:
                # Fallback: assume model predicts absolute deltas in °C.
                y_current = torch.tensor([[current_temp]], dtype=torch.float32, device=self.device)
                y_pred = self.model.predict_absolute(x_tensor, y_current)
                next_temp_tensor = y_pred[0, 0]
        
        if self.output_mean is not None and self.output_std is not None:
            next_temp_tensor = next_temp_tensor * self.output_std + self.output_mean
        
        next_temp = float(next_temp_tensor.cpu().item())
        
        next_temp = np.clip(next_temp, 30.0, 95.0)
        
        self._current_temp = next_temp
        
        return next_temp
    
    @property
    def warmup_steps(self) -> int:
        """PINN model is stateless, no warmup needed."""
        return 0

    def _load_feature_cols(self, feature_columns_path: Optional[Path]) -> Optional[List[str]]:
        if feature_columns_path is None:
            return None
        p = Path(feature_columns_path)
        with open(p, "r") as f:
            cols = json.load(f)
        if not isinstance(cols, list) or not cols:
            raise ValueError(f"Invalid feature columns file: {p}")
        return [str(c) for c in cols]

    def _load_target_scaler(self, scalers_path: Optional[Path]) -> None:
        if scalers_path is None:
            return
        p = Path(scalers_path)
        with open(p, "r") as f:
            obj = json.load(f)

        stats = obj.get("stats", {}) if isinstance(obj, dict) else {}
        # Prefer gpu_temp_c, else first stats entry.
        if "gpu_temp_c" in stats:
            s = stats["gpu_temp_c"]
        elif stats:
            first_key = next(iter(stats.keys()))
            s = stats[first_key]
        else:
            s = None

        if isinstance(s, dict) and "mean" in s and "std" in s:
            self._target_mean = float(s["mean"])
            self._target_std = float(s["std"])

    def _infer_warmup_len(self, feature_cols: List[str]) -> int:
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

    def _extract_base_columns(self, feature_cols: List[str]) -> set:
        base_cols = set()
        for col in feature_cols:
            if "_lag" in col:
                base_cols.add(col.rsplit("_lag", 1)[0])
            elif "_roll" in col:
                base_cols.add(col.split("_roll", 1)[0])
            else:
                base_cols.add(col)
        return base_cols

    def _base_value_from_state(
        self,
        base_col: str,
        state: np.ndarray,
        action: Optional[np.ndarray],
    ) -> float:
        temp, ambient, power, fan_speed, temp_delta = [float(x) for x in state[:5]]
        if action is not None:
            fan_speed = float(action[0])

        mapping = {
            "gpu_temp_current": temp,
            "gpu_temp_c": temp,
            "temp": temp,
            "temperature": temp,
            "ambient_temp_c": ambient,
            "ambient": ambient,
            "gpu_power_w": power,
            "power": power,
            "fan_speed_pct": fan_speed,
            "fan_speed": fan_speed,
            "temp_delta": temp_delta,
            "workload_intensity": 0.0,
            "workload_pct": 0.0,
        }
        return float(mapping.get(base_col, 0.0))
    
    def _load_model(self, config: Optional[Dict] = None) -> HybridPINN:
        """Load PINN model from checkpoint."""
        map_location = self.device
        if map_location.type == "cuda" and not torch.cuda.is_available():
            map_location = torch.device("cpu")

        try:
            checkpoint = torch.load(self.model_path, map_location=map_location)
        except RuntimeError:
            checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"))
            map_location = torch.device("cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            ckpt_cfg = checkpoint.get("config")
        elif isinstance(checkpoint, dict):
            # Sometimes the checkpoint is the state_dict itself
            state_dict = checkpoint
            ckpt_cfg = None
        else:
            raise ValueError("Unsupported checkpoint format for PINNAdapter")

        # Prefer explicit config passed in, otherwise use checkpoint config if present
        effective_cfg = config if config is not None else (ckpt_cfg or {})
        model_cfg = effective_cfg.get("model", {}) if isinstance(effective_cfg, dict) else {}

        # If no useful model config is present, infer architecture from state_dict
        if not model_cfg:
            model_cfg = self._infer_model_config_from_state_dict(state_dict)

        model = HybridPINN(
            input_dim=int(model_cfg.get("input_dim", 4)),
            output_dim=int(model_cfg.get("output_dim", 1)),
            hidden_dims=list(model_cfg.get("hidden_dims", [128, 128, 128])),
            activation=str(model_cfg.get("activation", "silu")),
            dropout=float(model_cfg.get("dropout", 0.0)),
            time_embedding_enabled=bool(model_cfg.get("time_embedding_enabled", False)),
            time_embedding_method=str(model_cfg.get("time_embedding_method", "fourier")),
            time_embedding_n_freqs=int(model_cfg.get("time_embedding_n_freqs", 16)),
            physics_head_enabled=bool(model_cfg.get("physics_head_enabled", False)),
        )

        model.load_state_dict(state_dict, strict=True)
        model.to(map_location)
        self.device = map_location
        return model

    def _infer_model_config_from_state_dict(self, state_dict: dict) -> Dict:
        """Infer a compatible HybridPINN config from a checkpoint state_dict."""
        # HybridPINN stores its MLP backbone as nn.Sequential with Linear + (Activation) + (Dropout)
        # repeating. That means Linear layers may appear at indices like 0,3,6,9 when dropout is used,
        # or 0,2,4,6 when dropout is not used. We must reproduce the same structure so state_dict keys match.
        linear_weight_keys = [
            k
            for k in state_dict.keys()
            if k.startswith("backbone.") and k.endswith(".weight") and k.split(".")[1].isdigit()
        ]
        if not linear_weight_keys:
            raise ValueError("Cannot infer PINN architecture: no backbone weights found in checkpoint")

        def layer_index(key: str) -> int:
            # backbone.{idx}.weight
            return int(key.split(".")[1])

        linear_weight_keys = sorted(linear_weight_keys, key=layer_index)
        linear_indices = [layer_index(k) for k in linear_weight_keys]

        first_w = state_dict[linear_weight_keys[0]]
        input_dim = int(first_w.shape[1])

        out_dims = [int(state_dict[k].shape[0]) for k in linear_weight_keys]
        output_dim = int(out_dims[-1])
        hidden_dims = out_dims[:-1]

        # Infer whether dropout existed by checking spacing between Linear layers.
        # diff==3 implies Linear, Activation, Dropout (3 modules per block)
        # diff==2 implies Linear, Activation (2 modules per block)
        diffs = [b - a for a, b in zip(linear_indices[:-1], linear_indices[1:])]
        has_dropout = bool(diffs) and all(d == 3 for d in diffs)
        dropout = 0.1 if has_dropout else 0.0

        # Heuristics: if time_emb parameters exist, time embedding was enabled.
        time_embedding_enabled = any(k.startswith("time_emb") for k in state_dict.keys())
        physics_head_enabled = any(k.startswith("physics_head") for k in state_dict.keys())

        # We cannot reliably infer activation; choose silu as default.
        return {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "activation": "silu",
            "dropout": dropout,
            "time_embedding_enabled": bool(time_embedding_enabled),
            "time_embedding_method": "fourier",
            "time_embedding_n_freqs": 16,
            "physics_head_enabled": bool(physics_head_enabled),
        }
    
    def _build_feature_vector(
        self,
        temp: float,
        ambient: float,
        power: float,
        fan_speed: float
    ) -> np.ndarray:
        """
        Build feature vector for PINN input.
        
        Assumes exogenous-only features: [ambient, power, fan_speed, ...]
        Adjust based on actual PINN training feature set.
        """
        if self._feature_cols is not None:
            if self._history is None:
                raise RuntimeError("PINNAdapter history is not initialized. Call reset() before predict_next().")

            state = np.array([temp, ambient, power, fan_speed, 0.0], dtype=np.float32)
            base_cols_needed = self._extract_base_columns(self._feature_cols)
            last_idx = self._history.index[-1]
            for base in base_cols_needed:
                if base in self._history.columns:
                    self._history.loc[last_idx, base] = self._base_value_from_state(base, state, action=None)
                else:
                    self._history[base] = 0.0
                    self._history.loc[last_idx, base] = self._base_value_from_state(base, state, action=None)

            X_full = materialize_features_from_list(self._history, self._feature_cols)
            x_last = X_full.iloc[-1]

            # Update rolling history with the new row for next time step
            new_row = {c: self._base_value_from_state(c, state, action=None) for c in self._history.columns}
            next_ts = self._history.index[-1] + self._cadence
            self._history = pd.concat(
                [
                    self._history.iloc[1:],
                    pd.DataFrame([new_row], index=[next_ts]),
                ],
                ignore_index=False,
            )

            return x_last.fillna(0.0).to_numpy(dtype=np.float32)

        # No feature schema provided: only allow padding fallback for smoke use.
        if self._strict_features:
            raise ValueError(
                "PINNAdapter requires 'feature_columns_path' for real inference (to match training feature schema)."
            )

        base = np.array([ambient, power, fan_speed, temp], dtype=np.float32)
        expected_dim = int(getattr(self.model, "input_dim", base.shape[0]))
        if expected_dim == base.shape[0]:
            return base

        x = np.zeros((expected_dim,), dtype=np.float32)
        n = min(expected_dim, base.shape[0])
        x[:n] = base[:n]
        return x
