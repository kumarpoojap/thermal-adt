"""
RC+NN (hybrid) surrogate adapter for RL.

Wraps an RCAdapter with a small neural network that predicts a residual
correction. The adapter conforms to the ThermalSurrogate interface used by
ThermalControlEnv.
"""
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import joblib

from .interface import ThermalSurrogate
from .rc_adapter import RCAdapter


class ResidualNN(nn.Module):
    """Simple MLP predicting residual given [temp, ambient, power, fan].

    Note: Attribute name is 'network' to match the training script's saved state_dict keys.
    """
    def __init__(self, input_dim: int = 4, hidden_dims = [32, 16], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RCNNAdapter(ThermalSurrogate):
    """
    Hybrid RC+NN surrogate.

    Either construct from components (rc_adapter + nn_model + normalization)
    or from a serialized bundle saved by scripts/training/train_rc_nn.py.
    """
    def __init__(
        self,
        rc_adapter: Optional[RCAdapter] = None,
        nn_model: Optional[nn.Module] = None,
        device: str = "cpu",
        input_mean: Optional[np.ndarray] = None,
        input_std: Optional[np.ndarray] = None,
        bundle_path: Optional[Path] = None,
    ) -> None:
        if bundle_path is not None:
            bundle = joblib.load(bundle_path)
            self.rc = RCAdapter(**bundle["rc_params"])  # type: ignore[arg-type]
            nn_cfg = bundle["nn_config"]
            model = ResidualNN(input_dim=nn_cfg.get("input_dim", 4), hidden_dims=nn_cfg.get("hidden_dims", [32, 16]))
            model.load_state_dict(bundle["nn_state_dict"])  # type: ignore[arg-type]
            self.nn = model
            self.input_mean = np.asarray(bundle.get("input_mean", None)) if bundle.get("input_mean", None) is not None else None
            self.input_std = np.asarray(bundle.get("input_std", None)) if bundle.get("input_std", None) is not None else None
        else:
            assert rc_adapter is not None and nn_model is not None, "Provide rc_adapter+nn_model or bundle_path"
            self.rc = rc_adapter
            self.nn = nn_model
            self.input_mean = None if input_mean is None else np.asarray(input_mean)
            self.input_std = None if input_std is None else np.asarray(input_std)

        self.device = device
        self.nn.to(self.device)
        self.nn.eval()

    def reset(self, seed: Optional[int] = None, init_state: Optional[np.ndarray] = None) -> None:
        self.rc.reset(seed=seed, init_state=init_state)

    def predict_next(self, state: np.ndarray, action: np.ndarray) -> float:
        # 1) Physics RC prediction
        temp_rc = float(self.rc.predict_next(state, action))

        # 2) NN residual features: [temp, ambient, power, fan]
        feats = np.array([state[0], state[1], state[2], action[0]], dtype=np.float32)
        if self.input_mean is not None and self.input_std is not None:
            feats = (feats - self.input_mean) / (self.input_std + 1e-8)
        x = torch.from_numpy(feats).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            residual = float(self.nn(x).cpu().item())

        # 3) Combine and clip
        temp_pred = np.clip(temp_rc + residual, 30.0, 95.0)
        return float(temp_pred)

    @property
    def warmup_steps(self) -> int:
        return 0
