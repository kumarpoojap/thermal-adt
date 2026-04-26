"""Physics residual losses for Hybrid PINN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


def extract_physics_drivers(
    X: torch.Tensor,
    feature_cols: list,
    supply_col: str,
    actuator_col: str,
    load_col: Optional[str] = None,
    window_size: int = 12,
) -> Dict[str, torch.Tensor]:
    """Extract drivers used by the physics loss from the feature tensor."""

    def col_idx(name: str) -> int:
        if name not in feature_cols:
            raise KeyError(f"Required physics feature column not found: {name}")
        return feature_cols.index(name)

    supply = X[:, col_idx(supply_col)]
    actuator = X[:, col_idx(actuator_col)]

    out = {
        "supply_air": supply,
        "cooling_actuator": actuator,
    }

    if load_col is not None:
        out["load_proxy"] = X[:, col_idx(load_col)]

    _ = window_size
    return out


class PhysicsODELoss(nn.Module):
    """Simple ODE residual loss: encourages predicted temperature to follow a first-order model."""

    def __init__(self, config: Optional[Dict] = None, **kwargs):
        super().__init__()
        merged: Dict = {}
        if config:
            merged.update(config)
        merged.update(kwargs)
        self.config = merged

    def forward(
        self,
        y_pred: torch.Tensor,
        y_current: torch.Tensor,
        physics_params: Dict[str, torch.Tensor],
        supply_air: torch.Tensor,
        cooling_actuator: torch.Tensor,
        load_proxy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # This matches the structure expected by the existing HybridPINN model code.
        # Residual is intentionally simple; your original repo contains a richer version.
        C = physics_params.get("C")
        h = physics_params.get("h")

        if C is None or h is None:
            return torch.tensor(0.0, device=y_pred.device)

        # Expand drivers
        supply_air = supply_air.unsqueeze(-1).expand_as(y_pred)
        cooling_actuator = cooling_actuator.unsqueeze(-1).expand_as(y_pred)

        # Optional load proxy
        if load_proxy is None:
            load_proxy = torch.zeros_like(supply_air[:, 0])
        load_proxy = load_proxy.unsqueeze(-1).expand_as(y_pred)

        # Very lightweight residual term
        # dT approx: -(h/C)*(T - T_supply) - alpha*actuator + beta*load
        alpha = physics_params.get("alpha", torch.zeros_like(C))
        beta = physics_params.get("beta", torch.zeros_like(C))

        rhs = (-(h / (C + 1e-6)) * (y_current - supply_air)) - alpha * cooling_actuator + beta * load_proxy
        residual = (y_pred - (y_current + rhs))
        return torch.mean(residual ** 2)
