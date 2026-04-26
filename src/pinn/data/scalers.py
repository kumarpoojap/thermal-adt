"""PINN-specific scaler utilities (thin re-export from src.common.scalers)."""

from src.common.scalers import (
    TargetScaler,
    compute_train_target_scaler,
    apply_target_normalization,
    invert_target_normalization,
)

__all__ = [
    "TargetScaler",
    "compute_train_target_scaler",
    "apply_target_normalization",
    "invert_target_normalization",
]
