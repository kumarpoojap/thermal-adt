"""Training/evaluation metrics for PINN."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_predictions(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in y_true.columns:
        e = (y_true[col].values - y_pred[col].values).astype(float)
        mae = float(np.mean(np.abs(e)))
        rmse = float(np.sqrt(np.mean(e**2)))
        rows.append({"target": col, "MAE": mae, "RMSE": rmse})

    # overall
    e_all = (y_true.values - y_pred.values).astype(float)
    rows.append(
        {
            "target": "__overall__",
            "MAE": float(np.mean(np.abs(e_all))),
            "RMSE": float(np.sqrt(np.mean(e_all**2))),
        }
    )
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_model_on_dataset(model, dataloader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    ys = []
    preds = []
    for X, y, t_idx in dataloader:
        X = X.to(device)
        y = y.to(device)
        t_idx = t_idx.to(device).float()
        out = model(X, t_idx, return_physics_params=False)["delta_y"]
        ys.append(y.detach().cpu().numpy())
        preds.append(out.detach().cpu().numpy())
    return {
        "y": np.concatenate(ys, axis=0),
        "y_pred": np.concatenate(preds, axis=0),
    }
