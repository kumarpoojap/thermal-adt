"""Plotting utilities for PINN training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt


def plot_loss_curves(history: Union[Dict, List], outpath: Path, val_losses: List = None) -> None:
    """Plot training and validation loss curves.
    
    Args:
        history: Either a dict with 'train_loss' and 'val_loss' keys, or a list of train losses
        outpath: Path to save plot
        val_losses: Optional list of val losses (only used if history is a list)
    """
    plt.figure(figsize=(6, 4))
    
    if isinstance(history, dict):
        train_losses = history.get("train_loss", [])
        val_losses = history.get("val_loss", [])
    else:
        train_losses = history
        val_losses = val_losses or []
    
    plt.plot(train_losses, label="train")
    if val_losses:
        plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
