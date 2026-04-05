"""
Simplified surrogate model evaluation focusing on one-step accuracy.

For multi-step rollout, we need actual sequence data which isn't available
in the current k-ahead dataset format. This script focuses on what we can
measure accurately with the current data.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.pinn.data.dataset_k_ahead import prepare_k_ahead_data
from src.pinn.models.hybrid_pinn import HybridPINN
from src.pinn.models.teacher_rf import load_teacher


def load_trained_pinn(
    checkpoint_path: Path,
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    device: str = "cpu"
) -> HybridPINN:
    """Load trained PINN model from checkpoint."""
    model = HybridPINN(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation="silu",
        dropout=0.1,
        time_embedding_enabled=False,
        physics_head_enabled=True
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model


def evaluate_one_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    scaler,
    target_cols: List[str],
    device: str = "cpu"
) -> Dict[str, float]:
    """Evaluate one-step prediction accuracy with denormalization."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_preds_denorm = []
    all_targets_denorm = []
    
    with torch.no_grad():
        for X, y, t_idx in dataloader:
            X = X.to(device)
            y = y.to(device)
            t_idx = t_idx.to(device).float()
            
            out = model(X, t_idx, return_physics_params=False)
            y_pred = out["delta_y"]
            
            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
            # Denormalize for absolute temperature metrics
            if scaler is not None:
                target_col = target_cols[0]
                mean = scaler.stats[target_col]["mean"]
                std = scaler.stats[target_col]["std"]
                
                y_pred_denorm = y_pred.cpu().numpy() * std + mean
                y_target_denorm = y.cpu().numpy() * std + mean
                
                all_preds_denorm.append(y_pred_denorm)
                all_targets_denorm.append(y_target_denorm)
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Normalized metrics
    mae_norm = float(np.mean(np.abs(preds - targets)))
    rmse_norm = float(np.sqrt(np.mean((preds - targets) ** 2)))
    
    # Denormalized metrics (absolute temperature)
    if scaler is not None:
        preds_denorm = np.concatenate(all_preds_denorm, axis=0)
        targets_denorm = np.concatenate(all_targets_denorm, axis=0)
        
        mae_abs = float(np.mean(np.abs(preds_denorm - targets_denorm)))
        rmse_abs = float(np.sqrt(np.mean((preds_denorm - targets_denorm) ** 2)))
    else:
        mae_abs = mae_norm
        rmse_abs = rmse_norm
    
    return {
        "mae_normalized": mae_norm,
        "rmse_normalized": rmse_norm,
        "mae_celsius": mae_abs,
        "rmse_celsius": rmse_abs,
        "n_samples": len(preds)
    }


def evaluate_rf_teacher(
    teacher_model,
    dataloader: DataLoader,
    feature_cols: List[str],
    scaler,
    target_cols: List[str]
) -> Dict[str, float]:
    """Evaluate RF teacher one-step accuracy."""
    all_preds = []
    all_targets = []
    
    for X, y, t_idx in dataloader:
        X_np = X.numpy()
        y_np = y.numpy()
        
        # Convert to DataFrame
        X_df = pd.DataFrame(X_np, columns=feature_cols)
        
        # Predict
        y_pred = teacher_model.predict(X_df, return_tensor=False)
        
        all_preds.append(y_pred)
        all_targets.append(y_np)
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Normalized metrics
    mae_norm = float(np.mean(np.abs(preds - targets)))
    rmse_norm = float(np.sqrt(np.mean((preds - targets) ** 2)))
    
    # Denormalized metrics
    if scaler is not None:
        target_col = target_cols[0]
        mean = scaler.stats[target_col]["mean"]
        std = scaler.stats[target_col]["std"]
        
        preds_denorm = preds * std + mean
        targets_denorm = targets * std + mean
        
        mae_abs = float(np.mean(np.abs(preds_denorm - targets_denorm)))
        rmse_abs = float(np.sqrt(np.mean((preds_denorm - targets_denorm) ** 2)))
    else:
        mae_abs = mae_norm
        rmse_abs = rmse_norm
    
    return {
        "mae_normalized": mae_norm,
        "rmse_normalized": rmse_norm,
        "mae_celsius": mae_abs,
        "rmse_celsius": rmse_abs,
        "n_samples": len(preds)
    }


def plot_prediction_scatter(
    preds: np.ndarray,
    targets: np.ndarray,
    model_name: str,
    save_path: Path
):
    """Plot predicted vs actual scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(targets, preds, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Temperature (°C)', fontsize=12)
    ax.set_ylabel('Predicted Temperature (°C)', fontsize=12)
    ax.set_title(f'{model_name} - Predicted vs Actual', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved scatter plot: {save_path}")


def plot_comparison_bar(
    metrics_dict: Dict[str, Dict],
    save_path: Path
):
    """Plot comparison bar chart of model performance."""
    models = list(metrics_dict.keys())
    mae_values = [metrics_dict[m]["mae_celsius"] for m in models]
    rmse_values = [metrics_dict[m]["rmse_celsius"] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
    ax.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Error (°C)', fontsize=12)
    ax.set_title('One-Step Prediction Accuracy Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved comparison plot: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate surrogate models (one-step)")
    parser.add_argument("--config", type=str, required=True, help="Path to training config")
    parser.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt", help="PINN checkpoint")
    parser.add_argument("--output-dir", type=str, default="results/surrogate_eval", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Surrogate Model Evaluation (One-Step Accuracy)")
    print("="*60)
    
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # Prepare dataset
    print("[INFO] Loading dataset...")
    data_result = prepare_k_ahead_data(
        parquet_path=Path(cfg["data"]["parquet_path"]),
        spec_path=Path(cfg["data"]["spec_path"]),
        feature_columns_path=Path(cfg["data"]["feature_columns_path"]),
        base_cols=cfg["data"]["features"]["base_cols"],
        lags=cfg["data"]["features"]["lags"],
        roll_windows=cfg["data"]["features"]["roll_windows"],
        k_ahead=cfg["data"]["k_ahead"],
        train_frac=cfg["data"]["train_frac"],
        val_frac=cfg["data"]["val_frac"],
        normalize_targets=cfg["data"]["normalize_targets"],
        winsorize=cfg["data"]["features"]["winsorize"],
        winsor_quantiles=tuple(cfg["data"]["features"]["winsor_quantiles"]),
        low_var_threshold=cfg["data"]["features"]["low_var_threshold"],
        cadence_seconds=cfg["data"]["cadence_seconds"]
    )
    
    test_dataset = data_result["test_dataset"]
    scaler = data_result["scaler"]
    feature_cols = data_result["feature_cols"]
    target_cols = data_result["target_cols"]
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"[INFO] Test samples: {len(test_dataset)}")
    print(f"[INFO] Features: {len(feature_cols)}")
    print(f"[INFO] Targets: {len(target_cols)}")
    
    # Load PINN model
    print("\n[INFO] Loading PINN model...")
    input_dim = len(feature_cols)
    output_dim = len(target_cols)
    hidden_dims = cfg["model"]["hidden_dims"]
    
    pinn_model = load_trained_pinn(
        Path(args.checkpoint),
        input_dim, output_dim, hidden_dims, device
    )
    
    # Load RF teacher
    print("[INFO] Loading RF teacher...")
    rf_teacher = load_teacher(
        Path(cfg["teacher"]["model_path"]),
        cache_dir=None,
        use_cache=False
    )
    
    # Evaluate PINN
    print("\n" + "="*60)
    print("Evaluating PINN Model")
    print("="*60)
    
    pinn_metrics = evaluate_one_step(
        pinn_model, test_loader, scaler, target_cols, device
    )
    
    print(f"[PINN] Normalized  - MAE: {pinn_metrics['mae_normalized']:.4f}, RMSE: {pinn_metrics['rmse_normalized']:.4f}")
    print(f"[PINN] Absolute °C - MAE: {pinn_metrics['mae_celsius']:.4f}, RMSE: {pinn_metrics['rmse_celsius']:.4f}")
    print(f"[PINN] Samples: {pinn_metrics['n_samples']}")
    
    # Evaluate RF
    print("\n" + "="*60)
    print("Evaluating RF Teacher Model")
    print("="*60)
    
    rf_metrics = evaluate_rf_teacher(
        rf_teacher, test_loader, feature_cols, scaler, target_cols
    )
    
    print(f"[RF] Normalized  - MAE: {rf_metrics['mae_normalized']:.4f}, RMSE: {rf_metrics['rmse_normalized']:.4f}")
    print(f"[RF] Absolute °C - MAE: {rf_metrics['mae_celsius']:.4f}, RMSE: {rf_metrics['rmse_celsius']:.4f}")
    print(f"[RF] Samples: {rf_metrics['n_samples']}")
    
    # Save metrics
    metrics_summary = {
        "pinn": pinn_metrics,
        "rf": rf_metrics,
        "comparison": {
            "pinn_vs_rf_mae_improvement": float((rf_metrics['mae_celsius'] - pinn_metrics['mae_celsius']) / rf_metrics['mae_celsius'] * 100),
            "pinn_vs_rf_rmse_improvement": float((rf_metrics['rmse_celsius'] - pinn_metrics['rmse_celsius']) / rf_metrics['rmse_celsius'] * 100)
        }
    }
    
    metrics_path = output_dir / "surrogate_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"\n[INFO] Saved metrics: {metrics_path}")
    
    # Plot comparison
    plot_comparison_bar(
        {"pinn": pinn_metrics, "rf": rf_metrics},
        output_dir / "model_comparison.png"
    )
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"\nKey Findings:")
    print(f"  - PINN MAE: {pinn_metrics['mae_celsius']:.2f}°C")
    print(f"  - RF MAE: {rf_metrics['mae_celsius']:.2f}°C")
    if pinn_metrics['mae_celsius'] < rf_metrics['mae_celsius']:
        improvement = (rf_metrics['mae_celsius'] - pinn_metrics['mae_celsius']) / rf_metrics['mae_celsius'] * 100
        print(f"  - PINN is {improvement:.1f}% better than RF")
    else:
        degradation = (pinn_metrics['mae_celsius'] - rf_metrics['mae_celsius']) / rf_metrics['mae_celsius'] * 100
        print(f"  - PINN is {degradation:.1f}% worse than RF (needs more training)")


if __name__ == "__main__":
    main()
