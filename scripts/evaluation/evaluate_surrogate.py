"""
Surrogate model evaluation utilities

This script supports two evaluation modes:

1) One-step evaluation (existing):
   - Evaluates PINN checkpoint and an RF/XGB Teacher bundle on one-step accuracy
   - Uses the PINN-style dataset configuration

2) Multi-step rollout evaluation (new):
   - Evaluates RC, RF/XGB (TeacherRF bundle via RFAdapter), and RC+NN bundles
   - Uses a raw parquet time series and runs autoregressive rollouts for K steps
   - Reports MAE per-horizon (step 1..K)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Ensure repository root (two levels up from scripts/evaluation) is on sys.path
try:
    _THIS_FILE = Path(__file__).resolve()
    _REPO_ROOT = _THIS_FILE.parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
except Exception:
    pass

from src.pinn.data.dataset_k_ahead import prepare_k_ahead_data
from src.pinn.models.hybrid_pinn import HybridPINN
from src.pinn.models.teacher_rf import load_teacher
from src.rl.surrogates.rc_adapter import RCAdapter
from src.rl.surrogates.rf_adapter import RFAdapter
import joblib
import numpy as np
import pandas as pd


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


def _require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in parquet: {missing}")


def _rollout_model(adapter, df: pd.DataFrame, start_indices, steps: int) -> np.ndarray:
    """Run autoregressive rollout for given adapter over multiple start indices.

    Returns mean absolute error per horizon step (shape [steps]).
    """
    errors = []
    for s in start_indices:
        # Initialize state from row s
        temp0 = float(df.loc[s, "gpu_temp_c"]) 
        amb0 = float(df.loc[s, "ambient_temp_c"]) 
        pow0 = float(df.loc[s, "gpu_power_w"]) 
        fan0 = float(df.loc[s, "fan_speed_pct"]) 
        state = np.array([temp0, amb0, pow0, fan0, 0.0], dtype=float)
        adapter.reset(init_state=state)

        preds = []
        trues = []
        prev_temp = temp0
        for k in range(steps):
            # Use recorded action at time s+k (open-loop replay)
            fan_k = float(df.loc[s + k, "fan_speed_pct"]) if (s + k) in df.index else fan0
            amb_k = float(df.loc[s + k, "ambient_temp_c"]) if (s + k) in df.index else amb0
            pow_k = float(df.loc[s + k, "gpu_power_w"]) if (s + k) in df.index else pow0
            action = np.array([fan_k], dtype=float)

            # Predict next temp
            next_temp_pred = float(adapter.predict_next(state, action))

            # Ground truth next temp at s+k+1
            if (s + k + 1) not in df.index:
                break
            next_temp_true = float(df.loc[s + k + 1, "gpu_temp_c"]) 
            preds.append(next_temp_pred)
            trues.append(next_temp_true)

            # Update state for next step (autoregressive)
            temp_delta = next_temp_pred - prev_temp
            state = np.array([next_temp_pred, amb_k, pow_k, fan_k, temp_delta], dtype=float)
            prev_temp = next_temp_pred

        if len(preds) == steps:
            errors.append(np.abs(np.array(preds) - np.array(trues)))

    if not errors:
        return np.full((steps,), np.nan)
    return np.mean(np.vstack(errors), axis=0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate surrogate models (one-step and multi-step)")

    # One-step (existing PINN/RF path)
    parser.add_argument("--config", type=str, help="Path to PINN training/eval config YAML")
    parser.add_argument("--checkpoint", type=str, default="artifacts/best_model.pt", help="PINN checkpoint path")

    # Multi-step rollout inputs
    parser.add_argument("--parquet", type=str, help="Path to raw parquet time-series for rollout")
    parser.add_argument("--rollout-steps", type=int, nargs="+", default=[], help="Rollout horizons to evaluate, e.g., 10 30")
    parser.add_argument("--output-dir", type=str, default="results/surrogate_eval", help="Output directory")

    # Surrogate selection and paths
    parser.add_argument("--eval-rc", action="store_true", help="Evaluate RC surrogate (multi-step)")
    parser.add_argument("--eval-rf", type=str, default=None, help="Path to RF Teacher bundle (joblib)")
    parser.add_argument("--eval-xgb", type=str, default=None, help="Path to XGB Teacher bundle (joblib)")
    parser.add_argument("--eval-rcnn", type=str, default=None, help="Path to RC+NN bundle (joblib)")

    # Optional RC params (fallback to sensible defaults)
    parser.add_argument("--rc-C", type=float, default=100.0)
    parser.add_argument("--rc-h", type=float, default=0.05)
    parser.add_argument("--rc-beta", type=float, default=-0.03)
    parser.add_argument("--rc-gamma", type=float, default=0.01)

    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Surrogate Model Evaluation (One-Step Accuracy)")
    print("="*60)
    
    # Branch: Multi-step rollout if requested
    if args.rollout_steps and args.parquet:
        print("\n" + "="*60)
        print("Multi-Step Rollout Evaluation")
        print("="*60)

        # Load raw parquet and prepare an integer index for convenience
        df = pd.read_parquet(args.parquet).reset_index(drop=True)
        _require_cols(df, ["gpu_temp_c", "ambient_temp_c", "gpu_power_w", "fan_speed_pct"])

        # Choose start indices from the last 20% of the data (test-like), spaced by 20 samples
        n = len(df)
        min_len = max(args.rollout_steps) + 2
        start = int(n * 0.8)
        start_indices = [i for i in range(start, n - min_len, 20)]
        print(f"[INFO] Using {len(start_indices)} start indices for rollouts (test region)")

        results = {}
        for steps in args.rollout_steps:
            print(f"\n-- Horizon: {steps} steps --")

            # RC
            if args.eval_rc:
                rc = RCAdapter(
                    thermal_capacity=args.rc_C,
                    heat_transfer_coeff=args.rc_h,
                    cooling_effectiveness=args.rc_beta,
                    power_to_heat=args.rc_gamma,
                    dt=1.0,
                )
                mae_by_h = _rollout_model(rc, df, start_indices, steps)
                results.setdefault("rc", {})[steps] = mae_by_h.tolist()
                print(f"  RC   MAE by step: {np.round(mae_by_h, 3)}")

            # RF/XGB via RFAdapter (Teacher bundle agnostic)
            if args.eval_rf:
                rf_adapter = RFAdapter(model_path=Path(args.eval_rf))
                mae_by_h = _rollout_model(rf_adapter, df, start_indices, steps)
                results.setdefault("rf", {})[steps] = mae_by_h.tolist()
                print(f"  RF   MAE by step: {np.round(mae_by_h, 3)}")

            if args.eval_xgb:
                xgb_adapter = RFAdapter(model_path=Path(args.eval_xgb))
                mae_by_h = _rollout_model(xgb_adapter, df, start_indices, steps)
                results.setdefault("xgb", {})[steps] = mae_by_h.tolist()
                print(f"  XGB  MAE by step: {np.round(mae_by_h, 3)}")

            # RC+NN
            if args.eval_rcnn:
                from scripts.training.train_rc_nn import ResidualNN, RCNNAdapter as RCNNAdapterEval
                bundle = joblib.load(args.eval_rcnn)
                rc = RCAdapter(**bundle["rc_params"])
                nn_cfg = bundle["nn_config"]
                nn = ResidualNN(input_dim=nn_cfg["input_dim"], hidden_dims=nn_cfg["hidden_dims"]) 
                nn.load_state_dict(bundle["nn_state_dict"])
                rcnn = RCNNAdapterEval(rc_adapter=rc, nn_model=nn, device="cpu", input_mean=bundle["input_mean"], input_std=bundle["input_std"]) 
                mae_by_h = _rollout_model(rcnn, df, start_indices, steps)
                results.setdefault("rcnn", {})[steps] = mae_by_h.tolist()
                print(f"  RC+NN MAE by step: {np.round(mae_by_h, 3)}")

        # Save results JSON and simple plot per model
        out_json = output_dir / "rollout_metrics.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Saved rollout metrics: {out_json}")

        # Plot per-model MAE vs horizon
        for model_name, horizons in results.items():
            for steps, mae_list in horizons.items():
                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(range(1, steps+1), mae_list, marker='o')
                ax.set_xlabel('Step')
                ax.set_ylabel('MAE (°C)')
                ax.set_title(f'{model_name.upper()} rollout MAE (K={steps})')
                ax.grid(True, alpha=0.3)
                save_path = output_dir / f"{model_name}_rollout_mae_k{steps}.png"
                plt.tight_layout()
                plt.savefig(save_path, dpi=150)
                plt.close()
                print(f"  Saved: {save_path}")

        # Combined comparison plots across models for each K
        if results:
            unique_horizons = sorted({k for horizons in results.values() for k in horizons.keys()})
            for steps in unique_horizons:
                # Line plot: MAE vs step for all models
                fig, ax = plt.subplots(figsize=(7,4))
                any_series = False
                for model_name, horizons in results.items():
                    mae_list = horizons.get(steps)
                    if mae_list is None:
                        continue
                    ax.plot(range(1, steps+1), mae_list, marker='o', label=model_name.upper())
                    any_series = True
                if any_series:
                    ax.set_xlabel('Step')
                    ax.set_ylabel('MAE (°C)')
                    ax.set_title(f'Combined rollout MAE (K={steps})')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    save_path = output_dir / f"combined_rollout_mae_k{steps}.png"
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=150)
                    plt.close()
                    print(f"  Saved: {save_path}")

                # Bar chart: last-step MAE per model
                labels, values = [], []
                for model_name, horizons in results.items():
                    mae_list = horizons.get(steps)
                    if mae_list is None:
                        continue
                    labels.append(model_name.upper())
                    values.append(float(mae_list[-1]))
                if labels:
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.bar(labels, values)
                    ax.set_xlabel('Model')
                    ax.set_ylabel('MAE at last step (°C)')
                    ax.set_title(f'Last-step MAE by model (K={steps})')
                    ax.grid(True, axis='y', alpha=0.3)
                    save_path = output_dir / f"combined_last_step_mae_k{steps}.png"
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=150)
                    plt.close()
                    print(f"  Saved: {save_path}")

        # Export CSV summaries
        tidy_rows = []
        last_rows = []
        for model_name, horizons in results.items():
            for steps, mae_list in horizons.items():
                for i, mae in enumerate(mae_list, start=1):
                    tidy_rows.append({
                        "model": model_name,
                        "horizon": int(steps),
                        "step": int(i),
                        "mae_c": float(mae),
                    })
                last_rows.append({
                    "model": model_name,
                    "horizon": int(steps),
                    "mae_last_step_c": float(mae_list[-1])
                })

        if tidy_rows:
            tidy_df = pd.DataFrame(tidy_rows)
            tidy_csv = output_dir / "rollout_mae_tidy.csv"
            tidy_df.to_csv(tidy_csv, index=False)
            print(f"[INFO] Saved tidy CSV: {tidy_csv}")

        if last_rows:
            last_df = pd.DataFrame(last_rows).sort_values(["horizon", "model"])
            last_csv = output_dir / "rollout_mae_last_step.csv"
            last_df.to_csv(last_csv, index=False)
            print(f"[INFO] Saved last-step CSV: {last_csv}")

        print("\n" + "="*60)
        print("Rollout Evaluation Complete!")
        print("="*60)
        return

    # Fallback: one-step PINN/RF evaluation if config is provided
    if not args.config:
        raise SystemExit("No rollout requested and no --config provided for one-step evaluation.")

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
    
    # Load RF teacher (could be RF or XGB bundle)
    print("[INFO] Loading Teacher bundle...")
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
    
    # Evaluate Teacher bundle
    print("\n" + "="*60)
    print("Evaluating Teacher Model (RF/XGB)")
    print("="*60)
    
    rf_metrics = evaluate_rf_teacher(
        rf_teacher, test_loader, feature_cols, scaler, target_cols
    )
    
    print(f"[TEACHER] Normalized  - MAE: {rf_metrics['mae_normalized']:.4f}, RMSE: {rf_metrics['rmse_normalized']:.4f}")
    print(f"[TEACHER] Absolute °C - MAE: {rf_metrics['mae_celsius']:.4f}, RMSE: {rf_metrics['rmse_celsius']:.4f}")
    print(f"[TEACHER] Samples: {rf_metrics['n_samples']}")
    
    # Save metrics
    metrics_summary = {
        "pinn": pinn_metrics,
        "teacher": rf_metrics,
        "comparison": {
            "pinn_vs_teacher_mae_improvement": float((rf_metrics['mae_celsius'] - pinn_metrics['mae_celsius']) / rf_metrics['mae_celsius'] * 100),
            "pinn_vs_teacher_rmse_improvement": float((rf_metrics['rmse_celsius'] - pinn_metrics['rmse_celsius']) / rf_metrics['rmse_celsius'] * 100)
        }
    }
    
    metrics_path = output_dir / "surrogate_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"\n[INFO] Saved metrics: {metrics_path}")
    
    # Plot comparison
    plot_comparison_bar(
        {"pinn": pinn_metrics, "teacher": rf_metrics},
        output_dir / "model_comparison.png"
    )
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"\nKey Findings:")
    print(f"  - PINN MAE: {pinn_metrics['mae_celsius']:.2f}°C")
    print(f"  - TEACHER MAE: {rf_metrics['mae_celsius']:.2f}°C")
    if pinn_metrics['mae_celsius'] < rf_metrics['mae_celsius']:
        improvement = (rf_metrics['mae_celsius'] - pinn_metrics['mae_celsius']) / rf_metrics['mae_celsius'] * 100
        print(f"  - PINN is {improvement:.1f}% better than Teacher")
    else:
        degradation = (pinn_metrics['mae_celsius'] - rf_metrics['mae_celsius']) / rf_metrics['mae_celsius'] * 100
        print(f"  - PINN is {degradation:.1f}% worse than Teacher (needs more training)")


if __name__ == "__main__":
    main()
