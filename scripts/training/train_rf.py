#!/usr/bin/env python3
"""
Self-contained RF Teacher Training Script
Trains a RandomForest surrogate model and exports a TeacherRF bundle for RL.
No dependencies on old repository structure - all logic is inline.

Usage:
    python scripts/training/train_rf.py --help
    
Example:
    python scripts/training/train_rf.py \
        --data data/synthetic/thermal_dataset.parquet \
        --config configs/data/gpu_thermal_spec.json \
        --output-dir results/rf_training \
        --bundle-path models/rf_teacher.pkl \
        --n-estimators 200 \
        --max-depth 16
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.features import (
    compute_winsor_bounds,
    apply_winsorization,
    drop_low_variance_features,
    ensure_datetime_index,
)
from src.common.scalers import (
    compute_train_target_scaler,
    apply_target_normalization,
    invert_target_normalization,
)
from src.common.data_utils import time_split_indices

plt.style.use("seaborn-v0_8")


# ============================================================================
# Feature Engineering (inline, no external dependencies)
# ============================================================================

def add_lag_features(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    """Add lag features for specified columns."""
    result = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            result[f"{col}_lag{lag}"] = df[col].shift(lag)
    return result


def build_supervised_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    lags: List[int],
    k_ahead: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Build supervised dataset with lag features and k-step-ahead targets.
    
    Returns:
        X: Feature DataFrame
        y: Target DataFrame (k steps ahead)
        feature_names: List of all feature column names
    """
    # Add lag features
    df_lagged = add_lag_features(df, feature_cols, lags)
    
    # Collect all feature columns (base + lags)
    all_feature_cols = []
    for col in feature_cols:
        all_feature_cols.append(col)
        for lag in lags:
            lag_col = f"{col}_lag{lag}"
            if lag_col in df_lagged.columns:
                all_feature_cols.append(lag_col)
    
    # Build X and y
    X = df_lagged[all_feature_cols].copy()
    
    # Shift targets k steps back (so y[t] = actual[t+k])
    y = df[target_cols].shift(-k_ahead).copy()
    
    # Drop rows with NaN (from lags and k_ahead)
    valid_idx = X.notna().all(axis=1) & y.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y, all_feature_cols


# ============================================================================
# Evaluation & Plotting
# ============================================================================

def evaluate_predictions(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """Compute MAE and RMSE for each target and overall."""
    metrics = []
    for col in y_true.columns:
        mae = mean_absolute_error(y_true[col], y_pred[col])
        rmse = np.sqrt(mean_squared_error(y_true[col], y_pred[col]))
        metrics.append({"target": col, "MAE": mae, "RMSE": rmse})
    
    # Overall
    mae_overall = mean_absolute_error(y_true.values.ravel(), y_pred.values.ravel())
    rmse_overall = np.sqrt(mean_squared_error(y_true.values.ravel(), y_pred.values.ravel()))
    metrics.append({"target": "__overall__", "MAE": mae_overall, "RMSE": rmse_overall})
    
    return pd.DataFrame(metrics)


def persistence_baseline(y_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Compute persistence baseline: y(t+k) ≈ y(t)."""
    y_true_future = y_df.shift(-k).dropna()
    y_hat = y_df.loc[y_true_future.index]
    
    metrics = []
    for col in y_df.columns:
        mae = mean_absolute_error(y_true_future[col], y_hat[col])
        rmse = np.sqrt(mean_squared_error(y_true_future[col], y_hat[col]))
        metrics.append({"target": col, "MAE": mae, "RMSE": rmse})
    
    mae_overall = mean_absolute_error(y_true_future.values.ravel(), y_hat.values.ravel())
    rmse_overall = np.sqrt(mean_squared_error(y_true_future.values.ravel(), y_hat.values.ravel()))
    metrics.append({"target": "__overall__", "MAE": mae_overall, "RMSE": rmse_overall})
    
    return pd.DataFrame(metrics)


def plot_feature_importance(model, feature_names: List[str], outpath: str, top_k: int = 25):
    """Plot top-k feature importances."""
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_k]
    
    plt.figure(figsize=(10, 8))
    plt.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1], color="#2a9d8f")
    plt.title("Feature Importance (RandomForest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")


def plot_predictions(y_true: pd.Series, y_pred: pd.Series, outpath: str, title: str):
    """Plot actual vs predicted time series."""
    plt.figure(figsize=(12, 4))
    plt.plot(y_true.index, y_true.values, label="Actual", lw=1.5)
    plt.plot(y_pred.index, y_pred.values, label="Predicted", lw=1.2, alpha=0.8)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved: {outpath}")


# ============================================================================
# TeacherRF Bundle Export (inline)
# ============================================================================

def export_teacher_bundle(
    model,
    feature_cols: List[str],
    target_cols: List[str],
    target_stats: Dict,
    winsor_bounds: Dict,
    k_ahead: int,
    cadence_s: float,
    output_path: str,
):
    """Export a TeacherRF-compatible joblib bundle."""
    
    def sha256_str(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    
    def fingerprint_list(items) -> str:
        return sha256_str("|".join([str(x) for x in items]))
    
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = "unknown"
    
    bundle = {
        "type": "rf_teacher_bundle",
        "framework": "sklearn",
        "sklearn_version": sklearn_version,
        "python_version": sys.version.split()[0],
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "created_at_epoch": int(time.time()),
        "k_ahead": int(k_ahead),
        "cadence_seconds": float(cadence_s),
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "model": model,
        "target_normalization_stats": target_stats,
        "feature_winsor_bounds": winsor_bounds,
        "feature_fingerprint": fingerprint_list(feature_cols),
        "target_fingerprint": fingerprint_list(target_cols),
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(bundle, output_path)
    print(f"\n✓ Exported TeacherRF bundle: {output_path}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Targets: {len(target_cols)}")
    print(f"  k_ahead: {k_ahead}, cadence: {cadence_s}s")


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Self-contained RF teacher training and export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument("--data", required=True, help="Path to parquet dataset")
    parser.add_argument("--config", required=True, help="Path to feature/target spec JSON")
    
    # Output
    parser.add_argument("--output-dir", default="results/rf_training", help="Output directory for artifacts")
    parser.add_argument("--bundle-path", default="models/rf_teacher.pkl", help="Path for exported TeacherRF bundle")
    
    # Feature engineering
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 3, 5, 10], help="Lag steps for features")
    parser.add_argument("--k-ahead", type=int, default=10, help="Prediction horizon (steps)")
    parser.add_argument("--cadence-s", type=float, default=1.0, help="Time cadence in seconds")
    
    # Model
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=None, help="Max tree depth (None=unlimited)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    
    # Normalization
    parser.add_argument("--normalize-targets", action="store_true", help="Normalize targets using train stats")
    
    args = parser.parse_args()
    
    # ========================================================================
    # 1. Load Data and Config
    # ========================================================================
    
    print("\n" + "="*70)
    print("RF TEACHER TRAINING - SELF-CONTAINED")
    print("="*70)
    
    print(f"\n[1/7] Loading dataset: {args.data}")
    df = pd.read_parquet(args.data)
    df = ensure_datetime_index(df)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    print(f"\n[2/7] Loading config: {args.config}")
    with open(args.config) as f:
        spec = json.load(f)
    
    feature_cols = spec.get("feature_cols", [])
    target_cols_raw = spec.get("target_cols_raw", [])
    
    # Validate columns
    missing_feats = [c for c in feature_cols if c not in df.columns]
    missing_targets = [c for c in target_cols_raw if c not in df.columns]
    
    if missing_feats:
        raise ValueError(f"Missing feature columns: {missing_feats}")
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Targets: {len(target_cols_raw)}")
    
    # ========================================================================
    # 2. Build Supervised Dataset
    # ========================================================================
    
    print(f"\n[3/7] Building supervised dataset (lags={args.lags}, k_ahead={args.k_ahead})")
    X, y, all_feature_cols = build_supervised_dataset(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols_raw,
        lags=args.lags,
        k_ahead=args.k_ahead,
    )
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Total features (with lags): {len(all_feature_cols)}")
    
    # ========================================================================
    # 3. Train/Val/Test Split
    # ========================================================================
    
    print(f"\n[4/7] Splitting data (70/15/15)")
    train_idx, val_idx, test_idx = time_split_indices(len(X), train_frac=0.7, val_frac=0.15)
    
    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_train_raw, y_val_raw, y_test_raw = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Winsorization (train-only bounds)
    print("  Computing winsorization bounds (train only)...")
    winsor_bounds = compute_winsor_bounds(X_train, q_low=0.01, q_high=0.99)
    X_train = apply_winsorization(X_train, winsor_bounds)
    X_val = apply_winsorization(X_val, winsor_bounds)
    X_test = apply_winsorization(X_test, winsor_bounds)
    
    # Drop low-variance features
    print("  Dropping low-variance features...")
    keep_cols = drop_low_variance_features(X_train, threshold=1e-8)
    X_train = X_train[keep_cols]
    X_val = X_val[keep_cols]
    X_test = X_test[keep_cols]
    all_feature_cols = keep_cols
    print(f"  Kept {len(all_feature_cols)} features")
    
    # Target normalization (optional)
    y_stats = None
    if args.normalize_targets:
        print("  Normalizing targets (train-only stats)...")
        y_stats = compute_train_target_scaler(y_train_raw)
        y_train = apply_target_normalization(y_train_raw, y_stats)
        y_val = apply_target_normalization(y_val_raw, y_stats)
        y_test = apply_target_normalization(y_test_raw, y_stats)
    else:
        y_train, y_val, y_test = y_train_raw, y_val_raw, y_test_raw
    
    # ========================================================================
    # 4. Train RandomForest
    # ========================================================================
    
    print(f"\n[5/7] Training RandomForest (n_estimators={args.n_estimators}, max_depth={args.max_depth})")
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_jobs=-1,
        random_state=args.random_state,
        min_samples_leaf=1,
    )
    
    rf.fit(X_train, y_train)
    print("  ✓ Training complete")
    
    # ========================================================================
    # 5. Evaluate
    # ========================================================================
    
    print(f"\n[6/7] Evaluating...")
    
    # Predict
    y_pred_train = pd.DataFrame(rf.predict(X_train), index=y_train.index, columns=y_train.columns)
    y_pred_val = pd.DataFrame(rf.predict(X_val), index=y_val.index, columns=y_val.columns)
    y_pred_test = pd.DataFrame(rf.predict(X_test), index=y_test.index, columns=y_test.columns)
    
    # Invert normalization if needed
    if args.normalize_targets:
        y_pred_train_c = invert_target_normalization(y_pred_train.values, y_stats, target_cols_raw)
        y_pred_val_c = invert_target_normalization(y_pred_val.values, y_stats, target_cols_raw)
        y_pred_test_c = invert_target_normalization(y_pred_test.values, y_stats, target_cols_raw)
        
        y_pred_train_c.index = y_train.index
        y_pred_val_c.index = y_val.index
        y_pred_test_c.index = y_test.index
        
        y_train_c, y_val_c, y_test_c = y_train_raw, y_val_raw, y_test_raw
    else:
        y_train_c, y_val_c, y_test_c = y_train_raw, y_val_raw, y_test_raw
        y_pred_train_c, y_pred_val_c, y_pred_test_c = y_pred_train, y_pred_val, y_pred_test
    
    # Metrics
    m_train = evaluate_predictions(y_train_c, y_pred_train_c).assign(split="train")
    m_val = evaluate_predictions(y_val_c, y_pred_val_c).assign(split="val")
    m_test = evaluate_predictions(y_test_c, y_pred_test_c).assign(split="test")
    
    metrics_summary = pd.concat([m_train, m_val, m_test], ignore_index=True)
    
    # Persistence baseline
    baseline_test = persistence_baseline(y_test_c, k=args.k_ahead)
    baseline_test["split"] = "test_persistence"
    
    # Print summary
    print("\n  Metrics Summary:")
    print(metrics_summary.to_string(index=False))
    print("\n  Persistence Baseline (test):")
    print(baseline_test.to_string(index=False))
    
    # ========================================================================
    # 6. Save Artifacts
    # ========================================================================
    
    print(f"\n[7/7] Saving artifacts to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Metrics
    metrics_path = os.path.join(args.output_dir, "metrics_summary.csv")
    metrics_summary.to_csv(metrics_path, index=False)
    print(f"  Saved: {metrics_path}")
    
    baseline_path = os.path.join(args.output_dir, "metrics_persistence_baseline.csv")
    baseline_test.to_csv(baseline_path, index=False)
    print(f"  Saved: {baseline_path}")
    
    # Model and metadata
    joblib.dump(rf, os.path.join(args.output_dir, "model_random_forest.pkl"))
    print(f"  Saved: {os.path.join(args.output_dir, 'model_random_forest.pkl')}")
    
    with open(os.path.join(args.output_dir, "feature_columns.json"), "w") as f:
        json.dump(all_feature_cols, f, indent=2)
    
    with open(os.path.join(args.output_dir, "target_columns.json"), "w") as f:
        json.dump(target_cols_raw, f, indent=2)
    
    with open(os.path.join(args.output_dir, "feature_winsor_bounds.json"), "w") as f:
        json.dump(winsor_bounds, f, indent=2)
    
    if y_stats:
        with open(os.path.join(args.output_dir, "target_normalization_stats.json"), "w") as f:
            json.dump(y_stats, f, indent=2)
    
    # Plots
    print("\n  Generating plots...")
    plot_feature_importance(
        rf,
        feature_names=all_feature_cols,
        outpath=os.path.join(args.output_dir, "feature_importance.png"),
        top_k=min(40, len(all_feature_cols))
    )
    
    # Plot first target
    one_target = target_cols_raw[0]
    plot_predictions(
        y_true=y_test_c[one_target],
        y_pred=y_pred_test_c[one_target],
        outpath=os.path.join(args.output_dir, f"predictions_{one_target}.png"),
        title=f"Actual vs Predicted (Test) — {one_target}"
    )
    
    # ========================================================================
    # 7. Export TeacherRF Bundle
    # ========================================================================
    
    export_teacher_bundle(
        model=rf,
        feature_cols=all_feature_cols,
        target_cols=target_cols_raw,
        target_stats=y_stats,
        winsor_bounds=winsor_bounds,
        k_ahead=args.k_ahead,
        cadence_s=args.cadence_s,
        output_path=args.bundle_path,
    )
    
    # ========================================================================
    # Done
    # ========================================================================
    
    print("\n" + "="*70)
    print("✓ RF TEACHER TRAINING COMPLETE")
    print("="*70)
    print(f"\nArtifacts saved to: {args.output_dir}")
    print(f"TeacherRF bundle: {args.bundle_path}")
    print("\nNext steps:")
    print(f"  1. Update configs/rl/sac_shielded.yaml to point to: {args.bundle_path}")
    print(f"  2. Run SAC training:")
    print(f"     python scripts/training/train_sac.py --config configs/rl/sac_shielded.yaml")
    print()


if __name__ == "__main__":
    main()
