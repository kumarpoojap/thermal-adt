#!/usr/bin/env python3
"""Hybrid PINN + Teacher curriculum training script.

Trains a physics-informed neural network with:
- Supervised data loss
- Optional teacher distillation
- Optional physics residual
- Optional monotonic cooling constraint
- Temporal smoothness penalty
- Curriculum learning (Phase 1 → Phase 2 → Phase 3)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pinn.data.dataset_k_ahead import prepare_k_ahead_data
from src.pinn.data.scalers import TargetScaler
from src.pinn.losses.monotonicity import create_monotonic_loss
from src.pinn.losses.physics import PhysicsODELoss, extract_physics_drivers
from src.pinn.losses.smoothness import TemporalSmoothnessLoss
from src.pinn.models.hybrid_pinn import HybridPINN
from src.pinn.models.teacher_rf import load_teacher
from src.pinn.training.baselines import compute_baseline_metrics, persistence_k_ahead_baseline
from src.pinn.training.metrics import evaluate_model_on_dataset, evaluate_predictions
from src.pinn.training.plotting import plot_loss_curves


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    loss_weights: Dict[str, float],
    teacher_preds: Optional[torch.Tensor] = None,
    physics_loss_fn: Optional[PhysicsODELoss] = None,
    physics_config: Optional[Dict] = None,
    feature_cols: Optional[list] = None,
    target_cols: Optional[list] = None,
    scaler: Optional[TargetScaler] = None,
    monotonic_loss_fn: Optional[nn.Module] = None,
    smoothness_loss_fn: Optional[TemporalSmoothnessLoss] = None,
    delta_bound_config: Optional[Dict] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    loss_data_sum = 0.0
    loss_teacher_sum = 0.0
    loss_physics_sum = 0.0
    loss_mono_sum = 0.0
    loss_smooth_sum = 0.0
    loss_delta_bound_sum = 0.0
    n_batches = 0
    
    for batch_idx, (X, y, t_idx) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        t_idx = t_idx.to(device).float()
        
        optimizer.zero_grad()
        
        # Forward pass
        out = model(X, t_idx, return_physics_params=(physics_loss_fn is not None))
        y_pred = out["delta_y"]

        # Build normalized delta target: y_delta = y(t+k) - y(t)
        y_current_norm = None
        if scaler is not None and feature_cols is not None and "gpu_temp_current" in feature_cols:
            target_col = target_cols[0] if target_cols else "gpu_temp_c"
            mean = scaler.stats[target_col]["mean"]
            std = scaler.stats[target_col]["std"]
            current_temp_idx = feature_cols.index("gpu_temp_current")
            y_current_abs = X[:, current_temp_idx].unsqueeze(-1)
            y_current_norm = (y_current_abs - mean) / std
            y_target = y - y_current_norm
        else:
            y_target = y

        # Data loss (MSE) on delta target
        loss_data = nn.functional.mse_loss(y_pred, y_target)
        loss = loss_weights["data"] * loss_data
        loss_data_sum += loss_data.item()
        
        # Teacher distillation loss
        if teacher_preds is not None and loss_weights.get("teacher", 0) > 0:
            batch_start = batch_idx * dataloader.batch_size
            batch_end = min(batch_start + len(X), len(teacher_preds))
            y_teacher = teacher_preds[batch_start:batch_end].to(device)
            
            if len(y_teacher) == len(y_pred):
                # Fix shape mismatch: squeeze both to [B] before computing loss
                y_pred_squeezed = y_pred.squeeze(-1)  # [B, 1] -> [B]
                y_teacher_squeezed = y_teacher.squeeze(-1)  # [B, 1] -> [B]
                if y_current_norm is not None:
                    # Teacher outputs absolute temperature in °C; convert to normalized, then to normalized delta
                    y_teacher_squeezed = ((y_teacher_squeezed - mean) / std) - y_current_norm.squeeze(-1)
                loss_teacher = nn.functional.mse_loss(y_pred_squeezed, y_teacher_squeezed)
                loss += loss_weights["teacher"] * loss_teacher
                loss_teacher_sum += loss_teacher.item()
        
        # Physics loss
        if physics_loss_fn is not None and loss_weights.get("physics", 0) > 0:
            # Extract physics drivers from features
            drivers = extract_physics_drivers(
                X, feature_cols,
                supply_col=physics_config.get("supply_col", "supply_air_c"),
                actuator_col=physics_config.get("actuator_col", "evap_fan_speed_pct"),
                load_col=physics_config.get("load_col", "cool_demand_kwh") if physics_config.get("use_load_proxy") else None,
                window_size=physics_config.get("window_size", 12)
            )
            
            physics_params = out.get("physics_params", {})
            if physics_params and scaler is not None:
                target_col = target_cols[0] if target_cols else "gpu_temp_c"
                mean = scaler.stats[target_col]["mean"]
                std = scaler.stats[target_col]["std"]

                # Current temperature in °C
                if feature_cols is not None and "gpu_temp_current" in feature_cols:
                    current_temp_idx = feature_cols.index("gpu_temp_current")
                    y_current_abs = X[:, current_temp_idx].unsqueeze(-1).expand_as(y_pred)
                else:
                    y_current_abs = drivers["supply_air"].unsqueeze(-1).expand_as(y_pred)

                # Reconstruct predicted absolute temperature in °C: T(t+k) = T(t) + ΔT
                y_pred_abs = (y_pred * std) + y_current_abs
                
                # Compute physics loss with absolute temperatures
                loss_physics_raw = physics_loss_fn(
                    y_pred=y_pred_abs,
                    y_current=y_current_abs,
                    physics_params=physics_params,
                    supply_air=drivers["supply_air"],
                    cooling_actuator=drivers["cooling_actuator"],
                    load_proxy=drivers.get("load_proxy")
                )
                
                # Normalize physics loss to same scale as data loss
                # Physics loss is in °C², data loss is in normalized units²
                # Convert to normalized scale by dividing by std²
                loss_physics = loss_physics_raw / (std ** 2)
                
                # Debug: log physics loss before clipping (first batch only)
                if batch_idx == 0:
                    # Show what's happening in the physics computation
                    delta_T_pred = (y_pred_abs - y_current_abs).mean().item()
                    print(f"  [DEBUG] Physics loss raw: {loss_physics_raw.item():.4f}, normalized: {loss_physics.item():.4f}")
                    print(f"  [DEBUG] Predicted delta_T: {delta_T_pred:.2f}°C")
                    print(f"  [DEBUG] Current temp range: [{y_current_abs.min().item():.1f}, {y_current_abs.max().item():.1f}]°C")
                    print(f"  [DEBUG] Predicted temp range: [{y_pred_abs.min().item():.1f}, {y_pred_abs.max().item():.1f}]°C")
                    if 'h' in physics_params:
                        print(f"  [DEBUG] Physics params - h: {physics_params['h'].mean().item():.6f}, C: {physics_params['C'].mean().item():.2f}")

                # Smoothly compress physics loss to avoid domination while preserving gradients
                loss_physics = torch.log1p(loss_physics)
                if batch_idx == 0:
                    print(f"  [DEBUG] Physics loss compressed (log1p): {loss_physics.item():.4f}")
                
                loss += loss_weights["physics"] * loss_physics
                loss_physics_sum += loss_physics.item()
        
        # Monotonic cooling loss
        if monotonic_loss_fn is not None and loss_weights.get("monotonic", 0) > 0:
            loss_mono = monotonic_loss_fn(model, X, t_idx)
            loss += loss_weights["monotonic"] * loss_mono
            loss_mono_sum += loss_mono.item()
        
        # Smoothness loss
        if loss_weights.get("smoothness", 0) > 0:
            loss_smooth = smoothness_loss_fn(y_pred)
            loss += loss_weights["smoothness"] * loss_smooth
            loss_smooth_sum += loss_smooth.item()

        if delta_bound_config is not None and loss_weights.get("delta_bound", 0) > 0 and scaler is not None:
            max_delta_c = float(delta_bound_config.get("max_delta_c", 6.0))
            target_col = target_cols[0] if target_cols else "gpu_temp_c"
            std = scaler.stats[target_col]["std"]
            delta_c = (y_pred * std).abs()
            excess = torch.relu(delta_c - max_delta_c)
            loss_delta_bound = torch.mean(excess ** 2)
            loss += loss_weights["delta_bound"] * loss_delta_bound
            loss_delta_bound_sum += loss_delta_bound.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return {
        "loss": total_loss / n_batches,
        "loss_data": loss_data_sum / n_batches,
        "loss_teacher": loss_teacher_sum / n_batches if teacher_preds is not None else 0.0,
        "loss_physics": loss_physics_sum / n_batches if physics_loss_fn is not None else 0.0,
        "loss_mono": loss_mono_sum / n_batches if monotonic_loss_fn is not None else 0.0,
        "loss_smooth": loss_smooth_sum / n_batches,
        "loss_delta_bound": loss_delta_bound_sum / n_batches if loss_weights.get("delta_bound", 0) > 0 else 0.0
    }


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    loss_weights: Dict[str, float],
    teacher_preds: Optional[torch.Tensor],
    physics_loss_fn: Optional[PhysicsODELoss],
    smoothness_loss_fn: TemporalSmoothnessLoss,
    feature_cols: list,
    target_cols: list,
    scaler: Optional[TargetScaler],
    physics_config: dict,
    delta_bound_config: Optional[Dict] = None
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    loss_data_sum = 0.0
    loss_teacher_sum = 0.0
    loss_physics_sum = 0.0
    loss_smooth_sum = 0.0
    loss_delta_bound_sum = 0.0
    n_batches = 0
    
    for batch_idx, (X, y, t_idx) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        t_idx = t_idx.to(device).float()
        
        out = model(X, t_idx, return_physics_params=(physics_loss_fn is not None))
        y_pred = out["delta_y"]

        # Build normalized delta target: y_delta = y(t+k) - y(t)
        y_current_norm = None
        if scaler is not None and feature_cols is not None and "gpu_temp_current" in feature_cols:
            target_col = target_cols[0] if target_cols else "gpu_temp_c"
            mean = scaler.stats[target_col]["mean"]
            std = scaler.stats[target_col]["std"]
            current_temp_idx = feature_cols.index("gpu_temp_current")
            y_current_abs = X[:, current_temp_idx].unsqueeze(-1)
            y_current_norm = (y_current_abs - mean) / std
            y_target = y - y_current_norm
        else:
            y_target = y

        loss_data = nn.functional.mse_loss(y_pred, y_target)
        loss = loss_weights["data"] * loss_data
        loss_data_sum += loss_data.item()
        
        if teacher_preds is not None and loss_weights.get("teacher", 0) > 0:
            batch_start = batch_idx * dataloader.batch_size
            batch_end = min(batch_start + len(X), len(teacher_preds))
            y_teacher = teacher_preds[batch_start:batch_end].to(device)
            
            if len(y_teacher) == len(y_pred):
                # Fix shape mismatch: squeeze both to [B] before computing loss
                y_pred_squeezed = y_pred.squeeze(-1)  # [B, 1] -> [B]
                y_teacher_squeezed = y_teacher.squeeze(-1)  # [B, 1] -> [B]
                if y_current_norm is not None:
                    # Teacher outputs absolute temperature in °C; convert to normalized, then to normalized delta
                    y_teacher_squeezed = ((y_teacher_squeezed - mean) / std) - y_current_norm.squeeze(-1)
                loss_teacher = nn.functional.mse_loss(y_pred_squeezed, y_teacher_squeezed)
                loss += loss_weights["teacher"] * loss_teacher
                loss_teacher_sum += loss_teacher.item()
        
        if physics_loss_fn is not None and loss_weights.get("physics", 0) > 0:
            drivers = extract_physics_drivers(
                X, feature_cols,
                supply_col=physics_config.get("supply_col", "supply_air_c"),
                actuator_col=physics_config.get("actuator_col", "evap_fan_speed_pct"),
                load_col=physics_config.get("load_col", "cool_demand_kwh") if physics_config.get("use_load_proxy") else None,
                window_size=physics_config.get("window_size", 12)
            )
            
            physics_params = out.get("physics_params", {})
            if physics_params and scaler is not None:
                target_col = target_cols[0] if target_cols else "gpu_temp_c"
                mean = scaler.stats[target_col]["mean"]
                std = scaler.stats[target_col]["std"]

                # Current temperature in °C
                if feature_cols is not None and "gpu_temp_current" in feature_cols:
                    current_temp_idx = feature_cols.index("gpu_temp_current")
                    y_current_abs = X[:, current_temp_idx].unsqueeze(-1).expand_as(y_pred)
                else:
                    y_current_abs = drivers["supply_air"].unsqueeze(-1).expand_as(y_pred)

                # Reconstruct predicted absolute temperature in °C: T(t+k) = T(t) + ΔT
                y_pred_abs = (y_pred * std) + y_current_abs
                
                # Compute physics loss with absolute temperatures
                loss_physics_raw = physics_loss_fn(
                    y_pred=y_pred_abs,
                    y_current=y_current_abs,
                    physics_params=physics_params,
                    supply_air=drivers["supply_air"],
                    cooling_actuator=drivers["cooling_actuator"],
                    load_proxy=drivers.get("load_proxy")
                )
                
                # Normalize physics loss to same scale as data loss
                # Physics loss is in °C², data loss is in normalized units²
                # Convert to normalized scale by dividing by std²
                loss_physics = loss_physics_raw / (std ** 2)

                # Smoothly compress physics loss to avoid domination while preserving gradients
                loss_physics = torch.log1p(loss_physics)
                
                loss += loss_weights["physics"] * loss_physics
                loss_physics_sum += loss_physics.item()
        
        if loss_weights.get("smoothness", 0) > 0:
            loss_smooth = smoothness_loss_fn(y_pred)
            loss += loss_weights["smoothness"] * loss_smooth
            loss_smooth_sum += loss_smooth.item()

        if delta_bound_config is not None and loss_weights.get("delta_bound", 0) > 0 and scaler is not None:
            max_delta_c = float(delta_bound_config.get("max_delta_c", 6.0))
            target_col = target_cols[0] if target_cols else "gpu_temp_c"
            std = scaler.stats[target_col]["std"]
            delta_c = (y_pred * std).abs()
            excess = torch.relu(delta_c - max_delta_c)
            loss_delta_bound = torch.mean(excess ** 2)
            loss += loss_weights["delta_bound"] * loss_delta_bound
            loss_delta_bound_sum += loss_delta_bound.item()
        
        total_loss += loss.item()
        n_batches += 1
    
    return {
        "loss": total_loss / n_batches,
        "loss_data": loss_data_sum / n_batches,
        "loss_teacher": loss_teacher_sum / n_batches if teacher_preds is not None else 0.0,
        "loss_physics": loss_physics_sum / n_batches if physics_loss_fn is not None else 0.0,
        "loss_smooth": loss_smooth_sum / n_batches,
        "loss_delta_bound": loss_delta_bound_sum / n_batches if loss_weights.get("delta_bound", 0) > 0 else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid PINN with curriculum learning")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dev-run", action="store_true", help="Dev run mode (small subset, few epochs)")
    parser.add_argument("--no-teacher", action="store_true", help="Disable teacher distillation")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = load_config(config_path)

    # Backward-compatible defaults for configs that predate the curriculum-style format
    if "dev_run" not in cfg or cfg["dev_run"] is None:
        cfg["dev_run"] = {"enabled": False}
    if "enabled" not in cfg["dev_run"]:
        cfg["dev_run"]["enabled"] = False
    if "training" in cfg and "num_workers" not in cfg["training"]:
        cfg["training"]["num_workers"] = 0
    
    # Override for dev run
    if args.dev_run:
        print("[DEV RUN MODE] Using small subset and few epochs")
        cfg["dev_run"]["enabled"] = True
    
    # Set seed
    set_seed(cfg["training"]["seed"])
    
    # Device
    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        device = "cpu"
    
    print(f"[INFO] Using device: {device}")
    
    # Prepare output directories
    artifacts_dir = Path(cfg["output"]["artifacts_dir"])
    metrics_dir = Path(cfg["output"]["metrics_dir"])
    plots_dir = Path(cfg["output"]["plots_dir"])
    logs_dir = Path(cfg["output"]["logs_dir"])
    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    
    for d in [artifacts_dir, metrics_dir, plots_dir, logs_dir, checkpoint_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Prepare dataset
    print("[INFO] Preparing k-ahead dataset...")
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
        cadence_seconds=cfg["data"]["cadence_seconds"],
        dev_run=cfg["dev_run"]["enabled"],
        max_samples=cfg["dev_run"].get("max_samples")
    )
    
    train_dataset = data_result["train_dataset"]
    val_dataset = data_result["val_dataset"]
    test_dataset = data_result["test_dataset"]
    scaler = data_result["scaler"]
    feature_cols = data_result["feature_cols"]
    target_cols = data_result["target_cols"]
    
    # Save scaler
    if scaler is not None:
        scaler.save(artifacts_dir / "scalers.json")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device == "cuda")
    )
    
    # Load teacher (optional)
    teacher = None
    teacher_train_preds = None
    teacher_val_preds = None
    
    if cfg["teacher"]["enabled"] and not args.no_teacher:
        teacher = load_teacher(
            model_path=Path(cfg["teacher"]["model_path"]),
            cache_dir=Path(cfg["teacher"]["cache_dir"]) if cfg["teacher"].get("cache_dir") else None,
            use_cache=cfg["teacher"].get("use_cache", True),
            allow_missing=True
        )
        
        if teacher is not None:
            # Filter out gpu_temp_current for teacher (RF was trained without it)
            teacher_feature_cols = [c for c in feature_cols if c != 'gpu_temp_current']
            
            # Validate compatibility with filtered features
            teacher.validate_compatibility(teacher_feature_cols, target_cols)
            
            # Get teacher predictions (cached if enabled)
            print("[INFO] Getting teacher predictions...")
            import pandas as pd
            
            # Build DataFrames from datasets for teacher prediction (exclude gpu_temp_current)
            teacher_feature_indices = [i for i, c in enumerate(feature_cols) if c != 'gpu_temp_current']
            X_train_teacher = train_dataset.X.numpy()[:, teacher_feature_indices]
            X_val_teacher = val_dataset.X.numpy()[:, teacher_feature_indices]
            
            X_train_df = pd.DataFrame(X_train_teacher, columns=teacher_feature_cols, index=train_dataset.timestamps)
            X_val_df = pd.DataFrame(X_val_teacher, columns=teacher_feature_cols, index=val_dataset.timestamps)
            
            teacher_train_preds = teacher.get_or_compute_predictions(X_train_df, "train", return_tensor=True)
            teacher_val_preds = teacher.get_or_compute_predictions(X_val_df, "val", return_tensor=True)
    
    # Create model
    print("[INFO] Creating Hybrid PINN model...")
    model = HybridPINN(
        input_dim=len(feature_cols),
        output_dim=len(target_cols),
        hidden_dims=cfg["model"]["hidden_dims"],
        activation=cfg["model"]["activation"],
        dropout=cfg["model"]["dropout"],
        time_embedding_enabled=cfg["model"]["time_embedding"]["enabled"],
        time_embedding_method=cfg["model"]["time_embedding"]["method"],
        time_embedding_n_freqs=cfg["model"]["time_embedding"]["n_freqs"],
        physics_head_enabled=cfg["model"]["physics_head"]["enabled"]
    ).to(device)
    
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss functions
    smoothness_loss_fn = TemporalSmoothnessLoss(
        order=cfg["losses"]["smoothness"]["order"]
    )
    
    physics_loss_fn = None
    if cfg["model"]["physics_head"]["enabled"]:
        # Physics residual must use the SAME horizon as the model's delta target.
        # Our model is trained on ΔT over k_ahead steps, so dt = k_ahead * cadence.
        dt_physics = cfg["data"]["k_ahead"] * cfg["data"]["cadence_seconds"]
        print(f"[INFO] Physics loss dt: {dt_physics}s (k_ahead={cfg['data']['k_ahead']}, cadence={cfg['data']['cadence_seconds']}s)")
        physics_loss_fn = PhysicsODELoss(
            window_size=cfg["losses"]["physics"]["window_size"],
            dt=dt_physics,
            use_load_proxy=cfg["losses"]["physics"]["use_load_proxy"]
        )
    
    monotonic_loss_fn = None
    if cfg["losses"]["monotonic"]["actuator_col"] in feature_cols:
        monotonic_loss_fn = create_monotonic_loss(
            feature_cols=feature_cols,
            actuator_col=cfg["losses"]["monotonic"]["actuator_col"],
            epsilon=cfg["losses"]["monotonic"]["epsilon"],
            penalty_type=cfg["losses"]["monotonic"]["penalty_type"]
        )
    
    # Physics config for driver extraction
    physics_config = {
        "supply_col": cfg["losses"]["physics"].get("supply_col", "supply_air_c"),
        "actuator_col": cfg["losses"]["physics"].get("actuator_col", "evap_fan_speed_pct"),
        "load_col": cfg["losses"]["physics"].get("load_proxy_col", "cool_demand_kwh"),
        "use_load_proxy": cfg["losses"]["physics"]["use_load_proxy"],
        "window_size": cfg["losses"]["physics"]["window_size"]
    }
    
    # Curriculum phases
    curriculum = cfg["training"]["curriculum"]
    phases = [
        ("phase1", curriculum["phase1"]),
        ("phase2", curriculum["phase2"]),
        ("phase3", curriculum["phase3"])
    ]
    
    # Override epochs for dev run
    if cfg["dev_run"]["enabled"]:
        max_epochs_dev = cfg["dev_run"].get("max_epochs", 2)
        for _, phase_cfg in phases:
            phase_cfg["epochs"] = min(phase_cfg["epochs"], max_epochs_dev)
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_loss_data": [],
        "val_loss_data": [],
        "train_loss_physics": [],
        "val_loss_physics": [],
        "val_skill_rmse": []
    }
    
    best_val_skill = -float("inf")
    patience_counter = 0
    
    # Curriculum training
    for phase_name, phase_cfg in phases:
        print(f"\n{'='*60}")
        print(f"PHASE: {phase_cfg['name'].upper()}")
        print(f"{'='*60}")
        
        loss_weights = phase_cfg["loss_weights"]
        n_epochs = phase_cfg["epochs"]
        
        # Freeze physics parameters for first 10 epochs of phase2 (physics_on)
        if phase_name == "phase2" and model.physics_head is not None:
            print("[INFO] Freezing physics parameters for first 10 epochs")
            for param in model.physics_head.parameters():
                param.requires_grad = False
        
        # Create optimizer and scheduler for this phase
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["training"]["lr_initial"],
            weight_decay=cfg["training"]["weight_decay"]
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs,
            eta_min=cfg["training"]["lr_final"]
        )
        
        for epoch in range(n_epochs):
            # Unfreeze physics parameters after 10 epochs in phase2
            if phase_name == "phase2" and epoch == 10 and model.physics_head is not None:
                print("[INFO] Unfreezing physics parameters")
                for param in model.physics_head.parameters():
                    param.requires_grad = True
                # Recreate optimizer to include physics parameters
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=optimizer.param_groups[0]['lr'],
                    weight_decay=cfg["training"]["weight_decay"]
                )
            # Train
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, device,
                loss_weights, teacher_train_preds,
                physics_loss_fn, physics_config,
                feature_cols, target_cols, scaler,
                monotonic_loss_fn, smoothness_loss_fn,
                delta_bound_config=cfg.get("losses", {}).get("delta_bound")
            )
            
            # Validate
            val_metrics = validate_epoch(
                model, val_loader, device,
                loss_weights, teacher_val_preds,
                physics_loss_fn, smoothness_loss_fn,
                feature_cols, target_cols, scaler,
                physics_config,
                delta_bound_config=cfg.get("losses", {}).get("delta_bound")
            )
            
            scheduler.step()
            
            # Compute validation skill (quick estimate)
            val_skill_rmse = 0.0  # Placeholder; full eval is expensive
            
            # Log
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_loss_data"].append(train_metrics["loss_data"])
            history["val_loss_data"].append(val_metrics["loss_data"])
            history["train_loss_physics"].append(train_metrics.get("loss_physics", 0.0))
            history["val_loss_physics"].append(val_metrics.get("loss_physics", 0.0))
            history["val_skill_rmse"].append(val_skill_rmse)
            
            print(f"[{phase_cfg['name']}] Epoch {epoch+1}/{n_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Data: {val_metrics['loss_data']:.4f} | "
                  f"Physics: {val_metrics.get('loss_physics', 0.0):.4f}")
            
            # Early stopping (based on val loss for simplicity)
            if cfg["training"]["early_stopping"]["enabled"]:
                if val_metrics["loss"] < best_val_skill or epoch == 0:
                    best_val_skill = val_metrics["loss"]
                    patience_counter = 0
                    
                    # Save best model
                    if cfg["output"]["save_best"]:
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "epoch": epoch,
                            "phase": phase_name,
                            "val_loss": val_metrics["loss"]
                        }, artifacts_dir / "best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= cfg["training"]["early_stopping"]["patience"]:
                        print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
                        break
    
    # Save final model
    if cfg["output"]["save_last"]:
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "phase": phase_name
        }, artifacts_dir / "last_model.pt")
    
    # Plot loss curves
    plot_loss_curves(history, plots_dir / "loss_curves.png")
    
    # Save training history
    with open(logs_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n[INFO] Training complete!")
    print(f"[INFO] Best model saved to: {artifacts_dir / 'best_model.pt'}")
    print(f"[INFO] Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
