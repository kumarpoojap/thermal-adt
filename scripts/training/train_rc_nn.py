#!/usr/bin/env python3
"""
Train RC+NN Hybrid Surrogate Model

Combines physics-based RC model with neural network residual correction.
Architecture: T_pred = T_RC + NN(residual)

Usage:
    python scripts/training/train_rc_nn.py \
        --data data/synthetic/thermal_dataset.parquet \
        --output-dir results/rc_nn_training \
        --bundle-path models/rc_nn_hybrid.pkl
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rl.surrogates.rc_adapter import RCAdapter


class ResidualNN(nn.Module):
    """
    Simple MLP to predict RC model residuals.
    
    Input: [temp, ambient, power, fan_speed]
    Output: temperature residual (correction to RC prediction)
    """
    
    def __init__(self, input_dim=4, hidden_dims=[32, 16], dropout=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (no activation - predicting residual)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class RCNNAdapter:
    """
    Hybrid RC+NN surrogate adapter.
    
    Combines physics-based RC prediction with learned NN residual.
    """
    
    def __init__(
        self,
        rc_adapter: RCAdapter,
        nn_model: ResidualNN,
        device: str = 'cpu',
        input_mean: np.ndarray = None,
        input_std: np.ndarray = None
    ):
        self.rc = rc_adapter
        self.nn = nn_model.to(device)
        self.device = device
        self.nn.eval()
        
        self.input_mean = input_mean
        self.input_std = input_std
    
    def reset(self, seed=None, init_state=None):
        self.rc.reset(seed=seed, init_state=init_state)
    
    def predict_next(self, state: np.ndarray, action: np.ndarray) -> float:
        """Predict next temperature using RC + NN correction."""
        # RC physics prediction
        temp_rc = self.rc.predict_next(state, action)
        
        # NN residual correction
        features = np.array([
            state[0],  # temp
            state[1],  # ambient
            state[2],  # power
            action[0]  # fan_speed
        ], dtype=np.float32)
        
        # Normalize
        if self.input_mean is not None and self.input_std is not None:
            features = (features - self.input_mean) / (self.input_std + 1e-8)
        
        x_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            residual = self.nn(x_tensor).cpu().item()
        
        # Combined prediction
        temp_pred = temp_rc + residual
        
        # Clip to physical bounds
        temp_pred = np.clip(temp_pred, 30.0, 95.0)
        
        return float(temp_pred)
    
    @property
    def warmup_steps(self) -> int:
        return 0


def prepare_data(df: pd.DataFrame):
    """
    Prepare data for RC+NN training.
    
    Returns:
        X: Features [temp, ambient, power, fan_speed]
        y: Target temperatures (next step)
    """
    # Ensure required columns exist
    required = ['gpu_temp_c', 'ambient_temp_c', 'gpu_power_w', 'fan_speed_pct']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Build feature matrix
    X = df[required].values
    
    # Target: next temperature (shift by -1)
    y = df['gpu_temp_c'].shift(-1).values
    
    # Remove last row (no next temp)
    X = X[:-1]
    y = y[:-1]
    
    # Remove NaNs
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    return X, y


def compute_rc_predictions(X: np.ndarray, rc_adapter: RCAdapter):
    """Compute RC predictions for all samples."""
    rc_preds = []
    
    for i in range(len(X)):
        temp, ambient, power, fan = X[i]
        
        # Build state vector for RC
        state = np.array([temp, ambient, power, fan, 0.0])
        action = np.array([fan])
        
        temp_rc = rc_adapter.predict_next(state, action)
        rc_preds.append(temp_rc)
    
    return np.array(rc_preds)


def train_nn(
    X_train, residuals_train,
    X_val, residuals_val,
    hidden_dims=[32, 16],
    lr=0.001,
    epochs=100,
    batch_size=256,
    device='cpu'
):
    """Train NN to predict residuals."""
    
    # Normalize inputs
    input_mean = X_train.mean(axis=0)
    input_std = X_train.std(axis=0)
    
    X_train_norm = (X_train - input_mean) / (input_std + 1e-8)
    X_val_norm = (X_val - input_mean) / (input_std + 1e-8)
    
    # Convert to tensors
    X_train_t = torch.from_numpy(X_train_norm).float()
    y_train_t = torch.from_numpy(residuals_train).float().unsqueeze(1)
    
    X_val_t = torch.from_numpy(X_val_norm).float()
    y_val_t = torch.from_numpy(residuals_val).float().unsqueeze(1)
    
    # Create model
    model = ResidualNN(input_dim=4, hidden_dims=hidden_dims).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\nTraining NN (epochs={epochs}, batch_size={batch_size}, lr={lr})")
    
    for epoch in range(epochs):
        model.train()
        
        # Mini-batch training
        indices = torch.randperm(len(X_train_t))
        train_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(X_train_t), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx].to(device)
            y_batch = y_train_t[batch_idx].to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(device))
            val_loss = criterion(val_outputs, y_val_t.to(device)).item()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, input_mean, input_std, history


def evaluate(y_true, y_pred, split_name):
    """Compute evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    max_err = np.abs(y_true - y_pred).max()
    
    return {
        'split': split_name,
        'mae': mae,
        'rmse': rmse,
        'max_error': max_err,
        'n_samples': len(y_true)
    }


def main():
    parser = argparse.ArgumentParser(description="Train RC+NN hybrid surrogate")
    
    # Data
    parser.add_argument("--data", required=True, help="Path to thermal dataset (parquet)")
    parser.add_argument("--output-dir", default="results/rc_nn_training", help="Output directory")
    parser.add_argument("--bundle-path", default="models/rc_nn_hybrid.pkl", help="Output bundle path")
    
    # RC parameters
    parser.add_argument("--rc-C", type=float, default=100.0, help="Thermal capacity")
    parser.add_argument("--rc-h", type=float, default=0.05, help="Heat transfer coeff")
    parser.add_argument("--rc-beta", type=float, default=-0.03, help="Cooling effectiveness")
    parser.add_argument("--rc-gamma", type=float, default=0.01, help="Power-to-heat conversion")
    
    # NN parameters
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[32, 16], help="Hidden layer sizes")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    
    # Other
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("RC+NN HYBRID SURROGATE TRAINING")
    print("="*80)
    
    # 1. Load data
    print(f"\n[1/6] Loading data: {args.data}")
    df = pd.read_parquet(args.data)
    print(f"  Loaded {len(df)} rows")
    
    # 2. Prepare data
    print("\n[2/6] Preparing features and targets")
    X, y = prepare_data(df)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=args.random_state
    )
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 3. Create RC adapter
    print("\n[3/6] Creating RC adapter")
    rc_adapter = RCAdapter(
        thermal_capacity=args.rc_C,
        heat_transfer_coeff=args.rc_h,
        cooling_effectiveness=args.rc_beta,
        power_to_heat=args.rc_gamma,
        dt=1.0
    )
    print(f"  RC params: C={args.rc_C}, h={args.rc_h}, beta={args.rc_beta}, gamma={args.rc_gamma}")
    
    # 4. Compute RC predictions and residuals
    print("\n[4/6] Computing RC predictions and residuals")
    y_rc_train = compute_rc_predictions(X_train, rc_adapter)
    y_rc_val = compute_rc_predictions(X_val, rc_adapter)
    y_rc_test = compute_rc_predictions(X_test, rc_adapter)
    
    residuals_train = y_train - y_rc_train
    residuals_val = y_val - y_rc_val
    residuals_test = y_test - y_rc_test
    
    print(f"  RC MAE (train): {mean_absolute_error(y_train, y_rc_train):.2f}°C")
    print(f"  RC MAE (val):   {mean_absolute_error(y_val, y_rc_val):.2f}°C")
    print(f"  RC MAE (test):  {mean_absolute_error(y_test, y_rc_test):.2f}°C")
    
    # 5. Train NN on residuals
    print("\n[5/6] Training NN to predict residuals")
    nn_model, input_mean, input_std, history = train_nn(
        X_train, residuals_train,
        X_val, residuals_val,
        hidden_dims=args.hidden_dims,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # 6. Evaluate hybrid model
    print("\n[6/6] Evaluating RC+NN hybrid")
    
    # Compute NN predictions
    X_train_norm = (X_train - input_mean) / (input_std + 1e-8)
    X_val_norm = (X_val - input_mean) / (input_std + 1e-8)
    X_test_norm = (X_test - input_mean) / (input_std + 1e-8)
    
    nn_model.eval()
    with torch.no_grad():
        residuals_pred_train = nn_model(torch.from_numpy(X_train_norm).float().to(args.device)).cpu().numpy().flatten()
        residuals_pred_val = nn_model(torch.from_numpy(X_val_norm).float().to(args.device)).cpu().numpy().flatten()
        residuals_pred_test = nn_model(torch.from_numpy(X_test_norm).float().to(args.device)).cpu().numpy().flatten()
    
    # Hybrid predictions
    y_hybrid_train = y_rc_train + residuals_pred_train
    y_hybrid_val = y_rc_val + residuals_pred_val
    y_hybrid_test = y_rc_test + residuals_pred_test
    
    # Metrics
    metrics_rc = [
        evaluate(y_train, y_rc_train, 'train_rc'),
        evaluate(y_val, y_rc_val, 'val_rc'),
        evaluate(y_test, y_rc_test, 'test_rc')
    ]
    
    metrics_hybrid = [
        evaluate(y_train, y_hybrid_train, 'train_hybrid'),
        evaluate(y_val, y_hybrid_val, 'val_hybrid'),
        evaluate(y_test, y_hybrid_test, 'test_hybrid')
    ]
    
    metrics_df = pd.DataFrame(metrics_rc + metrics_hybrid)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(metrics_df.to_string(index=False))
    
    improvement = (metrics_df[metrics_df['split']=='test_rc']['mae'].values[0] - 
                   metrics_df[metrics_df['split']=='test_hybrid']['mae'].values[0])
    print(f"\nTest MAE Improvement: {improvement:.2f}°C ({improvement/metrics_df[metrics_df['split']=='test_rc']['mae'].values[0]*100:.1f}%)")
    
    # Save results
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    
    # Plot training history
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('NN Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    print(f"\nSaved training history: {output_dir / 'training_history.png'}")
    
    # Save bundle
    bundle = {
        'rc_params': {
            'thermal_capacity': args.rc_C,
            'heat_transfer_coeff': args.rc_h,
            'cooling_effectiveness': args.rc_beta,
            'power_to_heat': args.rc_gamma,
            'dt': 1.0
        },
        'nn_state_dict': nn_model.state_dict(),
        'nn_config': {
            'input_dim': 4,
            'hidden_dims': args.hidden_dims
        },
        'input_mean': input_mean,
        'input_std': input_std,
        'metrics': metrics_df.to_dict('records'),
        'training_args': vars(args)
    }
    
    joblib.dump(bundle, args.bundle_path)
    print(f"\n✓ Saved RC+NN bundle: {args.bundle_path}")
    print(f"  RC MAE: {metrics_df[metrics_df['split']=='test_rc']['mae'].values[0]:.2f}°C")
    print(f"  Hybrid MAE: {metrics_df[metrics_df['split']=='test_hybrid']['mae'].values[0]:.2f}°C")
    print(f"  Improvement: {improvement:.2f}°C")


if __name__ == "__main__":
    main()
