# RC+NN vs PINN-lite: Decision Guide

**Date**: May 28, 2026  
**Context**: Need neural network-based surrogate to capture nonlinearities  
**Deadline**: Mid-June report (2 weeks)

---

## TL;DR Recommendation

✅ **Use RC+NN for mid-semester report**  
⭐ **Add PINN-lite post mid-sem if time permits**

**Reason**: RC+NN is faster to implement (1-2 days vs 3-5 days) and sufficient to demonstrate hybrid physics+ML approach.

---

## Comparison Table

| Criterion | RC+NN | PINN-lite | Winner |
|-----------|-------|-----------|--------|
| **Training Time** | 1-2 hours | 4-8 hours | RC+NN ✅ |
| **Implementation Time** | 1 day | 3-5 days | RC+NN ✅ |
| **Complexity** | Low (standard supervised) | High (physics loss tuning) | RC+NN ✅ |
| **Debugging Difficulty** | Easy | Hard (gradient issues) | RC+NN ✅ |
| **Data Efficiency** | Medium | High (physics guides) | PINN-lite ✅ |
| **Accuracy Potential** | Good | Excellent | PINN-lite ✅ |
| **Interpretability** | Medium (RC + black box) | High (physics + learned) | PINN-lite ✅ |
| **Dissertation Story** | Good (hybrid approach) | Better (physics-informed) | PINN-lite ✅ |
| **Risk** | Low (standard ML) | Medium (may not converge) | RC+NN ✅ |
| **Time to Results** | 1-2 days | 3-5 days | RC+NN ✅ |

**Score**: RC+NN wins 7/10 criteria for **time-constrained** scenario

---

## Detailed Comparison

### 1. Architecture

#### RC+NN (Residual Learning)
```
Input: [temp, ambient, power, fan_speed]
  ↓
RC Physics Model → T_RC
  ↓
Neural Network → residual
  ↓
Output: T_pred = T_RC + residual
```

**Pros**:
- Simple: NN only learns corrections to RC
- Fast: Standard supervised learning
- Stable: RC provides physics baseline

**Cons**:
- NN is a black box (less interpretable)
- Assumes RC captures main dynamics

#### PINN-lite (Physics-Informed)
```
Input: [temp, ambient, power, fan_speed]
  ↓
Neural Network → T_pred
  ↓
Physics Loss: Check if dT/dt matches RC equation
  ↓
Combined Loss: Data Loss + λ * Physics Loss
```

**Pros**:
- Physics constraints embedded in training
- Better generalization (physics guides learning)
- More interpretable (respects physics)

**Cons**:
- Complex: Need to tune physics loss weight (λ)
- Slow: Gradient computation through physics
- Risky: May not converge if λ too high/low

---

### 2. Training Procedure

#### RC+NN Training (Simple)

**Step 1**: Generate RC predictions
```python
# Compute RC predictions on all data
T_rc = rc_model.predict(X)
```

**Step 2**: Compute residuals
```python
# Residuals = actual - RC prediction
residuals = T_actual - T_rc
```

**Step 3**: Train NN on residuals
```python
# Standard supervised learning
nn_model.fit(X, residuals)
```

**Step 4**: Combine for inference
```python
# Final prediction
T_pred = rc_model.predict(X) + nn_model.predict(X)
```

**Time**: ~1-2 hours training, 1 day implementation

#### PINN-lite Training (Complex)

**Step 1**: Define physics loss
```python
def physics_loss(model, X, T_current):
    # Predict next temp
    T_pred = model(X)
    
    # Compute dT/dt from prediction
    dT_dt_pred = (T_pred - T_current) / dt
    
    # Compute dT/dt from RC physics
    dT_dt_physics = rc_equation(X)
    
    # Physics loss = difference
    loss_physics = MSE(dT_dt_pred, dT_dt_physics)
    return loss_physics
```

**Step 2**: Combined training
```python
for epoch in range(epochs):
    # Data loss
    T_pred = model(X)
    loss_data = MSE(T_pred, T_actual)
    
    # Physics loss
    loss_physics = physics_loss(model, X, T_current)
    
    # Combined loss
    loss = loss_data + lambda_physics * loss_physics
    
    # Backprop
    loss.backward()
    optimizer.step()
```

**Challenges**:
- Tuning λ (physics weight): too high → underfits data, too low → ignores physics
- Gradient issues: physics loss can explode
- Normalization: need to denormalize for physics loss

**Time**: ~4-8 hours training, 3-5 days implementation + debugging

---

### 3. Expected Performance

#### RC+NN Performance Estimate

Based on typical residual learning:
```
RC alone:        MAE ~1.5°C
NN correction:   Reduces error by 30-50%
RC+NN hybrid:    MAE ~0.8-1.0°C
```

**Why it works**:
- RC captures main thermal dynamics (heating, cooling, dissipation)
- NN learns systematic errors (e.g., nonlinear fan response, thermal lag)

#### PINN-lite Performance Estimate

Based on physics-informed learning:
```
Pure NN:         MAE ~1.2°C (may drift)
PINN-lite:       MAE ~0.6-0.8°C (stable)
```

**Why it's better**:
- Physics constraints prevent non-physical predictions
- Better generalization to unseen conditions
- More stable multi-step rollouts

---

### 4. Implementation Status

#### RC+NN: Ready to Go ✅

**What you have**:
- ✅ RC adapter working (`src/rl/surrogates/rc_adapter.py`)
- ✅ Training script created (`scripts/training/train_rc_nn.py`)
- ✅ Adapter interface defined (just needs RC+NN wrapper)

**What you need**:
1. Create RC+NN adapter (30 min)
2. Run training script (1-2 hours)
3. Test in RL environment (30 min)

**Total time**: 1 day

#### PINN-lite: Partially Done ⚠️

**What you have**:
- ✅ PINN model architecture (`src/pinn/models/hybrid_pinn.py`)
- ✅ Physics loss implementation (fixed in previous sessions)
- ⚠️ Training script exists but may need tuning

**What you need**:
1. Verify physics loss is stable (1 day)
2. Tune hyperparameters (λ, lr, epochs) (1-2 days)
3. Full training run (4-8 hours)
4. Debug if issues arise (1-2 days)
5. Create PINN adapter (already done)

**Total time**: 3-5 days (with risk of delays)

---

### 5. Dissertation Story

#### With RC+NN

**Narrative**:
> "We evaluate three surrogate approaches:
> 1. **RC (physics-only)**: Interpretable but limited accuracy (MAE 1.5°C)
> 2. **RF (data-only)**: Flexible but may drift (MAE 2.5°C)
> 3. **RC+NN (hybrid)**: Combines physics baseline with learned corrections (MAE 0.9°C)
>
> Results show hybrid approaches outperform pure physics or pure ML."

**Contributions**:
- ✅ Shows value of physics-guided ML
- ✅ Demonstrates hybrid approach
- ✅ Sufficient for M.Tech dissertation

#### With PINN-lite (Better Story)

**Narrative**:
> "We evaluate four surrogate approaches:
> 1. **RC (physics-only)**: Interpretable but limited (MAE 1.5°C)
> 2. **RF (data-only)**: Flexible but drifts (MAE 2.5°C)
> 3. **RC+NN (residual hybrid)**: Physics + learned corrections (MAE 0.9°C)
> 4. **PINN-lite (physics-informed)**: Physics-constrained learning (MAE 0.7°C)
>
> Results show physics-informed methods achieve best accuracy and stability."

**Contributions**:
- ✅ Shows value of physics-guided ML
- ✅ Compares two hybrid approaches
- ✅ Demonstrates physics-informed learning
- ✅ Stronger for publication

---

### 6. Risk Assessment

#### RC+NN Risks: LOW ✅

**Potential Issues**:
- NN might not improve much over RC (unlikely - there are always residuals)
- Overfitting (mitigated by dropout, validation)

**Mitigation**:
- Use small NN (32-16 hidden units)
- Early stopping on validation loss
- If improvement < 20%, still shows hybrid approach

**Worst case**: RC+NN = RC performance → Still have RC for experiments

#### PINN-lite Risks: MEDIUM ⚠️

**Potential Issues**:
- Physics loss explodes (seen in previous sessions)
- λ tuning takes multiple runs
- May not converge in time for mid-sem

**Mitigation**:
- Use gradient clipping
- Start with low λ (0.01) and increase gradually
- Have RC+NN as backup

**Worst case**: PINN doesn't converge → Fall back to RC+NN

---

## Recommended Strategy

### For Mid-Semester Report (2 weeks):

#### Week 1:
1. **Day 1**: Train RC+NN (use provided script)
2. **Day 2-4**: RL training with RC
3. **Day 5**: Create RC+NN adapter
4. **Day 6-7**: Run surrogate evaluation (RC vs RF vs RC+NN)

#### Week 2:
1. **Day 8-9**: Controller comparison experiments
2. **Day 10-11**: Analyze results
3. **Day 12-14**: Write mid-sem report

**Surrogates for mid-sem**: RC, RF, RC+NN ✅

### Post Mid-Semester (Optional):

If you have time after mid-sem report:
1. Train PINN-lite (3-5 days)
2. Re-run surrogate evaluation with PINN
3. Update final dissertation with 4-way comparison

**Surrogates for final**: RC, RF, RC+NN, PINN-lite ✅

---

## Decision Matrix

### Choose RC+NN if:
- ✅ You have < 2 weeks to mid-sem
- ✅ You want guaranteed results
- ✅ You're comfortable with standard ML
- ✅ You want low-risk approach

### Choose PINN-lite if:
- ✅ You have > 3 weeks to mid-sem
- ✅ You want stronger dissertation story
- ✅ You're comfortable debugging physics losses
- ✅ You can afford risk of delays

### Choose BOTH if:
- ✅ You have time after mid-sem report
- ✅ You want comprehensive comparison
- ✅ You're targeting publication

---

## Practical Steps to Implement RC+NN

### Step 1: Train RC+NN Model (1-2 hours)

```bash
python scripts/training/train_rc_nn.py \
  --data data/synthetic/thermal_dataset.parquet \
  --output-dir results/rc_nn_training \
  --bundle-path models/rc_nn_hybrid.pkl \
  --hidden-dims 32 16 \
  --epochs 100 \
  --lr 0.001
```

**Expected output**:
```
RC MAE:     1.5°C
Hybrid MAE: 0.9°C
Improvement: 0.6°C (40%)
```

### Step 2: Create RC+NN Adapter (30 min)

File: `src/rl/surrogates/rc_nn_adapter.py`

```python
"""RC+NN hybrid surrogate adapter."""

import joblib
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from src.rl.surrogates.rc_adapter import RCAdapter
from scripts.training.train_rc_nn import ResidualNN


class RCNNAdapter:
    """Adapter for RC+NN hybrid surrogate."""
    
    def __init__(self, bundle_path: Path, device: str = 'cpu'):
        # Load bundle
        bundle = joblib.load(bundle_path)
        
        # Create RC adapter
        self.rc = RCAdapter(**bundle['rc_params'])
        
        # Create NN model
        nn_config = bundle['nn_config']
        self.nn = ResidualNN(
            input_dim=nn_config['input_dim'],
            hidden_dims=nn_config['hidden_dims']
        )
        self.nn.load_state_dict(bundle['nn_state_dict'])
        self.nn.to(device)
        self.nn.eval()
        
        self.device = device
        self.input_mean = bundle['input_mean']
        self.input_std = bundle['input_std']
    
    def reset(self, seed=None, init_state=None):
        self.rc.reset(seed=seed, init_state=init_state)
    
    def predict_next(self, state: np.ndarray, action: np.ndarray) -> float:
        # RC prediction
        temp_rc = self.rc.predict_next(state, action)
        
        # NN residual
        features = np.array([state[0], state[1], state[2], action[0]])
        features_norm = (features - self.input_mean) / (self.input_std + 1e-8)
        
        with torch.no_grad():
            x_tensor = torch.from_numpy(features_norm).float().unsqueeze(0).to(self.device)
            residual = self.nn(x_tensor).cpu().item()
        
        # Combined
        temp_pred = temp_rc + residual
        return float(np.clip(temp_pred, 30.0, 95.0))
    
    @property
    def warmup_steps(self) -> int:
        return 0
```

### Step 3: Update Surrogate Factory (10 min)

Add to `src/rl/surrogates/factory.py`:

```python
elif surrogate_type == "rc_nn":
    from src.rl.surrogates.rc_nn_adapter import RCNNAdapter
    bundle_path = config.get("bundle_path", "models/rc_nn_hybrid.pkl")
    device = config.get("device", "cpu")
    return RCNNAdapter(bundle_path=Path(bundle_path), device=device)
```

### Step 4: Create Config (5 min)

File: `configs/rl/sac_unified_rc_nn.yaml`

```yaml
run_name: sac_rc_nn_hybrid
output_dir: runs/rl

surrogate:
  type: rc_nn
  bundle_path: models/rc_nn_hybrid.pkl
  device: cpu

env:
  max_steps: 300
  temp_warning: 80.0
  temp_critical: 90.0
  temp_target: 75.0
  # ... rest same as sac_unified_rc.yaml
```

### Step 5: Test (30 min)

```python
# Quick test
from src.rl.surrogates.factory import create_surrogate

config = {'type': 'rc_nn', 'bundle_path': 'models/rc_nn_hybrid.pkl'}
surrogate = create_surrogate(config)

state = np.array([70.0, 25.0, 200.0, 50.0, 0.0])
action = np.array([60.0])

temp_next = surrogate.predict_next(state, action)
print(f"Predicted next temp: {temp_next:.2f}°C")
```

---

## Final Recommendation

### For Your Situation:

**Timeline**: 2 weeks to mid-sem  
**Goal**: Demonstrate hybrid physics+ML approach  
**Risk tolerance**: Low (need guaranteed results)

**Recommendation**: ✅ **Use RC+NN**

### Action Plan:

1. **Today**: Train RC+NN (1-2 hours)
2. **Tomorrow**: Create adapter and test (1 hour)
3. **This week**: Run RL training with RC (already planned)
4. **Next week**: Evaluate RC vs RF vs RC+NN

### For Final Dissertation:

After mid-sem, if you have 1-2 weeks:
- Train PINN-lite
- Add to comparison
- Upgrade to 4-way surrogate comparison

This gives you:
- ✅ Guaranteed results for mid-sem (RC+NN)
- ✅ Stronger final dissertation (add PINN later)
- ✅ Low risk, high reward strategy

---

## Summary

| Aspect | RC+NN | PINN-lite |
|--------|-------|-----------|
| **For mid-sem** | ✅ Perfect | ⚠️ Risky |
| **For final** | ✅ Good | ✅ Better |
| **Time needed** | 1-2 days | 3-5 days |
| **Risk** | Low | Medium |
| **Story quality** | Good | Excellent |

**Bottom line**: Start with RC+NN, add PINN-lite later if time permits.

---

**Next Step**: Run the RC+NN training script I created!

```bash
python scripts/training/train_rc_nn.py \
  --data data/synthetic/thermal_dataset.parquet \
  --bundle-path models/rc_nn_hybrid.pkl
```
