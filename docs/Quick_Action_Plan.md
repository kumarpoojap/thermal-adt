# Quick Action Plan for Mid-June Report

**Date**: May 28, 2026  
**Deadline**: Mid-June (2 weeks)  
**Goal**: Complete critical experiments for mid-semester report

---

## TL;DR

✅ **Good News**: Your code is 85% correct already!  
⚠️ **One Fix Needed**: Retrain RF for k-ahead=1 (currently k-ahead=10)  
🎯 **Focus**: RL training + Controller comparison + Surrogate evaluation

---

## Critical Path (Next 2 Weeks)

### Week 1: Training & Setup

#### Day 1 (Today): Train RC+NN Hybrid Surrogate
```bash
# Option A: Train locally (CPU, ~1-2 hours)
python scripts/training/train_rc_nn.py \
  --data data/synthetic/thermal_dataset.parquet \
  --output-dir results/rc_nn_training \
  --bundle-path models/rc_nn_hybrid.pkl \
  --hidden-dims 32 16 \
  --epochs 100 \
  --batch-size 256 \
  --lr 0.001

# Option B: Train on Colab (GPU, ~5-10 min)
# Upload notebooks/RC_NN_Training_Colab.ipynb to Google Colab
# Update GitHub repo URL in cell 3
# Run all cells
# Download rc_nn_hybrid.pkl from Google Drive
```

**Expected Output**:
- RC alone: MAE ~1.5°C
- RC+NN hybrid: MAE ~0.8-1.0°C
- Improvement: 30-50%

**Also Today (if time)**: Verify RF k-ahead
```bash
# Check current RF k-ahead
python -c "import joblib; b=joblib.load('models/rf_teacher.pkl'); print(f'k_ahead={b.get(\"k_ahead\", \"unknown\")}')"

# If k-ahead=10, retrain for k-ahead=1 (optional, for comparison)
python scripts/training/train_rf.py \
  --data data/synthetic/thermal_dataset.parquet \
  --config configs/data/gpu_thermal_spec.json \
  --k-ahead 1 \
  --n-estimators 500 \
  --max-depth None \
  --bundle-path models/rf_teacher_1step.pkl
```

#### Day 2: Create RC+NN Adapter

Create `src/rl/surrogates/rc_nn_adapter.py`:
```python
import joblib
import torch
import numpy as np
from pathlib import Path
from src.rl.surrogates.rc_adapter import RCAdapter
from scripts.training.train_rc_nn import ResidualNN

class RCNNAdapter:
    """Adapter for RC+NN hybrid surrogate."""
    
    def __init__(self, bundle_path: Path, device: str = 'cpu'):
        bundle = joblib.load(bundle_path)
        self.rc = RCAdapter(**bundle['rc_params'])
        
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
        temp_rc = self.rc.predict_next(state, action)
        features = np.array([state[0], state[1], state[2], action[0]])
        features_norm = (features - self.input_mean) / (self.input_std + 1e-8)
        
        with torch.no_grad():
            x_tensor = torch.from_numpy(features_norm).float().unsqueeze(0).to(self.device)
            residual = self.nn(x_tensor).cpu().item()
        
        temp_pred = temp_rc + residual
        return float(np.clip(temp_pred, 30.0, 95.0))
    
    @property
    def warmup_steps(self) -> int:
        return 0
```

Update `src/rl/surrogates/factory.py`:
```python
# Add to create_surrogate function
elif surrogate_type == "rc_nn":
    from src.rl.surrogates.rc_nn_adapter import RCNNAdapter
    bundle_path = config.get("bundle_path", "models/rc_nn_hybrid.pkl")
    device = config.get("device", "cpu")
    return RCNNAdapter(bundle_path=Path(bundle_path), device=device)
```

Create `configs/rl/sac_unified_rc_nn.yaml`:
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
  initial_temp_range: [40.0, 60.0]
  ambient_range: [20.0, 30.0]
  power_range: [100.0, 300.0]
  reward_weights:
    thermal: 10.0
    energy: 0.1
    oscillation: 1.0
    headroom: 2.0

use_safety: true
# ... rest same as sac_unified_rc.yaml
```

#### Day 3-5: RL Training with RC Surrogate
```bash
# Train RL agent with RC surrogate
python scripts/training/train_sac_unified.py \
  --config configs/rl/sac_unified_rc.yaml \
  --output-dir runs/rl/sac_rc
```
**Expected**: 200k steps, ~2-3 days on CPU (faster on GPU)

**Parallel (if resources available)**: Start RL training with RC+NN
```bash
# Train RL agent with RC+NN surrogate (better performance expected)
python scripts/training/train_sac_unified.py \
  --config configs/rl/sac_unified_rc_nn.yaml \
  --output-dir runs/rl/sac_rc_nn
```

#### Day 6: Enhance Surrogate Evaluation Script
- **Note**: `scripts/evaluation/evaluate_surrogate.py` exists but only does 1-step
- Add multi-step rollout evaluation (10, 30 steps)
- Test on RC, RF, RC+NN (PINN optional)
- Generate comparison plots showing drift over time

**Key Addition**: Autoregressive rollout function
```python
def evaluate_rollout(surrogate, test_data, k_steps=10):
    """Evaluate k-step autoregressive rollout."""
    errors_by_step = []
    
    for traj in test_data:
        state = traj['initial_state']
        surrogate.reset(init_state=state)
        
        predictions = []
        actuals = []
        
        for t in range(k_steps):
            action = traj['actions'][t]
            pred_temp = surrogate.predict_next(state, action)
            predictions.append(pred_temp)
            actuals.append(traj['temperatures'][t+1])
            
            # Update state (autoregressive)
            state[0] = pred_temp
            state[3] = action[0]
        
        errors = np.abs(np.array(predictions) - np.array(actuals))
        errors_by_step.append(errors)
    
    return np.array(errors_by_step).mean(axis=0)  # MAE by step
```

#### Day 7: Run Experiment 1 (Surrogate Evaluation)
```bash
python scripts/evaluation/evaluate_surrogates.py \
  --test-data data/synthetic/thermal_dataset.parquet \
  --output-dir results/surrogate_eval
```

### Week 2: Experiments & Report

#### Day 8-9: Run Experiment 3 (Controller Comparison)

**Important**: MPC uses a surrogate for predictive planning! For fair comparison, use the same surrogate as RL.

```bash
# Evaluate MPC with RC surrogate
python scripts/evaluation/run_evaluation.py \
  --policy-type mpc \
  --policy-config configs/evaluation/mpc_baseline.yaml \
  --scenarios nominal stress \
  --n-episodes 5

# Evaluate RL-RC agent (trained on RC surrogate)
python scripts/evaluation/run_evaluation.py \
  --policy-type rl \
  --policy-config configs/rl/sac_unified_rc.yaml \
  --model-path runs/rl/sac_rc/best_model.zip \
  --scenarios nominal stress \
  --n-episodes 5

# Evaluate RL-RC+NN agent (trained on RC+NN surrogate)
python scripts/evaluation/run_evaluation.py \
  --policy-type rl \
  --policy-config configs/rl/sac_unified_rc_nn.yaml \
  --model-path runs/rl/sac_rc_nn/best_model.zip \
  --scenarios nominal stress \
  --n-episodes 5

# Evaluate Threshold baseline (no surrogate, reactive)
python scripts/evaluation/run_evaluation.py \
  --policy-type threshold \
  --scenarios nominal stress \
  --n-episodes 5
```

**Comparison Matrix**:
| Controller | Surrogate | Type | Expected Performance |
|------------|-----------|------|---------------------|
| MPC-RC | RC | Predictive | Good (optimal planning) |
| RL-RC | RC | Learned | Good (learned policy) |
| RL-RC+NN | RC+NN | Learned | Better (better surrogate) |
| Threshold | None | Reactive | Baseline (no prediction) |

#### Day 10-11: Analyze Results
- Generate comparison tables
- Create plots (temperature tracking, violations, energy)
- Statistical analysis

#### Day 12-14: Draft Mid-Semester Report
**Sections**:
1. Introduction & Background (from proposal)
2. Methodology (architecture, surrogates, RL, MPC)
3. Preliminary Results:
   - Experiment 1: Surrogate comparison
   - Experiment 3: Controller comparison
4. Remaining Work (PINN training, additional experiments)
5. Timeline to completion

---

## What You'll Have for Mid-Sem Report

### Completed ✅
1. **Surrogate Models**:
   - RC: Physics-based, validated
   - RC+NN: Hybrid (physics + learned residuals)
   - RF: Data-driven (optional, for comparison)

2. **Control Methods**:
   - RL-RC: SAC trained with RC surrogate
   - RL-RC+NN: SAC trained with RC+NN surrogate
   - MPC-RC: Predictive optimization with RC surrogate
   - Threshold: Reactive baseline

3. **Experiments**:
   - Experiment 1: Surrogate evaluation (RC vs RC+NN vs RF)
   - Experiment 3: Controller comparison (RL-RC vs RL-RC+NN vs MPC-RC vs Threshold)

4. **Results**:
   - Surrogate accuracy table (1-step, 10-step, 30-step MAE)
   - Controller performance table (violations, energy, headroom)
   - Comparison plots showing RC+NN advantage

### Deferred to Post-Mid-Sem 📅
- PINN-lite training and evaluation
- RL training with PINN surrogate
- Cross-evaluation experiments (robustness study)
- Experiment 2: RL learning curves analysis
- Experiment 4: Safety ablation studies
- Experiments 5-6: Additional robustness and stress tests

---

## Key Decisions Made

### 1. RF k-ahead: Use 1-step for RL/MPC ✅
**Reason**: RL environment and MPC use 1-step prediction repeatedly. Training for k-ahead=10 creates train-test mismatch.

**Impact**: Better RF accuracy in closed-loop control.

### 2. Use RC+NN Instead of PINN-lite (For Mid-Sem) ✅
**Reason**: RC+NN is faster to train (1-2 hours vs 4-8 hours) and easier to debug. Still captures nonlinearities via neural network.

**Impact**: Saves 3-5 days, reduces risk, sufficient for mid-sem report.

### 3. MPC Uses Surrogate for Predictive Planning ✅
**Important**: MPC is NOT model-free! It uses a surrogate to simulate future trajectories.

**How MPC Works**:
```python
# MPC optimizes over 10-step horizon
for k in range(10):
    next_temp = surrogate.predict_next(state, action[k])  # Uses surrogate!
    cost += (next_temp - target)**2
    state = update(state, next_temp, action[k])
```

**Which Surrogate**: MPC-RC uses RC surrogate (same as RL-RC for fair comparison).

**Impact**: Fair comparison requires same surrogate for RL and MPC.

### 3. Focus on RC Surrogate First ✅
**Reason**: RC is already working, stable, and interpretable. Gets you results fastest.

**Impact**: Can complete experiments even if PINN training delayed.

### 4. Add Multi-Step Rollout Evaluation ✅
**Reason**: 1-step prediction is trivial due to thermal inertia. Need to show 10-30 second forecasting.

**Impact**: Demonstrates surrogate quality meaningfully.

---

## Backup Plan (If Time Runs Short)

### Minimum Viable Mid-Sem Report:
1. **Surrogate Evaluation**: RC vs RF only (skip PINN if not ready)
2. **Controller Comparison**: RL (RC) vs MPC (RC) vs Threshold
3. **Narrative**: "Physics-guided RC enables reliable control; PINN training ongoing"

### What You Can Defer:
- PINN-lite training → Post mid-sem
- RL with RF/PINN → Post mid-sem
- Additional experiments → Post mid-sem
- Robustness tests → Post mid-sem

---

## Success Criteria

### For Mid-Semester Report:

✅ **Minimum Acceptable**:
- RC surrogate validated (multi-step rollout)
- RC+NN surrogate trained and validated
- RL agent trained with RC (200k steps)
- Controller comparison (RL-RC vs MPC-RC vs Threshold)
- Preliminary results showing RL effectiveness

✅ **Target**:
- All of above +
- RL agent trained with RC+NN (200k steps)
- Comprehensive surrogate comparison (RC vs RF vs RC+NN)
- Controller comparison showing RC+NN advantage
- Statistical analysis of results

✅ **Stretch**:
- All of above +
- PINN surrogate trained and evaluated
- RL trained with PINN surrogate
- Cross-evaluation (robustness study)
- Additional stress scenarios

---

## Daily Checklist

### Day 1 (Today):
- [ ] Read full analysis document
- [ ] Verify RF k-ahead setting
- [ ] Start RF retraining (if needed)
- [ ] Review RL training config

### Day 2:
- [ ] RF retraining complete
- [ ] Update configs to use 1-step RF
- [ ] Start RL training with RC
- [ ] Plan surrogate evaluation script

### Day 3:
- [ ] RL training running
- [ ] Implement surrogate evaluation script
- [ ] Test on RC surrogate

### Day 4:
- [ ] RL training monitoring
- [ ] Complete surrogate evaluation script
- [ ] Test on RF surrogate

### Day 5:
- [ ] RL training checkpoint
- [ ] Run full surrogate evaluation
- [ ] Generate comparison plots

### Day 6:
- [ ] RL training monitoring
- [ ] Analyze surrogate results
- [ ] Prepare for controller comparison

### Day 7:
- [ ] RL training complete (or near)
- [ ] Run MPC evaluation
- [ ] Start RL evaluation

### Day 8-14:
- [ ] Complete controller comparison
- [ ] Analyze all results
- [ ] Create tables and plots
- [ ] Draft mid-semester report
- [ ] Review and revise

---

## Questions to Answer in Report

### Experiment 1: Surrogate Evaluation
1. Which surrogate has lowest 1-step MAE?
2. Which surrogate has lowest 10-step MAE?
3. Which surrogate drifts least over 30 steps?
4. Does physics-guided (RC) beat pure ML (RF)?
5. Does hybrid (PINN) combine best of both?

### Experiment 3: Controller Comparison
1. Which controller has fewest violations?
2. Which controller uses least energy?
3. Which controller maintains best headroom?
4. Does RL beat MPC in closed-loop?
5. Does predictive (RL/MPC) beat reactive (Threshold)?

### Overall
1. Do physics-guided surrogates enable better control?
2. Can RL learn effective thermal policies?
3. Is the safety shield effective?
4. What are the trade-offs (performance vs energy)?

---

## Communication Plan

### If You Get Stuck:
1. Check `docs/Codebase_Analysis_and_Alignment.md` for details
2. Review error logs in `runs/` directory
3. Test individual components in isolation
4. Ask for help with specific error messages

### Progress Tracking:
- Daily: Update checklist above
- Weekly: Review against timeline
- Continuous: Document issues and solutions

---

## Final Reminder

**You're in good shape!** 

Your code is well-designed. The main task is:
1. Fix RF k-ahead (1 day)
2. Run training and experiments (10 days)
3. Analyze and write (3 days)

**You can do this!** 🚀

Focus on the critical path. Don't get distracted by optimizations. Get results first, refine later.

---

**Next Steps**: 
1. Read full analysis document
2. Start RF retraining
3. Follow daily checklist

**Good luck!** 🎯
