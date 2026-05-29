# RL Training with Different Surrogates: Strategy Guide

**Date**: May 28, 2026  
**Question**: Does it matter which surrogate is used for RL training if we evaluate with different surrogates later?

---

## TL;DR

**Yes, it matters significantly!** ✅

The RL agent learns a policy optimized for the **training surrogate's dynamics**. Evaluating with a different surrogate creates **model mismatch** and degrades performance.

---

## The Problem: Model Mismatch

### What Happens During RL Training

```
RL Agent learns from surrogate:
  "When I increase fan from 50% to 70%, temperature drops by 2°C in 10 seconds"
  
Policy is optimized for this specific dynamic response.
```

### What Happens During Evaluation

```
If evaluated on SAME surrogate:
  ✅ Policy works as expected (matched dynamics)
  
If evaluated on DIFFERENT surrogate:
  ⚠️ Policy is suboptimal (mismatched dynamics)
  
  Example:
  - Trained on RC: "Fan increase → 2°C drop"
  - Evaluated on RF: "Fan increase → 1.5°C drop"
  - Result: Policy over-cools or under-cools
```

---

## Analogy

**Learning to drive a sedan, then being tested in a truck:**
- Controls work differently
- Acceleration/braking response differs
- Turning radius is different
- You can still drive, but not optimally

**Same with RL:**
- Trained on RC dynamics
- Evaluated on RF dynamics
- Agent can still control, but not optimally

---

## Experimental Design Options

### Option A: Train Separate Agents (Recommended) ✅

**Approach**: Train one RL agent per surrogate

**Training**:
```bash
# Train RL with RC surrogate
python scripts/training/train_sac_unified.py \
    --config configs/rl/sac_unified_rc.yaml \
    --output-dir runs/rl/sac_rc

# Train RL with RF surrogate
python scripts/training/train_sac_unified.py \
    --config configs/rl/sac_unified_rf.yaml \
    --output-dir runs/rl/sac_rf

# Train RL with RC+NN surrogate
python scripts/training/train_sac_unified.py \
    --config configs/rl/sac_unified_rc_nn.yaml \
    --output-dir runs/rl/sac_rc_nn
```

**Evaluation**:
```bash
# Evaluate each agent on its own surrogate
python scripts/evaluation/run_evaluation.py \
    --policy-type rl \
    --model-path runs/rl/sac_rc/best_model.zip \
    --env-config configs/rl/sac_unified_rc.yaml  # Same surrogate!

python scripts/evaluation/run_evaluation.py \
    --policy-type rl \
    --model-path runs/rl/sac_rf/best_model.zip \
    --env-config configs/rl/sac_unified_rf.yaml

python scripts/evaluation/run_evaluation.py \
    --policy-type rl \
    --model-path runs/rl/sac_rc_nn/best_model.zip \
    --env-config configs/rl/sac_unified_rc_nn.yaml
```

**Comparison**:
- Compare all three RL agents against MPC/Threshold baselines
- Show which surrogate enables best RL performance

**Dissertation Story**:
> "We train separate RL agents using each surrogate as the environment model. Results show:
> - **RL-RC**: Good performance, stable control
> - **RL-RF**: Moderate performance (RF less accurate)
> - **RL-RC+NN**: Best performance (most accurate surrogate)
>
> This demonstrates that **surrogate quality directly impacts RL learning quality**."

**Key Finding**: Better surrogate → Better RL agent

**Pros**:
- ✅ Fair comparison (each agent optimized for its surrogate)
- ✅ Shows impact of surrogate quality on RL
- ✅ Clear dissertation narrative

**Cons**:
- ⚠️ Requires training 3 RL agents (3× training time)
- ⚠️ More computational resources

---

### Option B: Train on Best, Test Robustness (Alternative)

**Approach**: Train one RL agent on best surrogate, evaluate on all

**Training**:
```bash
# Train RL with best surrogate (RC or RC+NN)
python scripts/training/train_sac_unified.py \
    --config configs/rl/sac_unified_rc.yaml \
    --output-dir runs/rl/sac_rc_best
```

**Evaluation** (Cross-evaluation):
```bash
# Evaluate on RC (matched)
python scripts/evaluation/run_evaluation.py \
    --policy-type rl \
    --model-path runs/rl/sac_rc_best/best_model.zip \
    --env-config configs/rl/sac_unified_rc.yaml

# Evaluate on RF (mismatched)
python scripts/evaluation/run_evaluation.py \
    --policy-type rl \
    --model-path runs/rl/sac_rc_best/best_model.zip \
    --env-config configs/rl/sac_unified_rf.yaml  # Different surrogate!

# Evaluate on RC+NN (mismatched)
python scripts/evaluation/run_evaluation.py \
    --policy-type rl \
    --model-path runs/rl/sac_rc_best/best_model.zip \
    --env-config configs/rl/sac_unified_rc_nn.yaml
```

**Dissertation Story**:
> "We train RL on RC surrogate and evaluate robustness to model mismatch:
> - **RC environment**: 0 violations, 45% avg fan (matched)
> - **RF environment**: 2 violations, 50% avg fan (mismatched, -15% performance)
> - **RC+NN environment**: 1 violation, 47% avg fan (mismatched, -8% performance)
>
> Results show RL policies are **sensitive to model mismatch**, highlighting the importance of accurate surrogate models."

**Key Finding**: Model mismatch degrades RL performance

**Pros**:
- ✅ Only need to train 1 RL agent (saves time)
- ✅ Shows robustness analysis
- ✅ Demonstrates model mismatch problem

**Cons**:
- ⚠️ Not a fair comparison (agent not optimized for RF/RC+NN)
- ⚠️ May show artificially poor performance on mismatched surrogates

---

### Option C: Hybrid Approach (Best of Both)

**Approach**: Train on best surrogate + one other, cross-evaluate

**Training**:
```bash
# Train RL with RC (baseline)
python scripts/training/train_sac_unified.py \
    --config configs/rl/sac_unified_rc.yaml

# Train RL with RC+NN (best)
python scripts/training/train_sac_unified.py \
    --config configs/rl/sac_unified_rc_nn.yaml
```

**Evaluation**:
- Matched: RL-RC on RC, RL-RC+NN on RC+NN
- Cross: RL-RC on RC+NN, RL-RC+NN on RC

**Dissertation Story**:
> "We train RL agents on RC and RC+NN surrogates:
> 
> **Matched Evaluation** (optimal):
> - RL-RC on RC: 0 violations, 45% fan
> - RL-RC+NN on RC+NN: 0 violations, 42% fan (better efficiency!)
>
> **Cross Evaluation** (robustness):
> - RL-RC on RC+NN: 0 violations, 46% fan (-2% efficiency)
> - RL-RC+NN on RC: 1 violation, 43% fan (slight degradation)
>
> Results show: (1) Better surrogate enables better RL, (2) Some robustness to model mismatch."

**Pros**:
- ✅ Balanced approach (2 agents, not 3)
- ✅ Shows both quality impact and robustness
- ✅ Good dissertation narrative

**Cons**:
- ⚠️ Still requires 2 RL training runs

---

## Recommended Strategy for Your Timeline

### For Mid-Semester Report (2 weeks):

**Priority 1**: Train RL with RC
```bash
python scripts/training/train_sac_unified.py \
    --config configs/rl/sac_unified_rc.yaml
```

**Priority 2**: Train RC+NN surrogate (1 day)
```bash
python scripts/training/train_rc_nn.py \
    --data data/synthetic/thermal_dataset.parquet \
    --bundle-path models/rc_nn_hybrid.pkl
```

**Priority 3**: Evaluate RL-RC against MPC/Threshold
```bash
python scripts/evaluation/run_evaluation.py \
    --policy-type rl \
    --model-path runs/rl/sac_rc/best_model.zip \
    --scenarios nominal stress
```

**Mid-Sem Report Content**:
- ✅ RL agent trained with RC surrogate
- ✅ Comparison: RL vs MPC vs Threshold
- ✅ Surrogate evaluation: RC vs RF vs RC+NN
- ✅ Preliminary results showing RL effectiveness

### Post Mid-Semester (Optional):

**Priority 4**: Train RL with RC+NN
```bash
python scripts/training/train_sac_unified.py \
    --config configs/rl/sac_unified_rc_nn.yaml
```

**Priority 5**: Cross-evaluation experiments
- Evaluate RL-RC on RC+NN environment
- Evaluate RL-RC+NN on RC environment
- Analyze model mismatch effects

**Final Dissertation Content**:
- ✅ Two RL agents (RC and RC+NN)
- ✅ Matched and cross-evaluation
- ✅ Comprehensive analysis of surrogate impact

---

## Performance Expectations

### Matched Evaluation (Optimal)

| RL Agent | Eval Surrogate | Violations | Avg Fan | Performance |
|----------|---------------|------------|---------|-------------|
| RL-RC | RC | 0 | 45% | ✅ Optimal |
| RL-RF | RF | 1 | 52% | ✅ Optimal |
| RL-RC+NN | RC+NN | 0 | 42% | ✅ Optimal (best) |

**Interpretation**: Each agent performs best on its training surrogate.

### Cross Evaluation (Robustness)

| RL Agent | Eval Surrogate | Violations | Avg Fan | Performance |
|----------|---------------|------------|---------|-------------|
| RL-RC | RF | 2 | 48% | ⚠️ Degraded (-10%) |
| RL-RC | RC+NN | 1 | 46% | ⚠️ Degraded (-5%) |
| RL-RF | RC | 1 | 54% | ⚠️ Degraded (-8%) |
| RL-RC+NN | RC | 1 | 43% | ⚠️ Degraded (-3%) |

**Interpretation**: Model mismatch causes performance degradation.

---

## Why This Matters for Your Dissertation

### Key Contributions:

1. **Surrogate Quality Impact**:
   > "We demonstrate that RL agent performance is directly tied to surrogate model accuracy. Agents trained on more accurate surrogates (RC+NN, MAE 0.9°C) achieve 15% better energy efficiency than those trained on less accurate surrogates (RF, MAE 2.5°C)."

2. **Model Mismatch Analysis**:
   > "Cross-evaluation reveals that RL policies are sensitive to model mismatch, with 5-10% performance degradation when evaluated on different surrogates. This highlights the importance of accurate surrogate modeling for reliable RL deployment."

3. **Physics-Guided Advantage**:
   > "RL agents trained on physics-guided surrogates (RC, RC+NN) show better generalization and robustness compared to those trained on pure data-driven surrogates (RF), demonstrating the value of incorporating domain knowledge."

---

## Summary Table

| Strategy | RL Agents | Training Time | Dissertation Value | Recommended For |
|----------|-----------|---------------|-------------------|-----------------|
| **Option A** | 3 (RC, RF, RC+NN) | 3× | High (comprehensive) | Final dissertation |
| **Option B** | 1 (best) | 1× | Medium (robustness) | Time-constrained |
| **Option C** | 2 (RC, RC+NN) | 2× | High (balanced) | **Mid-sem + Final** ✅ |

---

## Practical Implementation

### Config Files Needed:

1. **`configs/rl/sac_unified_rc.yaml`** ✅ (already exists)
2. **`configs/rl/sac_unified_rf.yaml`** ✅ (already exists)
3. **`configs/rl/sac_unified_rc_nn.yaml`** ⚠️ (need to create)

### Create RC+NN Config:

```yaml
# configs/rl/sac_unified_rc_nn.yaml
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
safety:
  max_fan_delta: 20.0
  fan_min: 20.0
  fan_max: 100.0
  temp_emergency: 90.0
  temp_high: 85.0

sac:
  total_timesteps: 200000
  learning_rate: 0.0003
  buffer_size: 100000
  learning_starts: 1000
  batch_size: 256
  tau: 0.005
  gamma: 0.99
  train_freq: 1
  gradient_steps: 1
  policy_kwargs:
    net_arch: [256, 256]
```

---

## Final Recommendation

**For your 2-week timeline**:

1. ✅ **Train RL with RC** (Priority 1)
2. ✅ **Train RC+NN surrogate** (Priority 2, 1 day)
3. ✅ **Evaluate RL-RC vs MPC vs Threshold** (Priority 3)
4. ⭐ **Train RL with RC+NN** (Post mid-sem, if time)

**This gives you**:
- Guaranteed results for mid-sem (RL-RC working)
- Neural network surrogate (RC+NN) for comparison
- Option to add RL-RC+NN later for final dissertation

**Bottom line**: Start with RL-RC (safest), add RL-RC+NN later (better story).
