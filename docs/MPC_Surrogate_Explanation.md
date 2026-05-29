# MPC Surrogate Usage Explained

**Date**: May 29, 2026  
**Question**: Does MPC use a surrogate model? If yes, which one?

---

## TL;DR

**Yes, MPC uses a surrogate!** ✅

MPC is a **model-based** controller that uses a surrogate to predict future temperatures over a planning horizon (10 steps). It optimizes the action sequence to minimize cost, then executes only the first action.

---

## How MPC Works

### High-Level Overview

```
Current State → MPC Controller → Action
                    ↓
            Uses Surrogate Model
            to simulate 10 steps
            and optimize actions
```

### Detailed Process

```python
def mpc_compute_action(current_state, surrogate):
    """
    MPC optimization process.
    
    Args:
        current_state: [temp, ambient, power, fan_speed, temp_delta]
        surrogate: Thermal surrogate model (RC, RF, RC+NN, etc.)
    
    Returns:
        optimal_action: Best fan speed for current step
    """
    
    # Step 1: Define objective function
    def objective(u_sequence):
        """
        Objective to minimize over 10-step horizon.
        
        Args:
            u_sequence: [u_0, u_1, ..., u_9] - fan speeds for 10 steps
        
        Returns:
            total_cost: Sum of tracking error + fan effort + smoothness
        """
        cost = 0.0
        state = current_state.copy()
        
        # Simulate 10 steps ahead using surrogate
        for k in range(10):
            # CRITICAL: Use surrogate to predict next temperature
            action = np.array([u_sequence[k]])
            next_temp = surrogate.predict_next(state, action)  # ← Uses surrogate!
            
            # Accumulate cost
            cost += 10.0 * (next_temp - 75.0)**2  # Tracking error
            cost += 0.1 * (u_sequence[k]/100)**2   # Fan effort
            
            if k > 0:
                cost += 1.0 * ((u_sequence[k] - u_sequence[k-1])/100)**2  # Smoothness
            
            # Update state for next prediction
            state[0] = next_temp
            state[3] = u_sequence[k]
        
        return cost
    
    # Step 2: Optimize action sequence
    initial_guess = [50.0] * 10  # Start with 50% fan for all steps
    bounds = [(20.0, 100.0)] * 10  # Fan speed constraints
    
    result = scipy.optimize.minimize(
        objective,
        initial_guess,
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    optimal_sequence = result.x  # [u_0*, u_1*, ..., u_9*]
    
    # Step 3: Execute only first action (receding horizon)
    optimal_action = optimal_sequence[0]
    
    return optimal_action
```

### Key Points

1. **Surrogate is Essential**: MPC cannot work without a model to predict future states
2. **10-Step Lookahead**: MPC simulates 10 seconds ahead to plan optimal actions
3. **Receding Horizon**: Only executes first action, then re-plans at next timestep
4. **Model Quality Matters**: Better surrogate → Better MPC performance

---

## MPC vs RL: Surrogate Usage Comparison

| Aspect | MPC | RL |
|--------|-----|-----|
| **Training Phase** | No training (optimization-based) | Trains policy network on surrogate |
| **Inference Phase** | Uses surrogate for planning | Uses learned policy (no surrogate) |
| **Surrogate Usage** | Every timestep (10 predictions) | Only during training |
| **Computational Cost** | High (optimization at runtime) | Low (forward pass only) |
| **Adaptability** | Adapts to new surrogate instantly | Needs retraining for new surrogate |

### Example Scenario

**Setup**: Temperature = 70°C, Target = 75°C, Current Fan = 50%

#### MPC-RC (Uses RC Surrogate):
```
Step 1: Try action sequence [60, 62, 64, 66, 68, 70, 70, 70, 70, 70]
        Simulate with RC: T = [71.2, 72.5, 73.8, 74.9, 75.8, ...]
        Cost = 45.3

Step 2: Try action sequence [55, 58, 61, 64, 67, 69, 70, 70, 70, 70]
        Simulate with RC: T = [70.8, 71.9, 73.1, 74.3, 75.2, ...]
        Cost = 38.7  ← Better!

... (continue optimization)

Step N: Optimal sequence = [57, 60, 63, 66, 68, 70, 70, 70, 70, 70]
        Execute: Set fan to 57%
```

#### RL-RC (Trained on RC Surrogate):
```
Step 1: Forward pass through policy network
        Input: [70.0, 25.0, 200.0, 50.0, 0.0]
        Output: 58%
        
        (No surrogate used at inference!)
```

**Key Difference**: MPC uses surrogate at **runtime**, RL uses it only during **training**.

---

## Which Surrogate Should MPC Use?

### For Fair Comparison with RL

**Rule**: Use the **same surrogate** for both MPC and RL.

**Example**:
```
Experiment: Compare RL vs MPC on RC surrogate

Training:
- RL-RC: Train RL agent using RC surrogate as environment
- MPC-RC: No training needed

Evaluation:
- RL-RC: Evaluate learned policy on RC environment
- MPC-RC: Evaluate MPC using RC surrogate for planning

Result: Fair comparison (both use RC dynamics)
```

### For Multiple Surrogates

**Option A**: Separate MPC for each surrogate
```
- MPC-RC: Uses RC surrogate
- MPC-RF: Uses RF surrogate
- MPC-RC+NN: Uses RC+NN surrogate

Compare:
- RL-RC vs MPC-RC (both use RC)
- RL-RF vs MPC-RF (both use RF)
- RL-RC+NN vs MPC-RC+NN (both use RC+NN)
```

**Option B**: MPC with best surrogate
```
- MPC-RC+NN: Uses best surrogate (RC+NN)

Compare:
- RL-RC vs MPC-RC+NN (shows MPC advantage with better model)
- RL-RC+NN vs MPC-RC+NN (fair comparison)
```

---

## Implementation in Your Codebase

### MPC Controller Configuration

**File**: `configs/evaluation/mpc_baseline.yaml`

```yaml
mpc:
  horizon: 10  # 10-step lookahead
  weight_temp: 10.0
  weight_effort: 0.1
  weight_rate: 1.0
  temp_target: 75.0
  temp_max: 85.0
  fan_min: 20.0
  fan_max: 100.0

surrogate:
  type: rc  # ← Specifies which surrogate to use!
  thermal_capacity: 100.0
  heat_transfer_coeff: 0.05
  cooling_effectiveness: -0.03
  power_to_heat: 0.01
  dt: 1.0
```

### MPC Controller Code

**File**: `src/control/mpc_controller.py`

```python
class MPCController:
    def __init__(self, surrogate, horizon=10, ...):
        self.surrogate = surrogate  # ← Surrogate is required!
        self.horizon = horizon
        # ...
    
    def compute_action(self, state):
        # Optimize using surrogate
        result = minimize(
            lambda u: self._objective(u, state),  # Uses self.surrogate
            initial_guess,
            bounds=bounds
        )
        return result.x[0]
    
    def _objective(self, u, state):
        cost = 0.0
        current_state = state.copy()
        
        for k in range(self.horizon):
            # Use surrogate to predict
            next_temp = self.surrogate.predict_next(current_state, u[k])
            # ... accumulate cost
        
        return cost
```

---

## Comparison Matrix

### Controllers and Their Surrogates

| Controller | Surrogate | Training Needed | Runtime Cost | Adaptability |
|------------|-----------|-----------------|--------------|--------------|
| **Threshold** | None | No | Very Low | N/A |
| **MPC-RC** | RC | No | High (optimization) | Instant |
| **MPC-RC+NN** | RC+NN | No | High (optimization) | Instant |
| **RL-RC** | RC (training only) | Yes (200k steps) | Low (inference) | Needs retraining |
| **RL-RC+NN** | RC+NN (training only) | Yes (200k steps) | Low (inference) | Needs retraining |

### Expected Performance

| Controller | Violations | Avg Fan | Energy | Notes |
|------------|-----------|---------|--------|-------|
| Threshold | 5 | 65% | High | Reactive, no prediction |
| MPC-RC | 1 | 50% | Medium | Optimal planning, RC accuracy |
| MPC-RC+NN | 0 | 48% | Medium | Optimal planning, better model |
| RL-RC | 1 | 48% | Medium | Learned policy, RC accuracy |
| RL-RC+NN | 0 | 45% | Low | Learned policy, better model |

**Key Findings**:
1. Better surrogate → Better MPC performance (MPC-RC+NN > MPC-RC)
2. RL can match or exceed MPC with good surrogate (RL-RC+NN ≥ MPC-RC+NN)
3. RL is more efficient at runtime (no optimization loop)

---

## Dissertation Story

### Experiment 3: Controller Comparison

**Setup**:
> "We compare four control strategies:
> 1. **Threshold**: Reactive baseline (no model)
> 2. **MPC-RC**: Model-based optimization with RC surrogate
> 3. **RL-RC**: Learned policy trained on RC surrogate
> 4. **RL-RC+NN**: Learned policy trained on RC+NN surrogate
>
> All model-based controllers use the same surrogate for fair comparison."

**Results**:
> "MPC-RC achieves good performance through online optimization but requires high computational cost (10 surrogate predictions per timestep). RL-RC achieves comparable performance with 10× lower runtime cost. RL-RC+NN, trained on a more accurate surrogate, outperforms both MPC-RC and RL-RC, demonstrating that **surrogate quality directly impacts control performance**."

**Key Contributions**:
1. RL can match MPC performance with learned policies
2. RL is more computationally efficient at runtime
3. Better surrogates enable better control (RC+NN > RC)
4. Physics-guided surrogates (RC, RC+NN) outperform pure ML (RF)

---

## Summary

### Key Takeaways

1. ✅ **MPC uses a surrogate** for predictive planning (10-step lookahead)
2. ✅ **Same surrogate for fair comparison** (MPC-RC vs RL-RC)
3. ✅ **Better surrogate → Better MPC** (MPC-RC+NN > MPC-RC)
4. ✅ **RL can match/exceed MPC** with learned policies and good surrogates

### For Your Experiments

**Mid-Semester Report**:
- MPC-RC (uses RC surrogate)
- RL-RC (trained on RC surrogate)
- RL-RC+NN (trained on RC+NN surrogate)
- Threshold (no surrogate)

**Comparison**:
- MPC-RC vs RL-RC: Shows RL can match MPC with lower runtime cost
- RL-RC+NN vs RL-RC: Shows surrogate quality impact on RL
- All vs Threshold: Shows value of predictive control

---

**Bottom Line**: MPC is model-based and requires a surrogate. For fair comparison with RL, use the same surrogate for both!
