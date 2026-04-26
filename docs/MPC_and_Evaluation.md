# MPC Controller and Evaluation Harness

This document describes the Model Predictive Control (MPC) baseline controller and the unified evaluation harness for comparing thermal control policies.

## Overview

The MPC controller provides a classical control baseline for comparison with RL agents. The evaluation harness enables systematic testing of both MPC and RL policies across nominal and stress scenarios.

## MPC Controller

### Architecture

The MPC controller optimizes fan speed over a receding horizon to:
1. **Track temperature target** - Minimize deviation from desired temperature
2. **Minimize fan effort** - Reduce energy consumption
3. **Ensure smooth control** - Minimize fan speed rate-of-change

### Objective Function

```
J = Σ [w_temp * (T - T_target)² + w_effort * (fan/100)² + w_rate * (Δfan/100)²]
```

Where:
- `w_temp`: Temperature tracking weight (default: 10.0)
- `w_effort`: Fan effort weight (default: 0.1)
- `w_rate`: Fan rate-of-change weight (default: 1.0)

### Constraints

**Hard constraints:**
- Fan speed bounds: `[fan_min, fan_max]` (default: [20%, 100%])
- Rate limiting: `|Δfan| ≤ max_fan_delta` (default: 20%)

**Soft constraints:**
- Temperature limit: Large penalty for `T > temp_max`

### Implementation

```python
from src.control import MPCController
from src.rl.surrogates import RCAdapter

# Create surrogate for predictions
surrogate = RCAdapter(
    thermal_capacity=100.0,
    heat_transfer_coeff=0.05,
    cooling_effectiveness=-0.03,
    power_to_heat=0.01
)

# Create MPC controller
mpc = MPCController(
    surrogate=surrogate,
    horizon=10,              # 10-step prediction horizon
    temp_target=75.0,        # Target 75°C
    temp_max=85.0,           # Safety limit
    fan_min=20.0,
    fan_max=100.0,
    max_fan_delta=20.0,      # Max 20% change per step
    weight_temp=10.0,
    weight_effort=0.1,
    weight_rate=1.0
)

# Use in control loop
action, info = mpc.compute_action(state)
```

### Configuration

MPC parameters are specified in `configs/evaluation/mpc_baseline.yaml`:

```yaml
mpc:
  horizon: 10
  temp_target: 75.0
  temp_max: 85.0
  fan_min: 20.0
  fan_max: 100.0
  max_fan_delta: 20.0
  weight_temp: 10.0
  weight_effort: 0.1
  weight_rate: 1.0
```

## Evaluation Harness

### Features

The `EvaluationHarness` provides:
- **Unified interface** for RL and MPC policies
- **Comprehensive metrics** collection
- **Scenario management** (nominal and stress tests)
- **Trajectory recording** for detailed analysis
- **Automatic result saving** (CSV, JSON)

### Usage

```python
from src.evaluation import EvaluationHarness, create_scenarios

# Create harness
harness = EvaluationHarness(
    env=env,
    policy=mpc,
    policy_type="mpc",
    output_dir="results/evaluation",
    save_trajectory=True
)

# Create scenarios
scenarios = create_scenarios(["nominal", "stress"])

# Run evaluation
results_df = harness.evaluate_scenarios(
    scenarios=scenarios,
    n_episodes_per_scenario=5
)

# Print and save results
harness.print_summary(results_df)
harness.save_results(results_df, prefix="mpc_baseline")
```

### Metrics Collected

**Temperature Tracking:**
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Max deviation
- Standard deviation

**Safety:**
- Temperature violations (count and rate)
- Safety interventions (count and rate)

**Energy Efficiency:**
- Average fan speed
- Fan speed variance
- Min/max fan speed

**Control Smoothness:**
- Average fan speed change
- Max fan speed change
- Fan delta variance

## Test Scenarios

### Nominal Scenarios

1. **Baseline** - Normal operation, moderate workload
2. **Low Workload** - Idle/light tasks
3. **High Workload** - Compute-intensive tasks
4. **Variable Workload** - Periodic changes
5. **Warm Ambient** - Elevated ambient temperature

### Stress Scenarios

1. **Thermal High** - High initial temp + sustained high workload
2. **Ambient Extreme** - Very high ambient temperature
3. **Workload Spike** - Sudden idle → max transition
4. **Workload Oscillation** - Rapid low ↔ high oscillation
5. **Combined Extreme** - High temp + ambient + workload
6. **Recovery** - Start hot, workload drops
7. **Sustained Limit** - Sustained near-limit operation

### Scenario Configuration

Scenarios are defined in `src/evaluation/scenarios.py`:

```python
{
    "name": "stress_thermal_high",
    "description": "Thermal stress: high initial temp and sustained high workload",
    "initial_temp": 80.0,
    "ambient_temp": 28.0,
    "temp_target": 75.0,
    "temp_max": 85.0,
    "workload_profile": "constant",
    "power_range": [280.0, 300.0]
}
```

## Running Evaluation

### Quick MPC Test

```bash
python scripts/evaluation/test_mpc.py
```

This runs a 50-step smoke test with the MPC controller.

### Full Evaluation

```bash
# Evaluate MPC baseline
python scripts/evaluation/run_evaluation.py \
    --policy-type mpc \
    --policy-config configs/evaluation/mpc_baseline.yaml \
    --scenarios nominal stress \
    --n-episodes 5 \
    --output-dir results/evaluation/mpc

# Evaluate RL agent (after training)
python scripts/evaluation/run_evaluation.py \
    --policy-type rl \
    --policy-config configs/rl/sac_unified_rc.yaml \
    --model-path runs/rl/sac_rc_baseline/checkpoints/best_model.zip \
    --scenarios nominal stress \
    --n-episodes 5 \
    --output-dir results/evaluation/sac_rc
```

### Output

Results are saved to the specified output directory:

```
results/evaluation/
├── mpc_baseline_metrics_20250426_120000.csv
├── mpc_baseline_summary_20250426_120000.json
└── mpc_baseline_trajectories_20250426_120000.json
```

**Metrics CSV** contains per-episode results:
- Scenario name, seed
- Total reward, avg reward
- Temperature metrics (RMSE, MAE, violations)
- Fan metrics (avg speed, smoothness)
- Safety statistics

**Summary JSON** contains aggregate statistics:
- Overall averages across all episodes
- Per-scenario breakdowns
- Standard deviations

**Trajectories JSON** contains step-by-step data:
- Temperature, fan speed, power at each step
- Rewards, violations
- Full state history

## Comparison Workflow

1. **Train RL agents** with different surrogates:
   ```bash
   python scripts/training/train_sac_unified.py --config configs/rl/sac_unified_rc.yaml
   python scripts/training/train_sac_unified.py --config configs/rl/sac_unified_rf.yaml
   ```

2. **Evaluate MPC baseline**:
   ```bash
   python scripts/evaluation/run_evaluation.py --policy-type mpc
   ```

3. **Evaluate RL agents**:
   ```bash
   python scripts/evaluation/run_evaluation.py \
       --policy-type rl \
       --model-path runs/rl/sac_rc_baseline/checkpoints/best_model.zip
   ```

4. **Compare results**:
   - Load CSV files into pandas/Excel
   - Compare metrics across policies and scenarios
   - Identify strengths/weaknesses of each approach

## Key Insights

### MPC Advantages
- **Explicit constraints** - Hard safety guarantees
- **Interpretable** - Clear objective function
- **No training required** - Works immediately
- **Predictable behavior** - Deterministic control

### MPC Limitations
- **Model-dependent** - Requires accurate surrogate
- **Computational cost** - Online optimization at each step
- **Fixed objective** - Cannot adapt weights automatically
- **Horizon limited** - Short-term optimization only

### RL Advantages
- **Learns from data** - Can handle model mismatch
- **Adaptive** - Learns optimal trade-offs
- **Long-term optimization** - Considers full episode
- **Efficient inference** - Fast forward pass

### RL Limitations
- **Training required** - Needs data and compute
- **Black box** - Less interpretable
- **Safety challenges** - Requires careful reward design
- **Exploration risk** - May violate constraints during training

## Future Extensions

1. **Advanced MPC variants**:
   - Nonlinear MPC with PINN/RF surrogates
   - Adaptive horizon based on workload
   - Multi-objective optimization (Pareto front)

2. **Hybrid approaches**:
   - RL for long-term planning, MPC for safety
   - MPC-guided RL exploration
   - Ensemble policies

3. **Additional scenarios**:
   - Multi-GPU thermal coupling
   - Datacenter-scale scenarios
   - Hardware-in-the-loop testing

4. **Visualization**:
   - Real-time trajectory plots
   - Comparative dashboards
   - Performance profiles

## References

- **MPC Theory**: Camacho & Bordons, "Model Predictive Control"
- **Thermal Modeling**: Coskun et al., "Temperature Management in Multiprocessor SoCs"
- **RL for Control**: Levine et al., "Reinforcement Learning and Control as Probabilistic Inference"
