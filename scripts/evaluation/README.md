# Evaluation Scripts

This directory contains scripts for evaluating thermal control policies.

## Quick Start

### Test MPC Controller

Run a quick smoke test to verify MPC implementation:

```bash
python scripts/evaluation/test_mpc.py
```

This runs a 50-step episode with the MPC controller using an RC surrogate.

### Full Evaluation

Evaluate MPC baseline on all scenarios:

```bash
python scripts/evaluation/run_evaluation.py \
    --policy-type mpc \
    --policy-config configs/evaluation/mpc_baseline.yaml \
    --scenarios nominal stress \
    --n-episodes 5 \
    --output-dir results/evaluation/mpc
```

Evaluate trained RL agent:

```bash
python scripts/evaluation/run_evaluation.py \
    --policy-type rl \
    --policy-config configs/rl/sac_unified_rc.yaml \
    --model-path runs/rl/sac_rc_baseline/checkpoints/best_model.zip \
    --scenarios nominal stress \
    --n-episodes 5 \
    --output-dir results/evaluation/sac_rc
```

## Scripts

### `test_mpc.py`

Quick smoke test for MPC controller.

**Features:**
- Creates RC surrogate and environment
- Initializes MPC controller
- Runs 50-step episode
- Reports temperature tracking, fan control, and MPC statistics

**Usage:**
```bash
python scripts/evaluation/test_mpc.py
```

### `run_evaluation.py`

Comprehensive evaluation of thermal control policies.

**Features:**
- Supports both MPC and RL policies
- Evaluates on multiple scenarios (nominal and stress)
- Collects detailed metrics (temperature, energy, safety)
- Saves results to CSV, JSON, and trajectory files

**Arguments:**
- `--config`: Evaluation config (default: `configs/evaluation/eval_config.yaml`)
- `--policy-config`: Policy-specific config
- `--policy-type`: `mpc` or `rl`
- `--model-path`: Path to trained RL model (required for RL)
- `--scenarios`: Scenario types (`nominal`, `stress`, or both)
- `--n-episodes`: Episodes per scenario (default: 5)
- `--output-dir`: Results directory
- `--save-trajectory`: Save full trajectory data (default: True)

**Example:**
```bash
# MPC evaluation
python scripts/evaluation/run_evaluation.py \
    --policy-type mpc \
    --policy-config configs/evaluation/mpc_baseline.yaml

# RL evaluation
python scripts/evaluation/run_evaluation.py \
    --policy-type rl \
    --policy-config configs/rl/sac_unified_rc.yaml \
    --model-path runs/rl/sac_rc_baseline/checkpoints/best_model.zip
```

## Test Scenarios

### Nominal Scenarios (5)
1. **Baseline** - Normal operation, moderate workload
2. **Low Workload** - Idle/light tasks
3. **High Workload** - Compute-intensive tasks
4. **Variable Workload** - Periodic changes
5. **Warm Ambient** - Elevated ambient temperature

### Stress Scenarios (7)
1. **Thermal High** - High initial temp + sustained high workload
2. **Ambient Extreme** - Very high ambient temperature
3. **Workload Spike** - Sudden idle → max transition
4. **Workload Oscillation** - Rapid low ↔ high oscillation
5. **Combined Extreme** - High temp + ambient + workload
6. **Recovery** - Start hot, workload drops
7. **Sustained Limit** - Sustained near-limit operation

See `src/evaluation/scenarios.py` for full scenario definitions.

## Output Files

Results are saved to the specified output directory:

```
results/evaluation/
├── mpc_baseline_metrics_YYYYMMDD_HHMMSS.csv
├── mpc_baseline_summary_YYYYMMDD_HHMMSS.json
└── mpc_baseline_trajectories_YYYYMMDD_HHMMSS.json
```

### Metrics CSV

Per-episode results with columns:
- `scenario`: Scenario name
- `seed`: Random seed
- `total_steps`: Episode length
- `total_reward`, `avg_reward`: Reward metrics
- `temp_rmse`, `temp_mae`, `temp_max_error`: Temperature tracking
- `temp_violations`, `temp_violation_rate`: Safety violations
- `avg_fan_speed`, `fan_speed_std`: Energy efficiency
- `avg_fan_delta`, `max_fan_delta`: Control smoothness

### Summary JSON

Aggregate statistics:
- Overall averages across all episodes
- Per-scenario breakdowns
- Standard deviations

### Trajectories JSON

Step-by-step data for each episode:
- Temperature, fan speed, power at each step
- Rewards, violations
- Full state history

## Metrics

### Temperature Tracking
- **RMSE**: Root mean square error from target
- **MAE**: Mean absolute error from target
- **Max Error**: Maximum deviation from target
- **Std Dev**: Temperature variability

### Safety
- **Violations**: Count of steps exceeding temp_max
- **Violation Rate**: Fraction of steps with violations
- **Safety Interventions**: Safety shield activations

### Energy Efficiency
- **Avg Fan Speed**: Mean fan speed (lower = more efficient)
- **Fan Std Dev**: Fan speed variability
- **Min/Max Fan**: Range of fan speeds used

### Control Smoothness
- **Avg Fan Delta**: Mean fan speed change per step
- **Max Fan Delta**: Maximum single-step change
- **Fan Delta Std**: Variability in fan changes

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
   - Identify strengths/weaknesses

## Configuration

### MPC Config (`configs/evaluation/mpc_baseline.yaml`)

```yaml
mpc:
  horizon: 10              # Prediction horizon
  temp_target: 75.0        # Target temperature
  temp_max: 85.0           # Safety limit
  weight_temp: 10.0        # Temperature tracking weight
  weight_effort: 0.1       # Energy penalty weight
  weight_rate: 1.0         # Smoothness penalty weight
```

### Evaluation Config (`configs/evaluation/eval_config.yaml`)

```yaml
scenarios: [nominal, stress]
n_episodes_per_scenario: 5
save_trajectory: true
seeds: [42, 123, 456, 789, 1011]
```

## Troubleshooting

### Import Errors

Ensure project root is in Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/thermal-adt"
```

### Missing Dependencies

Install required packages:
```bash
pip install scipy gymnasium stable-baselines3 pyyaml pandas
```

### Model Not Found

For RL evaluation, ensure model path is correct:
```bash
ls runs/rl/sac_rc_baseline/checkpoints/best_model.zip
```

### Encoding Errors (Windows)

The scripts handle UTF-8 encoding automatically. If issues persist, run:
```bash
chcp 65001  # Set console to UTF-8
python scripts/evaluation/test_mpc.py
```

## See Also

- `docs/MPC_and_Evaluation.md` - Detailed documentation
- `src/control/mpc_controller.py` - MPC implementation
- `src/evaluation/harness.py` - Evaluation harness
- `src/evaluation/scenarios.py` - Scenario definitions
