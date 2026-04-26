# Unified Surrogate Interface

This document describes the unified surrogate interface that enables seamless switching between different thermal dynamics models (RC, RF, PINN) in the RL environment.

## Overview

The unified surrogate interface allows the same Gymnasium environment to use different thermal models interchangeably, enabling:
- **Fair ablation studies** comparing RC, RF, and PINN surrogates
- **Flexible experimentation** without changing environment code
- **Consistent reward and safety** across all surrogate types
- **Easy curriculum learning** with different model complexities

## Architecture

### Components

1. **`ThermalSurrogate` Protocol** (`src/rl/surrogates/interface.py`)
   - Defines the interface all surrogates must implement
   - Methods: `reset()`, `predict_next()`, `warmup_steps` property

2. **Surrogate Adapters**
   - **`RCAdapter`**: Analytical RC thermal model (stateless)
   - **`RFAdapter`**: Random Forest with feature materialization and history
   - **`PINNAdapter`**: Physics-Informed Neural Network with tensorization

3. **`ThermalControlEnv`** (`src/rl/environments/thermal_unified.py`)
   - Unified environment accepting any `ThermalSurrogate`
   - Handles reward, safety, curriculum, and logging
   - Agnostic to surrogate implementation details

4. **`create_surrogate()` Factory** (`src/rl/surrogates/factory.py`)
   - Creates surrogate adapters from configuration
   - Enables easy switching via config files

## Usage

### Basic Example

```python
from src.rl.surrogates import RCAdapter
from src.rl.environments.thermal_unified import ThermalControlEnv

# Create RC surrogate
surrogate = RCAdapter(
    thermal_capacity=100.0,
    heat_transfer_coeff=0.05,
    cooling_effectiveness=-0.03,
    power_to_heat=0.01,
    dt=1.0
)

# Create environment
env = ThermalControlEnv(surrogate=surrogate)

# Use environment
obs, info = env.reset(seed=42)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Using the Factory

```python
from src.rl.surrogates import create_surrogate
from src.rl.environments.thermal_unified import ThermalControlEnv

# RC surrogate
config_rc = {
    "type": "rc",
    "thermal_capacity": 100.0,
    "dt": 1.0
}
surrogate = create_surrogate("rc", config_rc)

# RF surrogate
config_rf = {
    "type": "rf",
    "model_path": "runs/rf_teacher/rf_model.pkl",
    "use_cache": False
}
surrogate = create_surrogate("rf", config_rf)

# PINN surrogate
config_pinn = {
    "type": "pinn",
    "model_path": "runs/pinn/hybrid/checkpoints/best_model.pt",
    "device": "cuda"
}
surrogate = create_surrogate("pinn", config_pinn)

# Use with environment
env = ThermalControlEnv(surrogate=surrogate)
```

### Training with Different Surrogates

Use the unified training script with different config files:

```bash
# Train with RC surrogate
python scripts/training/train_sac_unified.py --config configs/rl/sac_unified_rc.yaml

# Train with RF surrogate
python scripts/training/train_sac_unified.py --config configs/rl/sac_unified_rf.yaml

# Train with PINN surrogate
python scripts/training/train_sac_unified.py --config configs/rl/sac_unified_pinn.yaml
```

## Surrogate Adapters

### RCAdapter

**Description**: Analytical RC thermal model using circuit analogy.

**Characteristics**:
- Stateless (no history buffer)
- Zero warmup steps
- Fast inference
- Physics-based dynamics

**Configuration**:
```yaml
surrogate:
  type: rc
  thermal_capacity: 100.0        # C (J/°C)
  heat_transfer_coeff: 0.05      # h (W/°C)
  cooling_effectiveness: -0.03   # beta (°C/%)
  power_to_heat: 0.01            # gamma (°C/W)
  dt: 1.0                        # Time step (seconds)
  temp_min: 30.0                 # Min temperature (°C)
  temp_max: 95.0                 # Max temperature (°C)
```

**Dynamics**:
```
dT/dt = (gamma*P + beta*Fan - h*(T - T_amb)) / C
T(t+1) = T(t) + dT/dt * dt
```

### RFAdapter

**Description**: Random Forest surrogate with feature engineering.

**Characteristics**:
- Stateful (maintains history buffer)
- Warmup steps required for lag/rolling features
- Handles feature materialization internally
- Trained on real or synthetic data

**Configuration**:
```yaml
surrogate:
  type: rf
  model_path: runs/rf_teacher/rf_model.pkl
  cache_dir: null
  use_cache: false
```

**Features**:
- Automatically infers warmup length from feature columns
- Builds lag and rolling window features
- Maps state vector to RF feature space
- Handles history buffer updates

### PINNAdapter

**Description**: Physics-Informed Neural Network surrogate.

**Characteristics**:
- Stateless (delta prediction)
- Zero warmup steps
- GPU-accelerated inference
- Physics-informed training

**Configuration**:
```yaml
surrogate:
  type: pinn
  model_path: runs/pinn/hybrid/checkpoints/best_model.pt
  device: cuda
  input_mean: null   # Loaded from checkpoint
  input_std: null
  output_mean: null
  output_std: null
  pinn_config: null  # Loaded from checkpoint
```

**Features**:
- Tensorized input/output
- Device management (CPU/CUDA)
- Input normalization
- Output denormalization
- Delta prediction (T_next = T_current + ΔT)

## Protocol Definition

All surrogate adapters must implement:

```python
class ThermalSurrogate(Protocol):
    def reset(
        self,
        seed: Optional[int] = None,
        init_state: Optional[np.ndarray] = None
    ) -> None:
        """Reset internal state (e.g., history buffers)."""
        ...
    
    def predict_next(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> float:
        """Predict next GPU temperature."""
        ...
    
    @property
    def warmup_steps(self) -> int:
        """Number of warmup steps required."""
        ...
```

**State Vector**: `[temp, ambient, power, fan_speed, temp_delta]`  
**Action Vector**: `[fan_speed]`  
**Output**: Next GPU temperature (°C)

## Environment Configuration

The unified environment accepts a common configuration:

```yaml
env:
  max_steps: 300                    # Episode length
  temp_warning: 80.0                # Warning threshold (°C)
  temp_critical: 90.0               # Critical threshold (°C)
  temp_target: 75.0                 # Target temperature (°C)
  initial_temp_range: [40.0, 60.0]  # Initial temp sampling range
  ambient_range: [20.0, 30.0]       # Ambient temp range
  power_range: [100.0, 300.0]       # GPU power range (W)
  reward_weights:
    thermal: 10.0                   # Thermal violation penalty
    energy: 0.1                     # Fan energy penalty
    oscillation: 1.0                # Fan oscillation penalty
    headroom: 2.0                   # Thermal headroom bonus
```

## Testing

Run smoke tests to validate all adapters:

```bash
pytest tests/test_surrogate_adapters.py -v
```

Tests verify:
- Adapter creation and configuration
- Protocol conformance
- Environment compatibility
- Determinism with seeding
- Full episode execution

## Ablation Studies

The unified interface enables fair comparisons:

1. **Same environment**: Identical reward, safety, curriculum
2. **Same RL algorithm**: Same SAC hyperparameters
3. **Only surrogate differs**: RC vs RF vs PINN

Example workflow:
```bash
# Train with all three surrogates
python scripts/training/train_sac_unified.py --config configs/rl/sac_unified_rc.yaml
python scripts/training/train_sac_unified.py --config configs/rl/sac_unified_rf.yaml
python scripts/training/train_sac_unified.py --config configs/rl/sac_unified_pinn.yaml

# Compare results
python scripts/evaluation/compare_surrogates.py \
  --runs runs/rl/sac_rc_baseline \
         runs/rl/sac_rf_teacher \
         runs/rl/sac_pinn_hybrid
```

## Implementation Notes

### RFAdapter Details

- **History buffer**: Pandas DataFrame with base columns
- **Feature materialization**: Uses `materialize_features_from_list()`
- **Warmup inference**: Parses feature names for lag/rolling windows
- **State mapping**: Maps compact state vector to RF feature space

### PINNAdapter Details

- **Model loading**: Loads checkpoint with architecture config
- **Device management**: Auto-detects CUDA availability
- **Normalization**: Applies input/output scaling if provided
- **Delta prediction**: Adds predicted ΔT to current temperature

### Environment Responsibilities

The environment handles:
- **Reward computation**: Thermal, energy, oscillation, headroom
- **Safety checking**: Temperature thresholds, termination
- **Episode management**: Reset, step, metrics
- **Logging**: Episode history, statistics

The environment does NOT handle:
- Feature engineering (delegated to adapter)
- Model inference (delegated to adapter)
- History buffers (delegated to adapter)

## Future Extensions

Potential enhancements:
- **Multi-step prediction**: Extend protocol for k-ahead prediction
- **Uncertainty quantification**: Add prediction intervals
- **Ensemble surrogates**: Combine multiple models
- **Online learning**: Update surrogates during RL training
- **Transfer learning**: Pre-train on one surrogate, fine-tune on another

## References

- `src/rl/surrogates/`: Surrogate interface and adapters
- `src/rl/environments/thermal_unified.py`: Unified environment
- `scripts/training/train_sac_unified.py`: Unified training script
- `configs/rl/sac_unified_*.yaml`: Example configurations
- `tests/test_surrogate_adapters.py`: Smoke tests
