# Autonomous Thermal Orchestration in GPU-Dense Servers
## A Node-Level Agentic Digital Twin for Predictive Throttle Avoidance

**M.Tech Dissertation Project**  
**Student**: Pooja  
**Program**: M.Tech (AI/ML), BITS Pilani - Work Integrated Learning  
**Supervisor**: Dr. Tabet Said

---

## Overview

This repository contains the implementation of an **Agentic Digital Twin (ADT)** for predictive thermal orchestration in GPU-dense servers. The system combines physics-guided surrogate modeling with safe reinforcement learning to proactively avoid thermal throttling while optimizing energy efficiency.

## Key Components

- **Thermal Surrogates**: RC model, Random Forest, PINN-lite
- **RL Controller**: Soft Actor-Critic (SAC) with safety shield
- **Baselines**: Static fan curve, threshold controller, MPC
- **Evaluation**: 6 comprehensive experiments

## Repository Structure

```
├── data/              # Datasets (synthetic thermal data)
├── configs/           # Configuration files
├── src/               # Source code
│   ├── surrogates/    # Thermal surrogate models
│   ├── rl/            # RL components
│   └── baselines/     # Baseline controllers
├── scripts/           # Training and evaluation scripts
├── experiments/       # Experiment runners
├── results/           # Experimental results
├── notebooks/         # Analysis notebooks
└── docs/              # Documentation

```

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n thermal-adt python=3.10
conda activate thermal-adt

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python scripts/data/prepare_synthetic_data.py
```

### 3. Train Surrogates

```bash
# Train RF surrogate
python scripts/training/train_rf.py

# Train PINN-lite
python scripts/training/train_pinn.py
```

### 4. Train RL Agent

```bash
# Train SAC with RF surrogate (shielded)
python scripts/training/train_sac.py \
  --config configs/rl/sac_shielded.yaml \
  --surrogate rf \
  --seed 42
```

### 5. Run Experiments

```bash
# Run controller comparison experiment
python experiments/exp3_controller_comparison.py
```

## Experiments

1. **Surrogate Evaluation**: RC vs RF vs PINN-lite accuracy and stability
2. **RL Learning Curves**: Sample efficiency and convergence
3. **Controller Comparison**: RL vs baselines on multiple scenarios
4. **Safety Ablation**: Shielded vs unshielded RL
5. **Robustness Tests**: Ambient shifts, sensor noise, workload bursts
6. **Predictive Orchestration**: Early intervention analysis

## Results

All experimental results are organized in `results/` with:
- Metrics (JSON/CSV)
- Plots (PNG/PDF for thesis)
- Writeups (Markdown with thesis sections)

## Documentation

- [Dissertation Proposal](DISSERTATION_PROPOSAL.md)
- [TODO List](docs/TODO.md)
- [Session Summaries](docs/)

## Citation

```bibtex
@mastersthesis{pooja2026thermal,
  title={Autonomous Thermal Orchestration in GPU-Dense Servers: A Node-Level Agentic Digital Twin},
  author={Pooja},
  year={2026},
  school={BITS Pilani}
}
```

## License

This is academic research code. Please contact for usage permissions.
