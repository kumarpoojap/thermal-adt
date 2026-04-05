# Experiment {N}: {Title}

**Date**: {YYYY-MM-DD}  
**Status**: {Planning | In Progress | Complete | Published}  
**Corresponds to**: Dissertation Section {X.Y}

---

## 1. Objective

{Clear statement of what this experiment aims to demonstrate or measure}

**Research Questions**:
1. {Question 1}
2. {Question 2}

**Hypotheses**:
- H1: {Hypothesis 1}
- H2: {Hypothesis 2}

---

## 2. Experimental Setup

### 2.1 Configuration

**Surrogates Tested**:
- [ ] RC Model
- [ ] Random Forest
- [ ] PINN-lite

**Controllers/Agents**:
- [ ] SAC (Shielded)
- [ ] SAC (Unshielded)
- [ ] Static Fan Curve
- [ ] Threshold Controller
- [ ] MPC Baseline

**Scenarios**:
| Scenario | Workload Profile | Ambient Temp | Duration | Description |
|----------|------------------|--------------|----------|-------------|
| Nominal  | Moderate (30-60%) | 25°C | 300s | Typical operation |
| Stress   | Bursty (20-100%) | 30°C | 300s | High stress test |

**Seeds**: [42, 123, 456]  
**Total Runs**: {N surrogates × M controllers × K scenarios × 3 seeds}

### 2.2 Metrics

**Primary Metrics**:
- Throttle events (count per episode)
- Thermal headroom (°C, mean and min)
- Time-to-cool (seconds after spike)

**Secondary Metrics**:
- Energy usage (proxy: cumulative fan speed)
- Control stability (action variance)
- Predictive behavior (early intervention count)

### 2.3 Hardware/Environment

- **Platform**: {Local / Colab / HPC}
- **Compute**: {CPU / GPU specs}
- **Runtime**: {Estimated time}

---

## 3. Results

### 3.1 Summary Statistics

**Best Performing Configuration**:
- Surrogate: {RC / RF / PINN}
- Controller: {SAC / MPC / etc.}
- Scenario: {Nominal / Stress}

**Key Metrics** (mean ± std across 3 seeds):

| Metric | SAC (Shielded) | SAC (Unshielded) | Static Fan | Threshold | MPC |
|--------|----------------|------------------|------------|-----------|-----|
| Throttle Events | {X.X ± Y.Y} | {X.X ± Y.Y} | {X.X ± Y.Y} | {X.X ± Y.Y} | {X.X ± Y.Y} |
| Energy Usage | {X.X ± Y.Y} | {X.X ± Y.Y} | {X.X ± Y.Y} | {X.X ± Y.Y} | {X.X ± Y.Y} |
| Mean Temp (°C) | {X.X ± Y.Y} | {X.X ± Y.Y} | {X.X ± Y.Y} | {X.X ± Y.Y} | {X.X ± Y.Y} |
| Min Headroom (°C) | {X.X ± Y.Y} | {X.X ± Y.Y} | {X.X ± Y.Y} | {X.X ± Y.Y} | {X.X ± Y.Y} |

### 3.2 Statistical Significance

**Paired t-tests** (SAC vs baselines, p < 0.05):
- SAC vs Static Fan: {p-value}, {significant / not significant}
- SAC vs Threshold: {p-value}, {significant / not significant}
- SAC vs MPC: {p-value}, {significant / not significant}

**Effect sizes** (Cohen's d):
- Throttle reduction: {d = X.XX} ({small / medium / large})
- Energy improvement: {d = X.XX} ({small / medium / large})

### 3.3 Key Findings

1. **Finding 1**: {Description}
   - Evidence: {Metric/plot reference}
   - Interpretation: {What this means}

2. **Finding 2**: {Description}
   - Evidence: {Metric/plot reference}
   - Interpretation: {What this means}

3. **Finding 3**: {Description}
   - Evidence: {Metric/plot reference}
   - Interpretation: {What this means}

### 3.4 Figures

#### Figure 1: {Title}
![{Alt text}](plots/figure1.png)

**Caption**: {Detailed caption explaining what the figure shows}

**Observations**:
- {Observation 1}
- {Observation 2}

#### Figure 2: {Title}
![{Alt text}](plots/figure2.png)

**Caption**: {Detailed caption}

**Observations**:
- {Observation 1}
- {Observation 2}

---

## 4. Discussion

### 4.1 Interpretation

{Detailed interpretation of results in context of research questions}

**Hypothesis Testing**:
- H1: {Supported / Rejected} - {Explanation}
- H2: {Supported / Rejected} - {Explanation}

### 4.2 Comparison with Literature

{How do these results compare with existing work?}

### 4.3 Limitations

1. {Limitation 1}
2. {Limitation 2}
3. {Limitation 3}

### 4.4 Implications

**For Dissertation**:
- {Implication 1}
- {Implication 2}

**For Practice**:
- {Practical implication 1}
- {Practical implication 2}

---

## 5. Reproducibility

### 5.1 Exact Commands

```bash
# Run experiment
python experiments/exp{N}_{name}.py --config configs/experiments/exp{N}_{name}.yaml

# Generate plots
python scripts/evaluation/plot_results.py --results results/experiments/exp{N}_{name}/

# Statistical analysis
python scripts/evaluation/statistical_tests.py --results results/experiments/exp{N}_{name}/
```

### 5.2 Configuration Files

- **Experiment config**: `configs/experiments/exp{N}_{name}.yaml`
- **Surrogate configs**: `configs/surrogates/{rc,rf,pinn_lite}.yaml`
- **RL config**: `configs/rl/sac_shielded.yaml`

### 5.3 Data Files

- **Input data**: `data/synthetic/processed/synthetic_gpu_thermal.parquet`
- **Results**: `results/experiments/exp{N}_{name}/`
- **Checkpoints**: `results/rl_training/{agent}_{surrogate}_{shield}/run_{date}_{seed}/`

### 5.4 Random Seeds

All experiments use fixed seeds: [42, 123, 456]

### 5.5 Software Versions

- Python: {version}
- PyTorch: {version}
- Stable-Baselines3: {version}
- NumPy: {version}

---

## 6. Thesis Section (Draft)

### {Section Number}: {Section Title}

{Polished text ready to copy into thesis. This should be publication-quality prose.}

#### Experimental Setup

{Concise description of setup for thesis}

#### Results

{Key results presented clearly}

Table X: {Table caption}

| ... | ... |
|-----|-----|

Figure X: {Figure caption}

#### Discussion

{Interpretation and implications}

---

## 7. Next Steps

**Follow-up Experiments**:
- [ ] {Follow-up 1}
- [ ] {Follow-up 2}

**Improvements**:
- [ ] {Improvement 1}
- [ ] {Improvement 2}

**Questions Raised**:
- {Question 1}
- {Question 2}

---

## 8. Appendix

### 8.1 Detailed Results

{Link to full results CSV/JSON}

### 8.2 Additional Plots

{Links to supplementary plots}

### 8.3 Raw Data

{Links to raw experiment data}

### 8.4 Code Snapshot

{Git commit hash or tag for exact code version used}

---

**Last Updated**: {YYYY-MM-DD}  
**Author**: Pooja  
**Reviewer**: {Advisor name}
