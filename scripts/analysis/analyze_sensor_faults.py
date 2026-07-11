"""
Analyze sensor fault robustness results.
"""

import pandas as pd
import numpy as np

print("="*80)
print("SENSOR FAULT ROBUSTNESS ANALYSIS")
print("="*80)

# Load baseline (no fault) results
baseline = pd.read_csv('results/policy_eval_v2/combined_per_scenario.csv')
baseline_only = baseline[baseline['scenario'] == 'baseline']
baseline_summary = baseline_only.groupby('controller')['cumulative_reward'].mean()

print("\n" + "="*80)
print("BASELINE (NO FAULT)")
print("="*80)
print(baseline_summary.to_string())

# Load biased ambient results
biased = pd.read_csv('results/sensor_faults/biased_ambient/sensor_fault_results.csv')
biased_summary = biased.groupby('controller')['cumulative_reward'].mean()

print("\n" + "="*80)
print("BIASED AMBIENT SENSOR (+5°C)")
print("="*80)
print(biased_summary.to_string())

# Load noisy temp results
noisy = pd.read_csv('results/sensor_faults/noisy_temp/sensor_fault_results.csv')
noisy_summary = noisy.groupby('controller')['cumulative_reward'].mean()

print("\n" + "="*80)
print("NOISY TEMPERATURE SENSOR (±2°C)")
print("="*80)
print(noisy_summary.to_string())

# Calculate degradation
print("\n" + "="*80)
print("DEGRADATION ANALYSIS")
print("="*80)

results = []
for controller in ['mpc_rc', 'mpc_rcnn', 'rl_rc', 'rl_rcnn']:
    base_reward = baseline_summary[controller]
    biased_reward = biased_summary[controller]
    noisy_reward = noisy_summary[controller]
    
    biased_deg = 100 * (biased_reward - base_reward) / base_reward
    noisy_deg = 100 * (noisy_reward - base_reward) / base_reward
    
    results.append({
        'Controller': controller,
        'Baseline': f"{base_reward:.1f}",
        'Biased': f"{biased_reward:.1f}",
        'Biased_Deg_%': f"{biased_deg:+.1f}",
        'Noisy': f"{noisy_reward:.1f}",
        'Noisy_Deg_%': f"{noisy_deg:+.1f}",
    })

df_results = pd.DataFrame(results)
print("\n" + df_results.to_string(index=False))

# Key findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print("\n1. BIASED AMBIENT SENSOR (+5°C bias):")
print("-" * 40)

mpc_rc_bias_deg = 100 * (biased_summary['mpc_rc'] - baseline_summary['mpc_rc']) / baseline_summary['mpc_rc']
mpc_rcnn_bias_deg = 100 * (biased_summary['mpc_rcnn'] - baseline_summary['mpc_rcnn']) / baseline_summary['mpc_rcnn']
rl_rc_bias_deg = 100 * (biased_summary['rl_rc'] - baseline_summary['rl_rc']) / baseline_summary['rl_rc']
rl_rcnn_bias_deg = 100 * (biased_summary['rl_rcnn'] - baseline_summary['rl_rcnn']) / baseline_summary['rl_rcnn']

print(f"  MPC-RC:   {mpc_rc_bias_deg:+.1f}% degradation")
print(f"  MPC-RCNN: {mpc_rcnn_bias_deg:+.1f}% degradation")
print(f"  RL-RC:    {rl_rc_bias_deg:+.1f}% degradation")
print(f"  RL-RCNN:  {rl_rcnn_bias_deg:+.1f}% degradation")

if abs(mpc_rcnn_bias_deg) < abs(mpc_rc_bias_deg):
    print(f"\n  ✓ MPC-RCNN is MORE robust than MPC-RC to biased sensor")
    print(f"    ({abs(mpc_rcnn_bias_deg):.1f}% vs {abs(mpc_rc_bias_deg):.1f}%)")
else:
    print(f"\n  ✗ MPC-RC is more robust than MPC-RCNN")

if abs(rl_rcnn_bias_deg) < abs(rl_rc_bias_deg):
    print(f"  ✓ RL-RCNN is MORE robust than RL-RC to biased sensor")
    print(f"    ({abs(rl_rcnn_bias_deg):.1f}% vs {abs(rl_rc_bias_deg):.1f}%)")
else:
    print(f"  ✗ RL-RC is more robust than RL-RCNN")

print("\n2. NOISY TEMPERATURE SENSOR (±2°C noise):")
print("-" * 40)

mpc_rc_noise_deg = 100 * (noisy_summary['mpc_rc'] - baseline_summary['mpc_rc']) / baseline_summary['mpc_rc']
mpc_rcnn_noise_deg = 100 * (noisy_summary['mpc_rcnn'] - baseline_summary['mpc_rcnn']) / baseline_summary['mpc_rcnn']
rl_rc_noise_deg = 100 * (noisy_summary['rl_rc'] - baseline_summary['rl_rc']) / baseline_summary['rl_rc']
rl_rcnn_noise_deg = 100 * (noisy_summary['rl_rcnn'] - baseline_summary['rl_rcnn']) / baseline_summary['rl_rcnn']

print(f"  MPC-RC:   {mpc_rc_noise_deg:+.1f}% degradation")
print(f"  MPC-RCNN: {mpc_rcnn_noise_deg:+.1f}% degradation")
print(f"  RL-RC:    {rl_rc_noise_deg:+.1f}% degradation")
print(f"  RL-RCNN:  {rl_rcnn_noise_deg:+.1f}% degradation")

if abs(mpc_rcnn_noise_deg) < abs(mpc_rc_noise_deg):
    print(f"\n  ✓ MPC-RCNN is MORE robust than MPC-RC to noisy sensor")
    print(f"    ({abs(mpc_rcnn_noise_deg):.1f}% vs {abs(mpc_rc_noise_deg):.1f}%)")
else:
    print(f"  ✗ MPC-RC is more robust than MPC-RCNN")

if abs(rl_rcnn_noise_deg) < abs(rl_rc_noise_deg):
    print(f"  ✓ RL-RCNN is MORE robust than RL-RC to noisy sensor")
    print(f"    ({abs(rl_rcnn_noise_deg):.1f}% vs {abs(rl_rc_noise_deg):.1f}%)")
else:
    print(f"  ✗ RL-RC is more robust than RL-RCNN")

print("\n3. OVERALL ROBUSTNESS:")
print("-" * 40)

# Average absolute degradation across both faults
mpc_rc_avg = (abs(mpc_rc_bias_deg) + abs(mpc_rc_noise_deg)) / 2
mpc_rcnn_avg = (abs(mpc_rcnn_bias_deg) + abs(mpc_rcnn_noise_deg)) / 2
rl_rc_avg = (abs(rl_rc_bias_deg) + abs(rl_rc_noise_deg)) / 2
rl_rcnn_avg = (abs(rl_rcnn_bias_deg) + abs(rl_rcnn_noise_deg)) / 2

print(f"  MPC-RC:   {mpc_rc_avg:.1f}% average degradation")
print(f"  MPC-RCNN: {mpc_rcnn_avg:.1f}% average degradation")
print(f"  RL-RC:    {rl_rc_avg:.1f}% average degradation")
print(f"  RL-RCNN:  {rl_rcnn_avg:.1f}% average degradation")

# Rank by robustness
robustness = {
    'MPC-RC': mpc_rc_avg,
    'MPC-RCNN': mpc_rcnn_avg,
    'RL-RC': rl_rc_avg,
    'RL-RCNN': rl_rcnn_avg,
}
ranked = sorted(robustness.items(), key=lambda x: x[1])

print("\n  Robustness Ranking (most robust first):")
for i, (controller, deg) in enumerate(ranked, 1):
    print(f"    {i}. {controller}: {deg:.1f}% avg degradation")

print("\n" + "="*80)
print("THESIS INTERPRETATION")
print("="*80)

if mpc_rcnn_avg < mpc_rc_avg and rl_rcnn_avg < rl_rc_avg:
    print("\n✓ RCNN-based controllers are MORE ROBUST to sensor faults than RC-based!")
    print("  This validates the hybrid physics-ML approach for real-world deployment.")
elif mpc_rcnn_avg > mpc_rc_avg and rl_rcnn_avg > rl_rc_avg:
    print("\n✗ RC-based controllers are more robust than RCNN-based.")
    print("  This suggests pure physics models are less sensitive to sensor errors.")
else:
    print("\n⚠ Mixed results: Robustness depends on controller type (MPC vs RL).")
    print("  Further investigation needed.")

print("\n" + "="*80)
