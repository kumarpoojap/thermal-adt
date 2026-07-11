"""
Plot sensor fault robustness results.

Creates visualizations comparing controller performance under different sensor faults.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Output directory
output_dir = Path("results/sensor_faults/plots")
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("SENSOR FAULT ROBUSTNESS - VISUALIZATION")
print("="*80)

# Load comparison data
comparison_file = Path("results/sensor_faults/analysis/sensor_fault_comparison.csv")
if not comparison_file.exists():
    print(f"\n❌ Error: {comparison_file} not found!")
    print("Run analyze_sensor_faults.py first to generate comparison data.")
    exit(1)

df = pd.read_csv(comparison_file)

# ============================================================================
# Plot 1: Reward Degradation by Fault Type
# ============================================================================
print("\n📊 Generating Plot 1: Reward Degradation by Fault Type...")

fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data
fault_types = df['fault_type'].unique()
controllers = ['mpc_rc', 'mpc_rcnn', 'rl_rc', 'rl_rcnn']
x = np.arange(len(fault_types))
width = 0.2

# Plot bars for each controller
for i, controller in enumerate(controllers):
    controller_data = df[df['controller'] == controller]
    degradation = []
    for fault in fault_types:
        fault_data = controller_data[controller_data['fault_type'] == fault]
        if len(fault_data) > 0:
            degradation.append(fault_data['reward_degradation_pct'].values[0])
        else:
            degradation.append(0)
    
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, degradation, width, label=controller.upper().replace('_', '-'))

# Formatting
ax.set_xlabel('Fault Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Reward Degradation (%)', fontsize=12, fontweight='bold')
ax.set_title('Controller Performance Under Sensor Faults', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f.replace('_', ' ').title() for f in fault_types])
ax.legend(loc='upper left', frameon=True, shadow=True)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig(output_dir / "reward_degradation_by_fault.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'reward_degradation_by_fault.png'}")
plt.close()

# ============================================================================
# Plot 2: Temperature Control Under Faults
# ============================================================================
print("\n📊 Generating Plot 2: Temperature Control Under Faults...")

fig, ax = plt.subplots(figsize=(12, 6))

# Plot max temperature for each controller and fault
for i, controller in enumerate(controllers):
    controller_data = df[df['controller'] == controller]
    max_temps = []
    for fault in fault_types:
        fault_data = controller_data[controller_data['fault_type'] == fault]
        if len(fault_data) > 0:
            max_temps.append(fault_data['max_temp'].values[0])
        else:
            max_temps.append(0)
    
    offset = (i - 1.5) * width
    ax.bar(x + offset, max_temps, width, label=controller.upper().replace('_', '-'))

# Add threshold lines
ax.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='Warning (80°C)', alpha=0.7)
ax.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Critical (85°C)', alpha=0.7)

# Formatting
ax.set_xlabel('Fault Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Maximum Temperature (°C)', fontsize=12, fontweight='bold')
ax.set_title('Temperature Control Under Sensor Faults', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f.replace('_', ' ').title() for f in fault_types])
ax.legend(loc='upper left', frameon=True, shadow=True)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "temperature_control_under_faults.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'temperature_control_under_faults.png'}")
plt.close()

# ============================================================================
# Plot 3: Robustness Heatmap
# ============================================================================
print("\n📊 Generating Plot 3: Robustness Heatmap...")

# Create pivot table for heatmap
heatmap_data = df.pivot_table(
    values='reward_degradation_pct',
    index='controller',
    columns='fault_type',
    aggfunc='mean'
)

# Reorder for better visualization
heatmap_data = heatmap_data.reindex(['mpc_rc', 'mpc_rcnn', 'rl_rc', 'rl_rcnn'])

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='.1f',
    cmap='RdYlGn_r',
    cbar_kws={'label': 'Reward Degradation (%)'},
    linewidths=1,
    linecolor='white',
    ax=ax,
    vmin=0,
    vmax=30
)

ax.set_xlabel('Fault Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Controller', fontsize=12, fontweight='bold')
ax.set_title('Sensor Fault Robustness Heatmap', fontsize=14, fontweight='bold')
ax.set_yticklabels([label.get_text().upper().replace('_', '-') for label in ax.get_yticklabels()], rotation=0)
ax.set_xticklabels([label.get_text().replace('_', ' ').title() for label in ax.get_xticklabels()], rotation=45, ha='right')

plt.tight_layout()
plt.savefig(output_dir / "robustness_heatmap.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'robustness_heatmap.png'}")
plt.close()

# ============================================================================
# Plot 4: Violation Comparison
# ============================================================================
print("\n📊 Generating Plot 4: Safety Violations Under Faults...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Warning violations
for i, controller in enumerate(controllers):
    controller_data = df[df['controller'] == controller]
    violations = []
    for fault in fault_types:
        fault_data = controller_data[controller_data['fault_type'] == fault]
        if len(fault_data) > 0:
            violations.append(fault_data['warning_violations'].values[0])
        else:
            violations.append(0)
    
    offset = (i - 1.5) * width
    ax1.bar(x + offset, violations, width, label=controller.upper().replace('_', '-'))

ax1.set_xlabel('Fault Type', fontsize=11, fontweight='bold')
ax1.set_ylabel('Warning Violations (>80°C)', fontsize=11, fontweight='bold')
ax1.set_title('Warning Violations Under Sensor Faults', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f.replace('_', ' ').title() for f in fault_types], rotation=45, ha='right')
ax1.legend(loc='upper left', frameon=True, shadow=True, fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Critical violations
for i, controller in enumerate(controllers):
    controller_data = df[df['controller'] == controller]
    violations = []
    for fault in fault_types:
        fault_data = controller_data[controller_data['fault_type'] == fault]
        if len(fault_data) > 0:
            violations.append(fault_data['critical_violations'].values[0])
        else:
            violations.append(0)
    
    offset = (i - 1.5) * width
    ax2.bar(x + offset, violations, width, label=controller.upper().replace('_', '-'))

ax2.set_xlabel('Fault Type', fontsize=11, fontweight='bold')
ax2.set_ylabel('Critical Violations (>85°C)', fontsize=11, fontweight='bold')
ax2.set_title('Critical Violations Under Sensor Faults', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f.replace('_', ' ').title() for f in fault_types], rotation=45, ha='right')
ax2.legend(loc='upper left', frameon=True, shadow=True, fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "violations_under_faults.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'violations_under_faults.png'}")
plt.close()

# ============================================================================
# Plot 5: Controller Ranking
# ============================================================================
print("\n📊 Generating Plot 5: Overall Controller Ranking...")

# Calculate average degradation per controller
avg_degradation = df.groupby('controller')['reward_degradation_pct'].mean().sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2ecc71' if x < 10 else '#f39c12' if x < 20 else '#e74c3c' for x in avg_degradation.values]
bars = ax.barh(range(len(avg_degradation)), avg_degradation.values, color=colors)

# Add value labels
for i, (controller, value) in enumerate(avg_degradation.items()):
    ax.text(value + 0.5, i, f'{value:.1f}%', va='center', fontweight='bold')

ax.set_yticks(range(len(avg_degradation)))
ax.set_yticklabels([c.upper().replace('_', '-') for c in avg_degradation.index])
ax.set_xlabel('Average Reward Degradation (%)', fontsize=12, fontweight='bold')
ax.set_title('Overall Controller Robustness Ranking\n(Lower is Better)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='Excellent (<10%)'),
    Patch(facecolor='#f39c12', label='Good (10-20%)'),
    Patch(facecolor='#e74c3c', label='Poor (>20%)')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig(output_dir / "controller_ranking.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'controller_ranking.png'}")
plt.close()

print("\n" + "="*80)
print("✅ All sensor fault plots generated successfully!")
print(f"📁 Output directory: {output_dir}")
print("="*80)
