"""
Plot sensor fault robustness results for actual experiments.

Works with biased_ambient and noisy_temp results.
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

# Load both experiment results
biased_file = Path("results/sensor_faults/biased_ambient/sensor_fault_results.csv")
noisy_file = Path("results/sensor_faults/noisy_temp/sensor_fault_results.csv")

if not biased_file.exists() or not noisy_file.exists():
    print(f"\n❌ Error: Results files not found!")
    print(f"Looking for:")
    print(f"  - {biased_file}")
    print(f"  - {noisy_file}")
    exit(1)

# Load data
df_biased = pd.read_csv(biased_file)
df_noisy = pd.read_csv(noisy_file)

# Combine datasets
df = pd.concat([df_biased, df_noisy], ignore_index=True)

print(f"\n✅ Loaded {len(df_biased)} biased ambient results")
print(f"✅ Loaded {len(df_noisy)} noisy temp results")
print(f"✅ Total: {len(df)} results")

# Get unique values
controllers = sorted(df['controller'].unique())
fault_types = sorted(df['fault_type'].unique())

print(f"\nControllers: {controllers}")
print(f"Fault types: {fault_types}")

# ============================================================================
# Plot 1: Reward Comparison by Fault Type
# ============================================================================
print("\n📊 Generating Plot 1: Reward Comparison by Fault Type...")

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(fault_types))
width = 0.2

# Plot bars for each controller
for i, controller in enumerate(controllers):
    controller_data = df[df['controller'] == controller]
    rewards = []
    for fault in fault_types:
        fault_data = controller_data[controller_data['fault_type'] == fault]
        if len(fault_data) > 0:
            rewards.append(fault_data['cumulative_reward'].mean())
        else:
            rewards.append(0)
    
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, rewards, width, label=controller.upper().replace('_', '-'))

# Formatting
ax.set_xlabel('Fault Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Cumulative Reward', fontsize=12, fontweight='bold')
ax.set_title('Controller Performance Under Sensor Faults', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f.replace('_', ' ').title() for f in fault_types])
ax.legend(loc='lower left', frameon=True, shadow=True)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "reward_by_fault_type.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'reward_by_fault_type.png'}")
plt.close()

# ============================================================================
# Plot 2: Temperature Control Under Faults
# ============================================================================
print("\n📊 Generating Plot 2: Temperature Control Under Faults...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Max temperature
for i, controller in enumerate(controllers):
    controller_data = df[df['controller'] == controller]
    max_temps = []
    for fault in fault_types:
        fault_data = controller_data[controller_data['fault_type'] == fault]
        if len(fault_data) > 0:
            max_temps.append(fault_data['max_temp'].max())
        else:
            max_temps.append(0)
    
    offset = (i - 1.5) * width
    ax1.bar(x + offset, max_temps, width, label=controller.upper().replace('_', '-'))

# Add threshold lines
ax1.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='Warning (80°C)', alpha=0.7)
ax1.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Critical (85°C)', alpha=0.7)

ax1.set_xlabel('Fault Type', fontsize=11, fontweight='bold')
ax1.set_ylabel('Maximum Temperature (°C)', fontsize=11, fontweight='bold')
ax1.set_title('Max Temperature Under Faults', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f.replace('_', ' ').title() for f in fault_types])
ax1.legend(loc='upper left', frameon=True, shadow=True, fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Mean temperature
for i, controller in enumerate(controllers):
    controller_data = df[df['controller'] == controller]
    mean_temps = []
    for fault in fault_types:
        fault_data = controller_data[controller_data['fault_type'] == fault]
        if len(fault_data) > 0:
            mean_temps.append(fault_data['mean_temp'].mean())
        else:
            mean_temps.append(0)
    
    offset = (i - 1.5) * width
    ax2.bar(x + offset, mean_temps, width, label=controller.upper().replace('_', '-'))

ax2.axhline(y=65, color='green', linestyle='--', linewidth=2, label='Target (65°C)', alpha=0.7)
ax2.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='Warning (80°C)', alpha=0.7)

ax2.set_xlabel('Fault Type', fontsize=11, fontweight='bold')
ax2.set_ylabel('Mean Temperature (°C)', fontsize=11, fontweight='bold')
ax2.set_title('Mean Temperature Under Faults', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f.replace('_', ' ').title() for f in fault_types])
ax2.legend(loc='upper left', frameon=True, shadow=True, fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "temperature_under_faults.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'temperature_under_faults.png'}")
plt.close()

# ============================================================================
# Plot 3: Safety Violations
# ============================================================================
print("\n📊 Generating Plot 3: Safety Violations...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Warning entries
for i, controller in enumerate(controllers):
    controller_data = df[df['controller'] == controller]
    violations = []
    for fault in fault_types:
        fault_data = controller_data[controller_data['fault_type'] == fault]
        if len(fault_data) > 0:
            violations.append(fault_data['warning_entries'].sum())
        else:
            violations.append(0)
    
    offset = (i - 1.5) * width
    ax1.bar(x + offset, violations, width, label=controller.upper().replace('_', '-'))

ax1.set_xlabel('Fault Type', fontsize=11, fontweight='bold')
ax1.set_ylabel('Warning Entries (>80°C)', fontsize=11, fontweight='bold')
ax1.set_title('Warning Violations Under Faults', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f.replace('_', ' ').title() for f in fault_types])
ax1.legend(loc='upper left', frameon=True, shadow=True, fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Critical entries
for i, controller in enumerate(controllers):
    controller_data = df[df['controller'] == controller]
    violations = []
    for fault in fault_types:
        fault_data = controller_data[controller_data['fault_type'] == fault]
        if len(fault_data) > 0:
            violations.append(fault_data['critical_entries'].sum())
        else:
            violations.append(0)
    
    offset = (i - 1.5) * width
    ax2.bar(x + offset, violations, width, label=controller.upper().replace('_', '-'))

ax2.set_xlabel('Fault Type', fontsize=11, fontweight='bold')
ax2.set_ylabel('Critical Entries (>85°C)', fontsize=11, fontweight='bold')
ax2.set_title('Critical Violations Under Faults', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f.replace('_', ' ').title() for f in fault_types])
ax2.legend(loc='upper left', frameon=True, shadow=True, fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "violations_under_faults.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'violations_under_faults.png'}")
plt.close()

# ============================================================================
# Plot 4: Controller Comparison Heatmap
# ============================================================================
print("\n📊 Generating Plot 4: Performance Heatmap...")

# Create pivot table for heatmap
heatmap_data = df.groupby(['controller', 'fault_type'])['cumulative_reward'].mean().unstack()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='.0f',
    cmap='RdYlGn',
    cbar_kws={'label': 'Mean Cumulative Reward'},
    linewidths=1,
    linecolor='white',
    ax=ax
)

ax.set_xlabel('Fault Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Controller', fontsize=12, fontweight='bold')
ax.set_title('Controller Performance Heatmap Under Sensor Faults', fontsize=14, fontweight='bold')
ax.set_yticklabels([label.get_text().upper().replace('_', '-') for label in ax.get_yticklabels()], rotation=0)
ax.set_xticklabels([label.get_text().replace('_', ' ').title() for label in ax.get_xticklabels()], rotation=0)

plt.tight_layout()
plt.savefig(output_dir / "performance_heatmap.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'performance_heatmap.png'}")
plt.close()

# ============================================================================
# Plot 5: Fault Magnitude Analysis
# ============================================================================
print("\n📊 Generating Plot 5: Fault Magnitude Analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Mean fault magnitude
for i, controller in enumerate(controllers):
    controller_data = df[df['controller'] == controller]
    magnitudes = []
    for fault in fault_types:
        fault_data = controller_data[controller_data['fault_type'] == fault]
        if len(fault_data) > 0:
            magnitudes.append(fault_data['mean_fault_magnitude'].mean())
        else:
            magnitudes.append(0)
    
    offset = (i - 1.5) * width
    ax1.bar(x + offset, magnitudes, width, label=controller.upper().replace('_', '-'))

ax1.set_xlabel('Fault Type', fontsize=11, fontweight='bold')
ax1.set_ylabel('Mean Fault Magnitude', fontsize=11, fontweight='bold')
ax1.set_title('Average Fault Magnitude', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f.replace('_', ' ').title() for f in fault_types])
ax1.legend(loc='upper left', frameon=True, shadow=True, fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Max fault magnitude
for i, controller in enumerate(controllers):
    controller_data = df[df['controller'] == controller]
    magnitudes = []
    for fault in fault_types:
        fault_data = controller_data[controller_data['fault_type'] == fault]
        if len(fault_data) > 0:
            magnitudes.append(fault_data['max_fault_magnitude'].max())
        else:
            magnitudes.append(0)
    
    offset = (i - 1.5) * width
    ax2.bar(x + offset, magnitudes, width, label=controller.upper().replace('_', '-'))

ax2.set_xlabel('Fault Type', fontsize=11, fontweight='bold')
ax2.set_ylabel('Max Fault Magnitude', fontsize=11, fontweight='bold')
ax2.set_title('Maximum Fault Magnitude', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([f.replace('_', ' ').title() for f in fault_types])
ax2.legend(loc='upper left', frameon=True, shadow=True, fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "fault_magnitude_analysis.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'fault_magnitude_analysis.png'}")
plt.close()

# ============================================================================
# Plot 6: Summary Comparison
# ============================================================================
print("\n📊 Generating Plot 6: Summary Comparison...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Overall reward by controller
controller_rewards = df.groupby('controller')['cumulative_reward'].mean().sort_values(ascending=True)
colors = ['#2ecc71' if i >= len(controller_rewards)//2 else '#e74c3c' for i in range(len(controller_rewards))]
bars = ax1.barh(range(len(controller_rewards)), controller_rewards.values, color=colors, alpha=0.8)

for i, (controller, value) in enumerate(controller_rewards.items()):
    ax1.text(value + 50, i, f'{value:.0f}', va='center', fontweight='bold')

ax1.set_yticks(range(len(controller_rewards)))
ax1.set_yticklabels([c.upper().replace('_', '-') for c in controller_rewards.index])
ax1.set_xlabel('Mean Cumulative Reward', fontsize=11, fontweight='bold')
ax1.set_title('Overall Controller Performance', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Max temperature by controller
controller_temps = df.groupby('controller')['max_temp'].max().sort_values(ascending=True)
colors = ['#2ecc71' if x < 80 else '#f39c12' if x < 85 else '#e74c3c' for x in controller_temps.values]
bars = ax2.barh(range(len(controller_temps)), controller_temps.values, color=colors, alpha=0.8)

for i, (controller, value) in enumerate(controller_temps.items()):
    ax2.text(value + 0.5, i, f'{value:.1f}°C', va='center', fontweight='bold')

ax2.set_yticks(range(len(controller_temps)))
ax2.set_yticklabels([c.upper().replace('_', '-') for c in controller_temps.index])
ax2.set_xlabel('Maximum Temperature (°C)', fontsize=11, fontweight='bold')
ax2.set_title('Temperature Control Performance', fontsize=12, fontweight='bold')
ax2.axvline(x=80, color='orange', linestyle='--', linewidth=2, alpha=0.7)
ax2.axvline(x=85, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax2.grid(axis='x', alpha=0.3)

# Warning entries by controller
controller_warnings = df.groupby('controller')['warning_entries'].sum().sort_values(ascending=True)
bars = ax3.barh(range(len(controller_warnings)), controller_warnings.values, color='#f39c12', alpha=0.8)

for i, (controller, value) in enumerate(controller_warnings.items()):
    ax3.text(value + 0.5, i, f'{int(value)}', va='center', fontweight='bold')

ax3.set_yticks(range(len(controller_warnings)))
ax3.set_yticklabels([c.upper().replace('_', '-') for c in controller_warnings.index])
ax3.set_xlabel('Total Warning Entries', fontsize=11, fontweight='bold')
ax3.set_title('Warning Violations (>80°C)', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Critical entries by controller
controller_critical = df.groupby('controller')['critical_entries'].sum().sort_values(ascending=True)
bars = ax4.barh(range(len(controller_critical)), controller_critical.values, color='#e74c3c', alpha=0.8)

for i, (controller, value) in enumerate(controller_critical.items()):
    if value > 0:
        ax4.text(value + 0.1, i, f'{int(value)}', va='center', fontweight='bold')

ax4.set_yticks(range(len(controller_critical)))
ax4.set_yticklabels([c.upper().replace('_', '-') for c in controller_critical.index])
ax4.set_xlabel('Total Critical Entries', fontsize=11, fontweight='bold')
ax4.set_title('Critical Violations (>85°C)', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

plt.suptitle('Sensor Fault Robustness: Summary Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / "summary_comparison.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'summary_comparison.png'}")
plt.close()

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nMean Reward by Controller:")
print(df.groupby('controller')['cumulative_reward'].mean().sort_values(ascending=False))

print("\nMax Temperature by Controller:")
print(df.groupby('controller')['max_temp'].max().sort_values())

print("\nTotal Violations:")
print(f"  Warning entries: {df['warning_entries'].sum()}")
print(f"  Critical entries: {df['critical_entries'].sum()}")

print("\n" + "="*80)
print("✅ All sensor fault plots generated successfully!")
print(f"📁 Output directory: {output_dir}")
print("="*80)
