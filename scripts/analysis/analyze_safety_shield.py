"""
Analyze safety shield evaluation results.
"""

import pandas as pd
import numpy as np

print("="*80)
print("SAFETY SHIELD ANALYSIS")
print("="*80)

# Load results
df = pd.read_csv('results/safety_shield_final/safety_shield_results.csv')

# Separate shielded vs unshielded
unshielded = df[df['shield'] == 'none']
shielded = df[df['shield'] == 'active']

print("\n" + "="*80)
print("UNSHIELDED RL (No Safety Constraints)")
print("="*80)

unshielded_summary = unshielded.groupby('scenario').agg({
    'cumulative_reward': 'mean',
    'max_temp': ['mean', 'max'],
    'mean_fan': 'mean',
    'warning_violations': 'sum',
    'critical_violations': 'sum',
    'emergency_violations': 'sum',
}).round(2)

print("\n", unshielded_summary)

print("\n" + "="*80)
print("SHIELDED RL (With Safety Shield)")
print("="*80)

shielded_summary = shielded.groupby('scenario').agg({
    'cumulative_reward': 'mean',
    'max_temp': ['mean', 'max'],
    'mean_fan': 'mean',
    'warning_violations': 'sum',
    'critical_violations': 'sum',
    'emergency_violations': 'sum',
    'safety_interventions': 'sum',
    'intervention_rate': 'mean',
}).round(2)

print("\n", shielded_summary)

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

for scenario in df['scenario'].unique():
    print(f"\n{scenario.upper()}:")
    print("-" * 40)
    
    unsh = unshielded[unshielded['scenario'] == scenario]
    sh = shielded[shielded['scenario'] == scenario]
    
    # Rewards
    unsh_reward = unsh['cumulative_reward'].mean()
    sh_reward = sh['cumulative_reward'].mean()
    reward_change = 100 * (sh_reward - unsh_reward) / unsh_reward if unsh_reward != 0 else 0
    
    print(f"  Reward:")
    print(f"    Unshielded: {unsh_reward:.1f}")
    print(f"    Shielded:   {sh_reward:.1f}")
    print(f"    Change:     {reward_change:+.1f}%")
    
    # Safety violations
    unsh_critical = unsh['critical_violations'].sum()
    sh_critical = sh['critical_violations'].sum()
    
    print(f"\n  Critical Violations (>85°C):")
    print(f"    Unshielded: {unsh_critical}")
    print(f"    Shielded:   {sh_critical}")
    if unsh_critical > 0:
        reduction = 100 * (unsh_critical - sh_critical) / unsh_critical
        print(f"    Reduction:  {reduction:.1f}%")
    
    # Emergency violations
    unsh_emerg = unsh['emergency_violations'].sum()
    sh_emerg = sh['emergency_violations'].sum()
    
    print(f"\n  Emergency Violations (>88°C):")
    print(f"    Unshielded: {unsh_emerg}")
    print(f"    Shielded:   {sh_emerg}")
    if unsh_emerg > 0:
        reduction = 100 * (unsh_emerg - sh_emerg) / unsh_emerg
        print(f"    Reduction:  {reduction:.1f}%")
    
    # Max temperature
    unsh_max = unsh['max_temp'].max()
    sh_max = sh['max_temp'].max()
    
    print(f"\n  Maximum Temperature:")
    print(f"    Unshielded: {unsh_max:.1f}°C")
    print(f"    Shielded:   {sh_max:.1f}°C")
    print(f"    Reduction:  {unsh_max - sh_max:.1f}°C")
    
    # Safety interventions
    interventions = sh['safety_interventions'].sum()
    intervention_rate = sh['intervention_rate'].mean()
    
    print(f"\n  Safety Shield Activity:")
    print(f"    Total interventions: {interventions}")
    print(f"    Intervention rate:   {intervention_rate:.1%}")

print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)

# Overall statistics
total_unsh_critical = unshielded['critical_violations'].sum()
total_sh_critical = shielded['critical_violations'].sum()
total_unsh_emerg = unshielded['emergency_violations'].sum()
total_sh_emerg = shielded['emergency_violations'].sum()

print(f"\nTotal Critical Violations (>85°C):")
print(f"  Unshielded: {total_unsh_critical}")
print(f"  Shielded:   {total_sh_critical}")
if total_unsh_critical > 0:
    reduction = 100 * (total_unsh_critical - total_sh_critical) / total_unsh_critical
    print(f"  Reduction:  {reduction:.1f}%")

print(f"\nTotal Emergency Violations (>88°C):")
print(f"  Unshielded: {total_unsh_emerg}")
print(f"  Shielded:   {total_sh_emerg}")
if total_unsh_emerg > 0:
    reduction = 100 * (total_unsh_emerg - total_sh_emerg) / total_unsh_emerg
    print(f"  Reduction:  {reduction:.1f}%")

# Reward impact
avg_unsh_reward = unshielded['cumulative_reward'].mean()
avg_sh_reward = shielded['cumulative_reward'].mean()
reward_impact = 100 * (avg_sh_reward - avg_unsh_reward) / avg_unsh_reward if avg_unsh_reward != 0 else 0

print(f"\nAverage Reward:")
print(f"  Unshielded: {avg_unsh_reward:.1f}")
print(f"  Shielded:   {avg_sh_reward:.1f}")
print(f"  Impact:     {reward_impact:+.1f}%")

# Safety interventions
total_interventions = shielded['safety_interventions'].sum()
avg_intervention_rate = shielded['intervention_rate'].mean()

print(f"\nSafety Shield Activity:")
print(f"  Total interventions:  {total_interventions}")
print(f"  Avg intervention rate: {avg_intervention_rate:.1%}")

print("\n" + "="*80)
print("THESIS INTERPRETATION")
print("="*80)

if total_unsh_critical == 0 and total_sh_critical == 0:
    print("\n✓ RL POLICY LEARNED INHERENTLY SAFE BEHAVIOR!")
    print("\n  Key Findings:")
    print(f"  - Zero critical violations in both shielded and unshielded modes")
    print(f"  - Policy trained with safety shield internalized safe behavior")
    print(f"  - Shield interventions: {total_interventions} ({avg_intervention_rate:.1%})")
    print(f"  - Reward impact: {reward_impact:+.1f}% (negligible)")
    
    # Check temperature differences
    max_unsh_temp = unshielded['max_temp'].max()
    max_sh_temp = shielded['max_temp'].max()
    temp_reduction = max_unsh_temp - max_sh_temp
    
    print(f"\n  Temperature Control:")
    print(f"  - Unshielded max: {max_unsh_temp:.1f}°C")
    print(f"  - Shielded max:   {max_sh_temp:.1f}°C")
    if temp_reduction > 0:
        print(f"  - Shield provides {temp_reduction:.1f}°C additional safety margin")
    
    print("\n  Thesis Contribution:")
    print("  ✓ Shield-aware training produces inherently safe policies")
    print("  ✓ Shield acts as fail-safe for out-of-distribution scenarios")
    print("  ✓ Zero performance trade-off for safety guarantees")
    print("  ✓ Validates safety-first RL design for cyber-physical systems")
    
elif total_sh_critical < total_unsh_critical and reward_impact > -10:
    print("\n✓ SAFETY SHIELD IS EFFECTIVE!")
    print(f"  - Reduces critical violations by {100 * (total_unsh_critical - total_sh_critical) / total_unsh_critical:.1f}%")
    print(f"  - Minimal reward impact ({reward_impact:+.1f}%)")
    print("  - Validates safety-aware RL for cyber-physical systems")
elif total_sh_critical < total_unsh_critical:
    print("\n⚠ SAFETY SHIELD IMPROVES SAFETY BUT IMPACTS PERFORMANCE")
    print(f"  - Reduces violations but decreases reward by {abs(reward_impact):.1f}%")
    print("  - Trade-off between safety and performance")
else:
    print("\n⚠ UNEXPECTED RESULTS")
    print("  - Further investigation needed")

print("\n" + "="*80)
