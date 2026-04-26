"""
Test scenarios for thermal control evaluation.

Defines nominal and stress test scenarios with varying workload profiles,
ambient conditions, and initial states.
"""

from typing import List, Dict
import numpy as np


def create_scenarios(scenario_types: List[str] = None) -> List[Dict]:
    """
    Create evaluation scenarios.
    
    Args:
        scenario_types: List of scenario types to include
                       Options: ["nominal", "stress", "all"]
                       Default: ["all"]
    
    Returns:
        scenarios: List of scenario configurations
    """
    if scenario_types is None:
        scenario_types = ["all"]
    
    scenarios = []
    
    # Determine which scenarios to include
    include_nominal = "nominal" in scenario_types or "all" in scenario_types
    include_stress = "stress" in scenario_types or "all" in scenario_types
    
    # ========== NOMINAL SCENARIOS ==========
    if include_nominal:
        # 1. Baseline: Normal operation
        scenarios.append({
            "name": "nominal_baseline",
            "description": "Normal operating conditions with moderate workload",
            "initial_temp": 50.0,
            "ambient_temp": 25.0,
            "temp_target": 75.0,
            "temp_max": 85.0,
            "workload_profile": "constant",
            "power_range": [150.0, 200.0],
        })
        
        # 2. Low workload
        scenarios.append({
            "name": "nominal_low_workload",
            "description": "Low workload scenario (idle/light tasks)",
            "initial_temp": 45.0,
            "ambient_temp": 22.0,
            "temp_target": 75.0,
            "temp_max": 85.0,
            "workload_profile": "constant",
            "power_range": [100.0, 150.0],
        })
        
        # 3. High workload
        scenarios.append({
            "name": "nominal_high_workload",
            "description": "High workload scenario (compute-intensive tasks)",
            "initial_temp": 60.0,
            "ambient_temp": 25.0,
            "temp_target": 75.0,
            "temp_max": 85.0,
            "workload_profile": "constant",
            "power_range": [250.0, 300.0],
        })
        
        # 4. Variable workload
        scenarios.append({
            "name": "nominal_variable_workload",
            "description": "Variable workload with periodic changes",
            "initial_temp": 50.0,
            "ambient_temp": 25.0,
            "temp_target": 75.0,
            "temp_max": 85.0,
            "workload_profile": "variable",
            "power_range": [100.0, 300.0],
        })
        
        # 5. Warm ambient
        scenarios.append({
            "name": "nominal_warm_ambient",
            "description": "Normal workload with elevated ambient temperature",
            "initial_temp": 55.0,
            "ambient_temp": 30.0,
            "temp_target": 75.0,
            "temp_max": 85.0,
            "workload_profile": "constant",
            "power_range": [150.0, 200.0],
        })
    
    # ========== STRESS SCENARIOS ==========
    if include_stress:
        # 1. Thermal stress: High initial temp + high workload
        scenarios.append({
            "name": "stress_thermal_high",
            "description": "Thermal stress: high initial temp and sustained high workload",
            "initial_temp": 80.0,
            "ambient_temp": 28.0,
            "temp_target": 75.0,
            "temp_max": 85.0,
            "workload_profile": "constant",
            "power_range": [280.0, 300.0],
        })
        
        # 2. Thermal stress: Very high ambient
        scenarios.append({
            "name": "stress_ambient_extreme",
            "description": "Extreme ambient temperature stress",
            "initial_temp": 60.0,
            "ambient_temp": 35.0,
            "temp_target": 75.0,
            "temp_max": 85.0,
            "workload_profile": "constant",
            "power_range": [200.0, 250.0],
        })
        
        # 3. Workload spike
        scenarios.append({
            "name": "stress_workload_spike",
            "description": "Sudden workload spike from idle to max",
            "initial_temp": 45.0,
            "ambient_temp": 25.0,
            "temp_target": 75.0,
            "temp_max": 85.0,
            "workload_profile": "spike",
            "power_range": [100.0, 300.0],
        })
        
        # 4. Rapid workload oscillation
        scenarios.append({
            "name": "stress_workload_oscillation",
            "description": "Rapid oscillation between low and high workload",
            "initial_temp": 50.0,
            "ambient_temp": 25.0,
            "temp_target": 75.0,
            "temp_max": 85.0,
            "workload_profile": "oscillating",
            "power_range": [100.0, 300.0],
        })
        
        # 5. Combined stress: High temp + high ambient + high workload
        scenarios.append({
            "name": "stress_combined_extreme",
            "description": "Combined extreme stress conditions",
            "initial_temp": 78.0,
            "ambient_temp": 32.0,
            "temp_target": 75.0,
            "temp_max": 85.0,
            "workload_profile": "constant",
            "power_range": [280.0, 300.0],
        })
        
        # 6. Recovery test: Start hot, workload drops
        scenarios.append({
            "name": "stress_recovery",
            "description": "Recovery from high temperature with reduced workload",
            "initial_temp": 82.0,
            "ambient_temp": 25.0,
            "temp_target": 75.0,
            "temp_max": 85.0,
            "workload_profile": "decreasing",
            "power_range": [100.0, 250.0],
        })
        
        # 7. Sustained near-limit operation
        scenarios.append({
            "name": "stress_sustained_limit",
            "description": "Sustained operation near thermal limit",
            "initial_temp": 75.0,
            "ambient_temp": 28.0,
            "temp_target": 75.0,
            "temp_max": 85.0,
            "workload_profile": "constant",
            "power_range": [270.0, 290.0],
        })
    
    return scenarios


def get_workload_profile(profile_type: str, step: int, max_steps: int, power_range: List[float]) -> float:
    """
    Generate workload (power) value based on profile type.
    
    Args:
        profile_type: Type of workload profile
        step: Current step
        max_steps: Maximum steps in episode
        power_range: [min_power, max_power]
    
    Returns:
        power: GPU power consumption (W)
    """
    min_power, max_power = power_range
    
    if profile_type == "constant":
        # Constant workload (with small random variation)
        base_power = (min_power + max_power) / 2
        noise = np.random.uniform(-5.0, 5.0)
        return np.clip(base_power + noise, min_power, max_power)
    
    elif profile_type == "variable":
        # Sinusoidal variation
        period = max_steps / 3  # 3 cycles per episode
        phase = 2 * np.pi * step / period
        normalized = (np.sin(phase) + 1) / 2  # [0, 1]
        return min_power + normalized * (max_power - min_power)
    
    elif profile_type == "spike":
        # Sudden spike at 1/4 of episode
        spike_start = max_steps // 4
        spike_duration = max_steps // 8
        if spike_start <= step < spike_start + spike_duration:
            return max_power
        else:
            return min_power
    
    elif profile_type == "oscillating":
        # Rapid square wave oscillation
        period = max_steps // 10  # 10 cycles
        if (step // period) % 2 == 0:
            return max_power
        else:
            return min_power
    
    elif profile_type == "decreasing":
        # Linear decrease from max to min
        progress = step / max_steps
        return max_power - progress * (max_power - min_power)
    
    elif profile_type == "increasing":
        # Linear increase from min to max
        progress = step / max_steps
        return min_power + progress * (max_power - min_power)
    
    else:
        # Default: constant mid-range
        return (min_power + max_power) / 2


def create_scenario_summary() -> str:
    """
    Create a summary description of all available scenarios.
    
    Returns:
        summary: Formatted summary string
    """
    scenarios = create_scenarios(["all"])
    
    summary = "Available Evaluation Scenarios:\n"
    summary += "=" * 80 + "\n\n"
    
    # Group by type
    nominal = [s for s in scenarios if s["name"].startswith("nominal")]
    stress = [s for s in scenarios if s["name"].startswith("stress")]
    
    summary += f"NOMINAL SCENARIOS ({len(nominal)}):\n"
    summary += "-" * 80 + "\n"
    for s in nominal:
        summary += f"  {s['name']}:\n"
        summary += f"    {s['description']}\n"
        summary += f"    Initial: {s['initial_temp']}°C, Ambient: {s['ambient_temp']}°C\n"
        summary += f"    Power: {s['power_range'][0]}-{s['power_range'][1]}W\n\n"
    
    summary += f"\nSTRESS SCENARIOS ({len(stress)}):\n"
    summary += "-" * 80 + "\n"
    for s in stress:
        summary += f"  {s['name']}:\n"
        summary += f"    {s['description']}\n"
        summary += f"    Initial: {s['initial_temp']}°C, Ambient: {s['ambient_temp']}°C\n"
        summary += f"    Power: {s['power_range'][0]}-{s['power_range'][1]}W\n\n"
    
    return summary


if __name__ == "__main__":
    # Print scenario summary
    print(create_scenario_summary())
