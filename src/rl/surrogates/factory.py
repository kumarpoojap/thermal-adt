"""
Factory for creating surrogate adapters from configuration.

Enables easy switching between RC, RF, and PINN surrogates.
"""

from typing import Dict, Literal
from pathlib import Path

from .interface import ThermalSurrogate
from .rc_adapter import RCAdapter
from .rf_adapter import RFAdapter
from .pinn_adapter import PINNAdapter


def create_surrogate(
    surrogate_type: Literal["rc", "rf", "pinn"],
    config: Dict
) -> ThermalSurrogate:
    """
    Create a surrogate adapter from configuration.
    
    Args:
        surrogate_type: Type of surrogate ("rc", "rf", or "pinn")
        config: Configuration dict with surrogate-specific parameters
    
    Returns:
        Surrogate adapter conforming to ThermalSurrogate protocol
    
    Raises:
        ValueError: If surrogate_type is unknown
    
    Example config for RF:
        {
            "surrogate_type": "rf",
            "model_path": "/path/to/rf_model.pkl",
            "cache_dir": None,
            "use_cache": False
        }
    
    Example config for RC:
        {
            "surrogate_type": "rc",
            "thermal_capacity": 100.0,
            "heat_transfer_coeff": 0.05,
            "cooling_effectiveness": -0.03,
            "power_to_heat": 0.01,
            "dt": 1.0
        }
    
    Example config for PINN:
        {
            "surrogate_type": "pinn",
            "model_path": "/path/to/pinn_checkpoint.pt",
            "device": "cuda",
            "input_mean": [...],
            "input_std": [...],
            "output_mean": 0.0,
            "output_std": 1.0
        }
    """
    if surrogate_type == "rc":
        return RCAdapter(
            thermal_capacity=config.get("thermal_capacity", 100.0),
            heat_transfer_coeff=config.get("heat_transfer_coeff", 0.05),
            cooling_effectiveness=config.get("cooling_effectiveness", -0.03),
            power_to_heat=config.get("power_to_heat", 0.01),
            dt=config.get("dt", 1.0),
            temp_min=config.get("temp_min", 30.0),
            temp_max=config.get("temp_max", 95.0),
            config=config.get("rc_config", None)
        )
    
    elif surrogate_type == "rf":
        model_path = config.get("model_path")
        if model_path is None:
            raise ValueError("RF surrogate requires 'model_path' in config")
        
        return RFAdapter(
            model_path=Path(model_path),
            cache_dir=Path(config["cache_dir"]) if config.get("cache_dir") else None,
            use_cache=config.get("use_cache", False)
        )
    
    elif surrogate_type == "pinn":
        model_path = config.get("model_path")
        if model_path is None:
            raise ValueError("PINN surrogate requires 'model_path' in config")
        
        return PINNAdapter(
            model_path=Path(model_path),
            device=config.get("device", None),
            input_mean=config.get("input_mean", None),
            input_std=config.get("input_std", None),
            output_mean=config.get("output_mean", None),
            output_std=config.get("output_std", None),
            config=config.get("pinn_config", None),
            feature_columns_path=Path(config["feature_columns_path"]) if config.get("feature_columns_path") else None,
            scalers_path=Path(config["scalers_path"]) if config.get("scalers_path") else None,
            cadence_seconds=float(config.get("cadence_seconds", 1.0)),
            strict_features=bool(config.get("strict_features", True)),
        )
    
    else:
        raise ValueError(
            f"Unknown surrogate type: {surrogate_type}. "
            f"Must be one of: 'rc', 'rf', 'pinn'"
        )
