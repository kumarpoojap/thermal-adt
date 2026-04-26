"""
Unified surrogate interface for thermal dynamics models.

This module provides a common interface for different surrogate types
(RC, RF, PINN) to be used interchangeably in the RL environment.
"""

from .interface import ThermalSurrogate
from .rf_adapter import RFAdapter
from .rc_adapter import RCAdapter
from .pinn_adapter import PINNAdapter
from .factory import create_surrogate

__all__ = [
    "ThermalSurrogate",
    "RFAdapter",
    "RCAdapter",
    "PINNAdapter",
    "create_surrogate",
]
