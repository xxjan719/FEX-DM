"""
FEX-DM: Finite Expression Methods for Identifying Stochastic Dynamics

This package implements methods for identifying stochastic dynamical systems
from time series data using finite expression methods.
"""

__version__ = "0.1.0"
__author__ = "FEX-DM Contributors"

from .core.identifier import StochasticDynamicsIdentifier
from .core.symbolic_regressor import SymbolicRegressor
from .models.sde import StochasticDifferentialEquation

__all__ = [
    "StochasticDynamicsIdentifier",
    "SymbolicRegressor",
    "StochasticDifferentialEquation",
]
