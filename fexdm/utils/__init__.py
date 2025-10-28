"""Utilities package for FEX-DM."""

from .data_generation import (
    generate_ornstein_uhlenbeck,
    generate_double_well,
    calculate_mse,
    calculate_rmse,
    calculate_r2_score
)

__all__ = [
    "generate_ornstein_uhlenbeck",
    "generate_double_well",
    "calculate_mse",
    "calculate_rmse",
    "calculate_r2_score"
]
