"""
Tests for utility functions.
"""

import numpy as np
import pytest
from fexdm.utils import (
    generate_ornstein_uhlenbeck,
    generate_double_well,
    calculate_mse,
    calculate_rmse,
    calculate_r2_score
)


def test_generate_ornstein_uhlenbeck_shape():
    """Test that OU generator returns correct shape."""
    t, X = generate_ornstein_uhlenbeck(
        theta=1.0, mu=0.0, sigma=0.5,
        x0=0.0, t_span=(0, 10), dt=0.1, seed=42
    )
    
    expected_length = int((10 - 0) / 0.1) + 1
    assert len(t) == expected_length
    assert len(X) == expected_length


def test_generate_ornstein_uhlenbeck_reproducibility():
    """Test that OU generator is reproducible with seed."""
    t1, X1 = generate_ornstein_uhlenbeck(
        theta=1.0, mu=0.0, sigma=0.5,
        x0=0.0, t_span=(0, 5), dt=0.1, seed=42
    )
    
    t2, X2 = generate_ornstein_uhlenbeck(
        theta=1.0, mu=0.0, sigma=0.5,
        x0=0.0, t_span=(0, 5), dt=0.1, seed=42
    )
    
    np.testing.assert_array_equal(X1, X2)


def test_generate_ornstein_uhlenbeck_initial_value():
    """Test that OU process starts at correct initial value."""
    x0 = 3.5
    t, X = generate_ornstein_uhlenbeck(
        theta=1.0, mu=0.0, sigma=0.5,
        x0=x0, t_span=(0, 1), dt=0.1, seed=42
    )
    
    assert X[0] == x0


def test_generate_double_well_shape():
    """Test that double-well generator returns correct shape."""
    t, X = generate_double_well(
        alpha=1.0, beta=1.0, sigma=0.5,
        x0=0.5, t_span=(0, 10), dt=0.1, seed=42
    )
    
    expected_length = int((10 - 0) / 0.1) + 1
    assert len(t) == expected_length
    assert len(X) == expected_length


def test_generate_double_well_reproducibility():
    """Test that double-well generator is reproducible with seed."""
    t1, X1 = generate_double_well(
        alpha=1.0, beta=1.0, sigma=0.5,
        x0=0.5, t_span=(0, 5), dt=0.1, seed=42
    )
    
    t2, X2 = generate_double_well(
        alpha=1.0, beta=1.0, sigma=0.5,
        x0=0.5, t_span=(0, 5), dt=0.1, seed=42
    )
    
    np.testing.assert_array_equal(X1, X2)


def test_calculate_mse():
    """Test MSE calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    
    mse = calculate_mse(y_true, y_pred)
    assert mse == 0.0
    
    y_pred = np.array([2, 3, 4, 5, 6])
    mse = calculate_mse(y_true, y_pred)
    assert mse == 1.0


def test_calculate_rmse():
    """Test RMSE calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    
    rmse = calculate_rmse(y_true, y_pred)
    assert rmse == 0.0
    
    y_pred = np.array([2, 3, 4, 5, 6])
    rmse = calculate_rmse(y_true, y_pred)
    assert rmse == 1.0


def test_calculate_r2_score():
    """Test R2 score calculation."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    
    r2 = calculate_r2_score(y_true, y_pred)
    assert r2 == 1.0
    
    # Predicting the mean should give R2 = 0
    y_pred = np.full_like(y_true, np.mean(y_true), dtype=float)
    r2 = calculate_r2_score(y_true, y_pred)
    assert abs(r2) < 1e-10
