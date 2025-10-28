"""
Tests for the SymbolicRegressor class.
"""

import numpy as np
import pytest
from fexdm.core import SymbolicRegressor


def test_symbolic_regressor_initialization():
    """Test regressor initialization."""
    regressor = SymbolicRegressor(max_complexity=3)
    assert regressor.max_complexity == 3
    assert regressor.best_expression_ is None


def test_linear_function_identification():
    """Test identification of a simple linear function."""
    # Generate data from y = 2*x + 1
    X = np.linspace(-5, 5, 100)
    y = 2 * X + 1
    
    regressor = SymbolicRegressor(
        basis_functions=['1', 'x'],
        max_complexity=2
    )
    regressor.fit(X, y)
    
    # Predict
    y_pred = regressor.predict(X)
    
    # Check accuracy
    mse = np.mean((y - y_pred) ** 2)
    assert mse < 1e-10


def test_quadratic_function_identification():
    """Test identification of a quadratic function."""
    # Generate data from y = x^2 - 2*x + 3
    X = np.linspace(-5, 5, 100)
    y = X**2 - 2*X + 3
    
    regressor = SymbolicRegressor(
        basis_functions=['1', 'x', 'x**2'],
        max_complexity=3
    )
    regressor.fit(X, y)
    
    y_pred = regressor.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    assert mse < 1e-6


def test_get_expression():
    """Test getting the best expression."""
    X = np.linspace(-5, 5, 100)
    y = 3 * X + 2
    
    regressor = SymbolicRegressor(
        basis_functions=['1', 'x'],
        max_complexity=2
    )
    regressor.fit(X, y)
    
    expr = regressor.get_expression()
    assert expr is not None
    assert hasattr(expr, 'formula')
    assert hasattr(expr, 'error')


def test_unfitted_predict_raises_error():
    """Test that predict raises error when not fitted."""
    regressor = SymbolicRegressor()
    X = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="not fitted"):
        regressor.predict(X)


def test_noisy_data():
    """Test regressor on noisy data."""
    # Generate noisy linear data
    np.random.seed(42)
    X = np.linspace(-5, 5, 100)
    y = 2 * X + 1 + 0.1 * np.random.randn(100)
    
    regressor = SymbolicRegressor(
        basis_functions=['1', 'x'],
        max_complexity=2
    )
    regressor.fit(X, y)
    
    y_pred = regressor.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    
    # Should be close but not perfect due to noise
    assert mse < 0.1
