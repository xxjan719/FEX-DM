"""
Tests for the StochasticDynamicsIdentifier class.
"""

import numpy as np
import pytest
from fexdm import StochasticDynamicsIdentifier
from fexdm.utils import generate_ornstein_uhlenbeck


def test_identifier_initialization():
    """Test identifier initialization."""
    identifier = StochasticDynamicsIdentifier(max_complexity=3, dt=0.01)
    assert identifier.max_complexity == 3
    assert identifier.dt == 0.01


def test_identifier_fit_returns_self():
    """Test that fit returns self for method chaining."""
    t, X = generate_ornstein_uhlenbeck(
        theta=1.0, mu=0.0, sigma=0.5,
        t_span=(0, 5), dt=0.01, seed=42
    )
    
    identifier = StochasticDynamicsIdentifier(dt=0.01)
    result = identifier.fit(X, t)
    
    assert result is identifier


def test_ornstein_uhlenbeck_identification():
    """Test identification of Ornstein-Uhlenbeck process."""
    # Generate data
    theta = 1.0
    mu = 0.0
    sigma = 0.5
    
    t, X = generate_ornstein_uhlenbeck(
        theta=theta, mu=mu, sigma=sigma,
        x0=2.0, t_span=(0, 20), dt=0.01, seed=42
    )
    
    # Identify
    identifier = StochasticDynamicsIdentifier(max_complexity=3, dt=0.01)
    identifier.fit(X, t)
    
    # Check that we got expressions
    drift_expr = identifier.get_drift_expression()
    diffusion_expr = identifier.get_diffusion_expression()
    
    assert isinstance(drift_expr, str)
    assert isinstance(diffusion_expr, str)
    assert len(drift_expr) > 0
    assert len(diffusion_expr) > 0


def test_identifier_predict_methods():
    """Test predict_drift and predict_diffusion methods."""
    t, X = generate_ornstein_uhlenbeck(
        theta=1.0, mu=0.0, sigma=0.5,
        t_span=(0, 10), dt=0.01, seed=42
    )
    
    identifier = StochasticDynamicsIdentifier(dt=0.01)
    identifier.fit(X, t)
    
    x_test = np.array([0.0, 1.0, -1.0])
    drift_pred = identifier.predict_drift(x_test)
    diffusion_pred = identifier.predict_diffusion(x_test)
    
    assert drift_pred.shape == x_test.shape
    assert diffusion_pred.shape == x_test.shape


def test_get_sde():
    """Test getting the identified SDE model."""
    t, X = generate_ornstein_uhlenbeck(
        theta=1.0, mu=0.0, sigma=0.5,
        t_span=(0, 10), dt=0.01, seed=42
    )
    
    identifier = StochasticDynamicsIdentifier(dt=0.01)
    identifier.fit(X, t)
    
    sde = identifier.get_sde()
    
    assert hasattr(sde, 'simulate')
    assert hasattr(sde, 'drift')
    assert hasattr(sde, 'diffusion')


def test_unfitted_methods_raise_errors():
    """Test that methods raise errors when not fitted."""
    identifier = StochasticDynamicsIdentifier()
    
    with pytest.raises(ValueError, match="not fitted"):
        identifier.get_drift_expression()
    
    with pytest.raises(ValueError, match="not fitted"):
        identifier.get_diffusion_expression()
    
    with pytest.raises(ValueError, match="not fitted"):
        identifier.get_sde()
    
    with pytest.raises(ValueError, match="not fitted"):
        identifier.predict_drift(np.array([1, 2, 3]))


def test_2d_input_handling():
    """Test that identifier handles 2D input arrays."""
    t, X = generate_ornstein_uhlenbeck(
        theta=1.0, mu=0.0, sigma=0.5,
        t_span=(0, 5), dt=0.01, seed=42
    )
    
    # Reshape to 2D
    X_2d = X.reshape(-1, 1)
    
    identifier = StochasticDynamicsIdentifier(dt=0.01)
    identifier.fit(X_2d, t)
    
    # Should not raise error
    drift_expr = identifier.get_drift_expression()
    assert isinstance(drift_expr, str)
