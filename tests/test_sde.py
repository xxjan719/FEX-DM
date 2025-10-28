"""
Tests for the StochasticDifferentialEquation class.
"""

import numpy as np
import pytest
from fexdm.models import StochasticDifferentialEquation


def test_sde_initialization():
    """Test SDE initialization."""
    drift = lambda x: -x
    diffusion = lambda x: 0.5 * np.ones_like(x)
    
    sde = StochasticDifferentialEquation(drift, diffusion, dimension=1)
    
    assert sde.dimension == 1
    assert callable(sde.drift)
    assert callable(sde.diffusion)


def test_sde_simulation_shape():
    """Test that simulation returns correct shape."""
    drift = lambda x: -x
    diffusion = lambda x: 0.5 * np.ones_like(x)
    sde = StochasticDifferentialEquation(drift, diffusion, dimension=1)
    
    t, X = sde.simulate(
        x0=1.0,
        t_span=(0, 1),
        dt=0.01,
        n_trajectories=3,
        seed=42
    )
    
    assert len(t) == 101  # (1-0)/0.01 + 1
    assert X.shape == (101, 3)


def test_sde_simulation_reproducibility():
    """Test that simulations are reproducible with seed."""
    drift = lambda x: -x
    diffusion = lambda x: 0.5 * np.ones_like(x)
    sde = StochasticDifferentialEquation(drift, diffusion, dimension=1)
    
    t1, X1 = sde.simulate(x0=1.0, t_span=(0, 1), dt=0.01, seed=42)
    t2, X2 = sde.simulate(x0=1.0, t_span=(0, 1), dt=0.01, seed=42)
    
    np.testing.assert_array_almost_equal(X1, X2)


def test_sde_initial_condition():
    """Test that simulation starts at correct initial condition."""
    drift = lambda x: -x
    diffusion = lambda x: 0.5 * np.ones_like(x)
    sde = StochasticDifferentialEquation(drift, diffusion, dimension=1)
    
    x0 = 2.5
    t, X = sde.simulate(x0=x0, t_span=(0, 1), dt=0.01, seed=42)
    
    assert X[0] == x0


def test_ornstein_uhlenbeck_mean_reversion():
    """Test that Ornstein-Uhlenbeck process shows mean reversion."""
    theta = 2.0
    mu = 0.0
    sigma = 0.1
    
    drift = lambda x: theta * (mu - x)
    diffusion = lambda x: sigma * np.ones_like(x)
    sde = StochasticDifferentialEquation(drift, diffusion, dimension=1)
    
    # Start far from mean
    x0 = 5.0
    t, X = sde.simulate(x0=x0, t_span=(0, 5), dt=0.01, n_trajectories=10, seed=42)
    
    # Check that process moves toward mean
    final_mean = np.mean(X[-1])
    assert abs(final_mean) < abs(x0)  # Closer to mean than initial
