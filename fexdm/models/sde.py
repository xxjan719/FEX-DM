"""
Stochastic Differential Equation (SDE) model class.

This module provides a class for representing and simulating stochastic
differential equations of the form:
    dX = f(X)dt + g(X)dW
where f is the drift function and g is the diffusion function.
"""

import numpy as np
from typing import Callable, Optional, Tuple


class StochasticDifferentialEquation:
    """
    A class representing a stochastic differential equation (SDE).
    
    The SDE is of the form:
        dX = f(X)dt + g(X)dW
    
    where:
        - f(X) is the drift function
        - g(X) is the diffusion function
        - dW is a Wiener process increment
    """
    
    def __init__(
        self,
        drift: Callable[[np.ndarray], np.ndarray],
        diffusion: Callable[[np.ndarray], np.ndarray],
        dimension: int = 1
    ):
        """
        Initialize the SDE.
        
        Parameters
        ----------
        drift : callable
            The drift function f(X)
        diffusion : callable
            The diffusion function g(X)
        dimension : int, optional
            The dimension of the state space (default: 1)
        """
        self.drift = drift
        self.diffusion = diffusion
        self.dimension = dimension
    
    def simulate(
        self,
        x0: np.ndarray,
        t_span: Tuple[float, float],
        dt: float = 0.01,
        n_trajectories: int = 1,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate trajectories of the SDE using Euler-Maruyama method.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial condition(s)
        t_span : tuple
            Time span (t_start, t_end)
        dt : float, optional
            Time step (default: 0.01)
        n_trajectories : int, optional
            Number of trajectories to simulate (default: 1)
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        t : np.ndarray
            Time points
        X : np.ndarray
            Simulated trajectories with shape (n_steps, n_trajectories, dimension)
        """
        if seed is not None:
            np.random.seed(seed)
        
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt) + 1
        t = np.linspace(t_start, t_end, n_steps)
        
        # Initialize trajectory array
        if self.dimension == 1:
            X = np.zeros((n_steps, n_trajectories))
            X[0] = x0
        else:
            X = np.zeros((n_steps, n_trajectories, self.dimension))
            X[0] = x0
        
        # Euler-Maruyama method
        sqrt_dt = np.sqrt(dt)
        for i in range(1, n_steps):
            dW = np.random.randn(n_trajectories, self.dimension if self.dimension > 1 else 1)
            if self.dimension == 1:
                dW = dW.squeeze()
            
            x_current = X[i-1]
            drift_term = self.drift(x_current) * dt
            diffusion_term = self.diffusion(x_current) * sqrt_dt * dW
            
            X[i] = x_current + drift_term + diffusion_term
        
        return t, X
    
    def __repr__(self) -> str:
        return f"StochasticDifferentialEquation(dimension={self.dimension})"
