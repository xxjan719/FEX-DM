"""
Stochastic Dynamics Identifier using Finite Expression Methods.

This module implements the main algorithm for identifying drift and diffusion
functions from time series data of stochastic dynamical systems.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from .symbolic_regressor import SymbolicRegressor
from ..models.sde import StochasticDifferentialEquation


class StochasticDynamicsIdentifier:
    """
    Identifies stochastic dynamics from time series data.
    
    This class uses finite expression methods to discover the drift and
    diffusion functions of a stochastic differential equation from observed
    time series data.
    """
    
    def __init__(
        self,
        max_complexity: int = 3,
        dt: Optional[float] = None,
        basis_functions: Optional[list] = None
    ):
        """
        Initialize the identifier.
        
        Parameters
        ----------
        max_complexity : int, optional
            Maximum complexity for symbolic expressions (default: 3)
        dt : float, optional
            Time step between observations (default: auto-detect)
        basis_functions : list, optional
            Basis functions for symbolic regression
        """
        self.max_complexity = max_complexity
        self.dt = dt
        self.basis_functions = basis_functions
        
        self.drift_regressor_ = None
        self.diffusion_regressor_ = None
        self.identified_sde_ = None
    
    def _estimate_drift_diffusion(
        self,
        X: np.ndarray,
        t: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate local drift and diffusion from time series data.
        
        Parameters
        ----------
        X : np.ndarray
            Time series data
        t : np.ndarray, optional
            Time points (if None, assumes uniform spacing)
        
        Returns
        -------
        X_mid : np.ndarray
            Midpoint values
        drift_estimate : np.ndarray
            Local drift estimates
        diffusion_estimate : np.ndarray
            Local diffusion estimates
        """
        # Determine time step
        if t is not None:
            dt = np.mean(np.diff(t))
        elif self.dt is not None:
            dt = self.dt
        else:
            dt = 1.0
        
        # Compute increments
        dX = np.diff(X, axis=0)
        X_mid = (X[:-1] + X[1:]) / 2
        
        # Drift estimate: E[dX/dt]
        drift_estimate = dX / dt
        
        # Diffusion estimate: sqrt(Var[dX]/dt)
        # For simplicity, using local estimation
        window_size = min(10, len(dX) // 10 + 1)
        diffusion_estimate = np.zeros_like(drift_estimate)
        
        for i in range(len(drift_estimate)):
            start = max(0, i - window_size // 2)
            end = min(len(dX), i + window_size // 2 + 1)
            local_dX = dX[start:end]
            diffusion_estimate[i] = np.sqrt(np.var(local_dX) / dt)
        
        return X_mid, drift_estimate, diffusion_estimate
    
    def fit(
        self,
        X: np.ndarray,
        t: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> 'StochasticDynamicsIdentifier':
        """
        Identify drift and diffusion functions from time series data.
        
        Parameters
        ----------
        X : np.ndarray
            Time series data (can be 1D or 2D for multiple trajectories)
        t : np.ndarray, optional
            Time points
        verbose : bool, optional
            Print progress information
        
        Returns
        -------
        self
            The fitted identifier
        """
        X = np.asarray(X)
        
        # Handle multiple trajectories
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if verbose:
            print("Estimating local drift and diffusion...")
        
        # Estimate drift and diffusion
        X_mid, drift_est, diffusion_est = self._estimate_drift_diffusion(X, t)
        
        if verbose:
            print(f"Data points: {len(X_mid)}")
            print("Fitting drift function...")
        
        # Fit drift function using symbolic regression
        self.drift_regressor_ = SymbolicRegressor(
            basis_functions=self.basis_functions,
            max_complexity=self.max_complexity
        )
        self.drift_regressor_.fit(X_mid.ravel(), drift_est.ravel(), verbose=verbose)
        
        if verbose:
            print("\nFitting diffusion function...")
        
        # Fit diffusion function using symbolic regression
        self.diffusion_regressor_ = SymbolicRegressor(
            basis_functions=self.basis_functions,
            max_complexity=self.max_complexity
        )
        self.diffusion_regressor_.fit(X_mid.ravel(), diffusion_est.ravel(), verbose=verbose)
        
        # Create identified SDE
        drift_func = lambda x: self.drift_regressor_.predict(x)
        diffusion_func = lambda x: np.abs(self.diffusion_regressor_.predict(x))
        
        self.identified_sde_ = StochasticDifferentialEquation(
            drift=drift_func,
            diffusion=diffusion_func,
            dimension=1
        )
        
        if verbose:
            print("\n" + "="*60)
            print("Identification Complete!")
            print("="*60)
            print(f"Drift function: {self.drift_regressor_.best_expression_.formula}")
            print(f"Diffusion function: {self.diffusion_regressor_.best_expression_.formula}")
        
        return self
    
    def get_drift_expression(self) -> str:
        """Get the discovered drift function as a string."""
        if self.drift_regressor_ is None:
            raise ValueError("Model not fitted yet.")
        return self.drift_regressor_.best_expression_.formula
    
    def get_diffusion_expression(self) -> str:
        """Get the discovered diffusion function as a string."""
        if self.diffusion_regressor_ is None:
            raise ValueError("Model not fitted yet.")
        return self.diffusion_regressor_.best_expression_.formula
    
    def get_sde(self) -> StochasticDifferentialEquation:
        """Get the identified SDE model."""
        if self.identified_sde_ is None:
            raise ValueError("Model not fitted yet.")
        return self.identified_sde_
    
    def predict_drift(self, X: np.ndarray) -> np.ndarray:
        """Predict drift at given points."""
        if self.drift_regressor_ is None:
            raise ValueError("Model not fitted yet.")
        return self.drift_regressor_.predict(X)
    
    def predict_diffusion(self, X: np.ndarray) -> np.ndarray:
        """Predict diffusion at given points."""
        if self.diffusion_regressor_ is None:
            raise ValueError("Model not fitted yet.")
        return np.abs(self.diffusion_regressor_.predict(X))
    
    def __repr__(self) -> str:
        if self.drift_regressor_ is not None:
            return f"StochasticDynamicsIdentifier(fitted=True)"
        return "StochasticDynamicsIdentifier(fitted=False)"
